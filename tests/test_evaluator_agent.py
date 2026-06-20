"""Testes do Agente de Avaliação e Seleção (RF11)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, StratifiedKFold

from src.agents.evaluator_agent import EvaluatorAgent
from src.agents.feature_agent import FeatureAgent
from src.agents.trainer_agent import TrainerAgent


def _make_agent() -> EvaluatorAgent:
    return EvaluatorAgent(event_sink=None)


def _model_result(name, values, fit_seconds=0.01, extra=None):
    """Constrói um resultado no formato do Model Zoo."""
    fold_metrics = []
    for v in values:
        metrics = {"macro_f1": v}
        if extra:
            metrics.update({k: vals for k, vals in extra.items()})
        fold_metrics.append({"fold": len(fold_metrics), "metrics": metrics,
                             "fit_seconds": fit_seconds})
    return {
        "model_name": name,
        "model_family": "sklearn.test",
        "estimator": "sklearn.linear_model:LogisticRegression",
        "hyperparameters": {"max_iter": 1000},
        "status": "ok",
        "fold_metrics": fold_metrics,
    }


# ---------------------------------------------------------------------------
# Ranking e estatísticas (RF11)
# ---------------------------------------------------------------------------

class TestRanking:
    def test_ranks_by_primary_metric(self):
        results = [
            _model_result("weak", [0.60, 0.62, 0.61]),
            _model_result("strong", [0.80, 0.82, 0.81]),
        ]
        out = _make_agent().run({
            "model_results": results, "task_type": "classification",
            "primary_metric": "macro_f1",
        }).output
        assert out["ranking"][0]["model_name"] == "strong"
        assert out["ranking"][0]["rank"] == 1
        assert out["best_model"] == "strong"

    def test_includes_mean_std_ci_interval_time(self):
        out = _make_agent().run({
            "model_results": [_model_result("m", [0.70, 0.80, 0.90])],
            "task_type": "classification", "primary_metric": "macro_f1",
        }).output
        e = out["ranking"][0]
        assert e["primary_mean"] == pytest.approx(0.8, abs=1e-6)
        assert e["primary_std"] > 0
        assert e["ci_low"] < e["primary_mean"] < e["ci_high"]
        assert e["primary_min"] == 0.7 and e["primary_max"] == 0.9
        assert e["n_folds"] == 3
        assert e["mean_fit_seconds"] is not None

    def test_rmse_ranks_ascending(self):
        results = [
            _model_result("a", [2.0, 2.1, 1.9]),
            _model_result("b", [1.0, 1.1, 0.9]),
        ]
        # renomeia a métrica para rmse em cada fold
        for r in results:
            for fm in r["fold_metrics"]:
                fm["metrics"]["rmse"] = fm["metrics"].pop("macro_f1")
        out = _make_agent().run({
            "model_results": results, "task_type": "regression", "primary_metric": "rmse",
        }).output
        assert out["direction"] == "minimize"
        assert out["best_model"] == "b"  # menor RMSE

    def test_secondary_metrics_aggregated(self):
        out = _make_agent().run({
            "model_results": [_model_result("m", [0.8, 0.8, 0.8], extra={"roc_auc": 0.9})],
            "task_type": "classification", "primary_metric": "macro_f1",
        }).output
        assert out["ranking"][0]["secondary_means"]["roc_auc"] == pytest.approx(0.9)

    def test_output_is_json_serializable(self):
        out = _make_agent().run({
            "model_results": [_model_result("m", [0.8, 0.8, 0.8])],
            "task_type": "classification", "primary_metric": "macro_f1",
        }).output
        json.dumps(out)


# ---------------------------------------------------------------------------
# Seleção por critério claro (anti cherry-picking)
# ---------------------------------------------------------------------------

class TestSelection:
    def test_best_mean_tie_breaks_by_lower_std(self):
        results = [
            _model_result("stable", [0.80, 0.80, 0.80]),    # mesma média, menor desvio
            _model_result("noisy", [0.70, 0.80, 0.90]),
        ]
        out = _make_agent().run({
            "model_results": results, "task_type": "classification",
            "primary_metric": "macro_f1", "selection_rule": "best_mean",
        }).output
        assert out["best_model"] == "stable"

    def test_one_se_prefers_faster_within_range(self):
        # 'fast' está dentro de 1 SE de 'best' e é mais rápido => deve ser escolhido
        results = [
            _model_result("best", [0.80, 0.82, 0.84], fit_seconds=1.0),  # média 0.82, SE ~0.0115
            _model_result("fast", [0.81, 0.81, 0.81], fit_seconds=0.01),  # 0.81 ≥ 0.82 - SE
        ]
        out = _make_agent().run({
            "model_results": results, "task_type": "classification",
            "primary_metric": "macro_f1", "selection_rule": "one_se",
        }).output
        assert out["best_model"] == "fast"
        assert "one_se" in out["selection_reason"].lower() or "one_se" == out["selection_rule"]

    def test_overlapping_ci_warns_significance(self):
        results = [
            _model_result("a", [0.78, 0.80, 0.82]),
            _model_result("b", [0.77, 0.79, 0.81]),
        ]
        out = _make_agent().run({
            "model_results": results, "task_type": "classification",
            "primary_metric": "macro_f1",
        }).output
        assert "sobrep" in out["significance_note"].lower()

    def test_disjoint_ci_significant(self):
        results = [
            _model_result("strong", [0.90, 0.91, 0.92]),
            _model_result("weak", [0.50, 0.51, 0.52]),
        ]
        out = _make_agent().run({
            "model_results": results, "task_type": "classification",
            "primary_metric": "macro_f1",
        }).output
        assert "significativa" in out["significance_note"].lower()

    def test_invalid_rule_raises(self):
        with pytest.raises(ValueError, match="selection_rule"):
            _make_agent().run({
                "model_results": [_model_result("m", [0.8])],
                "task_type": "classification", "primary_metric": "macro_f1",
                "selection_rule": "cherry_pick",
            })


# ---------------------------------------------------------------------------
# Critério de sucesso e validações
# ---------------------------------------------------------------------------

class TestThresholdAndValidation:
    def test_meets_success_threshold(self):
        out = _make_agent().run({
            "model_results": [_model_result("m", [0.85, 0.86, 0.84])],
            "task_type": "classification", "primary_metric": "macro_f1",
            "success_threshold": 0.80,
        }).output
        assert out["meets_success_threshold"] is True

    def test_fails_success_threshold(self):
        out = _make_agent().run({
            "model_results": [_model_result("m", [0.60, 0.61, 0.59])],
            "task_type": "classification", "primary_metric": "macro_f1",
            "success_threshold": 0.80,
        }).output
        assert out["meets_success_threshold"] is False

    def test_reads_results_from_training_dict(self):
        out = _make_agent().run({
            "training": {"results": [_model_result("m", [0.8, 0.8, 0.8])]},
            "task_type": "classification", "primary_metric": "macro_f1",
        }).output
        assert out["best_model"] == "m"

    def test_missing_results_raises(self):
        with pytest.raises(ValueError, match="model_results"):
            _make_agent().run({"task_type": "classification", "primary_metric": "macro_f1"})

    def test_missing_primary_metric_raises(self):
        with pytest.raises(ValueError, match="primary_metric"):
            _make_agent().run({"model_results": [_model_result("m", [0.8])]})

    def test_no_model_with_primary_raises(self):
        results = [_model_result("m", [0.8])]
        with pytest.raises(ValueError, match="métrica primária"):
            _make_agent().run({
                "model_results": results, "task_type": "classification",
                "primary_metric": "roc_auc",  # ausente nos fold_metrics
            })


# ---------------------------------------------------------------------------
# Integração real Trainer -> Evaluator (com diagnósticos)
# ---------------------------------------------------------------------------

class TestIntegrationAndDiagnostics:
    @pytest.fixture
    def clf_df(self):
        rng = np.random.default_rng(0)
        n = 90
        x1 = rng.normal(size=n)
        label = (x1 + rng.normal(scale=0.3, size=n) > 0).astype(int)
        return pd.DataFrame({"x1": x1, "x2": rng.normal(size=n), "label": label})

    def test_confusion_matrix_diagnostic(self, clf_df):
        sp = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        folds = [(tr, te) for tr, te in sp.split(np.arange(len(clf_df)), clf_df["label"])]
        pipeline = FeatureAgent().run({"dataframe": clf_df, "target_column": "label"}).pipeline
        training = TrainerAgent().run({
            "dataframe": clf_df, "task_type": "classification", "target_column": "label",
            "primary_metric": "macro_f1", "folds": folds, "pipeline": pipeline,
            "include": ["logistic_regression", "decision_tree"],
        }).output

        result = _make_agent().run({
            "model_results": training["results"],
            "task_type": "classification", "primary_metric": "macro_f1",
            "dataframe": clf_df, "folds": folds, "pipeline": pipeline,
            "target_column": "label",
        })
        diag = result.output["diagnostics"]
        assert diag["type"] == "confusion_matrix"
        total = sum(sum(row) for row in diag["matrix"])
        assert total == len(clf_df)  # predições out-of-fold para toda a base

    def test_regression_residual_diagnostic(self):
        rng = np.random.default_rng(1)
        n = 90
        x1 = rng.normal(size=n)
        df = pd.DataFrame({"x1": x1, "x2": rng.normal(size=n), "price": 3 * x1 + rng.normal(scale=0.5, size=n)})
        folds = [(tr, te) for tr, te in KFold(3, shuffle=True, random_state=0).split(np.arange(n))]
        pipeline = FeatureAgent().run({"dataframe": df, "target_column": "price"}).pipeline
        training = TrainerAgent().run({
            "dataframe": df, "task_type": "regression", "target_column": "price",
            "primary_metric": "rmse", "folds": folds, "pipeline": pipeline,
            "include": ["linear_regression"],
        }).output
        out = _make_agent().run({
            "model_results": training["results"],
            "task_type": "regression", "primary_metric": "rmse",
            "dataframe": df, "folds": folds, "pipeline": pipeline, "target_column": "price",
        }).output
        assert out["diagnostics"]["type"] == "regression_residuals"
        assert "max_abs_error" in out["diagnostics"]


class TestEvent:
    def test_event_emitted_via_call(self):
        events: list[dict] = []

        class _Sink:
            def record_event(self, **kwargs):
                events.append(kwargs)

        agent = EvaluatorAgent(event_sink=_Sink())
        agent({
            "experiment_id": "exp-1",
            "model_results": [_model_result("m", [0.8, 0.8, 0.8])],
            "task_type": "classification", "primary_metric": "macro_f1",
        })
        assert len(events) == 1
        assert events[0]["event_type"] == "model_selection"
        assert events[0]["agent_name"] == "Agente de Avaliação"
