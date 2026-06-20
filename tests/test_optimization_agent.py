"""Testes do Agente de Otimização de Hiperparâmetros (Optuna)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, StratifiedKFold

from src.agents.feature_agent import FeatureAgent
from src.agents.optimization_agent import OptimizationAgent


def _make_agent() -> OptimizationAgent:
    return OptimizationAgent(event_sink=None)


def _folds(df, y=None, n_splits=3, seed=0):
    if y is not None:
        sp = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [(tr, te) for tr, te in sp.split(np.arange(len(df)), y)]
    sp = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in sp.split(np.arange(len(df)))]


def _pipeline(df, target):
    return FeatureAgent().run({"dataframe": df, "target_column": target}).pipeline


@pytest.fixture
def clf_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 90
    x1 = rng.normal(size=n)
    label = (x1 + rng.normal(scale=0.3, size=n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": rng.normal(size=n), "label": label})


@pytest.fixture
def reg_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 90
    x1 = rng.normal(size=n)
    y = 3.0 * x1 + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"x1": x1, "x2": rng.normal(size=n), "price": y})


# ---------------------------------------------------------------------------
# Classificação
# ---------------------------------------------------------------------------

class TestClassification:
    def _run(self, df, **ctx):
        return _make_agent().run({
            "dataframe": df,
            "task_type": "classification",
            "target_column": "label",
            "primary_metric": "macro_f1",
            "folds": _folds(df, df["label"].to_numpy()),
            "pipeline": _pipeline(df, "label"),
            "include": ["logistic_regression", "decision_tree"],
            "n_trials": 5,
            **ctx,
        })

    def test_returns_best_params_per_model(self, clf_df):
        out = self._run(clf_df).output
        names = {r["model_name"] for r in out["results"]}
        assert names == {"logistic_regression", "decision_tree"}
        lr = next(r for r in out["results"] if r["model_name"] == "logistic_regression")
        assert lr["status"] == "ok"
        assert "C" in lr["tuned_params"]            # hiperparâmetro buscado
        assert "C" in lr["best_params"]             # mesclado nos defaults
        assert lr["best_params"]["max_iter"] == 1000  # default preservado

    def test_best_value_within_metric_range(self, clf_df):
        out = self._run(clf_df).output
        for r in out["results"]:
            assert 0.0 <= r["best_value"] <= 1.0

    def test_direction_maximize_for_f1(self, clf_df):
        out = self._run(clf_df).output
        assert out["direction"] == "maximize"
        means = [r["best_value"] for r in out["ranking"]]
        assert means == sorted(means, reverse=True)

    def test_n_trials_respected(self, clf_df):
        out = self._run(clf_df, n_trials=4).output
        lr = next(r for r in out["results"] if r["model_name"] == "logistic_regression")
        assert lr["n_trials_completed"] + lr.get("n_trials_pruned", 0) == 4

    def test_reproducible(self, clf_df):
        a = self._run(clf_df).output["results"]
        b = self._run(clf_df).output["results"]
        a_lr = next(r for r in a if r["model_name"] == "logistic_regression")
        b_lr = next(r for r in b if r["model_name"] == "logistic_regression")
        assert a_lr["best_params"] == b_lr["best_params"]
        assert a_lr["best_value"] == b_lr["best_value"]

    def test_output_is_json_serializable(self, clf_df):
        json.dumps(self._run(clf_df).output)


# ---------------------------------------------------------------------------
# Regressão
# ---------------------------------------------------------------------------

class TestRegression:
    def test_direction_minimize_for_rmse(self, reg_df):
        out = _make_agent().run({
            "dataframe": reg_df,
            "task_type": "regression",
            "target_column": "price",
            "primary_metric": "rmse",
            "folds": _folds(reg_df),
            "pipeline": _pipeline(reg_df, "price"),
            "include": ["elastic_net", "decision_tree_regressor"],
            "n_trials": 5,
        }).output
        assert out["direction"] == "minimize"
        means = [r["best_value"] for r in out["ranking"]]
        assert means == sorted(means)  # menor RMSE primeiro


# ---------------------------------------------------------------------------
# Budget, seleção e robustez
# ---------------------------------------------------------------------------

class TestBudgetAndSelection:
    def test_model_without_search_space_is_skipped(self, reg_df):
        # linear_regression não tem hiperparâmetros tunáveis => sem search_space
        out = _make_agent().run({
            "dataframe": reg_df,
            "task_type": "regression",
            "target_column": "price",
            "primary_metric": "rmse",
            "folds": _folds(reg_df),
            "pipeline": _pipeline(reg_df, "price"),
            "include": ["linear_regression", "elastic_net"],
            "n_trials": 3,
        }).output
        skipped = {s["model_name"] for s in out["skipped_models"]}
        assert "linear_regression" in skipped
        assert {r["model_name"] for r in out["results"]} == {"elastic_net"}

    def test_global_timeout_skips_remaining(self, clf_df):
        out = _make_agent().run({
            "dataframe": clf_df,
            "task_type": "classification",
            "target_column": "label",
            "primary_metric": "macro_f1",
            "folds": _folds(clf_df, clf_df["label"].to_numpy()),
            "pipeline": _pipeline(clf_df, "label"),
            "include": ["logistic_regression", "random_forest", "svm", "knn", "mlp"],
            "n_trials": 50,
            "timeout_seconds": 0.0,  # orçamento esgotado de imediato
        }).output
        # com budget zero, nenhum modelo chega a otimizar
        assert len(out["skipped_models"]) == 5
        assert out["results"] == []

    def test_custom_search_space_override(self, clf_df):
        out = _make_agent().run({
            "dataframe": clf_df,
            "task_type": "classification",
            "target_column": "label",
            "primary_metric": "macro_f1",
            "folds": _folds(clf_df, clf_df["label"].to_numpy()),
            "pipeline": _pipeline(clf_df, "label"),
            "include": ["knn"],
            "n_trials": 4,
            "search_spaces": {"knn": {"n_neighbors": {"type": "int", "low": 1, "high": 5}}},
        }).output
        knn = out["results"][0]
        assert knn["search_space"] == {"n_neighbors": {"type": "int", "low": 1, "high": 5}}
        assert 1 <= knn["best_params"]["n_neighbors"] <= 5

    def test_missing_primary_metric_raises(self, clf_df):
        with pytest.raises(ValueError, match="primary_metric"):
            _make_agent().run({
                "dataframe": clf_df, "task_type": "classification", "target_column": "label",
                "folds": _folds(clf_df, clf_df["label"].to_numpy()),
            })

    def test_missing_folds_raises(self, clf_df):
        with pytest.raises(ValueError, match="folds"):
            _make_agent().run({
                "dataframe": clf_df, "task_type": "classification",
                "target_column": "label", "primary_metric": "macro_f1",
            })


class TestEvent:
    def test_event_emitted_via_call(self, clf_df):
        events: list[dict] = []

        class _Sink:
            def record_event(self, **kwargs):
                events.append(kwargs)

        agent = OptimizationAgent(event_sink=_Sink())
        agent({
            "experiment_id": "exp-1",
            "dataframe": clf_df,
            "task_type": "classification",
            "target_column": "label",
            "primary_metric": "macro_f1",
            "folds": _folds(clf_df, clf_df["label"].to_numpy()),
            "pipeline": _pipeline(clf_df, "label"),
            "include": ["logistic_regression"],
            "n_trials": 3,
        })
        assert len(events) == 1
        assert events[0]["event_type"] == "hyperparameter_optimization"
        assert events[0]["agent_name"] == "Agente de Otimização"
