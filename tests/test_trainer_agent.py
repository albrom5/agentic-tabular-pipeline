"""Testes do Agente de Model Zoo / Treinamento Cruzado (RF09)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, StratifiedKFold

from src.agents.feature_agent import FeatureAgent
from src.agents.trainer_agent import TrainerAgent


def _make_agent() -> TrainerAgent:
    return TrainerAgent(event_sink=None)


def _folds(df, y=None, n_splits=3, seed=0):
    if y is not None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [(tr, te) for tr, te in splitter.split(np.arange(len(df)), y)]
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in splitter.split(np.arange(len(df)))]


def _pipeline(df, target):
    return FeatureAgent().run({"dataframe": df, "target_column": target}).pipeline


@pytest.fixture
def clf_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 90
    x1 = rng.normal(size=n)
    # alvo correlacionado com x1 para que os modelos aprendam algo
    label = (x1 + rng.normal(scale=0.3, size=n) > 0).astype(int)
    return pd.DataFrame({
        "x1": x1,
        "x2": rng.normal(size=n),
        "city": rng.choice(["SP", "RJ"], size=n),
        "label": label,
    })


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
            **ctx,
        })

    def test_runs_selected_models(self, clf_df):
        out = self._run(clf_df).output
        names = {r["model_name"] for r in out["results"]}
        assert names == {"logistic_regression", "decision_tree"}
        assert all(r["status"] == "ok" for r in out["results"])

    def test_records_per_fold_metrics(self, clf_df):
        out = self._run(clf_df).output
        r = out["results"][0]
        assert len(r["fold_metrics"]) == 3
        assert "macro_f1" in r["fold_metrics"][0]["metrics"]
        assert "fit_seconds" in r["fold_metrics"][0]
        assert r["random_seed"] == 42

    def test_aggregates_mean_and_std(self, clf_df):
        out = self._run(clf_df).output
        r = out["results"][0]
        assert "macro_f1" in r["metrics_mean"]
        assert "macro_f1" in r["metrics_std"]
        assert 0.0 <= r["metrics_mean"]["macro_f1"] <= 1.0

    def test_records_hyperparameters_and_seed(self, clf_df):
        out = self._run(clf_df).output
        lr = next(r for r in out["results"] if r["model_name"] == "logistic_regression")
        assert lr["hyperparameters"]["max_iter"] == 1000
        assert lr["hyperparameters"]["random_state"] == 42  # seed injetada

    def test_ranking_by_primary_metric(self, clf_df):
        out = self._run(clf_df).output
        assert out["ranking"][0]["rank"] == 1
        # ranking decrescente para macro_f1 (maior é melhor)
        means = [r["primary_metric_mean"] for r in out["ranking"]]
        assert means == sorted(means, reverse=True)

    def test_learns_signal(self, clf_df):
        # com sinal claro, a acurácia média deve superar o acaso
        out = self._run(clf_df).output
        lr = next(r for r in out["results"] if r["model_name"] == "logistic_regression")
        assert lr["metrics_mean"]["accuracy"] > 0.7

    def test_roc_auc_present_for_binary(self, clf_df):
        out = self._run(clf_df).output
        lr = next(r for r in out["results"] if r["model_name"] == "logistic_regression")
        assert "roc_auc" in lr["metrics_mean"]

    def test_output_is_json_serializable(self, clf_df):
        json.dumps(self._run(clf_df).output)


# ---------------------------------------------------------------------------
# Regressão
# ---------------------------------------------------------------------------

class TestRegression:
    def _run(self, df, **ctx):
        return _make_agent().run({
            "dataframe": df,
            "task_type": "regression",
            "target_column": "price",
            "primary_metric": "rmse",
            "folds": _folds(df),
            "pipeline": _pipeline(df, "price"),
            "include": ["linear_regression", "decision_tree_regressor"],
            **ctx,
        })

    def test_regression_metrics(self, reg_df):
        out = self._run(reg_df).output
        r = out["results"][0]
        for key in ("rmse", "mae", "r2", "mse"):
            assert key in r["metrics_mean"]

    def test_ranking_rmse_ascending(self, reg_df):
        out = self._run(reg_df).output
        means = [r["primary_metric_mean"] for r in out["ranking"]]
        assert means == sorted(means)  # menor RMSE é melhor => ordem crescente

    def test_linear_fits_linear_signal(self, reg_df):
        out = self._run(reg_df).output
        lin = next(r for r in out["results"] if r["model_name"] == "linear_regression")
        assert lin["metrics_mean"]["r2"] > 0.8


# ---------------------------------------------------------------------------
# Seleção, robustez e auditabilidade
# ---------------------------------------------------------------------------

class TestSelectionAndRobustness:
    def test_selects_by_tier_when_no_include(self, clf_df):
        # sem 'include', usa o tier 'minimum' do catálogo (só sklearn)
        out = _make_agent().run({
            "dataframe": clf_df,
            "task_type": "classification",
            "target_column": "label",
            "primary_metric": "macro_f1",
            "folds": _folds(clf_df, clf_df["label"].to_numpy()),
            "pipeline": _pipeline(clf_df, "label"),
        }).output
        names = {r["model_name"] for r in out["results"]}
        assert {"logistic_regression", "random_forest", "naive_bayes"} <= names
        assert "xgboost" not in names  # tier bonus não entra por padrão

    def test_unknown_include_is_warned(self, clf_df):
        result = _make_agent().run({
            "dataframe": clf_df,
            "task_type": "classification",
            "target_column": "label",
            "primary_metric": "macro_f1",
            "folds": _folds(clf_df, clf_df["label"].to_numpy()),
            "pipeline": _pipeline(clf_df, "label"),
            "include": ["logistic_regression", "autoencoder_latent_classifier"],
        })
        assert any("autoencoder_latent_classifier" in w for w in result.warnings)
        names = {r["model_name"] for r in result.output["results"]}
        assert names == {"logistic_regression"}

    def test_missing_dependency_skips_gracefully(self, clf_df, monkeypatch):
        import src.agents.trainer_agent as ta

        real_import = ta.importlib.import_module

        def fake_import(name, *args, **kwargs):
            if name == "xgboost":
                raise ImportError("No module named 'xgboost'", name="xgboost")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(ta.importlib, "import_module", fake_import)
        result = _make_agent().run({
            "dataframe": clf_df,
            "task_type": "classification",
            "target_column": "label",
            "primary_metric": "macro_f1",
            "folds": _folds(clf_df, clf_df["label"].to_numpy()),
            "pipeline": _pipeline(clf_df, "label"),
            "include": ["logistic_regression", "xgboost"],
        })
        assert any(s["model_name"] == "xgboost" for s in result.output["skipped_models"])
        assert any(r["model_name"] == "logistic_regression" for r in result.output["results"])

    def test_missing_task_type_raises(self, clf_df):
        with pytest.raises(ValueError, match="task_type"):
            _make_agent().run({"dataframe": clf_df, "folds": _folds(clf_df)})

    def test_missing_folds_raises(self, clf_df):
        with pytest.raises(ValueError, match="folds"):
            _make_agent().run({
                "dataframe": clf_df, "task_type": "classification", "target_column": "label",
            })

    def test_reads_folds_from_split_dict(self, clf_df):
        folds = _folds(clf_df, clf_df["label"].to_numpy())
        split = {"folds": [
            {"fold": i, "train_idx": tr.tolist(), "test_idx": te.tolist()}
            for i, (tr, te) in enumerate(folds)
        ]}
        out = _make_agent().run({
            "dataframe": clf_df,
            "task_type": "classification",
            "target_column": "label",
            "primary_metric": "macro_f1",
            "split": split,
            "pipeline": _pipeline(clf_df, "label"),
            "include": ["logistic_regression"],
        }).output
        assert out["n_folds"] == 3

    def test_failed_model_is_recorded_not_raised(self, clf_df):
        # Catálogo customizado com um estimador que quebra no fit.
        zoo = {"classification": {"broken": {
            "estimator": "sklearn.linear_model:LogisticRegression",
            "tier": "minimum",
            "default_params": {"solver": "invalido"},
        }}}
        result = _make_agent().run({
            "dataframe": clf_df,
            "task_type": "classification",
            "target_column": "label",
            "primary_metric": "macro_f1",
            "folds": _folds(clf_df, clf_df["label"].to_numpy()),
            "pipeline": _pipeline(clf_df, "label"),
            "model_zoo": zoo,
        })
        r = result.output["results"][0]
        assert r["status"] == "failed"
        assert "error" in r


class TestAnomaly:
    def test_anomaly_with_labels(self):
        rng = np.random.default_rng(3)
        normal = rng.normal(size=(80, 2))
        outliers = rng.normal(loc=6.0, size=(8, 2))
        X = np.vstack([normal, outliers])
        df = pd.DataFrame({"a": X[:, 0], "b": X[:, 1]})
        df["is_anomaly"] = [0] * 80 + [1] * 8
        folds = _folds(df, df["is_anomaly"].to_numpy(), n_splits=2)
        out = _make_agent().run({
            "dataframe": df,
            "task_type": "anomaly",
            "target_column": "is_anomaly",
            "primary_metric": "roc_auc",
            "folds": folds,
            "pipeline": _pipeline(df, "is_anomaly"),
            "include": ["isolation_forest"],
        }).output
        r = out["results"][0]
        assert r["status"] == "ok"
        assert "roc_auc" in r["metrics_mean"]


class TestEvent:
    def test_event_emitted_via_call(self, clf_df):
        events: list[dict] = []

        class _Sink:
            def record_event(self, **kwargs):
                events.append(kwargs)

        agent = TrainerAgent(event_sink=_Sink())
        agent({
            "experiment_id": "exp-1",
            "dataframe": clf_df,
            "task_type": "classification",
            "target_column": "label",
            "primary_metric": "macro_f1",
            "folds": _folds(clf_df, clf_df["label"].to_numpy()),
            "pipeline": _pipeline(clf_df, "label"),
            "include": ["logistic_regression"],
        })
        assert len(events) == 1
        assert events[0]["event_type"] == "model_training"
        assert events[0]["agent_name"] == "Agente de Model Zoo"
