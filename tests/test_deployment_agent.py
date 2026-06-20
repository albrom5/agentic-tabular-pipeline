"""Testes do Agente de Implantação e Monitoramento (etapa 10)."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.agents.deployment_agent import DeploymentAgent
from src.agents.feature_agent import FeatureAgent


def _make_agent() -> DeploymentAgent:
    return DeploymentAgent(event_sink=None)


@pytest.fixture
def clf_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 120
    x1 = rng.normal(size=n)
    label = (x1 + rng.normal(scale=0.3, size=n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": rng.normal(size=n),
                         "city": rng.choice(["SP", "RJ"], size=n), "label": label})


def _pipeline(df, target):
    return FeatureAgent().run({"dataframe": df, "target_column": target}).pipeline


def _ctx(df, tmp_path, **extra):
    base = {
        "dataframe": df,
        "task_type": "classification",
        "target_column": "label",
        "pipeline": _pipeline(df, "label"),
        "selected_model": {
            "model_name": "logistic_regression",
            "estimator": "sklearn.linear_model:LogisticRegression",
            "hyperparameters": {"max_iter": 1000},
        },
        "artifact_dir": str(tmp_path),
    }
    base.update(extra)
    return base


# ---------------------------------------------------------------------------
# Empacotamento
# ---------------------------------------------------------------------------

class TestPackaging:
    def test_artifact_saved_and_hashed(self, clf_df, tmp_path):
        out = _make_agent().run(_ctx(clf_df, tmp_path)).output
        dep = out["deployment"]
        assert dep["artifact_uri"] is not None
        assert (tmp_path / "logistic_regression.joblib").exists()
        assert len(dep["artifact_hash"]) == 64  # sha256 hex

    def test_artifact_is_loadable_and_predicts(self, clf_df, tmp_path):
        result = _make_agent().run(_ctx(clf_df, tmp_path))
        model = joblib.load(result.output["deployment"]["artifact_uri"])
        preds = model.predict(clf_df[["x1", "x2", "city"]])
        assert len(preds) == len(clf_df)

    def test_metadata_recorded(self, clf_df, tmp_path):
        dep = _make_agent().run(_ctx(clf_df, tmp_path)).output["deployment"]
        assert dep["n_train_rows"] == len(clf_df)
        assert dep["random_seed"] == 42
        assert "scikit-learn" in dep["library_versions"]
        assert len(dep["train_data_hash"]) == 64

    def test_result_predict_callable(self, clf_df, tmp_path):
        result = _make_agent().run(_ctx(clf_df, tmp_path))
        preds = result.predict(clf_df)
        assert len(preds) == len(clf_df)

    def test_output_is_json_serializable(self, clf_df, tmp_path):
        json.dumps(_make_agent().run(_ctx(clf_df, tmp_path)).output)


# ---------------------------------------------------------------------------
# Seleção do modelo
# ---------------------------------------------------------------------------

class TestModelSelection:
    def test_from_evaluation_and_training(self, clf_df, tmp_path):
        ctx = {
            "dataframe": clf_df, "task_type": "classification", "target_column": "label",
            "pipeline": _pipeline(clf_df, "label"), "artifact_dir": str(tmp_path),
            "evaluation": {"best_model": "random_forest"},
            "training": {"results": [{
                "model_name": "random_forest",
                "estimator": "sklearn.ensemble:RandomForestClassifier",
                "hyperparameters": {"n_estimators": 50},
            }]},
        }
        out = _make_agent().run(ctx).output
        assert out["deployment"]["model_name"] == "random_forest"
        assert "RandomForestClassifier" in out["deployment"]["estimator"]

    def test_missing_model_raises(self, clf_df, tmp_path):
        with pytest.raises(ValueError, match="selected_model"):
            _make_agent().run({
                "dataframe": clf_df, "task_type": "classification", "target_column": "label",
                "artifact_dir": str(tmp_path),
            })


# ---------------------------------------------------------------------------
# Monitoramento de drift (PSI)
# ---------------------------------------------------------------------------

class TestDriftMonitoring:
    def test_reference_profile_built(self, clf_df, tmp_path):
        mon = _make_agent().run(_ctx(clf_df, tmp_path)).output["monitoring"]
        assert mon["method"] == "psi"
        assert "x1" in mon["reference_profile"]["specs"]
        assert mon["reference_profile"]["specs"]["city"]["type"] == "categorical"

    def test_no_drift_on_same_distribution(self, clf_df, tmp_path):
        # PSI exige um lote de monitoramento de tamanho razoável (com poucas amostras
        # a métrica é ruidosa por construção); usamos um lote representativo.
        rng = np.random.default_rng(1)
        fresh = pd.DataFrame({"x1": rng.normal(size=500), "x2": rng.normal(size=500),
                              "city": rng.choice(["SP", "RJ"], size=500)})
        out = _make_agent().run(_ctx(clf_df, tmp_path, monitor_data=fresh)).output
        report = out["monitoring"]["drift_report"]
        assert report["drift_detected"] is False

    def test_drift_detected_on_shift(self, clf_df, tmp_path):
        rng = np.random.default_rng(2)
        shifted = pd.DataFrame({
            "x1": rng.normal(loc=8.0, size=100),   # forte deslocamento
            "x2": rng.normal(size=100),
            "city": ["SP"] * 100,                  # categoria colapsada
        })
        out = _make_agent().run(_ctx(clf_df, tmp_path, monitor_data=shifted)).output
        report = out["monitoring"]["drift_report"]
        assert report["drift_detected"] is True
        assert "x1" in report["drifted_columns"]
        assert report["max_psi"] >= out["monitoring"]["drift_threshold"]

    def test_detect_drift_callable_exposed(self, clf_df, tmp_path):
        result = _make_agent().run(_ctx(clf_df, tmp_path))
        shifted = pd.DataFrame({"x1": np.full(50, 9.0), "x2": np.zeros(50),
                                "city": ["RJ"] * 50})
        report = result.detect_drift(shifted)
        assert report["drift_detected"] is True


# ---------------------------------------------------------------------------
# Inferência, retreinamento e auditoria
# ---------------------------------------------------------------------------

class TestInferenceRetrainingEvent:
    def test_inference_spec(self, clf_df, tmp_path):
        out = _make_agent().run(_ctx(clf_df, tmp_path, inference_mode="endpoint")).output
        inf = out["inference"]
        assert inf["mode"] == "endpoint"
        assert inf["prediction_type"] == "class_label"
        assert inf["input_columns"] == ["x1", "x2", "city"]

    def test_retraining_plan(self, clf_df, tmp_path):
        out = _make_agent().run(_ctx(clf_df, tmp_path, retraining_cadence_days=15)).output
        rt = out["retraining"]
        assert rt["cadence_days"] == 15
        assert any("PSI" in t for t in rt["triggers"])
        assert rt["next_review_date"]  # data ISO

    def test_event_emitted_via_call(self, clf_df, tmp_path):
        events: list[dict] = []

        class _Sink:
            def record_event(self, **kwargs):
                events.append(kwargs)

        agent = DeploymentAgent(event_sink=_Sink())
        agent({"experiment_id": "exp-1", **_ctx(clf_df, tmp_path)})
        assert len(events) == 1
        assert events[0]["event_type"] == "deployment"
        assert events[0]["agent_name"] == "Agente de Implantação e Monitoramento"
