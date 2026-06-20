"""Testes do Agente de Autoencoders (RF10, seção 9)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, StratifiedKFold

from src.agents.autoencoder_agent import AutoencoderAgent
from src.agents.feature_agent import FeatureAgent

# Configuração leve para manter os testes rápidos.
_FAST = {"epochs": 5, "latent_dim": 4, "batch_size": 32, "hidden_dims": [16]}


def _make_agent() -> AutoencoderAgent:
    return AutoencoderAgent(event_sink=None)


def _folds(df, y=None, n_splits=2, seed=0):
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
    n = 120
    x1 = rng.normal(size=n)
    label = (x1 + rng.normal(scale=0.3, size=n) > 0).astype(int)
    return pd.DataFrame({
        "x1": x1, "x2": rng.normal(size=n), "x3": rng.normal(size=n), "label": label,
    })


@pytest.fixture
def anomaly_df() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    normal = rng.normal(size=(110, 3))
    outliers = rng.normal(loc=6.0, size=(10, 3))
    X = np.vstack([normal, outliers])
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    df["is_anomaly"] = [0] * 110 + [1] * 10
    return df


# ---------------------------------------------------------------------------
# latent_features (representação)
# ---------------------------------------------------------------------------

class TestLatentFeatures:
    def _run(self, df, **ctx):
        return _make_agent().run({
            "dataframe": df,
            "task_type": "classification",
            "target_column": "label",
            "primary_metric": "macro_f1",
            "folds": _folds(df, df["label"].to_numpy()),
            "pipeline": _pipeline(df, "label"),
            "autoencoder": {"use_case": "latent_features", **_FAST},
            **ctx,
        })

    def test_compares_ae_and_baseline(self, clf_df):
        out = self._run(clf_df).output
        assert out["use_case"] == "latent_features"
        assert "macro_f1" in out["ae_metrics_mean"]
        assert "macro_f1" in out["baseline_metrics_mean"]
        assert out["comparison"]["conclusive"] is True

    def test_records_per_fold(self, clf_df):
        out = self._run(clf_df).output
        assert len(out["folds"]) == 2
        assert "train_loss_final" in out["folds"][0]
        assert "ae_metrics" in out["folds"][0] and "baseline_metrics" in out["folds"][0]

    def test_augment_adds_latent_dims(self, clf_df):
        # Não há forma direta de inspecionar a dimensão aqui, mas o modo augment
        # deve produzir comparação conclusiva (latente concatenado aos atributos).
        out = self._run(clf_df, autoencoder={"use_case": "latent_features",
                                             "latent_mode": "augment", **_FAST}).output
        assert out["config"]["latent_mode"] == "augment"
        assert out["comparison"]["conclusive"]

    def test_verdict_text(self, clf_df):
        out = self._run(clf_df).output
        assert "autoencoder" in out["verdict"].lower()

    def test_output_is_json_serializable(self, clf_df):
        json.dumps(self._run(clf_df).output)

    def test_reproducible(self, clf_df):
        a = self._run(clf_df).output["ae_metrics_mean"]
        b = self._run(clf_df).output["ae_metrics_mean"]
        assert a == b

    def test_requires_target(self, clf_df):
        with pytest.raises(ValueError, match="target_column"):
            _make_agent().run({
                "dataframe": clf_df,
                "task_type": "classification",
                "folds": _folds(clf_df, clf_df["label"].to_numpy()),
                "autoencoder": {"use_case": "latent_features", **_FAST},
            })


# ---------------------------------------------------------------------------
# denoising
# ---------------------------------------------------------------------------

class TestDenoising:
    def test_reconstruction_beats_mean_baseline(self, clf_df):
        out = _make_agent().run({
            "dataframe": clf_df,
            "target_column": "label",
            "folds": _folds(clf_df),
            "pipeline": _pipeline(clf_df, "label"),
            "autoencoder": {"use_case": "denoising", "noise_std": 0.1,
                            "epochs": 40, "latent_dim": 3, "batch_size": 32, "hidden_dims": [16]},
        }).output
        assert "reconstruction_mse" in out["ae_metrics_mean"]
        # o AE deve reconstruir melhor que prever a média (baseline trivial)
        assert out["comparison"]["ae_better"] is True
        assert out["comparison"]["lower_is_better"] is True


# ---------------------------------------------------------------------------
# anomaly_detection
# ---------------------------------------------------------------------------

class TestAnomalyDetection:
    def test_roc_auc_vs_isolation_forest(self, anomaly_df):
        out = _make_agent().run({
            "dataframe": anomaly_df,
            "task_type": "anomaly",
            "target_column": "is_anomaly",
            "primary_metric": "roc_auc",
            "folds": _folds(anomaly_df, anomaly_df["is_anomaly"].to_numpy()),
            "pipeline": _pipeline(anomaly_df, "is_anomaly"),
            "autoencoder": {"use_case": "anomaly_detection", "epochs": 30,
                            "latent_dim": 2, "batch_size": 32, "hidden_dims": [8]},
        }).output
        assert "roc_auc" in out["ae_metrics_mean"]
        assert "roc_auc" in out["baseline_metrics_mean"]  # Isolation Forest
        assert "precision_at_k" in out["ae_metrics_mean"]

    def test_without_labels_is_inconclusive(self, anomaly_df):
        df = anomaly_df.drop(columns="is_anomaly")
        result = _make_agent().run({
            "dataframe": df,
            "task_type": "anomaly",
            "folds": _folds(df),
            "pipeline": None,
            "autoencoder": {"use_case": "anomaly_detection", **_FAST},
        })
        out = result.output
        assert out["comparison"]["conclusive"] is False
        assert "ae_score_stats" in out["folds"][0]
        assert any("rótulos" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Validação e auditabilidade
# ---------------------------------------------------------------------------

class TestValidationAndEvent:
    def test_invalid_use_case_raises(self, clf_df):
        with pytest.raises(ValueError, match="use_case"):
            _make_agent().run({
                "dataframe": clf_df, "target_column": "label",
                "folds": _folds(clf_df), "autoencoder": {"use_case": "magic"},
            })

    def test_missing_folds_raises(self, clf_df):
        with pytest.raises(ValueError, match="folds"):
            _make_agent().run({
                "dataframe": clf_df, "task_type": "classification", "target_column": "label",
                "autoencoder": {"use_case": "latent_features", **_FAST},
            })

    def test_event_emitted_via_call(self, clf_df):
        events: list[dict] = []

        class _Sink:
            def record_event(self, **kwargs):
                events.append(kwargs)

        agent = AutoencoderAgent(event_sink=_Sink())
        agent({
            "experiment_id": "exp-1",
            "dataframe": clf_df,
            "task_type": "classification",
            "target_column": "label",
            "primary_metric": "macro_f1",
            "folds": _folds(clf_df, clf_df["label"].to_numpy()),
            "pipeline": _pipeline(clf_df, "label"),
            "autoencoder": {"use_case": "latent_features", **_FAST},
        })
        assert len(events) == 1
        assert events[0]["event_type"] == "autoencoder_training"
        assert events[0]["agent_name"] == "Agente de Autoencoders"
