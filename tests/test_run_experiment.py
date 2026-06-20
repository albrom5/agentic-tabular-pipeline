"""Testes do orquestrador run_experiment (encadeamento + persistência)."""

from __future__ import annotations

import json
import uuid

import numpy as np
import pandas as pd
import pytest

from src.pipelines.training import load_config, run_experiment


def _clf_df(seed: int = 0, n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    label = (x1 + rng.normal(scale=0.3, size=n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": rng.normal(size=n),
                         "city": rng.choice(["SP", "RJ"], size=n), "label": label})


def _reg_df(seed: int = 1, n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": rng.normal(size=n),
                         "price": 3 * x1 + rng.normal(scale=0.5, size=n)})


def _clf_config(**overrides) -> dict:
    cfg = {
        "experiment": {
            "name": "demo", "task_type": "classification",
            "target_column": "label", "primary_metric": "macro_f1", "random_seed": 42,
        },
        "validation": {"split_strategy": "stratified_kfold", "n_splits": 3},
        "preprocessing": {"scaling": "standard", "categorical_encoding": "one_hot"},
        "models": {"include": ["logistic_regression", "random_forest"]},
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# Encadeamento end-to-end
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_runs_all_stages(self):
        result = run_experiment(_clf_config(), dataframe=_clf_df())
        for stage in ("problem", "profile", "cleaning", "features", "split",
                      "training", "evaluation", "report"):
            assert result[stage] is not None
        assert result["best_model"] in {"logistic_regression", "random_forest"}
        assert result["report_markdown"].startswith("# Relatório Técnico")

    def test_ranking_present_and_learns(self):
        result = run_experiment(_clf_config(), dataframe=_clf_df())
        ranking = result["ranking"]
        assert len(ranking) == 2
        assert max(e["primary_mean"] for e in ranking) > 0.7  # há sinal aprendível

    def test_regression_experiment(self):
        cfg = {
            "experiment": {"name": "reg", "task_type": "regression",
                           "target_column": "price", "primary_metric": "rmse",
                           "random_seed": 42},
            "validation": {"split_strategy": "kfold", "n_splits": 3},
            "models": {"include": ["linear_regression", "decision_tree_regressor"]},
        }
        result = run_experiment(cfg, dataframe=_reg_df())
        assert result["evaluation"]["direction"] == "minimize"
        assert result["best_model"] in {"linear_regression", "decision_tree_regressor"}

    def test_autoencoder_stage_when_enabled(self):
        cfg = _clf_config(autoencoder={
            "enabled": True, "use_case": "latent_features",
            "epochs": 5, "latent_dim": 4, "hidden_dims": [16],
        })
        result = run_experiment(cfg, dataframe=_clf_df())
        assert result["autoencoder"] is not None
        assert "verdict" in result["autoencoder"]

    def test_deployment_stage_when_enabled(self, tmp_path):
        cfg = _clf_config(
            deployment={"enabled": True},
            storage={"artifact_dir": str(tmp_path)},
        )
        result = run_experiment(cfg, dataframe=_clf_df())
        assert result["deployment"] is not None
        assert result["deployment"]["deployment"]["artifact_uri"] is not None

    def test_report_markdown_contains_recommendation(self):
        result = run_experiment(_clf_config(), dataframe=_clf_df())
        assert result["best_model"] in result["report_markdown"]
        assert "## 9. Modelo recomendado" in result["report_markdown"]


# ---------------------------------------------------------------------------
# Eventos agentivos (auditabilidade — RNF03)
# ---------------------------------------------------------------------------

class TestEventEmission:
    def test_events_emitted_for_each_agent(self):
        events: list[dict] = []

        class _Sink:
            def record_event(self, **kwargs):
                events.append(kwargs)

        run_experiment(_clf_config(), dataframe=_clf_df(), event_sink=_Sink())
        types = {e["event_type"] for e in events}
        # Etapas obrigatórias devem ter gerado evento.
        assert {"problem_definition", "data_profile", "cleaning_decision",
                "feature_engineering", "split_strategy", "model_training",
                "model_selection", "final_report"} <= types


# ---------------------------------------------------------------------------
# Persistência (via recorder fake — sem PostgreSQL real)
# ---------------------------------------------------------------------------

class _FakeRecorder:
    """Captura as chamadas de persistência feitas pelo orquestrador."""

    def __init__(self):
        self.calls: list[str] = []
        self.event_sink = None
        self.model_rows = 0

    def create_experiment(self, **kw):
        self.calls.append("create_experiment")
        self.experiment_config = kw
        return uuid.uuid4()

    def create_run(self, **kw):
        self.calls.append("create_run")
        self.run_kwargs = kw
        return uuid.uuid4()

    def save_dataset(self, **kw):
        self.calls.append("save_dataset")
        self.dataset_kwargs = kw

    def save_model_results(self, *, run_id, results, artifacts=None):
        self.calls.append("save_model_results")
        self.model_rows = sum(len(r.get("fold_metrics", [])) for r in results)
        return self.model_rows

    def save_report(self, **kw):
        self.calls.append("save_report")
        self.report_kwargs = kw

    def finish_run(self, **kw):
        self.calls.append("finish_run")
        self.finish_kwargs = kw

    def set_experiment_status(self, **kw):
        self.calls.append("set_experiment_status")


class TestPersistence:
    def test_recorder_called_in_order(self):
        rec = _FakeRecorder()
        result = run_experiment(_clf_config(), dataframe=_clf_df(), recorder=rec)
        assert rec.calls == [
            "create_experiment", "save_dataset", "create_run",
            "save_model_results", "save_report", "finish_run", "set_experiment_status",
        ]
        assert result["experiment_id"] is not None
        assert result["run_id"] is not None

    def test_dataset_persists_profile_and_quality(self):
        rec = _FakeRecorder()
        run_experiment(_clf_config(), dataframe=_clf_df(), recorder=rec)
        assert rec.dataset_kwargs["profile_json"] is not None
        assert rec.dataset_kwargs["quality_report_json"] is not None
        assert rec.dataset_kwargs["content_hash"]

    def test_model_results_rows_match_folds(self):
        rec = _FakeRecorder()
        run_experiment(_clf_config(), dataframe=_clf_df(), recorder=rec)
        # 2 modelos × 3 folds = 6 linhas por fold
        assert rec.model_rows == 6

    def test_run_finished_with_metrics(self):
        rec = _FakeRecorder()
        run_experiment(_clf_config(), dataframe=_clf_df(), recorder=rec)
        assert rec.finish_kwargs["status"] == "completed"
        assert rec.finish_kwargs["metrics_json"]["best_model"]

    def test_report_persisted(self):
        rec = _FakeRecorder()
        run_experiment(_clf_config(), dataframe=_clf_df(), recorder=rec)
        assert rec.report_kwargs["content"].startswith("# Relatório Técnico")
        json.dumps(rec.report_kwargs["summary_json"])  # serializável


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

class TestConfig:
    def test_loads_example_config(self):
        cfg = load_config("configs/experiment_example.yaml")
        assert cfg["experiment"]["task_type"] == "classification"

    def test_missing_source_without_dataframe_raises(self):
        cfg = _clf_config()
        cfg["data"] = {"source_type": "csv"}  # sem source_uri
        with pytest.raises(ValueError, match="source_uri"):
            run_experiment(cfg)
