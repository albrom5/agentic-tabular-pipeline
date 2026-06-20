"""Testes do Agente de Formulação do Problema."""

from __future__ import annotations

import pytest

from src.agents.problem_agent import ProblemAgent


def _make_agent() -> ProblemAgent:
    return ProblemAgent(event_sink=None)


# ---------------------------------------------------------------------------
# Casos felizes
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_classification_minimal(self):
        result = _make_agent().run({
            "task_type": "classification",
            "target_column": "label",
            "primary_metric": "macro_f1",
        })
        assert result.output["task_type"] == "classification"
        assert result.output["target_column"] == "label"
        assert result.output["primary_metric"] == "macro_f1"
        assert result.output["split_strategy"] == "stratified_kfold"
        assert "macro_f1" not in result.output["secondary_metrics"]

    def test_regression_minimal(self):
        result = _make_agent().run({
            "task_type": "regression",
            "target_column": "price",
            "primary_metric": "rmse",
        })
        assert result.output["task_type"] == "regression"
        assert result.output["split_strategy"] == "kfold"

    def test_anomaly_without_target(self):
        result = _make_agent().run({
            "task_type": "anomaly",
            "primary_metric": "roc_auc",
        })
        assert result.output["target_column"] is None
        assert any("não supervisionado" in w for w in result.warnings)

    def test_time_column_forces_time_split(self):
        result = _make_agent().run({
            "task_type": "regression",
            "target_column": "sales",
            "primary_metric": "mae",
            "time_column": "date",
        })
        assert result.output["split_strategy"] == "time_split"

    def test_group_column_suggests_group_kfold(self):
        result = _make_agent().run({
            "task_type": "classification",
            "target_column": "outcome",
            "primary_metric": "macro_f1",
            "group_column": "patient_id",
        })
        assert result.output["split_strategy"] == "group_kfold"

    def test_explicit_split_strategy_respected(self):
        result = _make_agent().run({
            "task_type": "classification",
            "target_column": "y",
            "primary_metric": "roc_auc",
            "split_strategy": "holdout",
        })
        assert result.output["split_strategy"] == "holdout"

    def test_success_threshold_used_when_provided(self):
        result = _make_agent().run({
            "task_type": "classification",
            "target_column": "y",
            "primary_metric": "roc_auc",
            "success_threshold": 0.85,
        })
        assert result.output["success_threshold"] == 0.85

    def test_default_metric_filled_when_missing(self):
        result = _make_agent().run({
            "task_type": "classification",
            "target_column": "y",
        })
        assert result.output["primary_metric"] == "macro_f1"
        assert any("primary_metric" in w for w in result.warnings)

    def test_constraints_passed_through(self):
        constraints = {"max_fit_seconds": 30, "interpretability": "high"}
        result = _make_agent().run({
            "task_type": "classification",
            "target_column": "y",
            "primary_metric": "macro_f1",
            "constraints": constraints,
        })
        assert result.output["constraints"] == constraints

    def test_rationale_is_non_empty(self):
        result = _make_agent().run({
            "task_type": "regression",
            "target_column": "price",
            "primary_metric": "rmse",
        })
        assert len(result.rationale) > 20

    def test_secondary_metrics_exclude_primary(self):
        result = _make_agent().run({
            "task_type": "regression",
            "target_column": "price",
            "primary_metric": "mae",
        })
        assert "mae" not in result.output["secondary_metrics"]

    def test_call_dunder_works_without_sink(self):
        agent = _make_agent()
        result = agent({
            "task_type": "classification",
            "target_column": "y",
            "primary_metric": "accuracy",
        })
        assert result.output["task_type"] == "classification"


# ---------------------------------------------------------------------------
# Validação de erros críticos
# ---------------------------------------------------------------------------

class TestValidationErrors:
    def test_missing_task_type(self):
        with pytest.raises(ValueError, match="task_type"):
            _make_agent().run({"target_column": "y", "primary_metric": "macro_f1"})

    def test_invalid_task_type(self):
        with pytest.raises(ValueError, match="task_type"):
            _make_agent().run({"task_type": "clustering", "target_column": "y", "primary_metric": "macro_f1"})

    def test_missing_target_for_classification(self):
        with pytest.raises(ValueError, match="target_column"):
            _make_agent().run({"task_type": "classification", "primary_metric": "macro_f1"})

    def test_missing_target_for_regression(self):
        with pytest.raises(ValueError, match="target_column"):
            _make_agent().run({"task_type": "regression", "primary_metric": "rmse"})

    def test_wrong_metric_for_task(self):
        with pytest.raises(ValueError, match="primary_metric"):
            _make_agent().run({
                "task_type": "classification",
                "target_column": "y",
                "primary_metric": "rmse",  # métrica de regressão
            })

    def test_regression_metric_for_classification(self):
        with pytest.raises(ValueError, match="primary_metric"):
            _make_agent().run({
                "task_type": "regression",
                "target_column": "price",
                "primary_metric": "roc_auc",
            })

    def test_invalid_split_strategy(self):
        with pytest.raises(ValueError, match="split_strategy"):
            _make_agent().run({
                "task_type": "classification",
                "target_column": "y",
                "primary_metric": "macro_f1",
                "split_strategy": "random_split",
            })


# ---------------------------------------------------------------------------
# Warnings (não lançam exceção)
# ---------------------------------------------------------------------------

class TestWarnings:
    def test_time_column_without_time_split_warns(self):
        result = _make_agent().run({
            "task_type": "regression",
            "target_column": "sales",
            "primary_metric": "mae",
            "time_column": "date",
            "split_strategy": "kfold",
        })
        assert any("time_split" in w for w in result.warnings)

    def test_group_column_without_group_kfold_warns(self):
        result = _make_agent().run({
            "task_type": "classification",
            "target_column": "y",
            "primary_metric": "macro_f1",
            "group_column": "patient_id",
            "split_strategy": "holdout",
        })
        assert any("group_kfold" in w for w in result.warnings)

    def test_default_threshold_fills_warning(self):
        result = _make_agent().run({
            "task_type": "classification",
            "target_column": "y",
            "primary_metric": "roc_auc",
        })
        assert result.output["success_threshold"] is not None
        assert any("success_threshold" in w for w in result.warnings)
