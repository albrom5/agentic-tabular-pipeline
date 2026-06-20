"""Testes do Agente Relator e Auditor (RF12 / RF14 / RNF05)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.agents.cleaning_agent import CleaningAgent
from src.agents.data_profile_agent import DataProfileAgent
from src.agents.evaluator_agent import EvaluatorAgent
from src.agents.feature_agent import FeatureAgent
from src.agents.problem_agent import ProblemAgent
from src.agents.report_agent import ReportAgent
from src.agents.split_agent import SplitAgent
from src.agents.trainer_agent import TrainerAgent


def _make_agent() -> ReportAgent:
    return ReportAgent(event_sink=None)


def _model_result(name, values, fit_seconds=0.01):
    fold_metrics = [
        {"fold": i, "metrics": {"macro_f1": v}, "fit_seconds": fit_seconds}
        for i, v in enumerate(values)
    ]
    return {
        "model_name": name, "model_family": "sklearn.test", "status": "ok",
        "estimator": "sklearn.linear_model:LogisticRegression",
        "hyperparameters": {"max_iter": 1000}, "fold_metrics": fold_metrics,
    }


# ---------------------------------------------------------------------------
# Estrutura mínima do relatório
# ---------------------------------------------------------------------------

class TestReportStructure:
    def test_minimal_report_has_all_sections(self):
        out = _make_agent().run({"experiment_name": "demo"}).output
        md = out["report_markdown"]
        for header in [
            "# Relatório Técnico — demo",
            "## 1. Resumo executivo",
            "## 5. Estratégia de validação",
            "## 10. Riscos, vieses e restrições de uso",
            "## 11. Próximos passos",
            "## Model card (resumido)",
            "## Trilha de auditoria",
        ]:
            assert header in md

    def test_output_is_json_serializable(self):
        out = _make_agent().run({
            "problem": {"task_type": "classification", "target_column": "y",
                        "primary_metric": "macro_f1"},
        }).output
        json.dumps(out)

    def test_report_exposed_as_attribute(self):
        result = _make_agent().run({"experiment_name": "x"})
        assert result.report_markdown.startswith("# Relatório Técnico")

    def test_writes_file_when_path_given(self, tmp_path):
        path = tmp_path / "report.md"
        _make_agent().run({"experiment_name": "demo", "report_path": str(path)})
        assert path.exists()
        assert path.read_text(encoding="utf-8").startswith("# Relatório Técnico")


# ---------------------------------------------------------------------------
# Conteúdo a partir das seções
# ---------------------------------------------------------------------------

class TestContent:
    def _sections(self):
        return {
            "problem": {"task_type": "classification", "target_column": "label",
                        "primary_metric": "macro_f1", "success_threshold": 0.7,
                        "split_strategy": "stratified_kfold"},
            "evaluation": {
                "primary_metric": "macro_f1", "best_model": "random_forest",
                "selection_rule": "best_mean",
                "selection_reason": "melhor média de random_forest.",
                "meets_success_threshold": True,
                "significance_note": "ICs disjuntos: diferença significativa.",
                "ranking": [
                    {"model_name": "random_forest", "primary_mean": 0.85, "primary_std": 0.02,
                     "ci_low": 0.82, "ci_high": 0.88, "n_folds": 5, "mean_fit_seconds": 0.5},
                    {"model_name": "logistic_regression", "primary_mean": 0.80,
                     "primary_std": 0.03, "ci_low": 0.76, "ci_high": 0.84, "n_folds": 5,
                     "mean_fit_seconds": 0.1},
                ],
            },
        }

    def test_best_model_in_summary_and_recommendation(self):
        out = _make_agent().run(self._sections()).output
        md = out["report_markdown"]
        assert "random_forest" in md
        assert "atingido" in md  # critério de sucesso atendido

    def test_results_table_lists_models(self):
        md = _make_agent().run(self._sections()).output["report_markdown"]
        assert "| random_forest | 0.85 |" in md
        assert "logistic_regression" in md

    def test_model_card_populated(self):
        out = _make_agent().run(self._sections()).output
        card = out["model_card"]
        assert card["model"] == "random_forest"
        assert card["task_type"] == "classification"
        assert card["performance"]["mean"] == 0.85

    def test_autoencoder_section(self):
        sections = self._sections()
        sections["autoencoder"] = {"use_case": "latent_features",
                                   "verdict": "O autoencoder SUPERA o baseline em 'macro_f1'."}
        md = _make_agent().run(sections).output["report_markdown"]
        assert "latent_features" in md
        assert "SUPERA o baseline" in md


# ---------------------------------------------------------------------------
# Riscos, limitações e auditoria (RNF05 / revisável por humano)
# ---------------------------------------------------------------------------

class TestRisksAndAudit:
    def test_leakage_notes_become_risks(self):
        out = _make_agent().run({
            "problem": {"task_type": "classification", "primary_metric": "macro_f1"},
            "features": {"column_groups": {}, "leakage_notes": [
                "Atributo 'x' fortemente correlacionado com o alvo (r=0.99)."]},
        }).output
        assert any("0.99" in r for r in out["risks"])

    def test_unmet_threshold_is_limitation(self):
        out = _make_agent().run({
            "evaluation": {"primary_metric": "macro_f1", "best_model": "m",
                           "meets_success_threshold": False, "ranking": [
                               {"model_name": "m", "primary_mean": 0.5, "primary_std": 0.1,
                                "ci_low": 0.4, "ci_high": 0.6, "n_folds": 5,
                                "mean_fit_seconds": 0.1}]},
        }).output
        assert any("NÃO atinge" in x for x in out["limitations"])

    def test_audit_log_from_events(self):
        out = _make_agent().run({
            "agent_events": [
                {"agent_name": "Agente de Formulação do Problema",
                 "event_type": "problem_definition", "rationale": "Tarefa definida."},
            ],
        }).output
        assert out["audit_log"][0]["agent_name"] == "Agente de Formulação do Problema"
        assert "Trilha de auditoria" in out["report_markdown"]

    def test_event_emitted_via_call(self):
        events: list[dict] = []

        class _Sink:
            def record_event(self, **kwargs):
                events.append(kwargs)

        ReportAgent(event_sink=_Sink())({"experiment_id": "exp-1", "experiment_name": "x"})
        assert len(events) == 1
        assert events[0]["event_type"] == "final_report"
        assert events[0]["agent_name"] == "Agente Relator e Auditor"


# ---------------------------------------------------------------------------
# Integração end-to-end + explicabilidade (RF12)
# ---------------------------------------------------------------------------

class TestEndToEnd:
    @pytest.fixture
    def clf_df(self):
        rng = np.random.default_rng(0)
        n = 120
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        label = (x1 + 0.5 * x2 + rng.normal(scale=0.3, size=n) > 0).astype(int)
        return pd.DataFrame({"x1": x1, "x2": x2, "city": rng.choice(["SP", "RJ"], size=n),
                             "label": label})

    def test_full_pipeline_report(self, clf_df):
        problem = ProblemAgent().run({
            "task_type": "classification", "target_column": "label", "primary_metric": "macro_f1",
        }).output
        profile = DataProfileAgent().run({"dataframe": clf_df, "target_column": "label"}).output
        cleaning = CleaningAgent().run({"dataframe": clf_df, "target_column": "label"})
        clean_df = cleaning.cleaned_dataframe
        feat = FeatureAgent().run({"dataframe": clean_df, "target_column": "label"})
        split = SplitAgent().run({
            "dataframe": clean_df, "split_strategy": "stratified_kfold",
            "target_column": "label", "task_type": "classification", "n_splits": 3,
        })
        training = TrainerAgent().run({
            "dataframe": clean_df, "task_type": "classification", "target_column": "label",
            "primary_metric": "macro_f1", "folds": split.folds, "pipeline": feat.pipeline,
            "include": ["logistic_regression", "random_forest"],
        }).output
        evaluation = EvaluatorAgent().run({
            "model_results": training["results"], "task_type": "classification",
            "primary_metric": "macro_f1", "success_threshold": 0.6,
        }).output

        report = _make_agent().run({
            "experiment_name": "credit_demo",
            "problem": problem, "profile": profile, "cleaning": cleaning.output,
            "features": feat.output, "split": split.output, "training": training,
            "evaluation": evaluation,
            # dados para a explicabilidade do modelo recomendado
            "dataframe": clean_df, "folds": split.folds, "pipeline": feat.pipeline,
            "target_column": "label",
        })
        out = report.output
        md = out["report_markdown"]

        assert out["explainability"] is not None
        assert out["explainability"]["top_features"]
        assert "## 9. Modelo recomendado" in md
        assert evaluation["best_model"] in md
        assert "Explicabilidade" in md
        # auditoria mostra as etapas presentes quando não há agent_events
        assert any(e.get("stage") == "evaluation" for e in out["audit_log"])
        json.dumps(out)  # serializável
