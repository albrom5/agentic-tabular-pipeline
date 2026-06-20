"""Testes do Agente de Qualidade e Limpeza (RF04 / RF05)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.agents.cleaning_agent import CleaningAgent


def _make_agent() -> CleaningAgent:
    return CleaningAgent(event_sink=None)


@pytest.fixture
def dirty_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [25, 30, np.nan, 45, 30, 30],          # numérica com 1 faltante
            "income": [3000.0, 5000.0, 4000.0, np.nan, 5000.0, 5000.0],  # numérica c/ faltante
            "city": ["SP", "RJ", "SP", None, "SP", "SP"],  # categórica com faltante
            "const": [1, 1, 1, 1, 1, 1],                  # constante
            "label": [0, 1, 0, 1, 0, 0],                  # alvo
        }
    )


# ---------------------------------------------------------------------------
# Ingestão e contrato geral
# ---------------------------------------------------------------------------

class TestIngestion:
    def test_does_not_mutate_input(self, dirty_df):
        before = dirty_df.copy()
        _make_agent().run({"dataframe": dirty_df, "target_column": "label"})
        pd.testing.assert_frame_equal(dirty_df, before)

    def test_output_is_json_serializable(self, dirty_df):
        result = _make_agent().run({"dataframe": dirty_df, "target_column": "label"})
        json.dumps(result.output)  # não deve lançar

    def test_cleaned_dataframe_returned(self, dirty_df):
        result = _make_agent().run({"dataframe": dirty_df, "target_column": "label"})
        assert isinstance(result.cleaned_dataframe, pd.DataFrame)

    def test_no_columns_raises(self):
        with pytest.raises(ValueError, match="colunas"):
            _make_agent().run({"dataframe": pd.DataFrame()})

    def test_invalid_target_raises(self, dirty_df):
        with pytest.raises(ValueError, match="target_column"):
            _make_agent().run({"dataframe": dirty_df, "target_column": "nope"})

    def test_load_from_csv(self, dirty_df, tmp_path):
        path = tmp_path / "data.csv"
        dirty_df.to_csv(path, index=False)
        result = _make_agent().run({"data": {"source_type": "csv", "source_uri": str(path)}})
        assert result.output["n_rows_before"] == 6

    def test_invalid_strategy_raises(self, dirty_df):
        with pytest.raises(ValueError, match="numeric_imputation"):
            _make_agent().run({"dataframe": dirty_df, "numeric_imputation": "bogus"})


# ---------------------------------------------------------------------------
# Relatório de qualidade (RF04)
# ---------------------------------------------------------------------------

class TestQualityReport:
    def test_missing_detected(self, dirty_df):
        qr = _make_agent().run({"dataframe": dirty_df, "target_column": "label"}).output[
            "quality_report"
        ]
        assert qr["missing"]["age"]["n"] == 1
        assert qr["missing"]["city"]["n"] == 1

    def test_constant_detected(self, dirty_df):
        qr = _make_agent().run({"dataframe": dirty_df, "target_column": "label"}).output[
            "quality_report"
        ]
        assert "const" in qr["constant_columns"]

    def test_duplicates_detected(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        qr = _make_agent().run({"dataframe": df}).output["quality_report"]
        assert qr["n_duplicate_rows"] == 1

    def test_rare_categories_detected(self):
        df = pd.DataFrame({"c": ["a"] * 199 + ["rare"]})  # 0.5% < limiar de 1%
        qr = _make_agent().run({"dataframe": df}).output["quality_report"]
        assert "rare" in qr["rare_categories"]["c"]["categories"]

    def test_outliers_detected(self):
        df = pd.DataFrame({"x": [10, 11, 12, 13, 14, 15, 16, 1000]})
        qr = _make_agent().run({"dataframe": df}).output["quality_report"]
        assert qr["outliers"]["x"]["n"] == 1


# ---------------------------------------------------------------------------
# Limpeza reprodutível (RF05) — toda ação é registrada
# ---------------------------------------------------------------------------

class TestActions:
    def _ops(self, result):
        return [a["operation"] for a in result.output["actions"]]

    def test_median_imputation_default(self, dirty_df):
        result = _make_agent().run({"dataframe": dirty_df, "target_column": "label"})
        action = next(a for a in result.output["actions"] if a.get("column") == "age")
        assert action["operation"] == "median_imputation"
        assert "fill_value" in action  # reprodutível
        assert result.cleaned_dataframe["age"].isna().sum() == 0

    def test_mean_imputation_when_requested(self, dirty_df):
        result = _make_agent().run(
            {"dataframe": dirty_df, "target_column": "label", "numeric_imputation": "mean"}
        )
        action = next(a for a in result.output["actions"] if a.get("column") == "age")
        assert action["operation"] == "mean_imputation"

    def test_categorical_imputation(self, dirty_df):
        result = _make_agent().run({"dataframe": dirty_df, "target_column": "label"})
        action = next(a for a in result.output["actions"] if a.get("column") == "city")
        assert action["operation"] == "most_frequent_imputation"
        assert action["fill_value"] == "SP"
        assert result.cleaned_dataframe["city"].isna().sum() == 0

    def test_constant_column_dropped(self, dirty_df):
        result = _make_agent().run({"dataframe": dirty_df, "target_column": "label"})
        assert "drop_constant_column" in self._ops(result)
        assert "const" not in result.cleaned_dataframe.columns

    def test_duplicates_dropped(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        result = _make_agent().run({"dataframe": df})
        assert "drop_duplicates" in self._ops(result)
        assert len(result.cleaned_dataframe) == 2

    def test_rare_categories_grouped(self):
        df = pd.DataFrame({"c": ["a"] * 199 + ["rare"]})  # 0.5% < limiar de 1%
        result = _make_agent().run({"dataframe": df})
        assert "group_rare_categories" in self._ops(result)
        assert "__rare__" in set(result.cleaned_dataframe["c"])

    def test_missing_target_rows_dropped(self):
        df = pd.DataFrame({"x": [1, 2, 3], "label": [1, None, 0]})
        result = _make_agent().run({"dataframe": df, "target_column": "label"})
        assert "drop_missing_target_rows" in self._ops(result)
        assert len(result.cleaned_dataframe) == 2
        assert result.cleaned_dataframe["label"].isna().sum() == 0

    def test_high_missing_column_dropped(self):
        # 60% de faltantes e ≥ 2 valores distintos (não é constante)
        df = pd.DataFrame({"mostly_null": [1.0, 2.0, None, None, None], "y": [1, 2, 3, 4, 5]})
        result = _make_agent().run({"dataframe": df})
        assert "drop_high_missing_column" in self._ops(result)
        assert "mostly_null" not in result.cleaned_dataframe.columns

    def test_impossible_ranges_coerced(self):
        df = pd.DataFrame({"age": [25, 30, 200, 40]})  # 200 é impossível
        result = _make_agent().run(
            {"dataframe": df, "impossible_ranges": {"age": {"min": 0, "max": 120}}}
        )
        ops = self._ops(result)
        assert "coerce_impossible_to_nan" in ops
        # valor impossível vira NaN e é imputado depois
        assert result.cleaned_dataframe["age"].max() <= 120

    def test_outlier_clip_when_requested(self):
        df = pd.DataFrame({"x": [10, 11, 12, 13, 14, 15, 16, 1000]})
        result = _make_agent().run({"dataframe": df, "outlier_strategy": "clip"})
        assert "clip_outliers" in self._ops(result)
        assert result.cleaned_dataframe["x"].max() < 1000

    def test_outliers_not_touched_by_default(self):
        df = pd.DataFrame({"x": [10, 11, 12, 13, 14, 15, 16, 1000]})
        result = _make_agent().run({"dataframe": df})
        assert "clip_outliers" not in self._ops(result)
        assert "remove_outliers" not in self._ops(result)
        assert result.cleaned_dataframe["x"].max() == 1000


# ---------------------------------------------------------------------------
# Proteção do alvo e robustez (RNF04)
# ---------------------------------------------------------------------------

class TestTargetAndRobustness:
    def test_target_not_imputed_or_dropped(self, dirty_df):
        df = dirty_df.copy()
        df["label"] = df["label"].astype(float)
        result = _make_agent().run({"dataframe": df, "target_column": "label"})
        assert "label" in result.cleaned_dataframe.columns
        target_actions = [a for a in result.output["actions"] if a.get("column") == "label"]
        assert all(a["operation"] != "median_imputation" for a in target_actions)

    def test_constant_target_not_dropped(self):
        df = pd.DataFrame({"x": [1, 2, 3], "label": [1, 1, 1]})
        result = _make_agent().run({"dataframe": df, "target_column": "label"})
        assert "label" in result.cleaned_dataframe.columns

    def test_whitespace_normalized(self):
        df = pd.DataFrame({"c": [" SP ", "RJ", "  ", "MG"]})
        # Sem imputação/dedup para observar a normalização isoladamente.
        result = _make_agent().run(
            {"dataframe": df, "categorical_imputation": "none", "drop_duplicates": False}
        )
        cleaned = result.cleaned_dataframe["c"]
        assert "strip_whitespace" in [a["operation"] for a in result.output["actions"]]
        assert cleaned.iloc[0] == "SP"        # " SP " aparado
        assert pd.isna(cleaned.iloc[2])       # "  " virou faltante

    def test_numeric_string_coercion(self):
        df = pd.DataFrame({"n": ["1", "2", "3", "4"]})
        result = _make_agent().run({"dataframe": df})
        assert "coerce_to_numeric" in [a["operation"] for a in result.output["actions"]]
        assert pd.api.types.is_numeric_dtype(result.cleaned_dataframe["n"])


# ---------------------------------------------------------------------------
# Auditabilidade / evento (RNF03)
# ---------------------------------------------------------------------------

class TestRationaleAndEvent:
    def test_rationale_non_empty(self, dirty_df):
        result = _make_agent().run({"dataframe": dirty_df, "target_column": "label"})
        assert len(result.rationale) > 30

    def test_warnings_present_in_output(self):
        # 23% de faltantes em coluna numérica gera aviso de viés
        df = pd.DataFrame({"renda": [1.0, 2.0, 3.0, None, 5.0] + [6.0] * 5})
        result = _make_agent().run({"dataframe": df})
        assert result.output["warnings"] == result.warnings

    def test_event_emitted_via_call(self, dirty_df):
        events: list[dict] = []

        class _Sink:
            def record_event(self, **kwargs):
                events.append(kwargs)

        agent = CleaningAgent(event_sink=_Sink())
        agent({"experiment_id": "exp-1", "dataframe": dirty_df, "target_column": "label"})
        assert len(events) == 1
        assert events[0]["event_type"] == "cleaning_decision"
        assert events[0]["agent_name"] == "Agente de Qualidade e Limpeza"
