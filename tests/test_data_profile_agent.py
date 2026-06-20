"""Testes do Agente de Ingestão e Perfilamento (RF03)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.agents.data_profile_agent import DataProfileAgent


def _make_agent() -> DataProfileAgent:
    return DataProfileAgent(event_sink=None)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame(
        {
            "customer_id": range(n),  # identificador
            "age": rng.integers(18, 80, size=n),  # numérica
            "income": rng.normal(5000, 1500, size=n),  # numérica contínua
            "city": rng.choice(["SP", "RJ", "MG"], size=n),  # categórica
            "signup_date": pd.date_range("2020-01-01", periods=n, freq="D"),  # datetime
            "default": rng.choice([0, 1], size=n, p=[0.9, 0.1]),  # alvo desbalanceado
        }
    )


# ---------------------------------------------------------------------------
# Ingestão e estrutura geral
# ---------------------------------------------------------------------------

class TestIngestion:
    def test_basic_shape(self, sample_df):
        result = _make_agent().run({"dataframe": sample_df})
        assert result.output["n_rows"] == 100
        assert result.output["n_cols"] == 6

    def test_output_is_json_serializable(self, sample_df):
        result = _make_agent().run({"dataframe": sample_df, "target_column": "default"})
        # Não deve lançar — tudo precisa ser nativo p/ JSONB.
        json.dumps(result.output)

    def test_does_not_mutate_input(self, sample_df):
        before = sample_df.copy()
        _make_agent().run({"dataframe": sample_df})
        pd.testing.assert_frame_equal(sample_df, before)

    def test_content_hash_is_stable(self, sample_df):
        h1 = _make_agent().run({"dataframe": sample_df}).output["content_hash"]
        h2 = _make_agent().run({"dataframe": sample_df.copy()}).output["content_hash"]
        assert h1 == h2

    def test_content_hash_changes_with_data(self, sample_df):
        h1 = _make_agent().run({"dataframe": sample_df}).output["content_hash"]
        mutated = sample_df.copy()
        mutated.loc[0, "age"] = 999
        h2 = _make_agent().run({"dataframe": mutated}).output["content_hash"]
        assert h1 != h2

    def test_load_from_csv(self, sample_df, tmp_path):
        path = tmp_path / "data.csv"
        sample_df.to_csv(path, index=False)
        result = _make_agent().run(
            {"data": {"source_type": "csv", "source_uri": str(path)}}
        )
        assert result.output["n_rows"] == 100

    def test_missing_source_raises(self):
        with pytest.raises(ValueError, match="dataframe"):
            _make_agent().run({})

    def test_unsupported_source_type_raises(self):
        with pytest.raises(ValueError, match="source_type"):
            _make_agent().run({"data": {"source_type": "xml", "source_uri": "x"}})

    def test_non_dataframe_raises(self):
        with pytest.raises(TypeError):
            _make_agent().run({"dataframe": [[1, 2], [3, 4]]})


# ---------------------------------------------------------------------------
# Inferência de tipos
# ---------------------------------------------------------------------------

class TestTypeInference:
    def _schema(self, df, **ctx):
        return _make_agent().run({"dataframe": df, **ctx}).output["schema"]

    def test_numeric_detected(self, sample_df):
        schema = self._schema(sample_df)
        assert schema["age"] == "numeric"
        assert schema["income"] == "numeric"

    def test_categorical_detected(self, sample_df):
        assert self._schema(sample_df)["city"] == "categorical"

    def test_datetime_detected(self, sample_df):
        assert self._schema(sample_df)["signup_date"] == "datetime"

    def test_string_datetime_detected(self):
        df = pd.DataFrame({"d": ["2021-01-01", "2021-02-01", "2021-03-01"] * 5})
        assert self._schema(df)["d"] == "datetime"

    def test_text_detected(self):
        df = pd.DataFrame(
            {"comment": [f"this is a fairly long free text comment number {i}" for i in range(30)]}
        )
        assert self._schema(df)["comment"] == "text"

    def test_boolean_detected(self):
        df = pd.DataFrame({"flag": [True, False, True, False, True]})
        assert self._schema(df)["flag"] == "boolean"

    def test_constant_detected(self):
        df = pd.DataFrame({"const": [7] * 10})
        assert self._schema(df)["const"] == "constant"


# ---------------------------------------------------------------------------
# Faltantes, duplicatas, cardinalidade
# ---------------------------------------------------------------------------

class TestMissingAndDuplicates:
    def _col(self, output, name):
        return next(c for c in output["columns"] if c["name"] == name)

    def test_missing_counted(self):
        df = pd.DataFrame({"x": [1.0, None, 3.0, None]})
        out = _make_agent().run({"dataframe": df}).output
        col = self._col(out, "x")
        assert col["n_missing"] == 2
        assert col["pct_missing"] == 50.0

    def test_high_missing_warns(self):
        df = pd.DataFrame({"x": [1.0, None, None, None]})
        result = _make_agent().run({"dataframe": df})
        assert any("faltantes" in w for w in result.warnings)

    def test_duplicates_detected(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        out = _make_agent().run({"dataframe": df}).output
        assert out["duplicates"]["n_duplicate_rows"] == 1

    def test_no_duplicates(self, sample_df):
        out = _make_agent().run({"dataframe": sample_df}).output
        assert out["duplicates"]["n_duplicate_rows"] == 0

    def test_cardinality(self, sample_df):
        out = _make_agent().run({"dataframe": sample_df}).output
        assert self._col(out, "city")["n_unique"] == 3

    def test_id_like_flagged(self, sample_df):
        result = _make_agent().run({"dataframe": sample_df, "id_column": "customer_id"})
        col = self._col(result.output, "customer_id")
        assert col["is_id_like"] is True
        assert col["role"] == "id"

    def test_constant_warns(self):
        df = pd.DataFrame({"const": [1] * 5, "x": [1, 2, 3, 4, 5]})
        result = _make_agent().run({"dataframe": df})
        assert any("constante" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Estatísticas numéricas
# ---------------------------------------------------------------------------

class TestNumericStats:
    def test_stats_present(self, sample_df):
        out = _make_agent().run({"dataframe": sample_df}).output
        col = next(c for c in out["columns"] if c["name"] == "age")
        stats = col["stats"]
        for key in ("mean", "std", "min", "p25", "p50", "p75", "max"):
            assert key in stats
        assert stats["min"] <= stats["p50"] <= stats["max"]


# ---------------------------------------------------------------------------
# Análise do alvo / desbalanceamento
# ---------------------------------------------------------------------------

class TestTargetAnalysis:
    def test_classification_imbalance(self, sample_df):
        result = _make_agent().run(
            {"dataframe": sample_df, "target_column": "default", "task_type": "classification"}
        )
        tgt = result.output["target"]
        assert tgt["kind"] == "classification"
        assert tgt["n_classes"] == 2
        assert tgt["imbalance_ratio"] > 1

    def test_imbalance_warns(self, sample_df):
        # default ~ 90/10 => razão ~9; força desbalanceamento mais forte
        df = sample_df.copy()
        df["default"] = [0] * 95 + [1] * 5
        result = _make_agent().run(
            {"dataframe": df, "target_column": "default", "task_type": "classification"}
        )
        assert any("desbalanceado" in w for w in result.warnings)

    def test_regression_target(self):
        df = pd.DataFrame({"price": np.linspace(10, 1000, 100)})
        result = _make_agent().run(
            {"dataframe": df, "target_column": "price", "task_type": "regression"}
        )
        tgt = result.output["target"]
        assert tgt["kind"] == "regression"
        assert "mean" in tgt["stats"]

    def test_invalid_target_raises(self, sample_df):
        with pytest.raises(ValueError, match="target_column"):
            _make_agent().run({"dataframe": sample_df, "target_column": "nope"})

    def test_no_target_means_none(self, sample_df):
        out = _make_agent().run({"dataframe": sample_df}).output
        assert out["target"] is None


# ---------------------------------------------------------------------------
# Correlações
# ---------------------------------------------------------------------------

class TestCorrelations:
    def test_high_correlation_detected(self):
        x = np.arange(100, dtype=float)
        df = pd.DataFrame({"x": x, "y": x * 2 + 1, "z": np.random.default_rng(0).normal(size=100)})
        out = _make_agent().run({"dataframe": df}).output
        pairs = {(p["a"], p["b"]) for p in out["high_correlations"]}
        assert ("x", "y") in pairs

    def test_no_spurious_correlations(self, sample_df):
        out = _make_agent().run({"dataframe": sample_df}).output
        # Não deve haver pares perfeitamente correlacionados na base sintética
        assert all(abs(p["pearson"]) >= 0.95 for p in out["high_correlations"])


# ---------------------------------------------------------------------------
# Auditabilidade / evento
# ---------------------------------------------------------------------------

class TestRationaleAndEvent:
    def test_rationale_non_empty(self, sample_df):
        result = _make_agent().run({"dataframe": sample_df, "target_column": "default"})
        assert len(result.rationale) > 30

    def test_event_emitted_via_call(self, sample_df):
        events: list[dict] = []

        class _Sink:
            def record_event(self, **kwargs):
                events.append(kwargs)

        agent = DataProfileAgent(event_sink=_Sink())
        agent({"experiment_id": "exp-1", "dataframe": sample_df, "agent_input": {"src": "test"}})
        assert len(events) == 1
        assert events[0]["event_type"] == "data_profile"
        assert events[0]["agent_name"] == "Agente de Ingestão e Perfilamento"
