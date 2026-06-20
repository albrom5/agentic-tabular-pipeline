"""Testes do Agente de Engenharia de Atributos (RF07)."""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.agents.feature_agent import FeatureAgent


def _make_agent() -> FeatureAgent:
    return FeatureAgent(event_sink=None)


@pytest.fixture
def mixed_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 40
    return pd.DataFrame(
        {
            "customer_id": range(n),                               # identificador
            "age": rng.integers(18, 80, size=n).astype(float),     # numérica
            "income": rng.normal(5000, 1500, size=n),              # numérica
            "city": rng.choice(["SP", "RJ", "MG"], size=n),        # categórica baixa card.
            "signup_date": pd.date_range("2021-01-01", periods=n, freq="D"),  # datetime
            "const": [1] * n,                                      # constante
            "label": rng.choice([0, 1], size=n),                  # alvo
        }
    )


# ---------------------------------------------------------------------------
# Estrutura geral e contrato
# ---------------------------------------------------------------------------

class TestStructure:
    def test_returns_unfitted_pipeline(self, mixed_df):
        result = _make_agent().run({"dataframe": mixed_df, "target_column": "label"})
        assert isinstance(result.pipeline, Pipeline)
        # Não ajustado: o ColumnTransformer ainda não tem atributos pós-fit.
        assert not hasattr(result.pipeline.named_steps["preprocessor"], "transformers_")

    def test_output_is_json_serializable(self, mixed_df):
        result = _make_agent().run({"dataframe": mixed_df, "target_column": "label"})
        json.dumps(result.output)

    def test_pipeline_is_picklable(self, mixed_df):
        # "Gerar pipeline serializável" (RF07): pickle/joblib devem funcionar.
        result = _make_agent().run({"dataframe": mixed_df, "target_column": "label"})
        restored = pickle.loads(pickle.dumps(result.pipeline))
        assert isinstance(restored, Pipeline)

    def test_does_not_mutate_input(self, mixed_df):
        before = mixed_df.copy()
        _make_agent().run({"dataframe": mixed_df, "target_column": "label"})
        pd.testing.assert_frame_equal(mixed_df, before)

    def test_invalid_target_raises(self, mixed_df):
        with pytest.raises(ValueError, match="target_column"):
            _make_agent().run({"dataframe": mixed_df, "target_column": "nope"})

    def test_invalid_strategy_raises(self, mixed_df):
        with pytest.raises(ValueError, match="scaling"):
            _make_agent().run({"dataframe": mixed_df, "scaling": "bogus"})

    def test_no_features_raises(self):
        df = pd.DataFrame({"label": [0, 1, 0, 1]})
        with pytest.raises(ValueError, match="atributo"):
            _make_agent().run({"dataframe": df, "target_column": "label"})


# ---------------------------------------------------------------------------
# Classificação das colunas em grupos
# ---------------------------------------------------------------------------

class TestColumnGrouping:
    def test_groups_by_type(self, mixed_df):
        out = _make_agent().run({"dataframe": mixed_df, "target_column": "label"}).output
        groups = out["column_groups"]
        assert set(groups["numeric"]) == {"age", "income", "customer_id"}
        assert groups["categorical"] == ["city"]
        assert groups["datetime"] == ["signup_date"]

    def test_target_and_constant_excluded(self, mixed_df):
        out = _make_agent().run({"dataframe": mixed_df, "target_column": "label"}).output
        assert "label" in out["excluded_columns"]
        assert "const" in out["excluded_columns"]
        assert "label" not in out["feature_columns"]

    def test_id_like_excluded_via_profile(self, mixed_df):
        profile = {
            "schema": {c: t for c, t in [
                ("customer_id", "numeric"), ("age", "numeric"), ("income", "numeric"),
                ("city", "categorical"), ("signup_date", "datetime"),
                ("const", "constant"), ("label", "categorical"),
            ]},
            "columns": [{"name": "customer_id", "is_id_like": True, "is_target": False}],
        }
        out = _make_agent().run(
            {"dataframe": mixed_df, "target_column": "label", "data_profile": profile}
        ).output
        assert "customer_id" in out["excluded_columns"]
        assert "customer_id" not in out["column_groups"]["numeric"]


# ---------------------------------------------------------------------------
# Transformação efetiva (ajustando o pipeline em dados de "treino")
# ---------------------------------------------------------------------------

class TestTransformation:
    def test_fit_transform_produces_numeric_matrix(self, mixed_df):
        result = _make_agent().run({"dataframe": mixed_df, "target_column": "label"})
        X = mixed_df.drop(columns="label")
        matrix = result.pipeline.fit_transform(X)
        assert matrix.shape[0] == len(mixed_df)
        assert np.isfinite(np.asarray(matrix, dtype=float)).all()

    def test_onehot_expands_categorical(self, mixed_df):
        result = _make_agent().run({"dataframe": mixed_df, "target_column": "label"})
        names = result.output["output_feature_names"]
        # city tem 3 categorias => 3 colunas one-hot
        assert sum(n.startswith("city") for n in names) == 3

    def test_datetime_expanded_into_parts(self, mixed_df):
        out = _make_agent().run({"dataframe": mixed_df, "target_column": "label"}).output
        names = out["output_feature_names"]
        assert "signup_date__year" in names
        assert "signup_date__dayofweek" in names

    def test_handle_unknown_category_at_transform(self, mixed_df):
        result = _make_agent().run({"dataframe": mixed_df, "target_column": "label"})
        X = mixed_df.drop(columns="label")
        result.pipeline.fit(X)
        unseen = X.iloc[[0]].copy()
        unseen["city"] = "XX"  # categoria nunca vista no treino
        transformed = result.pipeline.transform(unseen)  # não deve lançar
        assert transformed.shape[0] == 1

    def test_high_cardinality_uses_ordinal(self):
        # 30 categorias repetidas (ratio baixa => categórica), acima do limiar 20.
        df = pd.DataFrame({
            "hc": [f"c{i}" for i in range(30)] * 4,
            "y": list(range(120)),
        })
        out = _make_agent().run(
            {"dataframe": df, "target_column": "y", "high_cardinality_threshold": 20}
        ).output
        groups = [d["group"] for d in out["transformers"]]
        assert "categorical_high_card" in groups

    def test_scaling_none_keeps_raw_scale(self, mixed_df):
        result = _make_agent().run(
            {"dataframe": mixed_df, "target_column": "label", "scaling": "none"}
        )
        steps = next(t for t in result.output["transformers"] if t["group"] == "numeric")["steps"]
        assert not any("Scaler" in s for s in steps)

    def test_interactions_added_when_requested(self, mixed_df):
        result = _make_agent().run(
            {"dataframe": mixed_df, "target_column": "label", "add_interactions": True}
        )
        steps = next(t for t in result.output["transformers"] if t["group"] == "numeric")["steps"]
        assert any("Polynomial" in s for s in steps)

    def test_text_stats_features(self):
        df = pd.DataFrame({
            "comment": [f"este é um comentário de texto livre número {i}" for i in range(30)],
            "y": list(range(30)),
        })
        result = _make_agent().run({"dataframe": df, "target_column": "y"})
        names = result.output["output_feature_names"]
        assert "comment__char_len" in names
        assert "comment__n_words" in names


# ---------------------------------------------------------------------------
# Leakage e auditabilidade
# ---------------------------------------------------------------------------

class TestLeakageAndEvent:
    def test_leakage_notes_present(self, mixed_df):
        out = _make_agent().run({"dataframe": mixed_df, "target_column": "label"}).output
        assert any("vazamento" in n.lower() for n in out["leakage_notes"])

    def test_target_correlation_flagged_from_profile(self, mixed_df):
        profile = {"high_correlations": [{"a": "income", "b": "label", "pearson": 0.99}]}
        out = _make_agent().run(
            {"dataframe": mixed_df, "target_column": "label", "data_profile": profile}
        ).output
        assert any("income" in n for n in out["leakage_notes"])

    def test_rationale_non_empty(self, mixed_df):
        result = _make_agent().run({"dataframe": mixed_df, "target_column": "label"})
        assert len(result.rationale) > 30

    def test_event_emitted_via_call(self, mixed_df):
        events: list[dict] = []

        class _Sink:
            def record_event(self, **kwargs):
                events.append(kwargs)

        agent = FeatureAgent(event_sink=_Sink())
        agent({"experiment_id": "exp-1", "dataframe": mixed_df, "target_column": "label"})
        assert len(events) == 1
        assert events[0]["event_type"] == "feature_engineering"
        assert events[0]["agent_name"] == "Agente de Engenharia de Atributos"
