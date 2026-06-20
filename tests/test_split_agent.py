"""Testes do Agente de Split e Validação (RF08)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.agents.split_agent import SplitAgent


def _make_agent() -> SplitAgent:
    return SplitAgent(event_sink=None)


@pytest.fixture
def clf_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 100
    return pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "label": rng.choice([0, 1], size=n, p=[0.7, 0.3]),
            "patient": rng.integers(0, 10, size=n),  # grupos
            "date": pd.date_range("2022-01-01", periods=n, freq="D"),
        }
    )


def _all_test_indices(result) -> list[int]:
    out: list[int] = []
    for fold in result.output["folds"]:
        out.extend(fold["test_idx"])
    return out


# ---------------------------------------------------------------------------
# Contrato geral
# ---------------------------------------------------------------------------

class TestStructure:
    def test_output_is_json_serializable(self, clf_df):
        result = _make_agent().run({
            "dataframe": clf_df, "split_strategy": "kfold",
        })
        json.dumps(result.output)

    def test_folds_exposed_as_arrays(self, clf_df):
        result = _make_agent().run({"dataframe": clf_df, "split_strategy": "kfold"})
        assert len(result.folds) == 5
        train_idx, test_idx = result.folds[0]
        assert isinstance(train_idx, np.ndarray)

    def test_seed_recorded(self, clf_df):
        result = _make_agent().run({
            "dataframe": clf_df, "split_strategy": "kfold", "random_seed": 123,
        })
        assert result.output["random_seed"] == 123

    def test_invalid_strategy_raises(self, clf_df):
        with pytest.raises(ValueError, match="split_strategy"):
            _make_agent().run({"dataframe": clf_df, "split_strategy": "random"})

    def test_missing_column_raises(self, clf_df):
        with pytest.raises(ValueError, match="group_column"):
            _make_agent().run({
                "dataframe": clf_df, "split_strategy": "group_kfold", "group_column": "nope",
            })

    def test_too_few_rows_raises(self):
        with pytest.raises(ValueError, match="2 linhas"):
            _make_agent().run({"dataframe": pd.DataFrame({"x": [1]}), "split_strategy": "kfold"})

    def test_reads_strategy_from_validation(self, clf_df):
        result = _make_agent().run({
            "dataframe": clf_df,
            "validation": {
                "split_strategy": "stratified_kfold",
                "task_type": "classification",
                "target_column": "label",
            },
        })
        assert result.output["split_strategy"] == "stratified_kfold"
        assert result.output["stratified"] is True


# ---------------------------------------------------------------------------
# Reprodutibilidade e ausência de contaminação
# ---------------------------------------------------------------------------

class TestReproducibilityAndContamination:
    def test_same_seed_same_folds(self, clf_df):
        a = _make_agent().run({"dataframe": clf_df, "split_strategy": "kfold", "random_seed": 7})
        b = _make_agent().run({"dataframe": clf_df, "split_strategy": "kfold", "random_seed": 7})
        assert a.output["folds"] == b.output["folds"]

    def test_different_seed_different_folds(self, clf_df):
        a = _make_agent().run({"dataframe": clf_df, "split_strategy": "kfold", "random_seed": 1})
        b = _make_agent().run({"dataframe": clf_df, "split_strategy": "kfold", "random_seed": 2})
        assert a.output["folds"] != b.output["folds"]

    def test_train_test_disjoint(self, clf_df):
        result = _make_agent().run({"dataframe": clf_df, "split_strategy": "kfold"})
        for train_idx, test_idx in result.folds:
            assert not (set(train_idx.tolist()) & set(test_idx.tolist()))
        assert result.output["no_contamination_verified"] is True

    def test_kfold_covers_all_samples_once_as_test(self, clf_df):
        result = _make_agent().run({"dataframe": clf_df, "split_strategy": "kfold"})
        test_indices = sorted(_all_test_indices(result))
        assert test_indices == list(range(len(clf_df)))


# ---------------------------------------------------------------------------
# Estratégias específicas
# ---------------------------------------------------------------------------

class TestStrategies:
    def test_holdout(self, clf_df):
        result = _make_agent().run({
            "dataframe": clf_df, "split_strategy": "holdout", "test_size": 0.25,
            "target_column": "label", "task_type": "classification",
        })
        assert result.output["n_splits"] == 1
        fold = result.output["folds"][0]
        assert fold["n_test"] == 25
        assert fold["n_train"] == 75
        assert result.output["stratified"] is True

    def test_stratified_preserves_class_ratio(self, clf_df):
        result = _make_agent().run({
            "dataframe": clf_df, "split_strategy": "stratified_kfold",
            "task_type": "classification", "target_column": "label",
        })
        y = clf_df["label"].to_numpy()
        overall = y.mean()
        for _, test_idx in result.folds:
            # proporção da classe positiva no fold próxima da global
            assert abs(y[test_idx].mean() - overall) < 0.15

    def test_group_kfold_groups_disjoint(self, clf_df):
        result = _make_agent().run({
            "dataframe": clf_df, "split_strategy": "group_kfold", "group_column": "patient",
        })
        groups = clf_df["patient"].to_numpy()
        assert result.output["grouped"] is True
        for train_idx, test_idx in result.folds:
            assert not (set(groups[train_idx]) & set(groups[test_idx]))

    def test_time_split_no_future_in_train(self, clf_df):
        result = _make_agent().run({
            "dataframe": clf_df, "split_strategy": "time_split", "time_column": "date",
        })
        assert result.output["temporal"] is True
        assert result.output["shuffle"] is False
        dates = clf_df["date"].to_numpy()
        for train_idx, test_idx in result.folds:
            # todo treino é anterior a todo teste (passado não vê o futuro)
            assert dates[train_idx].max() <= dates[test_idx].min()


# ---------------------------------------------------------------------------
# Robustez (RNF04) — fallbacks e limites
# ---------------------------------------------------------------------------

class TestRobustness:
    def test_n_splits_reduced_for_small_groups(self):
        df = pd.DataFrame({"x": range(12), "g": [0, 1, 2] * 4})  # só 3 grupos
        result = _make_agent().run({
            "dataframe": df, "split_strategy": "group_kfold", "group_column": "g", "n_splits": 5,
        })
        assert result.output["n_splits"] == 3
        assert any("reduzido" in w for w in result.warnings)

    def test_stratified_falls_back_for_rare_class(self):
        df = pd.DataFrame({"x": range(10), "y": [0] * 9 + [1]})  # classe rara (1 amostra)
        result = _make_agent().run({
            "dataframe": df, "split_strategy": "stratified_kfold",
            "task_type": "classification", "target_column": "y",
        })
        assert result.output["split_strategy"] == "kfold"
        assert any("estratificação" in w.lower() for w in result.warnings)

    def test_stratified_falls_back_for_regression(self, clf_df):
        result = _make_agent().run({
            "dataframe": clf_df, "split_strategy": "stratified_kfold",
            "task_type": "regression", "target_column": "x",
        })
        assert result.output["split_strategy"] == "kfold"

    def test_invalid_n_splits_raises(self, clf_df):
        with pytest.raises(ValueError, match="n_splits"):
            _make_agent().run({"dataframe": clf_df, "split_strategy": "kfold", "n_splits": 1})

    def test_group_kfold_single_group_raises(self):
        df = pd.DataFrame({"x": range(6), "g": [0] * 6})
        with pytest.raises(ValueError, match="grupos"):
            _make_agent().run({
                "dataframe": df, "split_strategy": "group_kfold", "group_column": "g",
            })


# ---------------------------------------------------------------------------
# Auditabilidade
# ---------------------------------------------------------------------------

class TestRationaleAndEvent:
    def test_rationale_non_empty(self, clf_df):
        result = _make_agent().run({"dataframe": clf_df, "split_strategy": "kfold"})
        assert len(result.rationale) > 30

    def test_event_emitted_via_call(self, clf_df):
        events: list[dict] = []

        class _Sink:
            def record_event(self, **kwargs):
                events.append(kwargs)

        agent = SplitAgent(event_sink=_Sink())
        agent({"experiment_id": "exp-1", "dataframe": clf_df, "split_strategy": "kfold"})
        assert len(events) == 1
        assert events[0]["event_type"] == "split_strategy"
        assert events[0]["agent_name"] == "Agente de Split e Validação"
