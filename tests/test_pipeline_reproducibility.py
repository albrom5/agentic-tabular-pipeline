"""Testes de reprodutibilidade do pipeline (RNF01 / RF15).

Garantem que reexecutar um experimento a partir da mesma configuração e seed
produz as mesmas métricas, e que o pré-processamento é ajustado apenas no fold
de treino (sem leakage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from src.agents.trainer_agent import TrainerAgent
from src.pipelines.training import run_experiment


def _clf_df(seed: int = 0, n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    label = (x1 + rng.normal(scale=0.3, size=n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": rng.normal(size=n),
                         "city": rng.choice(["SP", "RJ"], size=n), "label": label})


def _config() -> dict:
    return {
        "experiment": {
            "name": "repro", "task_type": "classification",
            "target_column": "label", "primary_metric": "macro_f1", "random_seed": 7,
        },
        "validation": {"split_strategy": "stratified_kfold", "n_splits": 3},
        "preprocessing": {"scaling": "standard", "categorical_encoding": "one_hot"},
        "models": {"include": ["logistic_regression", "random_forest"]},
    }


def test_same_seed_yields_same_metrics() -> None:
    df = _clf_df()
    a = run_experiment(_config(), dataframe=df)
    b = run_experiment(_config(), dataframe=df)

    # Mesmo modelo recomendado e mesmas médias por modelo no ranking.
    assert a["best_model"] == b["best_model"]
    rank_a = {e["model_name"]: e["primary_mean"] for e in a["ranking"]}
    rank_b = {e["model_name"]: e["primary_mean"] for e in b["ranking"]}
    assert rank_a == rank_b


def test_preprocessing_is_fit_only_on_train_fold() -> None:
    """Um transformador-espião registra quais linhas viu no fit de cada fold."""
    df = _clf_df(n=60)
    seen_per_fit: list[set[float]] = []

    class _Spy(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            # Registra os valores da 1ª coluna vistos neste ajuste.
            arr = np.asarray(X)
            seen_per_fit.append(set(np.round(arr[:, 0], 8).tolist()))
            self.fitted_ = True  # marca de "ajustado" exigida por check_is_fitted
            return self

        def transform(self, X):
            return np.asarray(X, dtype=float)

    feature_cols = ["x1", "x2"]
    pipeline = Pipeline([("spy", _Spy())])

    sk = __import__("sklearn.model_selection", fromlist=["StratifiedKFold"])
    folds = list(
        sk.StratifiedKFold(n_splits=3, shuffle=True, random_state=0).split(
            np.arange(len(df)), df["label"]
        )
    )

    TrainerAgent().run({
        "dataframe": df, "task_type": "classification", "target_column": "label",
        "primary_metric": "macro_f1", "folds": folds, "pipeline": pipeline,
        "feature_columns": feature_cols, "include": ["logistic_regression"],
    })

    # Um fit por fold; os valores vistos no fit nunca incluem os do teste do fold.
    assert len(seen_per_fit) == len(folds)
    for (train_idx, test_idx), seen in zip(folds, seen_per_fit):
        test_values = set(np.round(df.iloc[test_idx]["x1"].to_numpy(), 8).tolist())
        assert seen.isdisjoint(test_values)
        assert len(seen) <= len(train_idx)
