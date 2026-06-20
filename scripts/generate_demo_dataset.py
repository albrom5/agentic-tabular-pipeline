"""Gera a base sintética de demonstração ``data/raw/credit.csv``.

A base imita um problema de risco de crédito (classificação binária da variável
``default``) e foi desenhada para exercitar todas as etapas do pipeline:

* mistura de variáveis numéricas e categóricas;
* valores faltantes controlados (``income``, ``employment_length``);
* algumas linhas duplicadas exatas (para o Agente de Qualidade e Limpeza);
* uma categoria rara em ``purpose`` (para o tratamento de categorias raras);
* desbalanceamento da classe alvo (~20% de ``default = 1``), tornando
  ``macro_f1`` uma métrica mais informativa que a acurácia.

A relação alvo↔features é determinística dado o ``--seed`` (logística + ruído),
de modo que os modelos têm sinal aprendível e o resultado é reprodutível.

Uso:
    python -m scripts.generate_demo_dataset                 # data/raw/credit.csv
    python -m scripts.generate_demo_dataset --rows 2000 --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "raw" / "credit.csv"

_HOME_OWNERSHIP = ["rent", "mortgage", "own"]
_REGIONS = ["north", "south", "east", "west"]
# 'small_business' é deliberadamente rara (~1%) para exercitar o tratamento
# de categorias raras na limpeza/engenharia de atributos.
_PURPOSES = [
    "debt_consolidation",
    "credit_card",
    "home_improvement",
    "medical",
    "other",
    "small_business",
]
_PURPOSE_PROBS = [0.32, 0.28, 0.16, 0.12, 0.11, 0.01]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Constrói o DataFrame sintético de risco de crédito."""
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 76, size=rows)
    # Renda log-normal, correlacionada levemente com idade.
    income = np.round(
        np.exp(rng.normal(10.6, 0.45, size=rows)) + age * 120, 2
    )
    employment_length = np.clip(
        np.round(rng.gamma(shape=2.0, scale=3.5, size=rows), 1), 0, 40
    )
    loan_amount = np.round(rng.uniform(1_000, 40_000, size=rows), 2)
    num_credit_lines = rng.integers(0, 16, size=rows)
    # Razão dívida/renda anual em escala realista (tipicamente 0,1–0,6).
    debt_to_income = np.round(
        np.clip(rng.beta(2.0, 5.0, size=rows) * loan_amount / np.maximum(income, 1) * 2.0, 0, 1.5),
        4,
    )

    home_ownership = rng.choice(_HOME_OWNERSHIP, size=rows, p=[0.45, 0.40, 0.15])
    purpose = rng.choice(_PURPOSES, size=rows, p=_PURPOSE_PROBS)
    region = rng.choice(_REGIONS, size=rows)

    # ------------------------------------------------------------------
    # Variável-alvo: combinação linear + ruído passada por sigmoide.
    # Maior debt_to_income, menor renda/emprego e aluguel elevam o risco.
    # ------------------------------------------------------------------
    income_z = (np.log(np.maximum(income, 1)) - 11.0) / 0.5
    logit = (
        -2.4
        + 2.8 * debt_to_income
        - 0.55 * income_z
        - 0.05 * employment_length
        + 0.04 * (loan_amount / 10_000)
        + 0.35 * (home_ownership == "rent").astype(float)
        + 0.30 * (purpose == "small_business").astype(float)
        - 0.10 * (num_credit_lines >= 8).astype(float)
        + rng.normal(0, 0.6, size=rows)
    )
    default = (rng.uniform(size=rows) < _sigmoid(logit)).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, rows + 1),
            "age": age,
            "income": income,
            "employment_length": employment_length,
            "loan_amount": loan_amount,
            "debt_to_income": debt_to_income,
            "num_credit_lines": num_credit_lines,
            "home_ownership": home_ownership,
            "purpose": purpose,
            "region": region,
            "default": default,
        }
    )

    # Faltantes controlados (~8% em income, ~5% em employment_length).
    df.loc[rng.uniform(size=rows) < 0.08, "income"] = np.nan
    df.loc[rng.uniform(size=rows) < 0.05, "employment_length"] = np.nan

    # Algumas duplicatas exatas para o Agente de Qualidade e Limpeza.
    n_dup = max(5, rows // 200)
    dup_idx = rng.choice(df.index, size=n_dup, replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    # Embaralha para não deixar as duplicatas todas no final.
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1000, help="Número de linhas base.")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade.")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Caminho do CSV de saída."
    )
    args = parser.parse_args()

    df = generate(rows=args.rows, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    rate = df["default"].mean()
    print(f"Base gerada em {args.output}")
    print(f"  linhas={len(df)} colunas={df.shape[1]} taxa_default={rate:.1%}")
    print(f"  faltantes income={df['income'].isna().mean():.1%} "
          f"employment_length={df['employment_length'].isna().mean():.1%}")


if __name__ == "__main__":
    main()
