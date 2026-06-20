"""Dados sintéticos (mock) que alimentam a interface Streamlit.

Esta camada existe apenas para validar a aparência e a usabilidade da UI
(seção 12 do documento de apoio). Nada aqui toca o banco ou executa modelos:
quando o backend estiver pronto, estas funções serão substituídas por consultas
reais a ``experiments``, ``datasets``, ``agent_events`` e ``model_results``.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Catálogo de experimentos (mock da tabela `experiments`)
# ---------------------------------------------------------------------------

EXPERIMENTS: list[dict[str, Any]] = [
    {
        "id": "2d0b4c8a-1f3e-4a21-9c77-1a2b3c4d5e6f",
        "name": "baseline_credit_risk",
        "task_type": "classification",
        "target_column": "default",
        "primary_metric": "macro_f1",
        "status": "concluído",
        "created_at": "2026-06-15 09:12",
    },
    {
        "id": "7a1c9e22-88bb-4d10-a0f1-9e8d7c6b5a40",
        "name": "house_prices_v2",
        "task_type": "regression",
        "target_column": "sale_price",
        "primary_metric": "rmse",
        "status": "em execução",
        "created_at": "2026-06-16 14:33",
    },
    {
        "id": "f3e2d1c0-aaaa-4bbb-8ccc-0123456789ab",
        "name": "fraud_anomaly_poc",
        "task_type": "anomaly",
        "target_column": None,
        "primary_metric": "roc_auc",
        "status": "rascunho",
        "created_at": "2026-06-17 08:05",
    },
]

TASK_TYPES = ["classification", "regression", "anomaly"]

METRICS_BY_TASK: dict[str, list[str]] = {
    "classification": ["macro_f1", "weighted_f1", "roc_auc", "balanced_accuracy", "accuracy"],
    "regression": ["rmse", "mae", "r2", "mape"],
    "anomaly": ["roc_auc", "average_precision", "precision_at_k"],
}

SPLIT_STRATEGIES = ["holdout", "kfold", "stratified_kfold", "group_kfold", "time_split"]

SAMPLE_DATASETS = ["credit.csv", "house_prices.parquet", "transactions.csv"]


# ---------------------------------------------------------------------------
# Base de exemplo (mock da ingestão)
# ---------------------------------------------------------------------------

def sample_dataframe(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Gera uma base sintética de risco de crédito, plausível e reprodutível."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "age": rng.integers(18, 80, size=n),
            "income": np.round(rng.normal(5200, 1800, size=n).clip(800), 2),
            "credit_score": rng.integers(300, 850, size=n),
            "n_dependents": rng.integers(0, 5, size=n),
            "city": rng.choice(["SP", "RJ", "MG", "RS", "BA"], size=n,
                               p=[0.4, 0.25, 0.15, 0.12, 0.08]),
            "product": rng.choice(["cartão", "consignado", "pessoal", "imobiliário"], size=n),
            "signup_date": pd.to_datetime("2021-01-01")
            + pd.to_timedelta(rng.integers(0, 1500, size=n), unit="D"),
            "default": rng.choice([0, 1], size=n, p=[0.88, 0.12]),
        }
    )
    # Introduz faltantes realistas para exercitar os alertas de qualidade.
    df.loc[rng.choice(n, size=int(0.23 * n), replace=False), "income"] = np.nan
    df.loc[rng.choice(n, size=int(0.04 * n), replace=False), "age"] = np.nan
    return df


# ---------------------------------------------------------------------------
# Relatório de qualidade (mock do quality_report_json — RF04/RF06)
# ---------------------------------------------------------------------------

def quality_report() -> dict[str, Any]:
    return {
        "alerts": [
            {
                "severity": "alta",
                "rule": "Faltantes",
                "column": "income",
                "message": "23% de valores ausentes em variável numérica relevante.",
                "suggestion": "Imputação pela mediana dentro de cada fold.",
            },
            {
                "severity": "alta",
                "rule": "Leakage potencial",
                "column": "credit_score",
                "message": "Correlação 0.97 com o alvo 'default' — possível vazamento.",
                "suggestion": "Confirmar se a variável é conhecida antes do evento previsto.",
            },
            {
                "severity": "média",
                "rule": "Duplicatas",
                "column": "—",
                "message": "12 linhas duplicadas exatas detectadas.",
                "suggestion": "Remoção controlada com registro da decisão.",
            },
            {
                "severity": "média",
                "rule": "Desbalanceamento",
                "column": "default",
                "message": "Classe positiva representa apenas 12% das amostras.",
                "suggestion": "Usar PR-AUC/recall e considerar reamostragem.",
            },
            {
                "severity": "baixa",
                "rule": "Categorias raras",
                "column": "city",
                "message": "Categoria 'BA' aparece em <10% dos registros.",
                "suggestion": "Agrupar categorias raras em 'outros'.",
            },
        ],
        "summary": {"alta": 2, "média": 2, "baixa": 1},
    }


# ---------------------------------------------------------------------------
# Ações propostas pelos agentes (mock — item 12: aceitar/ajustar)
# ---------------------------------------------------------------------------

def proposed_actions() -> list[dict[str, Any]]:
    return [
        {
            "agent": "Agente de Qualidade e Limpeza",
            "action": "Imputar 'income' pela mediana",
            "reason": "23% de faltantes; variável numérica relevante.",
            "default_accepted": True,
        },
        {
            "agent": "Agente de Qualidade e Limpeza",
            "action": "Remover 12 duplicatas exatas",
            "reason": "Linhas idênticas inflam métricas.",
            "default_accepted": True,
        },
        {
            "agent": "Agente de Engenharia de Atributos",
            "action": "Agrupar categorias raras de 'city' em 'outros'",
            "reason": "Categoria 'BA' < 10% das amostras.",
            "default_accepted": True,
        },
        {
            "agent": "Agente de Engenharia de Atributos",
            "action": "Derivar 'account_age_days' de 'signup_date'",
            "reason": "Datas brutas não são informativas para modelos clássicos.",
            "default_accepted": True,
        },
        {
            "agent": "Agente de Split e Validação",
            "action": "Usar StratifiedKFold (k=5)",
            "reason": "Classificação com alvo desbalanceado.",
            "default_accepted": True,
        },
        {
            "agent": "Agente de Qualidade e Limpeza",
            "action": "Descartar 'credit_score' (suspeita de leakage)",
            "reason": "Correlação 0.97 com o alvo.",
            "default_accepted": False,
        },
    ]


# ---------------------------------------------------------------------------
# Ranking de modelos (mock do model_results — RF11)
# ---------------------------------------------------------------------------

def model_ranking() -> pd.DataFrame:
    rows = [
        ("HistGradientBoosting", 0.812, 0.031, 0.889, 4.2),
        ("Random Forest", 0.796, 0.028, 0.871, 6.8),
        ("Autoencoder + LogReg", 0.781, 0.041, 0.858, 9.1),
        ("Logistic Regression", 0.742, 0.022, 0.833, 0.6),
        ("SVM (RBF)", 0.738, 0.036, 0.820, 12.4),
        ("KNN", 0.701, 0.045, 0.788, 1.1),
        ("MLP", 0.726, 0.052, 0.811, 7.7),
    ]
    df = pd.DataFrame(
        rows, columns=["modelo", "macro_f1", "desvio", "roc_auc", "tempo_fit_s"]
    )
    return df.sort_values("macro_f1", ascending=False).reset_index(drop=True)


def fold_scores() -> pd.DataFrame:
    """Métrica primária por fold para os 3 melhores modelos (gráfico de linhas)."""
    rng = np.random.default_rng(7)
    folds = [f"fold {i}" for i in range(1, 6)]
    data = {
        "HistGradientBoosting": np.round(0.812 + rng.normal(0, 0.025, 5), 3),
        "Random Forest": np.round(0.796 + rng.normal(0, 0.025, 5), 3),
        "Autoencoder + LogReg": np.round(0.781 + rng.normal(0, 0.03, 5), 3),
    }
    return pd.DataFrame(data, index=folds)


def roc_curve() -> pd.DataFrame:
    fpr = np.linspace(0, 1, 50)
    tpr = np.clip(fpr ** 0.35 + np.linspace(0, 0.05, 50), 0, 1)
    return pd.DataFrame({"taxa de falsos positivos": fpr, "taxa de verdadeiros positivos": tpr})


def confusion_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [[372, 18], [29, 81]],
        index=["real: adimplente", "real: inadimplente"],
        columns=["prev: adimplente", "prev: inadimplente"],
    )


def feature_importance() -> pd.DataFrame:
    data = {
        "atributo": ["income", "age", "account_age_days", "product", "n_dependents", "city"],
        "importância": [0.31, 0.22, 0.18, 0.12, 0.10, 0.07],
    }
    return pd.DataFrame(data).set_index("atributo")


def recommendation() -> dict[str, Any]:
    return {
        "model": "HistGradientBoosting",
        "primary_metric": "macro_f1",
        "score": 0.812,
        "std": 0.031,
        "threshold": 0.70,
        "passes": True,
        "text": (
            "O **HistGradientBoosting** é recomendado: maior macro-F1 médio (0.812 ± 0.031), "
            "acima do critério de sucesso (≥ 0.70), com tempo de treino baixo (4.2 s). "
            "Supera a linha de base (Logistic Regression, 0.742) e o classificador com "
            "autoencoder (0.781), justificando a escolha sem cherry-picking."
        ),
    }


# ---------------------------------------------------------------------------
# Histórico de eventos agentivos (mock do agent_events — RNF03)
# ---------------------------------------------------------------------------

def agent_events() -> list[dict[str, Any]]:
    base = datetime(2026, 6, 15, 9, 12, 0)
    raw = [
        ("Agente de Formulação do Problema", "problem_definition",
         "Tarefa definida como classificação binária; métrica macro_f1; sucesso ≥ 0.70."),
        ("Agente de Ingestão e Perfilamento", "data_profile",
         "500 linhas × 9 colunas; 1 datetime, 5 numéricas, 2 categóricas; alvo desbalanceado (12%)."),
        ("Agente de Qualidade e Limpeza", "cleaning_decision",
         "Imputação de 'income' pela mediana; remoção de 12 duplicatas exatas."),
        ("Agente de Engenharia de Atributos", "feature_engineering",
         "Derivada 'account_age_days'; one-hot em categóricas; categorias raras agrupadas."),
        ("Agente de Split e Validação", "split_decision",
         "StratifiedKFold (k=5, seed=42) escolhido pelo desbalanceamento do alvo."),
        ("Agente de Model Zoo", "training_run",
         "7 famílias treinadas; parâmetros, seeds e métricas por fold registrados."),
        ("Agente de Autoencoders", "autoencoder_run",
         "Autoencoder denso (latent_dim=8) treinado por fold; features latentes comparadas à baseline."),
        ("Agente de Avaliação", "model_selection",
         "HistGradientBoosting selecionado por maior macro_f1 médio com baixo desvio."),
        ("Agente Relator e Auditor", "report_generated",
         "Relatório técnico e model card gerados; limitações e risco de leakage documentados."),
    ]
    events = []
    for i, (agent, etype, rationale) in enumerate(raw):
        events.append(
            {
                "timestamp": (base + timedelta(minutes=3 * i)).strftime("%Y-%m-%d %H:%M:%S"),
                "agent_name": agent,
                "event_type": etype,
                "rationale": rationale,
            }
        )
    return events
