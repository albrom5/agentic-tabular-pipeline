"""Pipeline de avaliação.

Agrega os resultados por modelo/fold em um ranking (média, desvio, intervalo e
tempo), calcula as métricas conforme o tipo de tarefa e compara o uso do
autoencoder contra o baseline.
"""

from __future__ import annotations

from typing import Any


def compute_metrics(task_type: str, y_true: Any, y_pred: Any, y_score: Any | None = None) -> dict[str, float]:
    """Calcula as métricas recomendadas para o tipo de tarefa (seção 10).

    - classification: F1/macro-F1, balanced accuracy, ROC-AUC, PR-AUC, precision, recall
    - regression: MAE, RMSE, R2, (MAPE quando fizer sentido)
    - anomaly: ROC-AUC, PR-AUC, precision@k, recall@k
    """
    # TODO: delegar para sklearn.metrics conforme task_type.
    raise NotImplementedError


def build_ranking(model_results: list[dict[str, Any]], primary_metric: str) -> list[dict[str, Any]]:
    """Ordena os modelos pela média da métrica primária, reportando a dispersão."""
    # TODO: agrupar por model_name, agregar média/desvio por fold e ordenar.
    raise NotImplementedError
