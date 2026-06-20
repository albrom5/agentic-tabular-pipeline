"""Agente de Formulação do Problema.

Converte o objetivo informado em uma tarefa de ML bem definida: variável-alvo,
tipo de tarefa, métrica primária, restrições e critério de sucesso.

Cuidado principal: evitar tarefa mal definida e métricas incoerentes.
"""

from __future__ import annotations

from typing import Any

from src.agents.base import AgentResult, BaseAgent

# Métricas válidas por tipo de tarefa
_VALID_METRICS: dict[str, set[str]] = {
    "classification": {
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "roc_auc",
        "average_precision",
        "log_loss",
        "balanced_accuracy",
        "kappa",
    },
    "regression": {
        "rmse",
        "mae",
        "mse",
        "r2",
        "mape",
        "medae",
    },
    "anomaly": {
        "roc_auc",
        "average_precision",
        "f1",
        "precision_at_k",
    },
}

_VALID_TASK_TYPES = set(_VALID_METRICS.keys())

# Métricas que dependem de rótulos verdadeiros e não fazem sentido para anomaly sem ground truth
_REQUIRES_TARGET = {"accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "kappa", "log_loss"}

# Estratégias de split permitidas
_VALID_SPLIT_STRATEGIES = {
    "holdout",
    "kfold",
    "stratified_kfold",
    "group_kfold",
    "time_split",
}


class ProblemAgent(BaseAgent):
    """Agente 1 — Formulação do Problema.

    Entradas esperadas em ``context``:
        - ``task_type`` (str): "classification", "regression" ou "anomaly".
        - ``target_column`` (str | None): nome da coluna alvo. Obrigatório para
          classification e regression; opcional para anomaly (detecção não supervisionada).
        - ``primary_metric`` (str): métrica principal de avaliação.
        - ``split_strategy`` (str, opcional): estratégia de particionamento;
          padrão "stratified_kfold" para classificação e "kfold" para os demais.
        - ``success_threshold`` (float, opcional): valor mínimo da métrica primária
          considerado sucesso. Se não informado, o agente sugere um default.
        - ``constraints`` (dict, opcional): restrições adicionais livres
          (ex.: {"max_fit_seconds": 60, "interpretability": "high"}).
        - ``time_column`` (str | None, opcional): coluna temporal; exige time_split.
        - ``group_column`` (str | None, opcional): coluna de grupo; sugere group_kfold.

    Saída em ``AgentResult.output``:
        - ``task_type``, ``target_column``, ``primary_metric``,
          ``split_strategy``, ``success_threshold``, ``constraints``,
          ``secondary_metrics`` (list[str]).
    """

    name = "Agente de Formulação do Problema"
    event_type = "problem_definition"

    def run(self, context: dict[str, Any]) -> AgentResult:
        warnings: list[str] = []

        # ------------------------------------------------------------------
        # 1. Extração e normalização dos campos de entrada
        # ------------------------------------------------------------------
        task_type: str = str(context.get("task_type", "")).strip().lower()
        target_column: str | None = context.get("target_column") or None
        primary_metric: str = str(context.get("primary_metric", "")).strip().lower()
        split_strategy: str | None = (context.get("split_strategy") or "").strip().lower() or None
        success_threshold: float | None = context.get("success_threshold")
        constraints: dict[str, Any] = dict(context.get("constraints") or {})
        time_column: str | None = context.get("time_column") or None
        group_column: str | None = context.get("group_column") or None

        errors: list[str] = []

        # ------------------------------------------------------------------
        # 2. Validação do tipo de tarefa
        # ------------------------------------------------------------------
        if not task_type:
            errors.append("'task_type' é obrigatório.")
        elif task_type not in _VALID_TASK_TYPES:
            errors.append(
                f"'task_type' inválido: '{task_type}'. "
                f"Valores aceitos: {sorted(_VALID_TASK_TYPES)}."
            )

        # ------------------------------------------------------------------
        # 3. Validação da variável-alvo
        # ------------------------------------------------------------------
        if task_type in ("classification", "regression") and not target_column:
            errors.append(
                f"'target_column' é obrigatório para task_type='{task_type}'."
            )
        if task_type == "anomaly" and not target_column:
            warnings.append(
                "Detecção de anomalias sem 'target_column': modo não supervisionado. "
                "Métricas de avaliação supervisionadas não estarão disponíveis."
            )

        # ------------------------------------------------------------------
        # 4. Validação de coerência entre task_type e primary_metric
        # ------------------------------------------------------------------
        if task_type in _VALID_METRICS and primary_metric:
            allowed = _VALID_METRICS[task_type]
            if primary_metric not in allowed:
                errors.append(
                    f"'primary_metric' '{primary_metric}' é incompatível com "
                    f"task_type='{task_type}'. "
                    f"Métricas válidas: {sorted(allowed)}."
                )
        elif not primary_metric and task_type in _VALID_METRICS:
            # Escolhe métrica padrão razoável
            defaults = {
                "classification": "macro_f1",
                "regression": "rmse",
                "anomaly": "roc_auc",
            }
            primary_metric = defaults[task_type]
            warnings.append(
                f"'primary_metric' não informada. Usando padrão: '{primary_metric}'."
            )

        # ------------------------------------------------------------------
        # 5. Validação e derivação da estratégia de split
        # ------------------------------------------------------------------
        if split_strategy and split_strategy not in _VALID_SPLIT_STRATEGIES:
            errors.append(
                f"'split_strategy' inválida: '{split_strategy}'. "
                f"Valores aceitos: {sorted(_VALID_SPLIT_STRATEGIES)}."
            )
        else:
            if time_column and split_strategy and split_strategy != "time_split":
                warnings.append(
                    f"'time_column' informado mas 'split_strategy' é '{split_strategy}'. "
                    "Considere usar 'time_split' para dados temporais."
                )
            if group_column and split_strategy and split_strategy not in ("group_kfold",):
                warnings.append(
                    f"'group_column' informado mas 'split_strategy' é '{split_strategy}'. "
                    "Considere usar 'group_kfold' para evitar vazamento de dados entre grupos."
                )

            # Deriva split_strategy padrão quando não informada
            if not split_strategy:
                if time_column:
                    split_strategy = "time_split"
                    warnings.append(
                        "'split_strategy' não informada. "
                        "Usando 'time_split' por causa de 'time_column'."
                    )
                elif group_column:
                    split_strategy = "group_kfold"
                    warnings.append(
                        "'split_strategy' não informada. "
                        "Usando 'group_kfold' por causa de 'group_column'."
                    )
                elif task_type == "classification":
                    split_strategy = "stratified_kfold"
                    warnings.append(
                        "'split_strategy' não informada. "
                        "Usando 'stratified_kfold' (padrão para classificação)."
                    )
                else:
                    split_strategy = "kfold"
                    warnings.append(
                        "'split_strategy' não informada. "
                        f"Usando 'kfold' (padrão para task_type='{task_type}')."
                    )

        # ------------------------------------------------------------------
        # 6. Threshold de sucesso
        # ------------------------------------------------------------------
        if success_threshold is not None:
            if not isinstance(success_threshold, (int, float)):
                errors.append("'success_threshold' deve ser numérico.")
            elif not (0.0 <= float(success_threshold) <= 1.0) and primary_metric not in (
                "rmse", "mae", "mse", "mape", "medae", "log_loss"
            ):
                warnings.append(
                    f"'success_threshold'={success_threshold} está fora do intervalo [0, 1]. "
                    "Verifique se está na mesma escala que a métrica primária."
                )
        else:
            # Sugere thresholds conservadores por convenção
            _default_thresholds: dict[str, float] = {
                "macro_f1": 0.70,
                "weighted_f1": 0.70,
                "accuracy": 0.80,
                "balanced_accuracy": 0.70,
                "roc_auc": 0.75,
                "average_precision": 0.60,
                "r2": 0.60,
                "kappa": 0.60,
            }
            success_threshold = _default_thresholds.get(primary_metric)
            if success_threshold is not None:
                warnings.append(
                    f"'success_threshold' não informado. "
                    f"Usando default conservador: {success_threshold} para '{primary_metric}'."
                )

        # ------------------------------------------------------------------
        # 7. Falha explícita em caso de erros críticos
        # ------------------------------------------------------------------
        if errors:
            raise ValueError(
                "Formulação do problema inválida:\n" + "\n".join(f"  • {e}" for e in errors)
            )

        # ------------------------------------------------------------------
        # 8. Métricas secundárias sugeridas
        # ------------------------------------------------------------------
        secondary_metrics = _secondary_metrics(task_type, primary_metric)

        # ------------------------------------------------------------------
        # 9. Monta output e justificativa
        # ------------------------------------------------------------------
        output: dict[str, Any] = {
            "task_type": task_type,
            "target_column": target_column,
            "primary_metric": primary_metric,
            "secondary_metrics": secondary_metrics,
            "split_strategy": split_strategy,
            "success_threshold": success_threshold,
            "constraints": constraints,
            "time_column": time_column,
            "group_column": group_column,
        }

        rationale = _build_rationale(output, warnings)

        return AgentResult(output=output, rationale=rationale, warnings=warnings)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _secondary_metrics(task_type: str, primary_metric: str) -> list[str]:
    """Retorna métricas complementares, excluindo a primária."""
    suggestions: dict[str, list[str]] = {
        "classification": ["roc_auc", "average_precision", "macro_f1", "balanced_accuracy"],
        "regression": ["mae", "rmse", "r2", "mape"],
        "anomaly": ["roc_auc", "average_precision", "f1"],
    }
    return [m for m in suggestions.get(task_type, []) if m != primary_metric]


def _build_rationale(output: dict[str, Any], warnings: list[str]) -> str:
    lines = [
        f"Tarefa definida como '{output['task_type']}' "
        f"com variável-alvo '{output['target_column']}'.",
        f"Métrica primária: {output['primary_metric']} "
        f"(critério de sucesso ≥ {output['success_threshold']}).",
        f"Métricas secundárias: {', '.join(output['secondary_metrics'])}.",
        f"Estratégia de particionamento: {output['split_strategy']}.",
    ]
    if output["constraints"]:
        lines.append(f"Restrições adicionais: {output['constraints']}.")
    if warnings:
        lines.append("Avisos: " + "; ".join(warnings))
    return " ".join(lines)
