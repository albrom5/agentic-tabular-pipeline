"""Agente de Formulação do Problema.

Converte o objetivo informado em uma tarefa de ML bem definida: variável-alvo,
tipo de tarefa, métrica primária, restrições e critério de sucesso.

Cuidado principal: evitar tarefa mal definida e métricas incoerentes.
"""

from __future__ import annotations

from typing import Any

from src.agents.base import AgentResult, BaseAgent


class ProblemAgent(BaseAgent):
    name = "Agente de Formulação do Problema"
    event_type = "problem_definition"

    def run(self, context: dict[str, Any]) -> AgentResult:
        # TODO: validar coerência entre task_type, target_column e primary_metric.
        raise NotImplementedError
