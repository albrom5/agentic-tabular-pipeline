"""Agente de Avaliação e Seleção.

Compara métricas (média e dispersão por fold), matriz de confusão, curvas ROC/PR
ou erros de regressão, e recomenda o modelo final por um critério claro.

Cuidado principal: selecionar por critério explícito, nunca por cherry-picking.
"""

from __future__ import annotations

from typing import Any

from src.agents.base import AgentResult, BaseAgent


class EvaluatorAgent(BaseAgent):
    name = "Agente de Avaliação"
    event_type = "model_selection"

    def run(self, context: dict[str, Any]) -> AgentResult:
        # TODO: agregar model_results em ranking (média, desvio, intervalo) e
        #       justificar a recomendação pela métrica primária (RF11).
        raise NotImplementedError
