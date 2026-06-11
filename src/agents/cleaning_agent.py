"""Agente de Qualidade e Limpeza.

Propõe e executa limpeza reprodutível: faltantes, duplicatas, outliers,
categorias raras e inconsistências.

Cuidado principal: nunca alterar dados sem registrar a transformação e a justificativa.
"""

from __future__ import annotations

from typing import Any

from src.agents.base import AgentResult, BaseAgent


class CleaningAgent(BaseAgent):
    name = "Agente de Qualidade e Limpeza"
    event_type = "cleaning_decision"

    def run(self, context: dict[str, Any]) -> AgentResult:
        # TODO: imputação, remoção controlada de duplicatas, tratamento de
        #       categorias raras; cada ação vira um item em output["actions"] (RF04/RF05).
        raise NotImplementedError
