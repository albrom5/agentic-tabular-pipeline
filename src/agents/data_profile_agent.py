"""Agente de Ingestão e Perfilamento.

Lê a base, infere schema, tipos, distribuições, cardinalidade, faltantes e
desbalanceamento.

Cuidado principal: gerar `data_profile.json` e registrá-lo no PostgreSQL.
"""

from __future__ import annotations

from typing import Any

from src.agents.base import AgentResult, BaseAgent


class DataProfileAgent(BaseAgent):
    name = "Agente de Ingestão e Perfilamento"
    event_type = "data_profile"

    def run(self, context: dict[str, Any]) -> AgentResult:
        # TODO: estatísticas descritivas, tipos inferidos, cardinalidade,
        #       faltantes, duplicatas, distribuições e correlações básicas (RF03).
        raise NotImplementedError
