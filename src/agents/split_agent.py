"""Agente de Split e Validação.

Define a estratégia de particionamento: holdout, k-fold, stratified k-fold,
group split ou time split, conforme a natureza do problema.

Cuidado principal: evitar contaminação entre treino e teste.
"""

from __future__ import annotations

from typing import Any

from src.agents.base import AgentResult, BaseAgent


class SplitAgent(BaseAgent):
    name = "Agente de Split e Validação"
    event_type = "split_strategy"

    def run(self, context: dict[str, Any]) -> AgentResult:
        # TODO: escolher o splitter do sklearn a partir de validation.split_strategy (RF08).
        #       Respeitar group_column (GroupKFold) e time_column (split temporal).
        raise NotImplementedError
