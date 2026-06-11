"""Agente de Engenharia de Atributos.

Monta transformadores para variáveis numéricas, categóricas, datas e texto curto,
gerando um pipeline serializável e configurável.

Cuidado principal: evitar leakage — todas as transformações devem ser ajustadas
apenas com dados de treino, dentro de cada fold.
"""

from __future__ import annotations

from typing import Any

from src.agents.base import AgentResult, BaseAgent


class FeatureAgent(BaseAgent):
    name = "Agente de Engenharia de Atributos"
    event_type = "feature_engineering"

    def run(self, context: dict[str, Any]) -> AgentResult:
        # TODO: ColumnTransformer com imputação/encoding/scaling + interações simples (RF07).
        raise NotImplementedError
