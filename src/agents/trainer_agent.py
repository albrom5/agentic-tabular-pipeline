"""Agente de Model Zoo (treinamento cruzado).

Executa as famílias de modelos clássicos aplicáveis ao tipo de tarefa, definidas
em `configs/model_zoo.yaml`.

Cuidado principal: registrar cada execução, fold, seed, hiperparâmetros e métricas.
"""

from __future__ import annotations

from typing import Any

from src.agents.base import AgentResult, BaseAgent


class TrainerAgent(BaseAgent):
    name = "Agente de Model Zoo"
    event_type = "model_training"

    def run(self, context: dict[str, Any]) -> AgentResult:
        # TODO: instanciar modelos a partir do model_zoo, treinar por fold e
        #       gravar cada resultado em model_results (RF09).
        raise NotImplementedError
