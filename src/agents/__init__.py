"""Agentes do pipeline.

Cada agente é um papel lógico separado, com estado intermediário persistido e
decisões justificadas (seção 6 do documento de apoio).
"""

from src.agents.base import AgentResult, BaseAgent
from src.agents.data_profile_agent import DataProfileAgent

__all__ = ["BaseAgent", "AgentResult", "DataProfileAgent"]
