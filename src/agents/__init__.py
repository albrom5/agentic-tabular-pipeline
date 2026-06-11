"""Agentes do pipeline.

Cada agente é um papel lógico separado, com estado intermediário persistido e
decisões justificadas (seção 6 do documento de apoio).
"""

from src.agents.base import AgentResult, BaseAgent

__all__ = ["BaseAgent", "AgentResult"]
