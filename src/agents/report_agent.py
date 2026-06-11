"""Agente Relator e Auditor.

Gera o relatório técnico final, um model card simplificado, logs e explicações
(importância de variáveis / SHAP quando viável).

Cuidado principal: tornar a decisão revisável por um humano.
"""

from __future__ import annotations

from typing import Any

from src.agents.base import AgentResult, BaseAgent


class ReportAgent(BaseAgent):
    name = "Agente Relator e Auditor"
    event_type = "final_report"

    def run(self, context: dict[str, Any]) -> AgentResult:
        # TODO: montar relatório (Markdown/HTML/PDF) com metodologia, resultados,
        #       limitações, riscos de leakage e próximos passos (RF12/RF14).
        raise NotImplementedError
