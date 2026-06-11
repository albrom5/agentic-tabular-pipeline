"""Classe base dos agentes.

Todo agente recebe um contexto de execução, produz um resultado e registra um
evento agentivo no PostgreSQL (input, output e justificativa). Isso garante a
auditabilidade exigida pelo RNF03.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentResult:
    """Resultado padronizado de um agente."""

    output: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    warnings: list[str] = field(default_factory=list)


class BaseAgent:
    """Contrato comum a todos os agentes do pipeline.

    Subclasses devem implementar :meth:`run`. O método :meth:`__call__` envolve a
    execução, registrando o evento agentivo correspondente no repositório de eventos.
    """

    #: Nome legível usado nos eventos persistidos (ex.: "Agente de Qualidade e Limpeza").
    name: str = "BaseAgent"
    #: Tipo do evento emitido (ex.: "cleaning_decision").
    event_type: str = "agent_event"

    def __init__(self, event_sink: Any | None = None) -> None:
        # `event_sink` é um objeto capaz de persistir eventos (ver src/db/models.py).
        self.event_sink = event_sink

    def run(self, context: dict[str, Any]) -> AgentResult:  # pragma: no cover - stub
        """Executa a lógica do agente. Deve ser implementado pelas subclasses."""
        raise NotImplementedError

    def __call__(self, context: dict[str, Any]) -> AgentResult:
        result = self.run(context)
        self._emit_event(context, result)
        return result

    def _emit_event(self, context: dict[str, Any], result: AgentResult) -> None:
        """Persiste o evento agentivo, se houver um sink configurado."""
        if self.event_sink is None:
            return
        self.event_sink.record_event(
            experiment_id=context.get("experiment_id"),
            agent_name=self.name,
            event_type=self.event_type,
            input_json=context.get("agent_input", {}),
            output_json=result.output,
            rationale=result.rationale,
        )
