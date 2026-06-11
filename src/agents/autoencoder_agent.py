"""Agente de Autoencoders.

Treina um autoencoder tabular (PyTorch) para uma de três aplicações: representação
(latent features), denoising/imputação experimental ou detecção de anomalias por
erro de reconstrução. Sempre compara contra uma linha de base sem autoencoder.

Restrição metodológica: o autoencoder deve ser ajustado apenas com dados de treino
dentro de cada fold/split — treinar com toda a base causa vazamento (seção 9).
"""

from __future__ import annotations

from typing import Any

from src.agents.base import AgentResult, BaseAgent


class AutoencoderAgent(BaseAgent):
    name = "Agente de Autoencoders"
    event_type = "autoencoder_training"

    def run(self, context: dict[str, Any]) -> AgentResult:
        # TODO: autoencoder denso (encoder/decoder) em PyTorch; expor vetor latente
        #       e erro de reconstrução conforme autoencoder.use_case (RF10).
        raise NotImplementedError
