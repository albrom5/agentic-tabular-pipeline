"""Pipeline de treinamento — ponto de entrada de um experimento.

Carrega a configuração YAML, instancia os agentes na ordem do fluxo agentivo,
executa o treinamento cruzado do model zoo (incluindo o autoencoder) e persiste
todos os eventos, métricas e artefatos no PostgreSQL.

Uso:
    python -m src.pipelines.training --config configs/experiment_example.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Lê o arquivo de configuração do experimento."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Executa o experimento de ponta a ponta a partir da configuração.

    Etapas (seção 5): formulação -> perfilamento -> limpeza -> features ->
    split -> treino cruzado -> autoencoder -> avaliação -> relatório.
    """
    # TODO: instanciar o EventSink (src/db/models.py), criar o experiment/run e
    #       encadear os agentes, propagando o contexto entre etapas.
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa um experimento do pipeline.")
    parser.add_argument("--config", required=True, help="Caminho do YAML de configuração.")
    args = parser.parse_args()
    config = load_config(args.config)
    run_experiment(config)


if __name__ == "__main__":
    main()
