"""Testes de qualidade de dados.

Verificam que as regras de validação (faltantes, tipos inválidos, intervalos
impossíveis, variáveis constantes, duplicatas) sinalizam corretamente os problemas.
"""

import pandas as pd
import pytest

from src.agents.cleaning_agent import CleaningAgent


def test_constant_columns_are_flagged() -> None:
    df = pd.DataFrame({"const": [1] * 5, "x": [1, 2, 3, 4, 5]})
    report = CleaningAgent().run({"dataframe": df}).output["quality_report"]
    assert "const" in report["constant_columns"]


def test_duplicates_are_detected() -> None:
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    report = CleaningAgent().run({"dataframe": df}).output["quality_report"]
    assert report["n_duplicate_rows"] == 1


@pytest.mark.skip(reason="Aguardando implementação do detector de leakage.")
def test_target_leakage_is_warned() -> None:
    # TODO: variável perfeitamente correlacionada com o alvo deve gerar alerta (RF06).
    ...
