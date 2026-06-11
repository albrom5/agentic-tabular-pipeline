"""Testes de qualidade de dados.

Verificam que as regras de validação (faltantes, tipos inválidos, intervalos
impossíveis, variáveis constantes, duplicatas) sinalizam corretamente os problemas.
"""

import pytest


@pytest.mark.skip(reason="Aguardando implementação do agente de qualidade/limpeza.")
def test_constant_columns_are_flagged() -> None:
    # TODO: dado um DataFrame com coluna constante, o relatório de qualidade deve sinalizá-la.
    ...


@pytest.mark.skip(reason="Aguardando implementação do agente de qualidade/limpeza.")
def test_duplicates_are_detected() -> None:
    # TODO: linhas duplicadas exatas devem ser contadas no quality_report.
    ...


@pytest.mark.skip(reason="Aguardando implementação do detector de leakage.")
def test_target_leakage_is_warned() -> None:
    # TODO: variável perfeitamente correlacionada com o alvo deve gerar alerta (RF06).
    ...
