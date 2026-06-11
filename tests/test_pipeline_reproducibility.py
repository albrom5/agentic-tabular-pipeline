"""Testes de reprodutibilidade do pipeline (RNF01 / RF15).

Garantem que reexecutar um experimento a partir da mesma configuração e seed
produz as mesmas métricas e o mesmo ranking.
"""

import pytest


@pytest.mark.skip(reason="Aguardando implementação de run_experiment.")
def test_same_seed_yields_same_metrics() -> None:
    # TODO: executar o pipeline duas vezes com a mesma seed e comparar métricas.
    ...


@pytest.mark.skip(reason="Aguardando implementação do split sem leakage.")
def test_preprocessing_is_fit_only_on_train_fold() -> None:
    # TODO: assegurar que o pré-processamento é ajustado apenas no fold de treino.
    ...
