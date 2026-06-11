"""Pipeline de pré-processamento.

Orquestra ingestão/perfilamento, validação/limpeza e engenharia de atributos,
produzindo um conjunto pronto para particionamento — sempre de forma que as
transformações possam ser reaplicadas dentro de cada fold (sem leakage).
"""

from __future__ import annotations

from typing import Any


def build_preprocessing_pipeline(config: dict[str, Any]) -> Any:
    """Monta o ColumnTransformer/Pipeline de pré-processamento a partir da config.

    Retorna um objeto serializável (scikit-learn Pipeline) ainda não ajustado;
    o `fit` deve ocorrer apenas com dados de treino de cada fold.
    """
    # TODO: imputação numérica/categórica, encoding, scaling e tratamento de
    #       categorias raras conforme a seção `preprocessing` da config.
    raise NotImplementedError
