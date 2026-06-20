"""Agente de Engenharia de Atributos.

Monta transformadores para variáveis numéricas, categóricas, datas e texto curto,
gerando um pipeline reprodutível, configurável e serializável (RF07).

Cuidado principal: evitar leakage — o pipeline é devolvido **não ajustado**
(``AgentResult.pipeline``). Ele deve ser ajustado apenas com dados de treino,
dentro de cada fold/split, pelos agentes a jusante. Como é um ``sklearn.Pipeline``,
pode ser serializado (pickle/joblib) e versionado como artefato.

O ``output`` (persistido como JSONB) descreve o plano de atributos — grupos de
colunas, transformadores por grupo, nomes das features geradas e notas de
vazamento — mas não carrega o objeto do pipeline, que viaja em
``AgentResult.pipeline``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
)

from src.agents.base import AgentResult, BaseAgent

# ---------------------------------------------------------------------------
# Limiares e convenções (todos sobreponíveis via contexto)
# ---------------------------------------------------------------------------

#: Cardinalidade (nº de categorias) acima da qual usa-se codificação ordinal em
#: vez de one-hot, para não explodir a dimensionalidade.
_HIGH_CARDINALITY_THRESHOLD = 20
#: Comprimento médio (em caracteres) a partir do qual uma coluna textual é
#: tratada como texto livre (alinha com os agentes de perfilamento/limpeza).
_TEXT_AVG_LENGTH = 40
#: Razão cardinalidade/linhas acima da qual uma coluna textual é "alta cardinalidade".
_HIGH_CARDINALITY_RATIO = 0.5
#: Nº máximo de termos extraídos por coluna de texto quando a estratégia é "tfidf".
_TFIDF_MAX_FEATURES = 50

_SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
_NUMERIC_IMPUTERS = {"median", "mean", "constant", "none"}
_CATEGORICAL_IMPUTERS = {"most_frequent", "constant"}
_CATEGORICAL_ENCODINGS = {"onehot", "ordinal"}
_TEXT_STRATEGIES = {"stats", "tfidf", "drop"}
_SCALING_OPTIONS = set(_SCALERS) | {"none"}


# ---------------------------------------------------------------------------
# Transformadores customizados (módulo-nível => pickl e serializáveis)
# ---------------------------------------------------------------------------

class DatetimeFeatures(BaseEstimator, TransformerMixin):
    """Extrai componentes de calendário de colunas de data/hora.

    Gera, por coluna de entrada: ``year``, ``month``, ``day``, ``dayofweek``,
    ``hour``, ``quarter`` e ``is_weekend``. Sem estado ajustável (stateless), o
    que o mantém imune a vazamento entre treino e teste.
    """

    _PARTS = ("year", "month", "day", "dayofweek", "hour", "quarter", "is_weekend")

    def fit(self, X: Any, y: Any = None) -> "DatetimeFeatures":
        self.feature_names_in_ = _column_names(X)
        return self

    def transform(self, X: Any) -> np.ndarray:
        frame = _as_frame(X, self.feature_names_in_)
        blocks: list[np.ndarray] = []
        for col in frame.columns:
            dt = pd.to_datetime(frame[col], errors="coerce", format="mixed")
            blocks.append(
                np.column_stack([
                    dt.dt.year, dt.dt.month, dt.dt.day, dt.dt.dayofweek,
                    dt.dt.hour, dt.dt.quarter, (dt.dt.dayofweek >= 5).astype("float"),
                ])
            )
        out = np.hstack(blocks) if blocks else np.empty((len(frame), 0))
        return np.nan_to_num(out.astype("float64"), nan=0.0)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        base = list(input_features) if input_features is not None else list(self.feature_names_in_)
        return np.array([f"{col}__{part}" for col in base for part in self._PARTS])


class TextStatsFeatures(BaseEstimator, TransformerMixin):
    """Resume colunas de texto curto em estatísticas simples e deterministas.

    Gera, por coluna: ``char_len`` (nº de caracteres) e ``n_words`` (nº de
    palavras). Útil quando uma vetorização completa (tf-idf) seria custosa demais.
    """

    _PARTS = ("char_len", "n_words")

    def fit(self, X: Any, y: Any = None) -> "TextStatsFeatures":
        self.feature_names_in_ = _column_names(X)
        return self

    def transform(self, X: Any) -> np.ndarray:
        frame = _as_frame(X, self.feature_names_in_)
        blocks: list[np.ndarray] = []
        for col in frame.columns:
            text = frame[col].fillna("").astype(str)
            blocks.append(
                np.column_stack([
                    text.str.len().to_numpy(dtype="float64"),
                    text.str.split().map(len).to_numpy(dtype="float64"),
                ])
            )
        return np.hstack(blocks) if blocks else np.empty((len(frame), 0))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        base = list(input_features) if input_features is not None else list(self.feature_names_in_)
        return np.array([f"{col}__{part}" for col in base for part in self._PARTS])


class FeatureAgent(BaseAgent):
    """Agente 4 — Engenharia de Atributos.

    Entradas esperadas em ``context`` (uma fonte de dados é obrigatória):
        - ``dataframe`` (pandas.DataFrame): base usada para inferir colunas/tipos.
        - ``data`` (dict): fonte alternativa (``source_type`` csv/parquet, ``source_uri``).
        - ``target_column`` (str | None): alvo; nunca entra como atributo.
        - ``data_profile`` (dict | None): perfil do Agente de Ingestão; quando
          presente, seu ``schema`` é reutilizado e colunas ``is_id_like`` são
          excluídas. ``high_correlations`` envolvendo o alvo viram notas de leakage.

    Sobreposições de estratégia (opcionais):
        - ``numeric_imputation``: "median" (padrão) | "mean" | "constant" | "none".
        - ``scaling``: "standard" (padrão) | "minmax" | "robust" | "none".
        - ``categorical_imputation``: "most_frequent" (padrão) | "constant".
        - ``categorical_encoding``: "onehot" (padrão) | "ordinal".
        - ``high_cardinality_threshold`` (int, padrão 20): acima disso usa-se
          codificação ordinal mesmo quando ``categorical_encoding="onehot"``.
        - ``one_hot_max_categories`` (int | None): limite de categorias no one-hot.
        - ``datetime_features`` (bool, padrão True): extrai componentes de data.
        - ``text_strategy``: "stats" (padrão) | "tfidf" | "drop".
        - ``tfidf_max_features`` (int, padrão 50).
        - ``add_interactions`` (bool, padrão False): cria interações (produtos)
          simples entre as numéricas via ``PolynomialFeatures``.

    Saída em ``AgentResult.output``:
        - ``feature_columns`` / ``excluded_columns``;
        - ``column_groups``: {"numeric": [...], "categorical": [...], ...};
        - ``transformers``: descrição dos passos por grupo;
        - ``output_feature_names`` / ``n_output_features`` (após ajuste de prova);
        - ``config`` efetiva, ``warnings`` e ``leakage_notes``.

    O pipeline (não ajustado) é exposto em ``AgentResult.pipeline``.
    """

    name = "Agente de Engenharia de Atributos"
    event_type = "feature_engineering"

    def run(self, context: dict[str, Any]) -> AgentResult:
        warnings: list[str] = []

        # ------------------------------------------------------------------
        # 1. Ingestão e configuração
        # ------------------------------------------------------------------
        df = _load_dataframe(context)
        if df.shape[1] == 0:
            raise ValueError("A base não possui colunas; nada para transformar.")

        cfg = _resolve_config(context)
        data_profile = context.get("data_profile") or {}
        target_column: str | None = context.get("target_column") or None
        if target_column and target_column not in df.columns:
            raise ValueError(
                f"'target_column' '{target_column}' não existe na base. "
                f"Colunas disponíveis: {list(df.columns)}."
            )

        # ------------------------------------------------------------------
        # 2. Classificação das colunas em grupos de transformação
        # ------------------------------------------------------------------
        types = _infer_types(df, data_profile)
        id_like = _id_like_columns(data_profile)
        groups, excluded = _group_columns(df, types, target_column, id_like)
        feature_columns = [c for g in groups.values() for c in g]
        if not feature_columns:
            raise ValueError(
                "Nenhuma coluna utilizável como atributo após excluir alvo/identificadores/constantes."
            )

        # ------------------------------------------------------------------
        # 3. Montagem dos transformadores por grupo
        # ------------------------------------------------------------------
        transformers: list[tuple[str, Any, Any]] = []
        descriptions: list[dict[str, Any]] = []

        if groups["numeric"]:
            pipe, steps = _numeric_pipeline(cfg)
            transformers.append(("numeric", pipe, groups["numeric"]))
            descriptions.append({"group": "numeric", "columns": groups["numeric"], "steps": steps})

        if groups["categorical"]:
            low, high = _split_by_cardinality(df, groups["categorical"], cfg)
            if low:
                pipe, steps = _categorical_pipeline(cfg, encoding=cfg["categorical_encoding"])
                transformers.append(("categorical", pipe, low))
                descriptions.append({"group": "categorical", "columns": low, "steps": steps})
            if high:
                # Alta cardinalidade sempre vai para ordinal (evita explosão de colunas).
                pipe, steps = _categorical_pipeline(cfg, encoding="ordinal")
                transformers.append(("categorical_high_card", pipe, high))
                descriptions.append(
                    {"group": "categorical_high_card", "columns": high, "steps": steps}
                )
                warnings.append(
                    f"Colunas de alta cardinalidade {high} codificadas como ordinais "
                    f"(> {cfg['high_cardinality_threshold']} categorias)."
                )

        if groups["datetime"] and cfg["datetime_features"]:
            transformers.append(("datetime", DatetimeFeatures(), groups["datetime"]))
            descriptions.append({
                "group": "datetime",
                "columns": groups["datetime"],
                "steps": ["DatetimeFeatures(year, month, day, dayofweek, hour, quarter, is_weekend)"],
            })

        if groups["text"]:
            _append_text_transformers(transformers, descriptions, groups["text"], cfg, warnings)

        if not transformers:
            raise ValueError("Nenhum transformador aplicável às colunas disponíveis.")

        # ------------------------------------------------------------------
        # 4. Pipeline final (não ajustado) — serializável e leakage-safe
        # ------------------------------------------------------------------
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )
        pipeline = Pipeline([("preprocessor", preprocessor)])

        # ------------------------------------------------------------------
        # 5. Ajuste de prova num clone (apenas para introspecção; descartado)
        # ------------------------------------------------------------------
        output_feature_names, n_output, fit_warning = _probe_output_features(
            pipeline, df[feature_columns]
        )
        if fit_warning:
            warnings.append(fit_warning)

        # ------------------------------------------------------------------
        # 6. Notas de vazamento (RF06) e montagem da saída
        # ------------------------------------------------------------------
        leakage_notes = _leakage_notes(data_profile, target_column, cfg)

        output: dict[str, Any] = {
            "feature_columns": feature_columns,
            "excluded_columns": excluded,
            "column_groups": groups,
            "transformers": descriptions,
            "n_input_features": len(feature_columns),
            "output_feature_names": output_feature_names,
            "n_output_features": n_output,
            "config": cfg,
            "warnings": warnings,
            "leakage_notes": leakage_notes,
        }
        output = _to_native(output)

        rationale = _build_rationale(output, target_column)
        result = AgentResult(output=output, rationale=rationale, warnings=warnings)
        # O pipeline (não ajustado) viaja fora do output, que é persistido como JSONB.
        result.pipeline = pipeline
        return result


# ---------------------------------------------------------------------------
# Ingestão e configuração
# ---------------------------------------------------------------------------

def _load_dataframe(context: dict[str, Any]) -> pd.DataFrame:
    """Obtém o DataFrame a partir de ``dataframe`` ou de ``data.source_*``."""
    df = context.get("dataframe")
    if df is not None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("'dataframe' deve ser um pandas.DataFrame.")
        return df.copy()

    data = context.get("data") or {}
    source_type = str(data.get("source_type", "")).strip().lower()
    source_uri = data.get("source_uri")
    if not source_uri:
        raise ValueError(
            "Forneça 'dataframe' no contexto ou 'data.source_uri' para carregar a base."
        )
    if source_type == "csv":
        return pd.read_csv(source_uri)
    if source_type == "parquet":
        return pd.read_parquet(source_uri)
    raise ValueError(
        f"source_type '{source_type}' não suportado pela engenharia de atributos "
        "(use 'csv' ou 'parquet', ou passe 'dataframe' diretamente)."
    )


def _resolve_config(context: dict[str, Any]) -> dict[str, Any]:
    """Resolve estratégias efetivas, validando as escolhas categóricas."""
    numeric_imp = str(context.get("numeric_imputation", "median")).strip().lower()
    scaling = str(context.get("scaling", "standard")).strip().lower()
    cat_imp = str(context.get("categorical_imputation", "most_frequent")).strip().lower()
    cat_enc = str(context.get("categorical_encoding", "onehot")).strip().lower()
    text_strategy = str(context.get("text_strategy", "stats")).strip().lower()

    _validate("numeric_imputation", numeric_imp, _NUMERIC_IMPUTERS)
    _validate("scaling", scaling, _SCALING_OPTIONS)
    _validate("categorical_imputation", cat_imp, _CATEGORICAL_IMPUTERS)
    _validate("categorical_encoding", cat_enc, _CATEGORICAL_ENCODINGS)
    _validate("text_strategy", text_strategy, _TEXT_STRATEGIES)

    return {
        "numeric_imputation": numeric_imp,
        "scaling": scaling,
        "categorical_imputation": cat_imp,
        "categorical_encoding": cat_enc,
        "high_cardinality_threshold": int(
            context.get("high_cardinality_threshold", _HIGH_CARDINALITY_THRESHOLD)
        ),
        "one_hot_max_categories": context.get("one_hot_max_categories"),
        "datetime_features": bool(context.get("datetime_features", True)),
        "text_strategy": text_strategy,
        "tfidf_max_features": int(context.get("tfidf_max_features", _TFIDF_MAX_FEATURES)),
        "add_interactions": bool(context.get("add_interactions", False)),
    }


def _validate(name: str, value: str, allowed: set[str]) -> None:
    if value not in allowed:
        raise ValueError(f"'{name}' inválido: '{value}'. Opções: {sorted(allowed)}.")


# ---------------------------------------------------------------------------
# Classificação das colunas
# ---------------------------------------------------------------------------

def _infer_types(df: pd.DataFrame, data_profile: dict[str, Any]) -> dict[str, str]:
    """Mapa coluna -> tipo semântico, reutilizando o schema do perfil quando houver."""
    profile_schema = (data_profile or {}).get("schema") or {}
    n_rows = len(df)
    types: dict[str, str] = {}
    for col in df.columns:
        types[col] = profile_schema.get(col) or _infer_one_type(df[col], n_rows)
    return types


def _infer_one_type(series: pd.Series, n_rows: int) -> str:
    non_null = series.dropna()
    n_unique = int(non_null.nunique())
    if n_unique <= 1:
        return "constant"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    avg_len = float(non_null.astype(str).str.len().mean()) if not non_null.empty else 0.0
    ratio = (n_unique / n_rows) if n_rows else 0.0
    if avg_len >= _TEXT_AVG_LENGTH or (ratio >= _HIGH_CARDINALITY_RATIO and avg_len > 1):
        return "text"
    return "categorical"


def _id_like_columns(data_profile: dict[str, Any]) -> set[str]:
    """Colunas que são identificadores (não servem de atributo).

    Considera o papel explícito ``id`` e a heurística ``is_id_like`` — mas NÃO
    descarta colunas numéricas de ponto flutuante: uma contínua com todos os
    valores distintos não é um identificador, ao contrário de inteiros/strings.
    """
    ids: set[str] = set()
    for col in (data_profile or {}).get("columns", []):
        if col.get("is_target"):
            continue
        if col.get("role") == "id":
            ids.add(col["name"])
        elif col.get("is_id_like") and not str(col.get("dtype", "")).startswith("float"):
            ids.add(col["name"])
    return ids


def _group_columns(
    df: pd.DataFrame,
    types: dict[str, str],
    target_column: str | None,
    id_like: set[str],
) -> tuple[dict[str, list[str]], list[str]]:
    groups: dict[str, list[str]] = {"numeric": [], "categorical": [], "datetime": [], "text": []}
    excluded: list[str] = []
    semantic_to_group = {
        "numeric": "numeric",
        "categorical": "categorical",
        "boolean": "categorical",
        "datetime": "datetime",
        "text": "text",
    }
    for col in df.columns:
        if col == target_column or col in id_like:
            excluded.append(col)
            continue
        group = semantic_to_group.get(types.get(col))
        if group is None:  # constant ou tipo sem transformador
            excluded.append(col)
            continue
        groups[group].append(col)
    return groups, excluded


def _split_by_cardinality(
    df: pd.DataFrame, columns: list[str], cfg: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Separa categóricas em baixa e alta cardinalidade conforme o limiar."""
    if cfg["categorical_encoding"] == "ordinal":
        return columns, []  # tudo ordinal; não há grupo "alta cardinalidade" à parte
    threshold = cfg["high_cardinality_threshold"]
    low, high = [], []
    for col in columns:
        (high if int(df[col].nunique(dropna=True)) > threshold else low).append(col)
    return low, high


# ---------------------------------------------------------------------------
# Construção dos sub-pipelines
# ---------------------------------------------------------------------------

def _numeric_pipeline(cfg: dict[str, Any]) -> tuple[Pipeline, list[str]]:
    steps: list[tuple[str, Any]] = []
    labels: list[str] = []

    imp = cfg["numeric_imputation"]
    if imp != "none":
        if imp == "constant":
            steps.append(("imputer", SimpleImputer(strategy="constant", fill_value=0)))
        else:
            steps.append(("imputer", SimpleImputer(strategy=imp)))
        labels.append(f"SimpleImputer({imp})")

    if cfg["scaling"] != "none":
        steps.append(("scaler", _SCALERS[cfg["scaling"]]()))
        labels.append(f"{_SCALERS[cfg['scaling']].__name__}")

    if cfg["add_interactions"]:
        steps.append((
            "interactions",
            PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
        ))
        labels.append("PolynomialFeatures(interaction_only)")

    if not steps:  # nenhuma transformação => passthrough explícito
        steps.append(("passthrough", SimpleImputer(strategy="median")))
        labels.append("SimpleImputer(median)")
    return Pipeline(steps), labels


def _categorical_pipeline(cfg: dict[str, Any], *, encoding: str) -> tuple[Pipeline, list[str]]:
    steps: list[tuple[str, Any]] = []
    labels: list[str] = []

    cat_imp = cfg["categorical_imputation"]
    if cat_imp == "constant":
        steps.append(("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")))
    else:
        steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
    labels.append(f"SimpleImputer({cat_imp})")

    if encoding == "onehot":
        steps.append((
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                max_categories=cfg["one_hot_max_categories"],
            ),
        ))
        labels.append("OneHotEncoder(handle_unknown=ignore)")
    else:
        steps.append((
            "encoder",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        ))
        labels.append("OrdinalEncoder(handle_unknown=use_encoded_value)")
    return Pipeline(steps), labels


def _append_text_transformers(
    transformers: list[tuple[str, Any, Any]],
    descriptions: list[dict[str, Any]],
    text_columns: list[str],
    cfg: dict[str, Any],
    warnings: list[str],
) -> None:
    strategy = cfg["text_strategy"]
    if strategy == "drop":
        warnings.append(f"Colunas de texto {text_columns} descartadas (text_strategy='drop').")
        return
    if strategy == "stats":
        transformers.append(("text", TextStatsFeatures(), text_columns))
        descriptions.append({
            "group": "text",
            "columns": text_columns,
            "steps": ["TextStatsFeatures(char_len, n_words)"],
        })
        return
    # tfidf: um vetorizador por coluna (TfidfVectorizer exige entrada 1-D)
    for col in text_columns:
        transformers.append((
            f"tfidf__{col}",
            TfidfVectorizer(max_features=cfg["tfidf_max_features"]),
            col,  # escalar => ColumnTransformer entrega uma série 1-D
        ))
        descriptions.append({
            "group": "text",
            "columns": [col],
            "steps": [f"TfidfVectorizer(max_features={cfg['tfidf_max_features']})"],
        })


# ---------------------------------------------------------------------------
# Introspecção e notas
# ---------------------------------------------------------------------------

def _probe_output_features(
    pipeline: Pipeline, X: pd.DataFrame
) -> tuple[list[str] | None, int | None, str | None]:
    """Ajusta um clone só para enumerar as features de saída; o original fica intacto."""
    try:
        fitted = clone(pipeline).fit(X)
        names = [str(n) for n in fitted.named_steps["preprocessor"].get_feature_names_out()]
        return names, len(names), None
    except Exception as exc:  # pragma: no cover - introspecção é best-effort
        return None, None, (
            "Não foi possível pré-ajustar o pipeline para contar as features "
            f"de saída ({type(exc).__name__}); os nomes serão conhecidos após o fit."
        )


def _leakage_notes(
    data_profile: dict[str, Any], target_column: str | None, cfg: dict[str, Any]
) -> list[str]:
    notes = [
        "Pipeline devolvido NÃO ajustado: ajuste apenas com dados de treino, dentro de "
        "cada fold/split, para evitar vazamento (RF06).",
    ]
    if cfg["scaling"] != "none":
        notes.append(
            "Escalonamento e imputação aprendem estatísticas dos dados; ajustá-los na base "
            "completa antes do split contaminaria a avaliação."
        )
    for pair in (data_profile or {}).get("high_correlations", []):
        if target_column in (pair.get("a"), pair.get("b")):
            other = pair["b"] if pair["a"] == target_column else pair["a"]
            notes.append(
                f"Atributo '{other}' é fortemente correlacionado com o alvo "
                f"(r={pair.get('pearson')}); verifique vazamento antes de incluí-lo."
            )
    return notes


# ---------------------------------------------------------------------------
# Helpers gerais
# ---------------------------------------------------------------------------

def _column_names(X: Any) -> list[str]:
    if isinstance(X, pd.DataFrame):
        return [str(c) for c in X.columns]
    arr = np.asarray(X)
    n_cols = arr.shape[1] if arr.ndim == 2 else 1
    return [f"x{i}" for i in range(n_cols)]


def _as_frame(X: Any, names: list[str]) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return pd.DataFrame(arr, columns=names[: arr.shape[1]])


def _to_native(obj: Any) -> Any:
    """Sanitiza recursivamente para garantir serialização em JSONB."""
    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _build_rationale(output: dict[str, Any], target_column: str | None) -> str:
    groups = output["column_groups"]
    parts = [f"{len(cols)} {name}" for name, cols in groups.items() if cols]
    lines = [
        f"Pipeline de atributos montado sobre {output['n_input_features']} coluna(s): "
        f"{', '.join(parts)}.",
    ]
    if output["n_output_features"] is not None:
        lines.append(f"Gera {output['n_output_features']} feature(s) após o ajuste.")
    cfg = output["config"]
    lines.append(
        f"Numéricas: imputação '{cfg['numeric_imputation']}' + escala '{cfg['scaling']}'; "
        f"categóricas: '{cfg['categorical_encoding']}'."
    )
    if target_column:
        lines.append(f"Alvo '{target_column}' excluído dos atributos.")
    lines.append("Pipeline não ajustado (ajuste por fold) e serializável.")
    if output["warnings"]:
        lines.append("Avisos: " + "; ".join(output["warnings"]))
    return " ".join(lines)
