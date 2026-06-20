"""Agente de Ingestão e Perfilamento.

Lê a base, infere schema, tipos, distribuições, cardinalidade, faltantes e
desbalanceamento (RF03). Produz um perfil (``data_profile``) JSON-serializável
que é persistido no PostgreSQL como evento agentivo e pode alimentar os campos
``schema_json`` / ``profile_json`` da tabela ``datasets``.

Cuidado principal: gerar `data_profile.json` reprodutível (com hash de conteúdo)
e registrá-lo, sem nunca modificar a base original.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import pandas as pd

from src.agents.base import AgentResult, BaseAgent

# ---------------------------------------------------------------------------
# Limiares e convenções de perfilamento (todos sobreponíveis via contexto)
# ---------------------------------------------------------------------------

#: Acima desta razão entre cardinalidade e nº de linhas, uma coluna categórica/
#: textual é considerada "alta cardinalidade" (candidata a identificador).
_HIGH_CARDINALITY_RATIO = 0.5
#: Comprimento médio (em caracteres) a partir do qual uma coluna textual é
#: tratada como texto livre, e não como categórica.
_TEXT_AVG_LENGTH = 40
#: Frequência relativa abaixo da qual uma categoria é considerada "rara".
_RARE_CATEGORY_THRESHOLD = 0.01
#: Nº máximo de categorias mais frequentes reportadas por coluna.
_MAX_TOP_CATEGORIES = 10
#: |correlação de Pearson| a partir da qual um par de numéricas é destacado.
_CORRELATION_THRESHOLD = 0.95
#: Razão de desbalanceamento (maioria/minoria) a partir da qual há aviso.
_IMBALANCE_WARN_RATIO = 10.0


class DataProfileAgent(BaseAgent):
    """Agente 2 — Ingestão e Perfilamento.

    Entradas esperadas em ``context`` (uma das fontes de dados é obrigatória):
        - ``dataframe`` (pandas.DataFrame): base já carregada. Tem prioridade.
        - ``data`` (dict): descrição da fonte quando não há ``dataframe``,
          com ``source_type`` ("csv" | "parquet") e ``source_uri`` (caminho).
        - ``target_column`` (str | None, opcional): coluna-alvo, para análise de
          distribuição/desbalanceamento.
        - ``task_type`` (str | None, opcional): "classification", "regression"
          ou "anomaly"; refina a análise do alvo.
        - ``id_column`` / ``time_column`` / ``group_column`` (str | None, opcional):
          colunas com papel especial, sinalizadas no perfil.
        - Sobreposições de limiares: ``rare_category_threshold``,
          ``correlation_threshold``, ``high_cardinality_ratio``,
          ``max_top_categories``.

    Saída em ``AgentResult.output`` (``data_profile``):
        - ``n_rows``, ``n_cols``, ``content_hash``, ``memory_bytes``;
        - ``duplicates``: {``n_duplicate_rows``, ``pct_duplicate_rows``};
        - ``schema``: mapa coluna -> tipo semântico inferido;
        - ``columns``: lista de perfis por coluna;
        - ``target``: análise do alvo (distribuição/desbalanceamento), se houver;
        - ``high_correlations``: pares de numéricas fortemente correlacionadas.
    """

    name = "Agente de Ingestão e Perfilamento"
    event_type = "data_profile"

    def run(self, context: dict[str, Any]) -> AgentResult:
        warnings: list[str] = []

        # ------------------------------------------------------------------
        # 1. Ingestão da base
        # ------------------------------------------------------------------
        df = _load_dataframe(context)
        if df.shape[1] == 0:
            raise ValueError("A base não possui colunas; nada a perfilar.")
        if df.shape[0] == 0:
            warnings.append("A base está vazia (0 linhas); o perfil será degenerado.")

        # Limiares efetivos (contexto sobrepõe defaults)
        rare_threshold = float(context.get("rare_category_threshold", _RARE_CATEGORY_THRESHOLD))
        corr_threshold = float(context.get("correlation_threshold", _CORRELATION_THRESHOLD))
        high_card_ratio = float(context.get("high_cardinality_ratio", _HIGH_CARDINALITY_RATIO))
        max_top = int(context.get("max_top_categories", _MAX_TOP_CATEGORIES))

        target_column: str | None = context.get("target_column") or None
        task_type: str | None = (context.get("task_type") or "").strip().lower() or None
        special_roles = {
            context.get("id_column"): "id",
            context.get("time_column"): "time",
            context.get("group_column"): "group",
        }
        special_roles = {col: role for col, role in special_roles.items() if col}

        if target_column and target_column not in df.columns:
            raise ValueError(
                f"'target_column' '{target_column}' não existe na base. "
                f"Colunas disponíveis: {list(df.columns)}."
            )
        for col, role in special_roles.items():
            if col not in df.columns:
                warnings.append(f"Coluna '{role}' '{col}' não existe na base; ignorada.")

        n_rows, n_cols = int(df.shape[0]), int(df.shape[1])

        # ------------------------------------------------------------------
        # 2. Perfil por coluna
        # ------------------------------------------------------------------
        columns: list[dict[str, Any]] = []
        for col in df.columns:
            profile = _profile_column(
                df[col],
                n_rows=n_rows,
                role=special_roles.get(col),
                is_target=(col == target_column),
                high_card_ratio=high_card_ratio,
                rare_threshold=rare_threshold,
                max_top=max_top,
            )
            columns.append(profile)
            if profile["is_constant"] and n_rows > 0:
                warnings.append(f"Coluna '{col}' é constante (cardinalidade ≤ 1).")
            if profile.get("is_id_like") and col != target_column:
                warnings.append(
                    f"Coluna '{col}' parece um identificador "
                    f"(cardinalidade = nº de linhas); candidata a ser descartada como atributo."
                )
            if profile["pct_missing"] >= 50.0:
                warnings.append(
                    f"Coluna '{col}' tem {profile['pct_missing']:.1f}% de faltantes."
                )

        schema = {c["name"]: c["inferred_type"] for c in columns}

        # ------------------------------------------------------------------
        # 3. Duplicatas
        # ------------------------------------------------------------------
        n_duplicates = int(df.duplicated().sum()) if n_rows else 0
        duplicates = {
            "n_duplicate_rows": n_duplicates,
            "pct_duplicate_rows": _pct(n_duplicates, n_rows),
        }
        if n_duplicates:
            warnings.append(
                f"{n_duplicates} linha(s) duplicada(s) exata(s) "
                f"({duplicates['pct_duplicate_rows']:.1f}%)."
            )

        # ------------------------------------------------------------------
        # 4. Correlações fortes entre numéricas (sinal de redundância/leakage)
        # ------------------------------------------------------------------
        high_correlations = _high_correlations(df, columns, corr_threshold)
        for pair in high_correlations:
            if target_column in (pair["a"], pair["b"]):
                warnings.append(
                    f"Atributo fortemente correlacionado com o alvo "
                    f"({pair['a']} ~ {pair['b']}, r={pair['pearson']}); verifique vazamento."
                )

        # ------------------------------------------------------------------
        # 5. Análise do alvo (desbalanceamento / distribuição)
        # ------------------------------------------------------------------
        target_profile = None
        if target_column:
            target_profile = _profile_target(
                df[target_column], task_type=task_type, max_top=max_top
            )
            imbalance = target_profile.get("imbalance_ratio")
            if imbalance is not None and imbalance >= _IMBALANCE_WARN_RATIO:
                warnings.append(
                    f"Alvo '{target_column}' desbalanceado "
                    f"(razão maioria/minoria = {imbalance}); considere métricas e "
                    f"reamostragem adequadas."
                )

        # ------------------------------------------------------------------
        # 6. Monta o perfil
        # ------------------------------------------------------------------
        content_hash = _content_hash(df)
        output: dict[str, Any] = {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "content_hash": content_hash,
            "memory_bytes": int(df.memory_usage(deep=True).sum()),
            "duplicates": duplicates,
            "schema": schema,
            "columns": columns,
            "target": target_profile,
            "high_correlations": high_correlations,
        }
        output = _to_native(output)

        rationale = _build_rationale(output, target_column, warnings)
        return AgentResult(output=output, rationale=rationale, warnings=warnings)


# ---------------------------------------------------------------------------
# Ingestão
# ---------------------------------------------------------------------------

def _load_dataframe(context: dict[str, Any]) -> pd.DataFrame:
    """Obtém o DataFrame a partir de ``dataframe`` ou de ``data.source_*``."""
    df = context.get("dataframe")
    if df is not None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("'dataframe' deve ser um pandas.DataFrame.")
        # Cópia defensiva: o agente nunca modifica a base original.
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
        f"source_type '{source_type}' não suportado pelo perfilamento "
        "(use 'csv' ou 'parquet', ou passe 'dataframe' diretamente)."
    )


# ---------------------------------------------------------------------------
# Perfil de coluna
# ---------------------------------------------------------------------------

def _profile_column(
    series: pd.Series,
    *,
    n_rows: int,
    role: str | None,
    is_target: bool,
    high_card_ratio: float,
    rare_threshold: float,
    max_top: int,
) -> dict[str, Any]:
    non_null = series.dropna()
    n_missing = int(series.isna().sum())
    n_unique = int(non_null.nunique())
    inferred = _infer_type(series, non_null, n_unique, high_card_ratio, n_rows)

    profile: dict[str, Any] = {
        "name": str(series.name),
        "dtype": str(series.dtype),
        "inferred_type": inferred,
        "role": role,
        "is_target": is_target,
        "n_missing": n_missing,
        "pct_missing": _pct(n_missing, n_rows),
        "n_unique": n_unique,
        "is_constant": n_unique <= 1,
        "is_id_like": bool(n_rows and n_unique == n_rows and n_missing == 0),
    }

    if inferred == "numeric" and not non_null.empty:
        profile["stats"] = _numeric_stats(non_null)
    elif inferred == "datetime" and not non_null.empty:
        profile["stats"] = _datetime_stats(non_null)
    elif inferred in ("categorical", "boolean", "text") and not non_null.empty:
        profile["stats"] = _categorical_stats(non_null, n_rows, rare_threshold, max_top)

    return profile


def _infer_type(
    series: pd.Series,
    non_null: pd.Series,
    n_unique: int,
    high_card_ratio: float,
    n_rows: int,
) -> str:
    """Infere o tipo semântico de uma coluna."""
    if n_unique <= 1:
        return "constant"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    # Colunas de objeto/string: tenta datetime, depois decide entre texto e categórica.
    if _looks_like_datetime(non_null):
        return "datetime"
    avg_len = float(non_null.astype(str).str.len().mean())
    ratio = (n_unique / n_rows) if n_rows else 0.0
    if avg_len >= _TEXT_AVG_LENGTH or (ratio >= high_card_ratio and avg_len > 1):
        return "text"
    return "categorical"


def _looks_like_datetime(non_null: pd.Series) -> bool:
    """Heurística leve: amostra valores e tenta convertê-los para datetime."""
    if pd.api.types.is_numeric_dtype(non_null):
        return False
    sample = non_null.head(200).astype(str)
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return bool(parsed.notna().mean() >= 0.9)


def _numeric_stats(non_null: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(non_null, errors="coerce").dropna()
    if values.empty:
        return {}
    desc = values.describe()
    stats = {
        "mean": float(desc["mean"]),
        "std": float(desc["std"]) if not math.isnan(desc["std"]) else 0.0,
        "min": float(desc["min"]),
        "p25": float(desc["25%"]),
        "p50": float(desc["50%"]),
        "p75": float(desc["75%"]),
        "max": float(desc["max"]),
        "n_zeros": int((values == 0).sum()),
        "n_negative": int((values < 0).sum()),
    }
    # Skewness/curtose só fazem sentido com variação e amostra suficiente.
    if len(values) > 2 and values.std() > 0:
        stats["skewness"] = float(values.skew())
        stats["kurtosis"] = float(values.kurtosis())
    return stats


def _datetime_stats(non_null: pd.Series) -> dict[str, Any]:
    dt = pd.to_datetime(non_null, errors="coerce", format="mixed").dropna()
    if dt.empty:
        return {}
    return {"min": dt.min().isoformat(), "max": dt.max().isoformat()}


def _categorical_stats(
    non_null: pd.Series, n_rows: int, rare_threshold: float, max_top: int
) -> dict[str, Any]:
    counts = non_null.value_counts()
    n_valid = int(counts.sum())
    top = [
        {"value": _scalar(idx), "count": int(cnt), "pct": _pct(int(cnt), n_valid)}
        for idx, cnt in counts.head(max_top).items()
    ]
    rare_mask = counts < (rare_threshold * n_valid) if n_valid else counts < 0
    return {
        "top": top,
        "n_rare_categories": int(rare_mask.sum()),
        "rare_threshold": rare_threshold,
    }


# ---------------------------------------------------------------------------
# Análise do alvo
# ---------------------------------------------------------------------------

def _profile_target(series: pd.Series, *, task_type: str | None, max_top: int) -> dict[str, Any]:
    non_null = series.dropna()
    n_missing = int(series.isna().sum())
    result: dict[str, Any] = {
        "name": str(series.name),
        "n_missing": n_missing,
        "pct_missing": _pct(n_missing, len(series)),
    }

    numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
    n_unique = int(non_null.nunique())
    # Regressão: alvo numérico e contínuo o suficiente; senão tratamos como classes.
    treat_as_regression = task_type == "regression" or (
        task_type is None and numeric and n_unique > 20
    )

    if treat_as_regression and not non_null.empty:
        result["kind"] = "regression"
        result["stats"] = _numeric_stats(non_null)
    elif not non_null.empty:
        result["kind"] = "classification"
        counts = non_null.value_counts()
        n_valid = int(counts.sum())
        result["n_classes"] = int(len(counts))
        result["class_distribution"] = [
            {"label": _scalar(idx), "count": int(cnt), "pct": _pct(int(cnt), n_valid)}
            for idx, cnt in counts.head(max_top).items()
        ]
        majority, minority = int(counts.max()), int(counts.min())
        result["imbalance_ratio"] = round(majority / minority, 3) if minority else None
        result["minority_class"] = _scalar(counts.idxmin())
        result["majority_class"] = _scalar(counts.idxmax())
    else:
        result["kind"] = "empty"

    return result


# ---------------------------------------------------------------------------
# Correlações
# ---------------------------------------------------------------------------

def _high_correlations(
    df: pd.DataFrame, columns: list[dict[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    numeric_cols = [c["name"] for c in columns if c["inferred_type"] == "numeric"]
    if len(numeric_cols) < 2:
        return []
    corr = df[numeric_cols].corr(numeric_only=True)
    pairs: list[dict[str, Any]] = []
    for i, a in enumerate(numeric_cols):
        for b in numeric_cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= threshold:
                pairs.append({"a": a, "b": b, "pearson": round(float(r), 4)})
    pairs.sort(key=lambda p: abs(p["pearson"]), reverse=True)
    return pairs


# ---------------------------------------------------------------------------
# Helpers gerais
# ---------------------------------------------------------------------------

def _content_hash(df: pd.DataFrame) -> str:
    """Hash determinístico do conteúdo da base, para versionamento de dados."""
    try:
        row_hashes = pd.util.hash_pandas_object(df, index=True).values
        digest = hashlib.sha256(row_hashes.tobytes())
        digest.update("|".join(map(str, df.columns)).encode("utf-8"))
        return digest.hexdigest()
    except Exception:  # pragma: no cover - fallback robusto p/ dtypes exóticos
        return hashlib.sha256(
            pd.util.hash_pandas_object(df.astype(str), index=True).values.tobytes()
        ).hexdigest()


def _pct(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 3) if whole else 0.0


def _scalar(value: Any) -> Any:
    """Converte rótulos/valores para tipos nativos JSON-serializáveis."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return str(value) if not isinstance(value, (int, float, bool, str)) else value


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


def _build_rationale(
    profile: dict[str, Any], target_column: str | None, warnings: list[str]
) -> str:
    type_counts: dict[str, int] = {}
    for t in profile["schema"].values():
        type_counts[t] = type_counts.get(t, 0) + 1
    types_str = ", ".join(f"{n} {t}" for t, n in sorted(type_counts.items()))

    lines = [
        f"Base perfilada: {profile['n_rows']} linhas × {profile['n_cols']} colunas "
        f"(hash {profile['content_hash'][:12]}…).",
        f"Tipos inferidos: {types_str}.",
        f"Duplicatas exatas: {profile['duplicates']['n_duplicate_rows']} "
        f"({profile['duplicates']['pct_duplicate_rows']}%).",
    ]
    if profile.get("target") and target_column:
        tgt = profile["target"]
        if tgt.get("kind") == "classification":
            lines.append(
                f"Alvo '{target_column}': {tgt.get('n_classes')} classes, "
                f"razão de desbalanceamento {tgt.get('imbalance_ratio')}."
            )
        elif tgt.get("kind") == "regression":
            lines.append(f"Alvo '{target_column}': contínuo (perfil numérico).")
    if profile["high_correlations"]:
        lines.append(
            f"{len(profile['high_correlations'])} par(es) de atributos fortemente "
            "correlacionado(s) detectado(s)."
        )
    if warnings:
        lines.append("Avisos: " + "; ".join(warnings))
    return " ".join(lines)
