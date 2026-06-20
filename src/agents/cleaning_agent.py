"""Agente de Qualidade e Limpeza.

Propõe e executa limpeza reprodutível: faltantes, duplicatas, outliers,
categorias raras e inconsistências (RF04 — validação de qualidade; RF05 —
limpeza reprodutível).

Cada decisão vira um item em ``output["actions"]`` com a operação aplicada, a
coluna afetada e a justificativa, no mesmo formato do evento ``cleaning_decision``
do documento de apoio (seção 11.2). O perfil de qualidade detectado é exposto em
``output["quality_report"]`` (alimenta ``datasets.quality_report_json``).

Cuidado principal: nunca alterar dados sem registrar a transformação e a
justificativa. O agente trabalha sobre uma cópia — a base original nunca é
mutada — e todas as escolhas de imputação guardam o valor usado, de modo que a
mesma limpeza possa ser reaplicada de forma determinística (RF15).

A base já limpa é devolvida no atributo ``AgentResult.cleaned_dataframe`` (não no
``output``, que é persistido como JSONB e deve permanecer serializável).
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import pandas as pd

from src.agents.base import AgentResult, BaseAgent

# ---------------------------------------------------------------------------
# Limiares e convenções de limpeza (todos sobreponíveis via contexto)
# ---------------------------------------------------------------------------

#: Acima desta fração de faltantes, a coluna é descartada em vez de imputada.
_DROP_COLUMN_MISSING_FRACTION = 0.6
#: Frequência relativa abaixo da qual uma categoria é considerada "rara".
_RARE_CATEGORY_THRESHOLD = 0.01
#: Rótulo usado ao agrupar categorias raras.
_RARE_LABEL = "__rare__"
#: Valor usado na imputação por constante de colunas categóricas/textuais.
_MISSING_LABEL = "__missing__"
#: Multiplicador do IQR para as cercas de outliers (regra de Tukey).
_IQR_MULTIPLIER = 1.5
#: Comprimento médio (em caracteres) a partir do qual uma coluna textual é
#: tratada como texto livre, e não como categórica (alinha com o perfilamento).
_TEXT_AVG_LENGTH = 40
#: Razão cardinalidade/linhas acima da qual uma coluna textual é "alta cardinalidade".
_HIGH_CARDINALITY_RATIO = 0.5
#: Fração mínima de valores que precisam parsear como número para coerção de tipo.
_NUMERIC_COERCION_FRACTION = 0.9

_NUMERIC_IMPUTERS = {"median", "mean", "constant", "none"}
_CATEGORICAL_IMPUTERS = {"most_frequent", "constant", "none"}
_OUTLIER_STRATEGIES = {"none", "clip", "remove"}
_RARE_STRATEGIES = {"group", "none"}


class CleaningAgent(BaseAgent):
    """Agente 3 — Validação e Limpeza.

    Entradas esperadas em ``context`` (uma das fontes de dados é obrigatória):
        - ``dataframe`` (pandas.DataFrame): base já carregada. Tem prioridade.
        - ``data`` (dict): fonte quando não há ``dataframe``, com ``source_type``
          ("csv" | "parquet") e ``source_uri`` (caminho).
        - ``target_column`` (str | None, opcional): alvo; nunca é imputado nem
          descartado, e linhas com alvo ausente são removidas (tarefa supervisionada).
        - ``data_profile`` (dict | None, opcional): perfil do Agente de Ingestão;
          quando presente, seu ``schema`` é reutilizado para inferir os tipos.

    Sobreposições de estratégia/limiar (todas opcionais):
        - ``numeric_imputation``: "median" (padrão) | "mean" | "constant" | "none".
        - ``categorical_imputation``: "most_frequent" (padrão) | "constant" | "none".
        - ``constant_fill_value``: valor usado quando a imputação é "constant".
        - ``drop_column_missing_fraction``: fração de faltantes p/ descartar coluna.
        - ``drop_duplicates`` (bool, padrão True).
        - ``drop_constant`` (bool, padrão True): descarta colunas constantes.
        - ``rare_category_strategy``: "group" (padrão) | "none".
        - ``rare_category_threshold`` (float, padrão 0.01).
        - ``outlier_strategy``: "none" (padrão) | "clip" | "remove".
        - ``impossible_ranges`` (dict): {coluna: {"min": x, "max": y}}; valores
          fora do intervalo viram faltantes (e depois são imputados).
        - ``normalize_strings`` (bool, padrão True): apara espaços e trata strings
          vazias como faltantes.
        - ``coerce_numeric_strings`` (bool, padrão True): converte colunas-objeto
          majoritariamente numéricas para numérico.

    Saída em ``AgentResult.output`` (formato ``cleaning_decision``):
        - ``actions``: lista de transformações aplicadas (operação, coluna, motivo);
        - ``warnings``: alertas que não impedem a limpeza;
        - ``quality_report``: problemas detectados (RF04);
        - ``n_rows_before`` / ``n_rows_after`` / ``n_cols_before`` / ``n_cols_after``;
        - ``content_hash_before`` / ``content_hash_after``.

    A base limpa é exposta em ``AgentResult.cleaned_dataframe``.
    """

    name = "Agente de Qualidade e Limpeza"
    event_type = "cleaning_decision"

    def run(self, context: dict[str, Any]) -> AgentResult:
        warnings: list[str] = []
        actions: list[dict[str, Any]] = []

        # ------------------------------------------------------------------
        # 1. Ingestão (cópia defensiva: a base original nunca é mutada)
        # ------------------------------------------------------------------
        df = _load_dataframe(context)
        if df.shape[1] == 0:
            raise ValueError("A base não possui colunas; nada a limpar.")

        n_rows_before, n_cols_before = int(df.shape[0]), int(df.shape[1])
        content_hash_before = _content_hash(df)

        # Configuração efetiva (contexto sobrepõe defaults)
        cfg = _resolve_config(context)

        target_column: str | None = context.get("target_column") or None
        if target_column and target_column not in df.columns:
            raise ValueError(
                f"'target_column' '{target_column}' não existe na base. "
                f"Colunas disponíveis: {list(df.columns)}."
            )

        # ------------------------------------------------------------------
        # 2. Normalização de representação (apara espaços, vazios -> NaN)
        # ------------------------------------------------------------------
        if cfg["normalize_strings"]:
            normalized = _normalize_strings(df)
            if normalized:
                actions.append({
                    "operation": "strip_whitespace",
                    "columns": normalized,
                    "reason": "Espaços removidos; strings vazias tratadas como faltantes.",
                })

        # ------------------------------------------------------------------
        # 3. Coerção de tipos inválidos (objeto majoritariamente numérico)
        # ------------------------------------------------------------------
        if cfg["coerce_numeric_strings"]:
            for col in df.columns:
                if col == target_column:
                    continue
                coerced = _maybe_coerce_numeric(df, col)
                if coerced is not None:
                    df[col] = coerced
                    actions.append({
                        "operation": "coerce_to_numeric",
                        "column": col,
                        "reason": (
                            f"≥{int(_NUMERIC_COERCION_FRACTION * 100)}% dos valores "
                            "parseiam como número; coluna convertida para numérica."
                        ),
                    })

        # Tipos semânticos por coluna (reutiliza schema do perfil se disponível)
        types = _infer_types(df, context.get("data_profile"))

        # ------------------------------------------------------------------
        # 4. Relatório de qualidade (RF04) sobre o estado de entrada normalizado
        # ------------------------------------------------------------------
        quality_report = _build_quality_report(df, types, target_column, cfg)

        # ------------------------------------------------------------------
        # 5. Linhas com alvo ausente: removidas (tarefa supervisionada)
        # ------------------------------------------------------------------
        if target_column:
            n_missing_target = int(df[target_column].isna().sum())
            if n_missing_target:
                df = df[df[target_column].notna()].copy()
                actions.append({
                    "operation": "drop_missing_target_rows",
                    "column": target_column,
                    "rows": n_missing_target,
                    "reason": "Linhas sem rótulo não são utilizáveis em tarefa supervisionada.",
                })
                warnings.append(
                    f"{n_missing_target} linha(s) com '{target_column}' ausente foram removidas."
                )

        # ------------------------------------------------------------------
        # 6. Colunas constantes (sem poder preditivo)
        # ------------------------------------------------------------------
        if cfg["drop_constant"]:
            for col in quality_report["constant_columns"]:
                if col == target_column:
                    continue
                df = df.drop(columns=col)
                actions.append({
                    "operation": "drop_constant_column",
                    "column": col,
                    "reason": "Coluna constante (cardinalidade ≤ 1); não informa o modelo.",
                })

        # ------------------------------------------------------------------
        # 7. Valores impossíveis -> faltantes (depois imputados)
        # ------------------------------------------------------------------
        for col, bounds in cfg["impossible_ranges"].items():
            if col not in df.columns:
                continue
            n_bad = _coerce_impossible_to_nan(df, col, bounds)
            if n_bad:
                actions.append({
                    "operation": "coerce_impossible_to_nan",
                    "column": col,
                    "rows": n_bad,
                    "bounds": bounds,
                    "reason": f"{n_bad} valor(es) fora do intervalo plausível {bounds}.",
                })

        # ------------------------------------------------------------------
        # 8. Faltantes: descarte de coluna muito vazia ou imputação registrada
        # ------------------------------------------------------------------
        for col in list(df.columns):
            if col == target_column:
                continue
            n_missing = int(df[col].isna().sum())
            if n_missing == 0:
                continue
            frac = n_missing / len(df) if len(df) else 0.0
            if frac >= cfg["drop_column_missing_fraction"]:
                df = df.drop(columns=col)
                actions.append({
                    "operation": "drop_high_missing_column",
                    "column": col,
                    "pct_missing": round(100.0 * frac, 3),
                    "reason": (
                        f"{round(100.0 * frac, 1)}% de faltantes "
                        f"(≥ {round(100.0 * cfg['drop_column_missing_fraction'])}%); "
                        "imputar introduziria viés excessivo."
                    ),
                })
                continue
            action = _impute_column(df, col, types.get(col, "categorical"), cfg)
            if action is not None:
                action["n_missing"] = n_missing
                actions.append(action)
                if types.get(col) == "numeric" and frac >= 0.2:
                    warnings.append(
                        f"Coluna '{col}' imputada com {frac:.0%} de faltantes; "
                        "avaliar possibilidade de viés."
                    )
            else:
                warnings.append(
                    f"Coluna '{col}' tem {n_missing} faltante(s) não tratado(s) "
                    f"(tipo '{types.get(col)}' sem estratégia de imputação)."
                )

        # ------------------------------------------------------------------
        # 9. Categorias raras
        # ------------------------------------------------------------------
        if cfg["rare_category_strategy"] == "group":
            for col, info in quality_report["rare_categories"].items():
                if col == target_column or col not in df.columns:
                    continue
                n_grouped = _group_rare_categories(df, col, info["categories"])
                if n_grouped:
                    actions.append({
                        "operation": "group_rare_categories",
                        "column": col,
                        "categories": info["categories"],
                        "label": _RARE_LABEL,
                        "reason": (
                            f"{len(info['categories'])} categoria(s) com frequência < "
                            f"{cfg['rare_category_threshold']:.1%} agrupada(s) em "
                            f"'{_RARE_LABEL}'."
                        ),
                    })

        # ------------------------------------------------------------------
        # 10. Outliers (somente se solicitado; reportados sempre no quality_report)
        # ------------------------------------------------------------------
        if cfg["outlier_strategy"] != "none":
            for col, info in quality_report["outliers"].items():
                if col == target_column or col not in df.columns:
                    continue
                action = _handle_outliers(df, col, info, cfg["outlier_strategy"])
                if action is not None:
                    actions.append(action)
            # 'remove' pode ter eliminado linhas: reindexa
            df = df.reset_index(drop=True)

        # ------------------------------------------------------------------
        # 11. Duplicatas exatas (remoção controlada)
        # ------------------------------------------------------------------
        if cfg["drop_duplicates"]:
            n_dups = int(df.duplicated().sum())
            if n_dups:
                df = df.drop_duplicates(ignore_index=True)
                actions.append({
                    "operation": "drop_duplicates",
                    "rows": n_dups,
                    "reason": "Linhas duplicadas exatas removidas.",
                })

        # ------------------------------------------------------------------
        # 12. Monta saída (JSON-serializável) e devolve a base limpa
        # ------------------------------------------------------------------
        n_rows_after, n_cols_after = int(df.shape[0]), int(df.shape[1])
        output: dict[str, Any] = {
            "actions": actions,
            "warnings": warnings,
            "quality_report": quality_report,
            "n_rows_before": n_rows_before,
            "n_rows_after": n_rows_after,
            "n_cols_before": n_cols_before,
            "n_cols_after": n_cols_after,
            "content_hash_before": content_hash_before,
            "content_hash_after": _content_hash(df),
        }
        output = _to_native(output)

        rationale = _build_rationale(output, target_column)
        result = AgentResult(output=output, rationale=rationale, warnings=warnings)
        # A base limpa viaja fora do output (que é persistido como JSONB).
        result.cleaned_dataframe = df
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
        f"source_type '{source_type}' não suportado pela limpeza "
        "(use 'csv' ou 'parquet', ou passe 'dataframe' diretamente)."
    )


def _resolve_config(context: dict[str, Any]) -> dict[str, Any]:
    """Resolve estratégias/limiares efetivos, validando as escolhas categóricas."""
    numeric_imp = str(context.get("numeric_imputation", "median")).strip().lower()
    cat_imp = str(context.get("categorical_imputation", "most_frequent")).strip().lower()
    outlier = str(context.get("outlier_strategy", "none")).strip().lower()
    rare = str(context.get("rare_category_strategy", "group")).strip().lower()

    if numeric_imp not in _NUMERIC_IMPUTERS:
        raise ValueError(
            f"'numeric_imputation' inválida: '{numeric_imp}'. Opções: {sorted(_NUMERIC_IMPUTERS)}."
        )
    if cat_imp not in _CATEGORICAL_IMPUTERS:
        raise ValueError(
            f"'categorical_imputation' inválida: '{cat_imp}'. "
            f"Opções: {sorted(_CATEGORICAL_IMPUTERS)}."
        )
    if outlier not in _OUTLIER_STRATEGIES:
        raise ValueError(
            f"'outlier_strategy' inválida: '{outlier}'. Opções: {sorted(_OUTLIER_STRATEGIES)}."
        )
    if rare not in _RARE_STRATEGIES:
        raise ValueError(
            f"'rare_category_strategy' inválida: '{rare}'. Opções: {sorted(_RARE_STRATEGIES)}."
        )

    return {
        "numeric_imputation": numeric_imp,
        "categorical_imputation": cat_imp,
        "constant_fill_value": context.get("constant_fill_value", _MISSING_LABEL),
        "drop_column_missing_fraction": float(
            context.get("drop_column_missing_fraction", _DROP_COLUMN_MISSING_FRACTION)
        ),
        "drop_duplicates": bool(context.get("drop_duplicates", True)),
        "drop_constant": bool(context.get("drop_constant", True)),
        "rare_category_strategy": rare,
        "rare_category_threshold": float(
            context.get("rare_category_threshold", _RARE_CATEGORY_THRESHOLD)
        ),
        "outlier_strategy": outlier,
        "impossible_ranges": dict(context.get("impossible_ranges") or {}),
        "normalize_strings": bool(context.get("normalize_strings", True)),
        "coerce_numeric_strings": bool(context.get("coerce_numeric_strings", True)),
    }


# ---------------------------------------------------------------------------
# Inferência de tipos
# ---------------------------------------------------------------------------

def _infer_types(df: pd.DataFrame, data_profile: dict[str, Any] | None) -> dict[str, str]:
    """Mapa coluna -> tipo semântico, reutilizando o schema do perfil quando houver."""
    profile_schema = (data_profile or {}).get("schema") or {}
    types: dict[str, str] = {}
    n_rows = len(df)
    for col in df.columns:
        if col in profile_schema:
            types[col] = profile_schema[col]
            continue
        types[col] = _infer_one_type(df[col], n_rows)
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


# ---------------------------------------------------------------------------
# Normalização e coerção
# ---------------------------------------------------------------------------

def _is_textual(series: pd.Series) -> bool:
    """True para colunas de texto (``object`` ou ``StringDtype`` do pandas ≥ 2.1)."""
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def _normalize_strings(df: pd.DataFrame) -> list[str]:
    """Apara espaços de colunas textuais e troca strings vazias por NaN. Muta ``df``."""
    changed: list[str] = []
    for col in df.columns:
        if not _is_textual(df[col]):
            continue
        original = df[col]
        # Apara apenas valores string; preserva NaN e demais tipos como estão.
        stripped = original.map(lambda v: v.strip() if isinstance(v, str) else v)
        stripped = stripped.replace({"": np.nan})
        if not stripped.equals(original):
            df[col] = stripped
            changed.append(col)
    return changed


def _maybe_coerce_numeric(df: pd.DataFrame, col: str) -> pd.Series | None:
    """Devolve a coluna convertida para numérica se a maioria dos valores parsear."""
    series = df[col]
    if not _is_textual(series):
        return None
    non_null = series.dropna()
    if non_null.empty:
        return None
    parsed = pd.to_numeric(non_null, errors="coerce")
    if parsed.notna().mean() >= _NUMERIC_COERCION_FRACTION and parsed.notna().sum() > 0:
        return pd.to_numeric(series, errors="coerce")
    return None


# ---------------------------------------------------------------------------
# Relatório de qualidade (RF04)
# ---------------------------------------------------------------------------

def _build_quality_report(
    df: pd.DataFrame,
    types: dict[str, str],
    target_column: str | None,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    n_rows = len(df)
    missing: dict[str, dict[str, Any]] = {}
    rare_categories: dict[str, dict[str, Any]] = {}
    outliers: dict[str, dict[str, Any]] = {}
    constant_columns: list[str] = []

    for col in df.columns:
        series = df[col]
        n_missing = int(series.isna().sum())
        if n_missing:
            missing[col] = {"n": n_missing, "pct": _pct(n_missing, n_rows)}

        col_type = types.get(col)
        non_null = series.dropna()

        if int(non_null.nunique()) <= 1:
            constant_columns.append(col)

        if col_type in ("categorical", "boolean") and not non_null.empty and col != target_column:
            rare = _rare_categories(non_null, cfg["rare_category_threshold"])
            if rare:
                rare_categories[col] = {"n_rare": len(rare), "categories": rare}

        if col_type == "numeric" and col != target_column:
            fences = _iqr_fences(non_null)
            if fences is not None:
                lower, upper = fences
                mask = (non_null < lower) | (non_null > upper)
                n_out = int(mask.sum())
                if n_out:
                    outliers[col] = {
                        "n": n_out,
                        "pct": _pct(n_out, n_rows),
                        "lower": round(float(lower), 6),
                        "upper": round(float(upper), 6),
                    }

    n_duplicates = int(df.duplicated().sum()) if n_rows else 0
    return {
        "n_rows": n_rows,
        "n_cols": int(df.shape[1]),
        "missing": missing,
        "n_duplicate_rows": n_duplicates,
        "pct_duplicate_rows": _pct(n_duplicates, n_rows),
        "constant_columns": constant_columns,
        "rare_categories": rare_categories,
        "outliers": outliers,
        "target_missing": (
            int(df[target_column].isna().sum()) if target_column in df.columns else 0
        ),
    }


def _rare_categories(non_null: pd.Series, threshold: float) -> list[Any]:
    counts = non_null.value_counts()
    n_valid = int(counts.sum())
    if not n_valid:
        return []
    rare = counts[counts < threshold * n_valid]
    return [_scalar(idx) for idx in rare.index]


def _iqr_fences(non_null: pd.Series) -> tuple[float, float] | None:
    values = pd.to_numeric(non_null, errors="coerce").dropna()
    if len(values) < 4:
        return None
    q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
    iqr = q3 - q1
    if iqr <= 0:
        return None
    return q1 - _IQR_MULTIPLIER * iqr, q3 + _IQR_MULTIPLIER * iqr


# ---------------------------------------------------------------------------
# Transformações de limpeza (RF05)
# ---------------------------------------------------------------------------

def _coerce_impossible_to_nan(df: pd.DataFrame, col: str, bounds: dict[str, Any]) -> int:
    """Troca por NaN valores fora de [min, max]. Muta ``df``; devolve nº afetado."""
    series = pd.to_numeric(df[col], errors="coerce")
    mask = pd.Series(False, index=df.index)
    if bounds.get("min") is not None:
        mask |= series < float(bounds["min"])
    if bounds.get("max") is not None:
        mask |= series > float(bounds["max"])
    n_bad = int(mask.sum())
    if n_bad:
        df.loc[mask, col] = np.nan
    return n_bad


def _impute_column(
    df: pd.DataFrame, col: str, col_type: str, cfg: dict[str, Any]
) -> dict[str, Any] | None:
    """Imputa faltantes de ``col`` conforme o tipo. Muta ``df``; devolve a ação."""
    series = df[col]
    if col_type == "numeric":
        strategy = cfg["numeric_imputation"]
        if strategy == "none":
            return None
        if strategy == "constant":
            fill: Any = cfg["constant_fill_value"]
            fill = 0 if not isinstance(fill, (int, float)) else fill
            operation = "constant_imputation"
        else:
            numeric = pd.to_numeric(series, errors="coerce")
            fill = float(numeric.median() if strategy == "median" else numeric.mean())
            operation = f"{strategy}_imputation"
        df[col] = series.fillna(fill)
        return {
            "operation": operation,
            "column": col,
            "fill_value": _scalar(fill),
            "reason": f"Faltantes numéricos imputados por '{strategy}' (valor reprodutível).",
        }

    # categóricas, booleanas e texto
    strategy = cfg["categorical_imputation"]
    if strategy == "none":
        return None
    if strategy == "most_frequent":
        non_null = series.dropna()
        if non_null.empty:
            return None
        fill = non_null.mode().iloc[0]
        operation = "most_frequent_imputation"
        reason = "Faltantes categóricos imputados pela categoria mais frequente."
    else:
        fill = cfg["constant_fill_value"]
        operation = "constant_imputation"
        reason = f"Faltantes categóricos imputados pela constante '{fill}'."
    df[col] = series.fillna(fill)
    return {
        "operation": operation,
        "column": col,
        "fill_value": _scalar(fill),
        "reason": reason,
    }


def _group_rare_categories(df: pd.DataFrame, col: str, rare: list[Any]) -> int:
    """Agrupa categorias raras sob ``_RARE_LABEL``. Muta ``df``; devolve nº de linhas."""
    rare_set = set(rare)
    mask = df[col].map(lambda v: _scalar(v) in rare_set if pd.notna(v) else False)
    n = int(mask.sum())
    if n:
        # Categóricas do pandas precisam do rótulo registrado antes da atribuição.
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype("object")
        df.loc[mask, col] = _RARE_LABEL
    return n


def _handle_outliers(
    df: pd.DataFrame, col: str, info: dict[str, Any], strategy: str
) -> dict[str, Any] | None:
    lower, upper = float(info["lower"]), float(info["upper"])
    numeric = pd.to_numeric(df[col], errors="coerce")
    mask = (numeric < lower) | (numeric > upper)
    n_out = int(mask.sum())
    if not n_out:
        return None
    if strategy == "clip":
        df[col] = numeric.clip(lower=lower, upper=upper)
        return {
            "operation": "clip_outliers",
            "column": col,
            "rows": n_out,
            "bounds": {"lower": round(lower, 6), "upper": round(upper, 6)},
            "reason": f"{n_out} outlier(s) limitados às cercas de Tukey (IQR).",
        }
    # remove
    df.drop(index=df.index[mask], inplace=True)
    return {
        "operation": "remove_outliers",
        "column": col,
        "rows": n_out,
        "bounds": {"lower": round(lower, 6), "upper": round(upper, 6)},
        "reason": f"{n_out} linha(s) com outlier(s) removida(s) (cercas de Tukey).",
    }


# ---------------------------------------------------------------------------
# Helpers gerais (espelham as convenções dos demais agentes)
# ---------------------------------------------------------------------------

def _content_hash(df: pd.DataFrame) -> str:
    """Hash determinístico do conteúdo da base, para versionamento de dados."""
    try:
        row_hashes = pd.util.hash_pandas_object(df, index=False).values
        digest = hashlib.sha256(row_hashes.tobytes())
        digest.update("|".join(map(str, df.columns)).encode("utf-8"))
        return digest.hexdigest()
    except Exception:  # pragma: no cover - fallback robusto p/ dtypes exóticos
        return hashlib.sha256(
            pd.util.hash_pandas_object(df.astype(str), index=False).values.tobytes()
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


def _build_rationale(output: dict[str, Any], target_column: str | None) -> str:
    qr = output["quality_report"]
    op_counts: dict[str, int] = {}
    for action in output["actions"]:
        op_counts[action["operation"]] = op_counts.get(action["operation"], 0) + 1

    lines = [
        f"Qualidade avaliada: {len(qr['missing'])} coluna(s) com faltantes, "
        f"{qr['n_duplicate_rows']} duplicata(s), "
        f"{len(qr['constant_columns'])} coluna(s) constante(s), "
        f"{len(qr['rare_categories'])} coluna(s) com categorias raras, "
        f"{len(qr['outliers'])} coluna(s) com outliers.",
    ]
    if op_counts:
        ops_str = ", ".join(f"{n}× {op}" for op, n in sorted(op_counts.items()))
        lines.append(f"Ações aplicadas: {ops_str}.")
    else:
        lines.append("Nenhuma ação de limpeza foi necessária.")
    lines.append(
        f"Base: {output['n_rows_before']}×{output['n_cols_before']} → "
        f"{output['n_rows_after']}×{output['n_cols_after']} "
        f"(hash {output['content_hash_after'][:12]}…)."
    )
    if target_column:
        lines.append(f"Alvo preservado: '{target_column}' não foi imputado nem descartado.")
    if output["warnings"]:
        lines.append("Avisos: " + "; ".join(output["warnings"]))
    return " ".join(lines)
