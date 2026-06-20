"""Agente de Split e Validação.

Define a estratégia de particionamento — holdout, k-fold, stratified k-fold,
group split ou time split — e materializa os índices concretos de treino/teste
com splitters do scikit-learn (RF08).

Cuidado principal: evitar contaminação entre treino e teste. O agente verifica
que treino e teste são disjuntos em cada fold; para ``group_kfold`` garante que
nenhum grupo aparece nos dois lados; para ``time_split`` não embaralha passado e
futuro. A seed é sempre registrada (RNF01), tornando a partição reproduzível.

O ``output`` (persistido como JSONB) descreve a estratégia e os folds (índices
incluídos para reexecução). O objeto splitter do sklearn (quando aplicável) viaja
em ``AgentResult.splitter``, e a lista de folds materializados em
``AgentResult.folds`` (pares de arrays numpy ``(train_idx, test_idx)``).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)

from src.agents.base import AgentResult, BaseAgent

# ---------------------------------------------------------------------------
# Convenções (sobreponíveis via contexto)
# ---------------------------------------------------------------------------

_DEFAULT_SEED = 42
_DEFAULT_N_SPLITS = 5
_DEFAULT_TEST_SIZE = 0.2

_VALID_STRATEGIES = {"holdout", "kfold", "stratified_kfold", "group_kfold", "time_split"}


class SplitAgent(BaseAgent):
    """Agente 5 — Split e Validação Cruzada.

    Entradas esperadas em ``context`` (uma fonte de dados é obrigatória):
        - ``dataframe`` (pandas.DataFrame): base (já limpa) a ser particionada.
        - ``data`` (dict): fonte alternativa (``source_type`` csv/parquet, ``source_uri``).
        - ``validation`` / ``problem`` (dict | None): saída do Agente de Formulação;
          dela são lidos ``split_strategy``, ``task_type``, ``target_column``,
          ``group_column`` e ``time_column`` quando não vierem no topo do contexto.
        - Chaves diretas equivalentes (têm prioridade): ``split_strategy``,
          ``task_type``, ``target_column``, ``group_column``, ``time_column``.

    Sobreposições (opcionais):
        - ``random_seed`` (int, padrão 42).
        - ``n_splits`` (int, padrão 5): nº de folds nas estratégias de CV.
        - ``test_size`` (float, padrão 0.2): proporção de teste no holdout.
        - ``shuffle`` (bool, padrão True): embaralhar antes de partir (ignorado em
          ``time_split``).
        - ``include_indices`` (bool, padrão True): incluir os índices no ``output``.

    Saída em ``AgentResult.output``:
        - ``split_strategy``, ``n_splits``, ``test_size``, ``random_seed``,
          ``shuffle``, ``stratified``, ``grouped``, ``temporal``, ``n_samples``;
        - ``folds``: lista de {``fold``, ``n_train``, ``n_test``, ``train_idx``,
          ``test_idx``};
        - ``no_contamination_verified`` (bool), ``warnings``.

    O splitter do sklearn é exposto em ``AgentResult.splitter`` (None para holdout)
    e os folds materializados em ``AgentResult.folds``.
    """

    name = "Agente de Split e Validação"
    event_type = "split_strategy"

    def run(self, context: dict[str, Any]) -> AgentResult:
        warnings: list[str] = []

        # ------------------------------------------------------------------
        # 1. Ingestão e parâmetros
        # ------------------------------------------------------------------
        df = _load_dataframe(context)
        n = int(df.shape[0])
        if n < 2:
            raise ValueError("A base precisa de ao menos 2 linhas para ser particionada.")

        spec = _resolve_spec(context)
        seed = int(context.get("random_seed", _DEFAULT_SEED))
        n_splits = int(context.get("n_splits", _DEFAULT_N_SPLITS))
        test_size = float(context.get("test_size", _DEFAULT_TEST_SIZE))
        shuffle = bool(context.get("shuffle", True))

        strategy = spec["split_strategy"]
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"'split_strategy' inválida: '{strategy}'. Opções: {sorted(_VALID_STRATEGIES)}."
            )

        # Valida a existência das colunas declaradas
        for key in ("target_column", "group_column", "time_column"):
            col = spec[key]
            if col and col not in df.columns:
                raise ValueError(f"'{key}' '{col}' não existe na base.")

        # ------------------------------------------------------------------
        # 2. Materialização dos folds conforme a estratégia
        # ------------------------------------------------------------------
        strategy, folds, splitter, meta = _make_folds(
            df=df,
            strategy=strategy,
            spec=spec,
            n_splits=n_splits,
            test_size=test_size,
            shuffle=shuffle,
            seed=seed,
            warnings=warnings,
        )

        # ------------------------------------------------------------------
        # 3. Verificação de contaminação (treino ∩ teste = ∅; grupos disjuntos)
        # ------------------------------------------------------------------
        _verify_no_contamination(
            folds, df, spec["group_column"] if meta["grouped"] else None
        )

        # ------------------------------------------------------------------
        # 4. Monta a saída JSON-serializável
        # ------------------------------------------------------------------
        include_indices = bool(context.get("include_indices", True))
        fold_records: list[dict[str, Any]] = []
        for i, (train_idx, test_idx) in enumerate(folds):
            record: dict[str, Any] = {
                "fold": i,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            }
            if include_indices:
                record["train_idx"] = [int(x) for x in train_idx]
                record["test_idx"] = [int(x) for x in test_idx]
            fold_records.append(record)

        output: dict[str, Any] = {
            "split_strategy": strategy,
            "n_splits": len(folds),
            "test_size": test_size if strategy == "holdout" else None,
            "random_seed": seed,
            "shuffle": meta["shuffle"],
            "stratified": meta["stratified"],
            "grouped": meta["grouped"],
            "temporal": meta["temporal"],
            "n_samples": n,
            "target_column": spec["target_column"],
            "group_column": spec["group_column"],
            "time_column": spec["time_column"],
            "folds": fold_records,
            "no_contamination_verified": True,
            "warnings": warnings,
        }
        output = _to_native(output)

        rationale = _build_rationale(output)
        result = AgentResult(output=output, rationale=rationale, warnings=warnings)
        result.splitter = splitter
        result.folds = folds
        return result


# ---------------------------------------------------------------------------
# Ingestão e especificação
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
        f"source_type '{source_type}' não suportado pelo particionamento "
        "(use 'csv' ou 'parquet', ou passe 'dataframe' diretamente)."
    )


def _resolve_spec(context: dict[str, Any]) -> dict[str, Any]:
    """Lê estratégia e colunas-chave do contexto (topo) ou da saída do ProblemAgent."""
    upstream = context.get("validation") or context.get("problem") or {}

    def pick(key: str) -> Any:
        value = context.get(key)
        if value in (None, ""):
            value = upstream.get(key)
        return value or None

    strategy = pick("split_strategy")
    task_type = (pick("task_type") or "")
    task_type = str(task_type).strip().lower() or None
    target_column = pick("target_column")
    group_column = pick("group_column")
    time_column = pick("time_column")

    # Deriva uma estratégia padrão quando não informada (espelha o ProblemAgent).
    if not strategy:
        if time_column:
            strategy = "time_split"
        elif group_column:
            strategy = "group_kfold"
        elif task_type == "classification":
            strategy = "stratified_kfold"
        else:
            strategy = "kfold"
    return {
        "split_strategy": str(strategy).strip().lower(),
        "task_type": task_type,
        "target_column": target_column,
        "group_column": group_column,
        "time_column": time_column,
    }


# ---------------------------------------------------------------------------
# Geração dos folds
# ---------------------------------------------------------------------------

def _make_folds(
    *,
    df: pd.DataFrame,
    strategy: str,
    spec: dict[str, Any],
    n_splits: int,
    test_size: float,
    shuffle: bool,
    seed: int,
    warnings: list[str],
) -> tuple[str, list[tuple[np.ndarray, np.ndarray]], Any, dict[str, bool]]:
    n = len(df)
    indices = np.arange(n)
    meta = {"stratified": False, "grouped": False, "temporal": False, "shuffle": shuffle}

    # --- time_split: requer ordenação temporal; nunca embaralha ---------------
    if strategy == "time_split":
        if not spec["time_column"]:
            raise ValueError("'time_split' exige 'time_column'.")
        meta["temporal"] = True
        meta["shuffle"] = False
        order = _time_order(df[spec["time_column"]])
        eff = _bounded_splits(n_splits, n - 1, "time_split", warnings)
        splitter = TimeSeriesSplit(n_splits=eff)
        folds = [
            (order[train_pos], order[test_pos])
            for train_pos, test_pos in splitter.split(order)
        ]
        return strategy, folds, splitter, meta

    # --- group_kfold: grupos não podem cruzar treino/teste --------------------
    if strategy == "group_kfold":
        if not spec["group_column"]:
            raise ValueError("'group_kfold' exige 'group_column'.")
        groups = df[spec["group_column"]].to_numpy()
        n_groups = int(pd.Series(groups).nunique())
        if n_groups < 2:
            raise ValueError(
                f"'group_kfold' precisa de ≥ 2 grupos distintos; encontrados {n_groups}."
            )
        meta["grouped"] = True
        eff = _bounded_splits(n_splits, n_groups, "group_kfold", warnings)
        splitter = GroupKFold(n_splits=eff, shuffle=shuffle, random_state=seed if shuffle else None)
        folds = [
            (indices[train_pos], indices[test_pos])
            for train_pos, test_pos in splitter.split(indices, groups=groups)
        ]
        return strategy, folds, splitter, meta

    # --- stratified_kfold: preserva a proporção das classes -------------------
    if strategy == "stratified_kfold":
        strategy, folds, splitter, meta = _stratified_folds(
            df, spec, indices, n_splits, shuffle, seed, meta, warnings
        )
        return strategy, folds, splitter, meta

    # --- holdout: uma única partição treino/teste -----------------------------
    if strategy == "holdout":
        stratify, do_strat = _stratify_target(df, spec, n_splits=2, warnings=warnings)
        meta["stratified"] = do_strat
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            shuffle=shuffle,
            random_state=seed if shuffle else None,
            stratify=stratify if do_strat else None,
        )
        return strategy, [(train_idx, test_idx)], None, meta

    # --- kfold: padrão genérico -----------------------------------------------
    eff = _bounded_splits(n_splits, n, "kfold", warnings)
    splitter = KFold(n_splits=eff, shuffle=shuffle, random_state=seed if shuffle else None)
    folds = [
        (indices[train_pos], indices[test_pos])
        for train_pos, test_pos in splitter.split(indices)
    ]
    return strategy, folds, splitter, meta


def _stratified_folds(
    df: pd.DataFrame,
    spec: dict[str, Any],
    indices: np.ndarray,
    n_splits: int,
    shuffle: bool,
    seed: int,
    meta: dict[str, bool],
    warnings: list[str],
) -> tuple[str, list[tuple[np.ndarray, np.ndarray]], Any, dict[str, bool]]:
    """StratifiedKFold com salvaguardas; faz fallback para KFold quando inviável."""
    target = spec["target_column"]
    if spec["task_type"] == "regression":
        warnings.append(
            "Estratificação não se aplica a regressão; usando 'kfold' simples."
        )
        return _fallback_kfold(indices, n_splits, shuffle, seed, meta, warnings)
    if not target:
        warnings.append(
            "'stratified_kfold' exige 'target_column'; sem alvo, usando 'kfold' simples."
        )
        return _fallback_kfold(indices, n_splits, shuffle, seed, meta, warnings)

    y = df[target]
    if y.isna().any():
        raise ValueError(
            f"Alvo '{target}' contém faltantes; trate-os (limpeza) antes do particionamento."
        )

    min_class = int(y.value_counts().min())
    if min_class < 2:
        warnings.append(
            f"Classe minoritária de '{target}' tem {min_class} amostra(s); "
            "estratificação impossível, usando 'kfold' simples."
        )
        return _fallback_kfold(indices, n_splits, shuffle, seed, meta, warnings)

    eff = _bounded_splits(n_splits, min_class, "stratified_kfold", warnings)
    meta["stratified"] = True
    splitter = StratifiedKFold(n_splits=eff, shuffle=shuffle, random_state=seed if shuffle else None)
    folds = [
        (indices[train_pos], indices[test_pos])
        for train_pos, test_pos in splitter.split(indices, y.to_numpy())
    ]
    return "stratified_kfold", folds, splitter, meta


def _fallback_kfold(
    indices: np.ndarray,
    n_splits: int,
    shuffle: bool,
    seed: int,
    meta: dict[str, bool],
    warnings: list[str],
) -> tuple[str, list[tuple[np.ndarray, np.ndarray]], Any, dict[str, bool]]:
    eff = _bounded_splits(n_splits, len(indices), "kfold", warnings)
    splitter = KFold(n_splits=eff, shuffle=shuffle, random_state=seed if shuffle else None)
    folds = [
        (indices[train_pos], indices[test_pos])
        for train_pos, test_pos in splitter.split(indices)
    ]
    return "kfold", folds, splitter, meta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bounded_splits(n_splits: int, upper: int, strategy: str, warnings: list[str]) -> int:
    """Garante 2 ≤ n_splits ≤ upper, avisando quando precisa reduzir."""
    if n_splits < 2:
        raise ValueError(f"'n_splits' deve ser ≥ 2 para '{strategy}' (recebido {n_splits}).")
    if upper < 2:
        raise ValueError(
            f"Amostras/grupos insuficientes para '{strategy}': é preciso ao menos 2."
        )
    if n_splits > upper:
        warnings.append(
            f"'n_splits'={n_splits} reduzido para {upper} em '{strategy}' "
            "(limite de amostras/grupos/classes)."
        )
        return upper
    return n_splits


def _time_order(time_series: pd.Series) -> np.ndarray:
    """Índices originais ordenados cronologicamente (estável)."""
    parsed = time_series
    if not pd.api.types.is_datetime64_any_dtype(parsed):
        parsed = pd.to_datetime(parsed, errors="coerce", format="mixed")
    if parsed.isna().any():
        raise ValueError(
            "Coluna temporal contém valores não conversíveis para data; "
            "limpe-a antes do 'time_split'."
        )
    return parsed.to_numpy().argsort(kind="stable")


def _stratify_target(
    df: pd.DataFrame, spec: dict[str, Any], n_splits: int, warnings: list[str]
) -> tuple[Any, bool]:
    """Decide se o holdout pode estratificar pelo alvo (classificação balanceável)."""
    target = spec["target_column"]
    if not target or spec["task_type"] == "regression":
        return None, False
    y = df[target]
    if y.isna().any() or int(y.value_counts().min()) < 2:
        warnings.append(
            f"Holdout sem estratificação: alvo '{target}' tem classe(s) com < 2 amostras "
            "ou faltantes."
        )
        return None, False
    return y, True


def _verify_no_contamination(
    folds: list[tuple[np.ndarray, np.ndarray]],
    df: pd.DataFrame,
    group_column: str | None,
) -> None:
    """Confere que treino e teste são disjuntos (e grupos não vazam, se aplicável)."""
    groups = df[group_column].to_numpy() if group_column else None
    for i, (train_idx, test_idx) in enumerate(folds):
        if set(train_idx) & set(test_idx):
            raise AssertionError(f"Contaminação no fold {i}: índices em treino e teste.")
        if groups is not None:
            if set(groups[train_idx]) & set(groups[test_idx]):
                raise AssertionError(
                    f"Contaminação de grupo no fold {i}: grupo presente em treino e teste."
                )


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


def _build_rationale(output: dict[str, Any]) -> str:
    lines = [
        f"Particionamento '{output['split_strategy']}' com {output['n_splits']} "
        f"{'partição' if output['split_strategy'] == 'holdout' else 'fold(s)'} "
        f"sobre {output['n_samples']} amostras (seed {output['random_seed']}).",
    ]
    traits = []
    if output["stratified"]:
        traits.append("estratificado pelo alvo")
    if output["grouped"]:
        traits.append(f"agrupado por '{output['group_column']}'")
    if output["temporal"]:
        traits.append(f"ordenado por '{output['time_column']}' (sem embaralhamento)")
    if traits:
        lines.append("Modo: " + ", ".join(traits) + ".")
    lines.append("Disjunção treino/teste verificada (sem contaminação).")
    if output["warnings"]:
        lines.append("Avisos: " + "; ".join(output["warnings"]))
    return " ".join(lines)
