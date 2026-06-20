"""Agente de Otimização.

Executa busca de hiperparâmetros (Optuna) para os modelos candidatos do model zoo,
otimizando a métrica primária por validação cruzada sobre os folds do SplitAgent.

Cuidado principal: controlar tempo e budget computacional. A busca é limitada por
``n_trials`` por modelo e por orçamentos de tempo (``timeout_seconds`` global e
``per_model_timeout_seconds``); ao estourar o orçamento global, os modelos
restantes são pulados e registrados.

A busca de hiperparâmetros é mantida separada da avaliação final (seção 10): o
agente devolve apenas os melhores hiperparâmetros por modelo (``best_params``),
que devem ser reavaliados pelos agentes de Model Zoo/Avaliação. Cada avaliação de
fold reaproveita o pipeline de atributos ajustado apenas no treino (anti-leakage),
exatamente como no TrainerAgent — cujas rotinas de cálculo são reutilizadas aqui.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from sklearn.base import clone

from src.agents.base import AgentResult, BaseAgent
from src.agents.trainer_agent import (
    _LOWER_IS_BETTER,
    _anomaly_metrics,
    _classification_metrics,
    _family,
    _fit_transform_fold,
    _instantiate,
    _load_zoo,
    _regression_metrics,
    _select_models,
)

# ---------------------------------------------------------------------------
# Convenções
# ---------------------------------------------------------------------------

_DEFAULT_SEED = 42
_DEFAULT_N_TRIALS = 20

#: Espaços de busca padrão por modelo, usados quando o catálogo (ou o contexto)
#: não define um ``search_space``. Formato por hiperparâmetro:
#:   {"type": "int"|"float"|"categorical", ...}
_DEFAULT_SEARCH_SPACES: dict[str, dict[str, dict[str, Any]]] = {
    # Classificação
    "logistic_regression": {"C": {"type": "float", "low": 1e-2, "high": 1e2, "log": True}},
    "decision_tree": {
        "max_depth": {"type": "int", "low": 2, "high": 20},
        "min_samples_split": {"type": "int", "low": 2, "high": 20},
    },
    "random_forest": {
        "n_estimators": {"type": "int", "low": 100, "high": 500, "step": 50},
        "max_depth": {"type": "int", "low": 2, "high": 24},
        "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]},
    },
    "hist_gradient_boosting": {
        "learning_rate": {"type": "float", "low": 1e-2, "high": 3e-1, "log": True},
        "max_leaf_nodes": {"type": "int", "low": 15, "high": 63},
        "l2_regularization": {"type": "float", "low": 1e-6, "high": 1.0, "log": True},
    },
    "svm": {
        "C": {"type": "float", "low": 1e-1, "high": 1e2, "log": True},
        "gamma": {"type": "categorical", "choices": ["scale", "auto"]},
    },
    "knn": {
        "n_neighbors": {"type": "int", "low": 3, "high": 30},
        "weights": {"type": "categorical", "choices": ["uniform", "distance"]},
    },
    "naive_bayes": {"var_smoothing": {"type": "float", "low": 1e-12, "high": 1e-6, "log": True}},
    "mlp": {
        "alpha": {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
        "learning_rate_init": {"type": "float", "low": 1e-4, "high": 1e-1, "log": True},
    },
    # Regressão
    "elastic_net": {
        "alpha": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
        "l1_ratio": {"type": "float", "low": 0.0, "high": 1.0},
    },
    "decision_tree_regressor": {"max_depth": {"type": "int", "low": 2, "high": 20}},
    "random_forest_regressor": {
        "n_estimators": {"type": "int", "low": 100, "high": 500, "step": 50},
        "max_depth": {"type": "int", "low": 2, "high": 24},
    },
    "hist_gradient_boosting_regressor": {
        "learning_rate": {"type": "float", "low": 1e-2, "high": 3e-1, "log": True},
        "max_leaf_nodes": {"type": "int", "low": 15, "high": 63},
    },
    "svr": {
        "C": {"type": "float", "low": 1e-1, "high": 1e2, "log": True},
        "gamma": {"type": "categorical", "choices": ["scale", "auto"]},
    },
    "knn_regressor": {
        "n_neighbors": {"type": "int", "low": 3, "high": 30},
        "weights": {"type": "categorical", "choices": ["uniform", "distance"]},
    },
    "mlp_regressor": {"alpha": {"type": "float", "low": 1e-6, "high": 1e-1, "log": True}},
    # Anomalia
    "isolation_forest": {
        "n_estimators": {"type": "int", "low": 100, "high": 400, "step": 50},
        "contamination": {"type": "float", "low": 0.01, "high": 0.3},
    },
    "one_class_svm": {
        "nu": {"type": "float", "low": 0.01, "high": 0.5},
        "gamma": {"type": "categorical", "choices": ["scale", "auto"]},
    },
    "local_outlier_factor": {"n_neighbors": {"type": "int", "low": 5, "high": 40}},
}


class OptimizationAgent(BaseAgent):
    """Agente 8 — Otimização de Hiperparâmetros (Optuna).

    Entradas esperadas em ``context``:
        - ``dataframe`` (pandas.DataFrame): base já limpa.
        - ``task_type`` / ``target_column`` / ``primary_metric``: da Formulação
          (lidos de ``context['problem']`` se ausentes no topo).
        - ``folds``: pares ``(train_idx, test_idx)`` (SplitAgent) — ou
          ``context['split']['folds']``.
        - ``pipeline`` (sklearn, opcional): pipeline de atributos NÃO ajustado.
        - ``feature_columns`` (list, opcional).

    Sobreposições (opcionais):
        - ``random_seed`` (int, padrão 42).
        - ``include`` / ``exclude`` / ``tiers``: seleção de modelos (como no zoo).
        - ``n_trials`` (int, padrão 20): trials Optuna por modelo.
        - ``timeout_seconds`` (float): teto de tempo TOTAL (RNF08); ao estourar,
          os modelos restantes são pulados.
        - ``per_model_timeout_seconds`` (float): teto por modelo.
        - ``search_spaces`` (dict): espaços de busca por modelo (sobrepõem catálogo
          e defaults internos).
        - ``model_zoo`` (dict) / ``model_zoo_path`` (str): catálogo alternativo.

    Saída em ``AgentResult.output``:
        - ``results``: por modelo — ``best_params`` (mesclado c/ defaults),
          ``tuned_params``, ``best_value``, ``n_trials_completed/pruned``,
          ``elapsed_seconds``, ``search_space``, ``status``;
        - ``ranking`` pela métrica primária, ``skipped_models``, ``direction``,
          ``warnings``.
    """

    name = "Agente de Otimização"
    event_type = "hyperparameter_optimization"

    def run(self, context: dict[str, Any]) -> AgentResult:
        warnings: list[str] = []
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # ------------------------------------------------------------------
        # 1. Entradas
        # ------------------------------------------------------------------
        df = _load_dataframe(context)
        problem = context.get("problem") or {}
        task_type = str(context.get("task_type") or problem.get("task_type") or "").strip().lower()
        if task_type not in {"classification", "regression", "anomaly"}:
            raise ValueError(
                f"'task_type' inválido ou ausente: '{task_type}'. "
                "Use 'classification', 'regression' ou 'anomaly'."
            )
        target_column = context.get("target_column") or problem.get("target_column") or None
        if task_type in ("classification", "regression") and not target_column:
            raise ValueError(f"'target_column' é obrigatório para task_type='{task_type}'.")
        if target_column and target_column not in df.columns:
            raise ValueError(f"'target_column' '{target_column}' não existe na base.")

        primary_metric = str(
            context.get("primary_metric") or problem.get("primary_metric") or ""
        ).strip().lower()
        if not primary_metric:
            raise ValueError("'primary_metric' é obrigatório para otimizar.")

        seed = int(context.get("random_seed", _DEFAULT_SEED))
        n_trials = int(context.get("n_trials", _DEFAULT_N_TRIALS))
        if n_trials < 1:
            raise ValueError(f"'n_trials' deve ser ≥ 1 (recebido {n_trials}).")
        timeout = context.get("timeout_seconds")
        per_model_timeout = context.get("per_model_timeout_seconds")
        folds = _resolve_folds(context, len(df))
        feature_columns = context.get("feature_columns") or [
            c for c in df.columns if c != target_column
        ]
        pipeline = context.get("pipeline")
        direction = "minimize" if primary_metric in _LOWER_IS_BETTER else "maximize"

        # ------------------------------------------------------------------
        # 2. Seleção dos modelos a otimizar
        # ------------------------------------------------------------------
        zoo = _load_zoo(context)
        selected = _select_models(zoo, task_type, context, warnings)
        if not selected:
            raise ValueError(f"Nenhum modelo aplicável a task_type='{task_type}'.")

        # ------------------------------------------------------------------
        # 3. Busca por modelo, respeitando o budget
        # ------------------------------------------------------------------
        results: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        start = time.perf_counter()

        for name, model_spec in selected:
            elapsed_total = time.perf_counter() - start
            if timeout is not None and elapsed_total >= float(timeout):
                skipped.append({"model_name": name, "reason": "timeout_seconds (global) excedido."})
                continue

            space = _resolve_search_space(name, model_spec, context)
            if not space:
                skipped.append({"model_name": name, "reason": "sem search_space definido."})
                continue

            # Orçamento de tempo efetivo deste modelo (mínimo entre limites).
            model_timeout = _effective_timeout(timeout, elapsed_total, per_model_timeout)

            record = _optimize_model(
                name=name,
                model_spec=model_spec,
                space=space,
                df=df,
                folds=folds,
                feature_columns=feature_columns,
                target_column=target_column,
                pipeline=pipeline,
                task_type=task_type,
                primary_metric=primary_metric,
                direction=direction,
                seed=seed,
                n_trials=n_trials,
                model_timeout=model_timeout,
            )
            if record["status"] == "failed":
                warnings.append(f"Otimização de '{name}' sem trials válidos.")
            results.append(record)

        # ------------------------------------------------------------------
        # 4. Ranking e saída
        # ------------------------------------------------------------------
        ranking = _rank(results, direction)
        output: dict[str, Any] = {
            "task_type": task_type,
            "target_column": target_column,
            "primary_metric": primary_metric,
            "direction": direction,
            "random_seed": seed,
            "n_trials": n_trials,
            "timeout_seconds": timeout,
            "per_model_timeout_seconds": per_model_timeout,
            "results": results,
            "ranking": ranking,
            "skipped_models": skipped,
            "warnings": warnings,
        }
        output = _to_native(output)

        rationale = _build_rationale(output)
        return AgentResult(output=output, rationale=rationale, warnings=warnings)


# ---------------------------------------------------------------------------
# Otimização de um modelo
# ---------------------------------------------------------------------------

def _optimize_model(
    *,
    name: str,
    model_spec: dict[str, Any],
    space: dict[str, dict[str, Any]],
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    feature_columns: list[str],
    target_column: str | None,
    pipeline: Any,
    task_type: str,
    primary_metric: str,
    direction: str,
    seed: int,
    n_trials: int,
    model_timeout: float | None,
) -> dict[str, Any]:
    estimator = model_spec["estimator"]
    default_params = dict(model_spec.get("default_params") or {})

    def objective(trial: optuna.Trial) -> float:
        tuned = _suggest(trial, space)
        params = {**default_params, **tuned}
        try:
            return _score_params(
                estimator, params, df, folds, feature_columns,
                target_column, pipeline, task_type, primary_metric, seed,
            )
        except optuna.TrialPruned:
            raise
        except Exception:  # noqa: BLE001 - combinação inválida => poda o trial
            raise optuna.TrialPruned() from None

    study = optuna.create_study(direction=direction, sampler=TPESampler(seed=seed))
    t0 = time.perf_counter()
    try:
        study.optimize(objective, n_trials=n_trials, timeout=model_timeout)
    except Exception as exc:  # noqa: BLE001 - protege o pipeline de falhas do Optuna
        return _failed_record(name, estimator, default_params, space, exc)
    elapsed = time.perf_counter() - t0

    completed = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    pruned = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    if not completed:
        return _failed_record(name, estimator, default_params, space, None)

    tuned = dict(study.best_trial.params)
    return {
        "model_name": name,
        "model_family": _family(estimator),
        "estimator": estimator,
        "search_space": space,
        "default_params": default_params,
        "tuned_params": tuned,
        "best_params": {**default_params, **tuned},
        "best_value": round(float(study.best_value), 6),
        "n_trials_completed": len(completed),
        "n_trials_pruned": len(pruned),
        "elapsed_seconds": round(elapsed, 4),
        "status": "ok",
    }


def _score_params(
    estimator: str,
    params: dict[str, Any],
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    feature_columns: list[str],
    target_column: str | None,
    pipeline: Any,
    task_type: str,
    primary_metric: str,
    seed: int,
) -> float:
    """Média da métrica primária nos folds para um conjunto de hiperparâmetros."""
    template, _ = _instantiate(estimator, {"default_params": params}, seed)
    scores: list[float] = []
    for train_idx, test_idx in folds:
        X_train = df.iloc[train_idx][feature_columns]
        X_test = df.iloc[test_idx][feature_columns]
        y_train = df.iloc[train_idx][target_column] if target_column else None
        y_test = df.iloc[test_idx][target_column] if target_column else None

        Xtr, Xte = _fit_transform_fold(pipeline, X_train, X_test, y_train)
        model = clone(template)
        if task_type == "anomaly":
            model.fit(Xtr)
            metrics = _anomaly_metrics(model, Xte, y_test)
        else:
            model.fit(Xtr, y_train)
            metrics = (
                _classification_metrics(model, Xte, y_test)
                if task_type == "classification"
                else _regression_metrics(model, Xte, y_test)
            )
        value = metrics.get(primary_metric)
        if value is None:
            raise optuna.TrialPruned()
        scores.append(value)
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Espaço de busca
# ---------------------------------------------------------------------------

def _resolve_search_space(
    name: str, model_spec: dict[str, Any], context: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Prioridade: contexto > catálogo (search_space) > defaults internos."""
    overrides = context.get("search_spaces") or {}
    if name in overrides:
        return overrides[name]
    if model_spec.get("search_space"):
        return model_spec["search_space"]
    return _DEFAULT_SEARCH_SPACES.get(name, {})


def _suggest(trial: optuna.Trial, space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for param, spec in space.items():
        kind = spec.get("type")
        if kind == "int":
            if spec.get("log"):
                params[param] = trial.suggest_int(param, spec["low"], spec["high"], log=True)
            else:
                params[param] = trial.suggest_int(
                    param, spec["low"], spec["high"], step=spec.get("step", 1)
                )
        elif kind == "float":
            params[param] = trial.suggest_float(
                param, spec["low"], spec["high"], log=bool(spec.get("log", False))
            )
        elif kind == "categorical":
            params[param] = trial.suggest_categorical(param, spec["choices"])
        else:
            raise ValueError(f"Tipo de hiperparâmetro inválido em '{param}': '{kind}'.")
    return params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _effective_timeout(
    timeout: Any, elapsed_total: float, per_model_timeout: Any
) -> float | None:
    limits = []
    if timeout is not None:
        limits.append(max(0.0, float(timeout) - elapsed_total))
    if per_model_timeout is not None:
        limits.append(float(per_model_timeout))
    return min(limits) if limits else None


def _failed_record(
    name: str,
    estimator: str,
    default_params: dict[str, Any],
    space: dict[str, dict[str, Any]],
    exc: Exception | None,
) -> dict[str, Any]:
    record = {
        "model_name": name,
        "model_family": _family(estimator),
        "estimator": estimator,
        "search_space": space,
        "default_params": default_params,
        "best_params": default_params,
        "best_value": None,
        "n_trials_completed": 0,
        "status": "failed",
    }
    if exc is not None:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def _rank(results: list[dict[str, Any]], direction: str) -> list[dict[str, Any]]:
    ranked = [
        {"model_name": r["model_name"], "best_value": r["best_value"]}
        for r in results
        if r["status"] == "ok" and r["best_value"] is not None
    ]
    ranked.sort(key=lambda r: r["best_value"], reverse=(direction == "maximize"))
    for pos, r in enumerate(ranked, start=1):
        r["rank"] = pos
    return ranked


def _load_dataframe(context: dict[str, Any]) -> pd.DataFrame:
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
        f"source_type '{source_type}' não suportado pela otimização "
        "(use 'csv' ou 'parquet', ou passe 'dataframe' diretamente)."
    )


def _resolve_folds(context: dict[str, Any], n_rows: int) -> list[tuple[np.ndarray, np.ndarray]]:
    raw = context.get("folds")
    if raw is None:
        split = context.get("split") or {}
        records = split.get("folds")
        if records:
            raw = [(r["train_idx"], r["test_idx"]) for r in records]
    if not raw:
        raise ValueError(
            "Forneça 'folds' (saída do SplitAgent) ou 'split' com os índices dos folds."
        )
    folds = [(np.asarray(tr, dtype=int), np.asarray(te, dtype=int)) for tr, te in raw]
    for i, (tr, te) in enumerate(folds):
        if tr.max(initial=-1) >= n_rows or te.max(initial=-1) >= n_rows:
            raise ValueError(f"Índices do fold {i} excedem o número de linhas da base.")
    return folds


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
    ok = [r for r in output["results"] if r["status"] == "ok"]
    lines = [
        f"Otimização de {len(ok)} modelo(s) por Optuna ({output['n_trials']} trials/modelo, "
        f"direção '{output['direction']}' de '{output['primary_metric']}', seed "
        f"{output['random_seed']}).",
    ]
    if output["ranking"]:
        best = output["ranking"][0]
        lines.append(f"Melhor: {best['model_name']} ({output['primary_metric']}={best['best_value']}).")
    if output["timeout_seconds"] is not None:
        lines.append(f"Budget total: {output['timeout_seconds']}s.")
    if output["skipped_models"]:
        lines.append(f"Pulados: {', '.join(s['model_name'] for s in output['skipped_models'])}.")
    lines.append("Busca separada da avaliação final; reavaliar best_params no Model Zoo/Avaliação.")
    if output["warnings"]:
        lines.append("Avisos: " + "; ".join(output["warnings"]))
    return " ".join(lines)
