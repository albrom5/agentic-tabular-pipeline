"""Agente de Model Zoo (treinamento cruzado).

Executa as famílias de modelos clássicos aplicáveis ao tipo de tarefa, definidas
em ``configs/model_zoo.yaml`` (RF09). O catálogo é extensível: adicionar um modelo
no YAML não exige reescrever este agente.

Cada modelo é treinado **por fold**, ajustando o pipeline de atributos apenas com
os dados de treino daquele fold (evita leakage, seção 5 do documento). Para cada
execução o agente registra fold, seed, hiperparâmetros, métricas e tempo — o
cuidado principal exigido na tabela de agentes e a base da tabela ``model_results``.

O ``output`` (JSONB) traz os resultados por modelo (métricas por fold, média e
desvio) e um ranking de conveniência pela métrica primária. Os estimadores
treinados não são serializados aqui: a seleção/explicação fica a cargo dos agentes
de Avaliação e Relator.
"""

from __future__ import annotations

import importlib
import inspect
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)

from src.agents.base import AgentResult, BaseAgent

# ---------------------------------------------------------------------------
# Convenções
# ---------------------------------------------------------------------------

_DEFAULT_SEED = 42
_DEFAULT_ZOO_PATH = Path(__file__).resolve().parents[2] / "configs" / "model_zoo.yaml"
#: Tiers incluídos por padrão (os "bonus" dependem de libs opcionais como XGBoost).
_DEFAULT_TIERS = {"minimum"}
#: Métricas em que valores menores são melhores (afeta a direção do ranking).
_LOWER_IS_BETTER = {"rmse", "mae", "mse", "mape", "medae", "log_loss"}


class TrainerAgent(BaseAgent):
    """Agente 6 — Model Zoo / Treinamento Cruzado.

    Entradas esperadas em ``context``:
        - ``dataframe`` (pandas.DataFrame): base já limpa.
        - ``task_type`` (str) e ``target_column`` (str): da Formulação do Problema
          (lidos também de ``context['problem']`` se ausentes no topo).
        - ``primary_metric`` (str): métrica para o ranking de conveniência.
        - ``folds``: lista de pares ``(train_idx, test_idx)`` (saída do SplitAgent,
          ``AgentResult.folds``). Alternativamente ``context['split']['folds']`` com
          ``train_idx``/``test_idx``.
        - ``pipeline`` (sklearn, opcional): pipeline de atributos NÃO ajustado
          (``AgentResult.pipeline`` do FeatureAgent); é clonado e ajustado por fold.
        - ``feature_columns`` (list, opcional): colunas usadas como atributos.

    Sobreposições (opcionais):
        - ``random_seed`` (int, padrão 42).
        - ``include`` / ``exclude`` (list[str]): seleção explícita de modelos.
        - ``tiers`` (list[str]): tiers do catálogo a considerar (padrão {"minimum"}).
        - ``model_zoo`` (dict) ou ``model_zoo_path`` (str): catálogo alternativo.
        - ``time_budget_seconds`` (float): teto de tempo total (RNF08); modelos
          restantes são pulados ao estourar.

    Saída em ``AgentResult.output``:
        - ``results``: por modelo — ``model_name``, ``model_family``, ``estimator``,
          ``hyperparameters``, ``tier``, ``fold_metrics``, ``metrics_mean``,
          ``metrics_std``, ``primary_metric_mean/std``, ``total_fit_seconds``,
          ``status`` ("ok"/"failed");
        - ``ranking``: modelos ok ordenados pela métrica primária;
        - ``skipped_models``, ``n_models``, ``n_folds``, ``random_seed``, ``warnings``.
    """

    name = "Agente de Model Zoo"
    event_type = "model_training"

    def run(self, context: dict[str, Any]) -> AgentResult:
        warnings: list[str] = []

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
        ).strip().lower() or None

        seed = int(context.get("random_seed", _DEFAULT_SEED))
        folds = _resolve_folds(context, len(df))
        feature_columns = context.get("feature_columns") or [
            c for c in df.columns if c != target_column
        ]
        pipeline = context.get("pipeline")

        # ------------------------------------------------------------------
        # 2. Seleção dos modelos aplicáveis ao tipo de tarefa
        # ------------------------------------------------------------------
        zoo = _load_zoo(context)
        selected = _select_models(zoo, task_type, context, warnings)
        if not selected:
            raise ValueError(
                f"Nenhum modelo aplicável a task_type='{task_type}' após a seleção."
            )

        # ------------------------------------------------------------------
        # 3. Treinamento cruzado por modelo / fold
        # ------------------------------------------------------------------
        results: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        budget = context.get("time_budget_seconds")
        start = time.perf_counter()

        for name, model_spec in selected:
            if budget is not None and (time.perf_counter() - start) > float(budget):
                skipped.append({"model_name": name, "reason": "time_budget_seconds excedido."})
                continue
            try:
                template, params = _instantiate(model_spec["estimator"], model_spec, seed)
            except ImportError as exc:
                warnings.append(f"Modelo '{name}' ignorado (dependência ausente: {exc.name}).")
                skipped.append({"model_name": name, "reason": f"ImportError: {exc.name}"})
                continue

            record = _train_model(
                name=name,
                template=template,
                params=params,
                model_spec=model_spec,
                df=df,
                folds=folds,
                feature_columns=feature_columns,
                target_column=target_column,
                pipeline=pipeline,
                task_type=task_type,
                seed=seed,
            )
            if record["status"] == "failed":
                warnings.append(f"Modelo '{name}' falhou: {record['error']}")
            results.append(record)

        # ------------------------------------------------------------------
        # 4. Ranking de conveniência pela métrica primária
        # ------------------------------------------------------------------
        ranking = _rank(results, primary_metric)

        output: dict[str, Any] = {
            "task_type": task_type,
            "target_column": target_column,
            "primary_metric": primary_metric,
            "random_seed": seed,
            "n_models": len(results),
            "n_folds": len(folds),
            "results": results,
            "ranking": ranking,
            "skipped_models": skipped,
            "warnings": warnings,
        }
        output = _to_native(output)

        rationale = _build_rationale(output)
        return AgentResult(output=output, rationale=rationale, warnings=warnings)


# ---------------------------------------------------------------------------
# Treino de um modelo em todos os folds
# ---------------------------------------------------------------------------

def _train_model(
    *,
    name: str,
    template: Any,
    params: dict[str, Any],
    model_spec: dict[str, Any],
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    feature_columns: list[str],
    target_column: str | None,
    pipeline: Any,
    task_type: str,
    seed: int,
) -> dict[str, Any]:
    fold_metrics: list[dict[str, Any]] = []
    total_seconds = 0.0
    try:
        for i, (train_idx, test_idx) in enumerate(folds):
            X_train = df.iloc[train_idx][feature_columns]
            X_test = df.iloc[test_idx][feature_columns]
            y_train = df.iloc[train_idx][target_column] if target_column else None
            y_test = df.iloc[test_idx][target_column] if target_column else None

            # Pipeline de atributos ajustado SÓ no treino do fold (anti-leakage).
            Xtr, Xte = _fit_transform_fold(pipeline, X_train, X_test, y_train)

            model = clone(template)
            t0 = time.perf_counter()
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
            fit_seconds = time.perf_counter() - t0
            total_seconds += fit_seconds
            fold_metrics.append({"fold": i, "metrics": metrics, "fit_seconds": round(fit_seconds, 4)})
    except Exception as exc:  # noqa: BLE001 - falha de um modelo não derruba o zoo
        return {
            "model_name": name,
            "model_family": _family(model_spec["estimator"]),
            "estimator": model_spec["estimator"],
            "hyperparameters": params,
            "tier": model_spec.get("tier"),
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "fold_metrics": fold_metrics,
        }

    mean, std = _aggregate(fold_metrics)
    return {
        "model_name": name,
        "model_family": _family(model_spec["estimator"]),
        "estimator": model_spec["estimator"],
        "hyperparameters": params,
        "tier": model_spec.get("tier"),
        "status": "ok",
        "random_seed": seed,
        "fold_metrics": fold_metrics,
        "metrics_mean": mean,
        "metrics_std": std,
    }


def _fit_transform_fold(
    pipeline: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: Any
) -> tuple[np.ndarray, np.ndarray]:
    if pipeline is None:
        # Sem pipeline: assume atributos já numéricos.
        return X_train.to_numpy(dtype=float), X_test.to_numpy(dtype=float)
    pipe = clone(pipeline)
    Xtr = pipe.fit_transform(X_train, y_train)
    Xte = pipe.transform(X_test)
    return Xtr, Xte


# ---------------------------------------------------------------------------
# Métricas por tarefa
# ---------------------------------------------------------------------------

def _classification_metrics(model: Any, X_test: Any, y_true: pd.Series) -> dict[str, float]:
    y_pred = model.predict(X_test)
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    classes = list(getattr(model, "classes_", []))
    proba, scores = _probabilities(model, X_test)
    try:
        if len(classes) == 2:
            y_bin = (np.asarray(y_true) == classes[1]).astype(int)
            s = proba[:, 1] if proba is not None else scores
            if s is not None:
                metrics["roc_auc"] = float(roc_auc_score(y_bin, s))
                metrics["average_precision"] = float(average_precision_score(y_bin, s))
        elif proba is not None:
            metrics["roc_auc"] = float(
                roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
            )
    except Exception:  # noqa: BLE001 - métrica de score é best-effort
        pass
    return metrics


def _regression_metrics(model: Any, X_test: Any, y_true: pd.Series) -> dict[str, float]:
    y_pred = model.predict(X_test)
    y_true = np.asarray(y_true, dtype=float)
    rmse = float(root_mean_squared_error(y_true, y_pred))
    metrics: dict[str, float] = {
        "rmse": rmse,
        "mse": float(rmse**2),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
    if np.all(y_true != 0):  # MAPE só faz sentido sem zeros no alvo
        metrics["mape"] = float(mean_absolute_percentage_error(y_true, y_pred))
    return metrics


def _anomaly_metrics(model: Any, X_test: Any, y_true: pd.Series | None) -> dict[str, float]:
    score = _anomaly_score(model, X_test)
    metrics: dict[str, float] = {}
    if y_true is None or score is None:
        return metrics  # sem rótulos não há métrica supervisionada (declarar limitação)
    arr = np.asarray(y_true)
    positive = _anomaly_positive_label(arr)
    y_bin = (arr == positive).astype(int)
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_bin, score))
        metrics["average_precision"] = float(average_precision_score(y_bin, score))
    except Exception:  # noqa: BLE001
        pass
    return metrics


def _probabilities(model: Any, X_test: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
    if hasattr(model, "predict_proba"):
        try:
            return np.asarray(model.predict_proba(X_test)), None
        except Exception:  # noqa: BLE001
            pass
    if hasattr(model, "decision_function"):
        try:
            return None, np.asarray(model.decision_function(X_test))
        except Exception:  # noqa: BLE001
            pass
    return None, None


def _anomaly_score(model: Any, X_test: Any) -> np.ndarray | None:
    """Score em que valores MAIORES indicam mais anômalo."""
    if hasattr(model, "score_samples"):
        return -np.asarray(model.score_samples(X_test))
    if hasattr(model, "decision_function"):
        return -np.asarray(model.decision_function(X_test))
    return None


def _anomaly_positive_label(y: np.ndarray) -> Any:
    """Classe positiva (anômala): 1/True quando presente, senão a minoritária."""
    values, counts = np.unique(y, return_counts=True)
    value_set = set(values.tolist())
    if value_set <= {0, 1}:
        return 1
    if value_set <= {True, False}:
        return True
    return values[int(np.argmin(counts))]


# ---------------------------------------------------------------------------
# Catálogo e instanciação
# ---------------------------------------------------------------------------

def _load_zoo(context: dict[str, Any]) -> dict[str, Any]:
    zoo = context.get("model_zoo")
    if zoo is not None:
        return zoo
    path = Path(context.get("model_zoo_path") or _DEFAULT_ZOO_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Catálogo de modelos não encontrado em '{path}'.")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _select_models(
    zoo: dict[str, Any], task_type: str, context: dict[str, Any], warnings: list[str]
) -> list[tuple[str, dict[str, Any]]]:
    catalog: dict[str, Any] = zoo.get(task_type) or {}
    include = context.get("include")
    exclude = set(context.get("exclude") or [])
    tiers = set(context.get("tiers") or _DEFAULT_TIERS)

    selected: list[tuple[str, dict[str, Any]]] = []
    if include:
        # Seleção explícita do usuário: ignora filtro de tier, avisa nomes ausentes.
        for name in include:
            if name in catalog and name not in exclude:
                selected.append((name, catalog[name]))
            elif name not in catalog:
                warnings.append(
                    f"Modelo '{name}' do 'include' não está no catálogo de '{task_type}'; ignorado."
                )
    else:
        for name, spec in catalog.items():
            if name in exclude:
                continue
            if spec.get("tier", "minimum") in tiers:
                selected.append((name, spec))
    return selected


def _instantiate(
    estimator_path: str, model_spec: dict[str, Any], seed: int
) -> tuple[Any, dict[str, Any]]:
    module_name, _, class_name = estimator_path.partition(":")
    module = importlib.import_module(module_name)
    klass = getattr(module, class_name)

    params = dict(model_spec.get("default_params") or {})
    sig = inspect.signature(klass)
    # Injeta a seed para reprodutibilidade quando o estimador a aceita.
    if "random_state" in sig.parameters and "random_state" not in params:
        params["random_state"] = seed
    # LocalOutlierFactor precisa de novelty=True para pontuar dados de teste.
    if class_name == "LocalOutlierFactor" and "novelty" not in params:
        params["novelty"] = True
    return klass(**params), params


def _family(estimator_path: str) -> str:
    """Família do modelo a partir do módulo do estimador (ex.: 'sklearn.ensemble')."""
    return estimator_path.split(":", 1)[0]


# ---------------------------------------------------------------------------
# Agregação e ranking
# ---------------------------------------------------------------------------

def _aggregate(fold_metrics: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
    keys: set[str] = set()
    for fm in fold_metrics:
        keys.update(fm["metrics"].keys())
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    for key in sorted(keys):
        vals = [
            fm["metrics"][key]
            for fm in fold_metrics
            if fm["metrics"].get(key) is not None
        ]
        if vals:
            mean[key] = round(float(np.mean(vals)), 6)
            std[key] = round(float(np.std(vals)), 6)
    return mean, std


def _rank(results: list[dict[str, Any]], primary_metric: str | None) -> list[dict[str, Any]]:
    if not primary_metric:
        return []
    ranked = [
        {
            "model_name": r["model_name"],
            "primary_metric_mean": r["metrics_mean"][primary_metric],
            "primary_metric_std": r["metrics_std"].get(primary_metric),
        }
        for r in results
        if r["status"] == "ok" and primary_metric in r.get("metrics_mean", {})
    ]
    ascending = primary_metric in _LOWER_IS_BETTER
    ranked.sort(key=lambda r: r["primary_metric_mean"], reverse=not ascending)
    for pos, r in enumerate(ranked, start=1):
        r["rank"] = pos
    return ranked


# ---------------------------------------------------------------------------
# Ingestão e helpers
# ---------------------------------------------------------------------------

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
        f"source_type '{source_type}' não suportado pelo treinamento "
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
        f"Model zoo executou {len(ok)}/{output['n_models']} modelo(s) em "
        f"{output['n_folds']} fold(s) para tarefa '{output['task_type']}' "
        f"(seed {output['random_seed']}).",
    ]
    if output["ranking"]:
        best = output["ranking"][0]
        lines.append(
            f"Melhor por '{output['primary_metric']}': {best['model_name']} "
            f"({best['primary_metric_mean']} ± {best['primary_metric_std']})."
        )
    if output["skipped_models"]:
        names = ", ".join(s["model_name"] for s in output["skipped_models"])
        lines.append(f"Modelos pulados: {names}.")
    lines.append("Cada execução registra fold, seed, hiperparâmetros e métricas.")
    if output["warnings"]:
        lines.append("Avisos: " + "; ".join(output["warnings"]))
    return " ".join(lines)
