"""Agente de Avaliação e Seleção.

Compara os modelos treinados pelo Model Zoo a partir das métricas por fold,
produzindo um ranking com média, desvio, intervalo de confiança, intervalo
[mín, máx] e tempo de execução (RF11). Recomenda o modelo final por um critério
explícito e determinístico.

Cuidado principal: selecionar por critério claro, nunca por cherry-picking. O
agente aplica uma regra documentada (``best_mean`` ou ``one_se``), desempata de
forma estável (menor desvio, depois menor tempo, depois nome) e **sinaliza quando
a diferença para o segundo colocado não é estatisticamente significativa** (ICs
sobrepostos) — alertando contra escolhas baseadas em ruído.

Quando ``dataframe``/``folds``/``pipeline`` são fornecidos, o agente também
reavalia o modelo selecionado para gerar diagnósticos (matriz de confusão na
classificação; resíduos na regressão), sem nunca ajustar fora do treino do fold.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import confusion_matrix

from src.agents.base import AgentResult, BaseAgent
from src.agents.trainer_agent import _fit_transform_fold, _instantiate

# ---------------------------------------------------------------------------
# Convenções
# ---------------------------------------------------------------------------

#: Métricas em que valores menores são melhores (define a direção do ranking).
_LOWER_IS_BETTER = {"rmse", "mae", "mse", "mape", "medae", "log_loss"}
_DEFAULT_CONFIDENCE = 0.95
_VALID_RULES = {"best_mean", "one_se"}


class EvaluatorAgent(BaseAgent):
    """Agente 9 — Avaliação e Seleção.

    Entradas esperadas em ``context``:
        - ``model_results`` (list) | ``training`` (dict com ``results``) |
          ``results`` (list): resultados do Model Zoo, cada um com ``model_name``,
          ``status`` e ``fold_metrics`` (lista de {``metrics``, ``fit_seconds``}).
        - ``task_type`` / ``primary_metric`` / ``success_threshold``: da Formulação
          (lidos de ``context['problem']`` se ausentes no topo).

    Sobreposições (opcionais):
        - ``selection_rule``: "best_mean" (padrão) | "one_se" (regra do 1 erro-padrão,
          escolhe o modelo mais eficiente dentro de 1 SE do melhor).
        - ``confidence_level`` (float, padrão 0.95).
        - Para diagnósticos do modelo escolhido: ``dataframe``, ``folds``,
          ``pipeline``, ``feature_columns``, ``target_column``, ``random_seed``.

    Saída em ``AgentResult.output``:
        - ``ranking``: por modelo — média, desvio, IC, [mín, máx], tempo, métricas
          secundárias, ``rank``;
        - ``best_model`` + ``selection_reason`` + ``selection_rule``;
        - ``meets_success_threshold``, ``significance_note``, ``diagnostics``,
          ``warnings``.
    """

    name = "Agente de Avaliação"
    event_type = "model_selection"

    def run(self, context: dict[str, Any]) -> AgentResult:
        warnings: list[str] = []

        problem = context.get("problem") or {}
        task_type = str(context.get("task_type") or problem.get("task_type") or "").strip().lower()
        primary_metric = str(
            context.get("primary_metric") or problem.get("primary_metric") or ""
        ).strip().lower()
        if not primary_metric:
            raise ValueError("'primary_metric' é obrigatório para avaliar.")

        success_threshold = context.get("success_threshold")
        if success_threshold is None:
            success_threshold = problem.get("success_threshold")

        confidence = float(context.get("confidence_level", _DEFAULT_CONFIDENCE))
        rule = str(context.get("selection_rule", "best_mean")).strip().lower()
        if rule not in _VALID_RULES:
            raise ValueError(f"'selection_rule' inválida: '{rule}'. Opções: {sorted(_VALID_RULES)}.")

        results = _extract_results(context)
        lower_is_better = primary_metric in _LOWER_IS_BETTER

        # ------------------------------------------------------------------
        # 1. Agregação por modelo (média, desvio, IC, intervalo, tempo)
        # ------------------------------------------------------------------
        entries: list[dict[str, Any]] = []
        for res in results:
            if res.get("status") and res["status"] != "ok":
                warnings.append(f"Modelo '{res.get('model_name')}' ignorado (status != ok).")
                continue
            entry = _aggregate_model(res, primary_metric, confidence)
            if entry is None:
                warnings.append(
                    f"Modelo '{res.get('model_name')}' sem métrica primária "
                    f"'{primary_metric}'; ignorado no ranking."
                )
                continue
            entries.append(entry)

        if not entries:
            raise ValueError(
                f"Nenhum modelo avaliável com a métrica primária '{primary_metric}'."
            )

        # ------------------------------------------------------------------
        # 2. Ranking determinístico (direção + desempate estável)
        # ------------------------------------------------------------------
        entries.sort(key=_sort_key(lower_is_better))
        for pos, e in enumerate(entries, start=1):
            e["rank"] = pos

        # ------------------------------------------------------------------
        # 3. Seleção por critério explícito + nota de significância
        # ------------------------------------------------------------------
        best, reason = _select(entries, rule, lower_is_better)
        significance_note = _significance_note(entries, lower_is_better)
        meets = _meets_threshold(best, success_threshold, lower_is_better)

        # ------------------------------------------------------------------
        # 4. Diagnósticos do modelo escolhido (opcional)
        # ------------------------------------------------------------------
        diagnostics = _diagnostics(context, best, results, task_type, warnings)

        output: dict[str, Any] = {
            "task_type": task_type or None,
            "primary_metric": primary_metric,
            "direction": "minimize" if lower_is_better else "maximize",
            "selection_rule": rule,
            "confidence_level": confidence,
            "n_models": len(entries),
            "ranking": entries,
            "best_model": best["model_name"],
            "best_primary_mean": best["primary_mean"],
            "selection_reason": reason,
            "meets_success_threshold": meets,
            "success_threshold": success_threshold,
            "significance_note": significance_note,
            "diagnostics": diagnostics,
            "warnings": warnings,
        }
        output = _to_native(output)

        rationale = _build_rationale(output)
        return AgentResult(output=output, rationale=rationale, warnings=warnings)


# ---------------------------------------------------------------------------
# Agregação
# ---------------------------------------------------------------------------

def _extract_results(context: dict[str, Any]) -> list[dict[str, Any]]:
    results = context.get("model_results")
    if results is None:
        training = context.get("training") or {}
        results = context.get("results") or training.get("results")
    if not results:
        raise ValueError(
            "Forneça 'model_results' (ou 'training' com 'results') do Agente de Model Zoo."
        )
    return list(results)


def _aggregate_model(
    res: dict[str, Any], primary_metric: str, confidence: float
) -> dict[str, Any] | None:
    fold_metrics = res.get("fold_metrics") or []
    primary_values = [
        fm["metrics"][primary_metric]
        for fm in fold_metrics
        if fm.get("metrics", {}).get(primary_metric) is not None
    ]
    # Fallback: usa metrics_mean quando não há detalhe por fold.
    if not primary_values:
        mean = (res.get("metrics_mean") or {}).get(primary_metric)
        if mean is None:
            return None
        return {
            "model_name": res.get("model_name"),
            "model_family": res.get("model_family"),
            "primary_mean": round(float(mean), 6),
            "primary_std": None,
            "ci_low": None,
            "ci_high": None,
            "primary_min": None,
            "primary_max": None,
            "n_folds": 0,
            "mean_fit_seconds": None,
            "total_fit_seconds": None,
            "secondary_means": dict(res.get("metrics_mean") or {}),
        }

    values = np.asarray(primary_values, dtype=float)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    ci_low, ci_high = _confidence_interval(values, confidence)
    fit_times = [fm.get("fit_seconds") for fm in fold_metrics if fm.get("fit_seconds") is not None]

    return {
        "model_name": res.get("model_name"),
        "model_family": res.get("model_family"),
        "primary_mean": round(mean, 6),
        "primary_std": round(std, 6),
        "ci_low": round(ci_low, 6),
        "ci_high": round(ci_high, 6),
        "primary_min": round(float(values.min()), 6),
        "primary_max": round(float(values.max()), 6),
        "n_folds": len(values),
        "mean_fit_seconds": round(float(np.mean(fit_times)), 4) if fit_times else None,
        "total_fit_seconds": round(float(np.sum(fit_times)), 4) if fit_times else None,
        "secondary_means": _secondary_means(fold_metrics, exclude=primary_metric),
    }


def _confidence_interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return mean, mean
    sem = float(stats.sem(values))
    if sem == 0:
        return mean, mean
    margin = sem * float(stats.t.ppf((1 + confidence) / 2, df=n - 1))
    return mean - margin, mean + margin


def _secondary_means(fold_metrics: list[dict[str, Any]], exclude: str) -> dict[str, float]:
    keys: set[str] = set()
    for fm in fold_metrics:
        keys.update(fm.get("metrics", {}).keys())
    keys.discard(exclude)
    means: dict[str, float] = {}
    for key in sorted(keys):
        vals = [
            fm["metrics"][key]
            for fm in fold_metrics
            if fm.get("metrics", {}).get(key) is not None
        ]
        if vals:
            means[key] = round(float(np.mean(vals)), 6)
    return means


# ---------------------------------------------------------------------------
# Ranking, seleção e significância
# ---------------------------------------------------------------------------

def _sort_key(lower_is_better: bool):
    sign = 1 if lower_is_better else -1

    def key(e: dict[str, Any]) -> tuple:
        # Critério primário (direção) e desempates estáveis e justificáveis.
        return (
            sign * e["primary_mean"],
            e["primary_std"] if e["primary_std"] is not None else math.inf,
            e["mean_fit_seconds"] if e["mean_fit_seconds"] is not None else math.inf,
            str(e["model_name"]),
        )

    return key


def _select(
    entries: list[dict[str, Any]], rule: str, lower_is_better: bool
) -> tuple[dict[str, Any], str]:
    best_mean = entries[0]  # já ordenado
    if rule == "best_mean":
        reason = (
            f"Regra 'best_mean': melhor média de '{best_mean['model_name']}' "
            f"({best_mean['primary_mean']}), com desempate por menor desvio e menor tempo."
        )
        return best_mean, reason

    # one_se: candidatos dentro de 1 erro-padrão do melhor; escolhe o mais eficiente.
    se = (best_mean["primary_std"] or 0.0) / math.sqrt(max(best_mean["n_folds"], 1))
    if lower_is_better:
        threshold = best_mean["primary_mean"] + se
        within = [e for e in entries if e["primary_mean"] <= threshold]
    else:
        threshold = best_mean["primary_mean"] - se
        within = [e for e in entries if e["primary_mean"] >= threshold]
    # Mais eficiente (menor tempo médio) entre os estatisticamente comparáveis.
    chosen = min(
        within,
        key=lambda e: (
            e["mean_fit_seconds"] if e["mean_fit_seconds"] is not None else math.inf,
            e["rank"],
        ),
    )
    reason = (
        f"Regra 'one_se': dentro de 1 erro-padrão do melhor ({best_mean['model_name']}, "
        f"{best_mean['primary_mean']}), escolheu-se o modelo mais eficiente "
        f"'{chosen['model_name']}' ({chosen['primary_mean']})."
    )
    return chosen, reason


def _significance_note(entries: list[dict[str, Any]], lower_is_better: bool) -> str:
    if len(entries) < 2:
        return "Apenas um modelo avaliado; sem comparação de significância."
    a, b = entries[0], entries[1]
    if a["ci_low"] is None or b["ci_low"] is None:
        return "Intervalos de confiança indisponíveis (poucos folds) para teste de significância."
    overlap = not (a["ci_high"] < b["ci_low"] or b["ci_high"] < a["ci_low"])
    if overlap:
        return (
            f"Os ICs de '{a['model_name']}' e '{b['model_name']}' se sobrepõem: a diferença "
            "pode não ser significativa — evite cherry-picking; considere custo/simplicidade."
        )
    return (
        f"'{a['model_name']}' supera '{b['model_name']}' com ICs disjuntos: "
        "diferença provavelmente significativa."
    )


def _meets_threshold(
    best: dict[str, Any], success_threshold: Any, lower_is_better: bool
) -> bool | None:
    if success_threshold is None:
        return None
    value = best["primary_mean"]
    threshold = float(success_threshold)
    return value <= threshold if lower_is_better else value >= threshold


# ---------------------------------------------------------------------------
# Diagnósticos do modelo escolhido (opcional)
# ---------------------------------------------------------------------------

def _diagnostics(
    context: dict[str, Any],
    best: dict[str, Any],
    results: list[dict[str, Any]],
    task_type: str,
    warnings: list[str],
) -> dict[str, Any] | None:
    df = context.get("dataframe")
    folds = context.get("folds")
    if df is None or not folds or task_type not in ("classification", "regression"):
        return None

    res = next((r for r in results if r.get("model_name") == best["model_name"]), None)
    if not res or not res.get("estimator"):
        return None

    target_column = context.get("target_column") or (context.get("problem") or {}).get(
        "target_column"
    )
    if not target_column:
        return None
    feature_columns = context.get("feature_columns") or [
        c for c in df.columns if c != target_column
    ]
    pipeline = context.get("pipeline")
    seed = int(context.get("random_seed", 42))
    params = res.get("hyperparameters") or {}

    try:
        template, _ = _instantiate(res["estimator"], {"default_params": params}, seed)
        y_true_all: list[Any] = []
        y_pred_all: list[Any] = []
        for tr, te in folds:
            tr, te = np.asarray(tr, dtype=int), np.asarray(te, dtype=int)
            X_train, X_test = df.iloc[tr][feature_columns], df.iloc[te][feature_columns]
            y_train, y_test = df.iloc[tr][target_column], df.iloc[te][target_column]
            Xtr, Xte = _fit_transform_fold(pipeline, X_train, X_test, y_train)
            model = clone(template)
            model.fit(Xtr, y_train)
            y_true_all.extend(np.asarray(y_test).tolist())
            y_pred_all.extend(np.asarray(model.predict(Xte)).tolist())
    except Exception as exc:  # noqa: BLE001 - diagnóstico é best-effort
        warnings.append(f"Diagnóstico do modelo escolhido falhou: {type(exc).__name__}.")
        return None

    if task_type == "classification":
        labels = sorted(set(y_true_all))
        cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)
        return {
            "type": "confusion_matrix",
            "labels": [_native(label) for label in labels],
            "matrix": cm.tolist(),
            "note": "Matriz somada sobre todos os folds (out-of-fold).",
        }
    residuals = np.asarray(y_true_all, dtype=float) - np.asarray(y_pred_all, dtype=float)
    return {
        "type": "regression_residuals",
        "mean_error": round(float(residuals.mean()), 6),
        "std_error": round(float(residuals.std(ddof=1)) if len(residuals) > 1 else 0.0, 6),
        "max_abs_error": round(float(np.abs(residuals).max()), 6),
        "note": "Resíduos (y - ŷ) agregados sobre todos os folds (out-of-fold).",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _native(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value if isinstance(value, (int, float, bool, str)) else str(value)


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
        f"Avaliados {output['n_models']} modelo(s) por '{output['primary_metric']}' "
        f"(direção '{output['direction']}', regra '{output['selection_rule']}').",
        output["selection_reason"],
    ]
    best = next((e for e in output["ranking"] if e["model_name"] == output["best_model"]), None)
    if best and best["ci_low"] is not None:
        lines.append(
            f"Melhor: {best['model_name']} = {best['primary_mean']} "
            f"(IC {output['confidence_level']:.0%}: [{best['ci_low']}, {best['ci_high']}], "
            f"desvio {best['primary_std']})."
        )
    if output["meets_success_threshold"] is not None:
        atende = "atinge" if output["meets_success_threshold"] else "NÃO atinge"
        lines.append(f"Critério de sucesso ({output['success_threshold']}): {atende}.")
    lines.append(output["significance_note"])
    if output["warnings"]:
        lines.append("Avisos: " + "; ".join(output["warnings"]))
    return " ".join(lines)
