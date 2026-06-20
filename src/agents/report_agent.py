"""Agente Relator e Auditor.

Monta o relatório técnico final em Markdown (RF14) a partir das saídas dos demais
agentes, um model card simplificado, a explicabilidade do modelo recomendado
(importância de atributos / permutation importance — RF12) e uma trilha de
auditoria das decisões agentivas.

Cuidado principal: tornar a decisão revisável por um humano. O relatório explicita
metodologia, critério de seleção, comparação entre modelos, e — conforme RNF05 —
limitações, riscos de leakage, vieses e restrições do dataset.

Todas as seções são opcionais: o agente preenche o que estiver disponível no
contexto e sinaliza o que faltar, permitindo gerar o relatório de forma incremental.
"""

from __future__ import annotations

import datetime
import math
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np

from src.agents.base import AgentResult, BaseAgent
from src.agents.trainer_agent import _fit_transform_fold, _instantiate

_NA = "_não disponível_"
_DEFAULT_TOP_K = 15


class ReportAgent(BaseAgent):
    """Agente 10 — Relator e Auditor.

    Entradas esperadas em ``context`` (todas opcionais; quanto mais, mais completo):
        - ``problem`` / ``profile`` (ou ``data_profile``) / ``cleaning`` /
          ``features`` (ou ``feature_engineering``) / ``split`` /
          ``training`` (ou ``model_results``) / ``optimization`` / ``autoencoder`` /
          ``evaluation``: saídas (``AgentResult.output``) dos respectivos agentes.
        - ``experiment_name`` (str), ``random_seed`` (int).
        - ``agent_events`` (list): eventos persistidos para a trilha de auditoria
          (cada um com ``agent_name``, ``event_type``, ``rationale``).
        - Para a explicabilidade do modelo recomendado: ``dataframe``, ``folds``,
          ``pipeline``, ``feature_columns``, ``target_column``, ``top_k_features``.
        - ``report_path`` (str): se fornecido, grava o Markdown nesse caminho.

    Saída em ``AgentResult.output``:
        - ``report_markdown`` (str), ``report_json`` (dict estruturado),
          ``model_card`` (dict), ``explainability`` (dict | None),
          ``audit_log`` (list), ``limitations`` (list), ``risks`` (list),
          ``warnings`` (list).

    O Markdown também é exposto em ``AgentResult.report_markdown``.
    """

    name = "Agente Relator e Auditor"
    event_type = "final_report"

    def run(self, context: dict[str, Any]) -> AgentResult:
        warnings: list[str] = []

        sections = _collect_sections(context)
        seed = context.get("random_seed") or (sections["problem"] or {}).get("random_seed")
        experiment_name = context.get("experiment_name", "experimento")

        # Explicabilidade do modelo recomendado (RF12), quando há dados suficientes.
        explainability = _explainability(context, sections, warnings)

        # Riscos / limitações agregados das várias etapas (RNF05).
        risks, limitations = _gather_risks_and_limitations(sections)

        model_card = _build_model_card(sections, experiment_name, seed)
        audit_log = _build_audit_log(context, sections)

        report_markdown = _build_markdown(
            experiment_name=experiment_name,
            sections=sections,
            explainability=explainability,
            risks=risks,
            limitations=limitations,
            audit_log=audit_log,
            model_card=model_card,
            seed=seed,
        )

        report_json = {
            "experiment_name": experiment_name,
            "task": sections["problem"],
            "best_model": (sections["evaluation"] or {}).get("best_model"),
            "selection_reason": (sections["evaluation"] or {}).get("selection_reason"),
            "ranking": (sections["evaluation"] or {}).get("ranking"),
            "autoencoder_verdict": (sections["autoencoder"] or {}).get("verdict"),
            "explainability": explainability,
            "risks": risks,
            "limitations": limitations,
        }

        # Grava em disco, se solicitado.
        report_path = context.get("report_path")
        if report_path:
            try:
                with open(report_path, "w", encoding="utf-8") as fh:
                    fh.write(report_markdown)
            except OSError as exc:
                warnings.append(f"Não foi possível gravar o relatório em '{report_path}': {exc}.")

        output: dict[str, Any] = {
            "report_markdown": report_markdown,
            "report_json": _to_native(report_json),
            "model_card": _to_native(model_card),
            "explainability": _to_native(explainability),
            "audit_log": audit_log,
            "limitations": limitations,
            "risks": risks,
            "warnings": warnings,
        }

        rationale = (
            f"Relatório técnico de '{experiment_name}' gerado com "
            f"{sum(1 for v in sections.values() if v)} etapa(s) disponível(is); "
            f"modelo recomendado: {report_json['best_model'] or _NA}. "
            "Inclui metodologia, resultados, explicabilidade, riscos e trilha de auditoria."
        )
        result = AgentResult(output=output, rationale=rationale, warnings=warnings)
        result.report_markdown = report_markdown
        return result


# ---------------------------------------------------------------------------
# Coleta das seções
# ---------------------------------------------------------------------------

def _collect_sections(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "problem": context.get("problem"),
        "profile": context.get("profile") or context.get("data_profile"),
        "cleaning": context.get("cleaning"),
        "features": context.get("features") or context.get("feature_engineering"),
        "split": context.get("split"),
        "training": context.get("training")
        or ({"results": context["model_results"]} if context.get("model_results") else None),
        "optimization": context.get("optimization"),
        "autoencoder": context.get("autoencoder"),
        "evaluation": context.get("evaluation"),
    }


# ---------------------------------------------------------------------------
# Explicabilidade (RF12)
# ---------------------------------------------------------------------------

def _explainability(
    context: dict[str, Any], sections: dict[str, Any], warnings: list[str]
) -> dict[str, Any] | None:
    evaluation = sections["evaluation"] or {}
    training = sections["training"] or {}
    best_name = evaluation.get("best_model")
    df = context.get("dataframe")
    folds = context.get("folds")
    if not best_name or df is None or not folds:
        return None

    results = training.get("results") or context.get("model_results") or []
    res = next((r for r in results if r.get("model_name") == best_name), None)
    if not res or not res.get("estimator"):
        return None

    target_column = context.get("target_column") or (sections["problem"] or {}).get(
        "target_column"
    )
    task_type = (sections["problem"] or {}).get("task_type")
    if not target_column:
        return None
    feature_columns = context.get("feature_columns") or [
        c for c in df.columns if c != target_column
    ]
    pipeline = context.get("pipeline")
    seed = int(context.get("random_seed", 42))
    top_k = int(context.get("top_k_features", _DEFAULT_TOP_K))

    # Usa o primeiro fold: treina no treino e mede importância no teste (sem leakage).
    tr, te = np.asarray(folds[0][0], dtype=int), np.asarray(folds[0][1], dtype=int)
    X_train, X_test = df.iloc[tr][feature_columns], df.iloc[te][feature_columns]
    y_train, y_test = df.iloc[tr][target_column], df.iloc[te][target_column]

    try:
        names = _feature_names(pipeline, X_train, feature_columns)
        Xtr, Xte = _fit_transform_fold(pipeline, X_train, X_test, y_train)
        template, _ = _instantiate(res["estimator"], {"default_params": res.get("hyperparameters") or {}}, seed)
        model = template
        if task_type == "anomaly":
            model.fit(Xtr)
        else:
            model.fit(Xtr, y_train)
        importances, method = _model_importances(model, Xte, y_test, names, seed, task_type)
    except Exception as exc:  # noqa: BLE001 - explicabilidade é best-effort
        warnings.append(f"Explicabilidade indisponível: {type(exc).__name__}.")
        return None

    top = sorted(importances, key=lambda d: abs(d["importance"]), reverse=True)[:top_k]
    return {"model_name": best_name, "method": method, "top_features": top}


def _feature_names(pipeline: Any, X_train: Any, feature_columns: list[str]) -> list[str]:
    if pipeline is None:
        return list(feature_columns)
    from sklearn.base import clone

    pipe = clone(pipeline).fit(X_train)
    try:
        return [str(n) for n in pipe.get_feature_names_out()]
    except Exception:  # noqa: BLE001
        return [str(n) for n in pipe.named_steps["preprocessor"].get_feature_names_out()]


def _model_importances(
    model: Any, X_test: Any, y_test: Any, names: list[str], seed: int, task_type: str | None
) -> tuple[list[dict[str, Any]], str]:
    """Importâncias nativas (coef_/feature_importances_) ou permutation importance."""
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        method = "feature_importances"
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        values = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
        method = "coef_abs"
    else:
        from sklearn.inspection import permutation_importance

        scoring = None  # usa o score padrão do estimador
        result = permutation_importance(
            model, X_test, y_test, n_repeats=5, random_state=seed, scoring=scoring
        )
        values = np.asarray(result.importances_mean, dtype=float)
        method = "permutation_importance"

    n = min(len(names), len(values))
    return (
        [{"feature": names[i], "importance": round(float(values[i]), 6)} for i in range(n)],
        method,
    )


# ---------------------------------------------------------------------------
# Riscos, limitações e model card
# ---------------------------------------------------------------------------

def _gather_risks_and_limitations(sections: dict[str, Any]) -> tuple[list[str], list[str]]:
    risks: list[str] = []
    limitations: list[str] = []

    features = sections["features"] or {}
    for note in features.get("leakage_notes", []):
        risks.append(note)

    profile = sections["profile"] or {}
    target = profile.get("target") or {}
    if target.get("imbalance_ratio") and target["imbalance_ratio"] >= 10:
        risks.append(
            f"Alvo desbalanceado (razão {target['imbalance_ratio']}): métricas como "
            "acurácia podem enganar; priorize PR-AUC/recall."
        )

    evaluation = sections["evaluation"] or {}
    if evaluation.get("significance_note"):
        limitations.append(evaluation["significance_note"])
    if evaluation.get("meets_success_threshold") is False:
        limitations.append(
            "O modelo recomendado NÃO atinge o critério de sucesso definido."
        )

    split = sections["split"] or {}
    n_samples = split.get("n_samples")
    if n_samples is not None and n_samples < 200:
        limitations.append(
            f"Base pequena ({n_samples} amostras): resultados instáveis; evite conclusões absolutas."
        )

    # Avisos relevantes das etapas viram limitações documentadas.
    for key in ("cleaning", "profile", "split"):
        for w in (sections[key] or {}).get("warnings", []) or []:
            limitations.append(f"[{key}] {w}")

    return _dedupe(risks), _dedupe(limitations)


def _build_model_card(
    sections: dict[str, Any], experiment_name: str, seed: Any
) -> dict[str, Any]:
    problem = sections["problem"] or {}
    evaluation = sections["evaluation"] or {}
    best_name = evaluation.get("best_model")
    best_entry = next(
        (e for e in evaluation.get("ranking", []) if e["model_name"] == best_name), None
    )
    performance: dict[str, Any] = {}
    if best_entry:
        performance = {
            "primary_metric": evaluation.get("primary_metric"),
            "mean": best_entry.get("primary_mean"),
            "std": best_entry.get("primary_std"),
            "ci": [best_entry.get("ci_low"), best_entry.get("ci_high")],
        }
    return {
        "experiment": experiment_name,
        "date": datetime.date.today().isoformat(),
        "model": best_name,
        "task_type": problem.get("task_type"),
        "target_column": problem.get("target_column"),
        "primary_metric": problem.get("primary_metric") or evaluation.get("primary_metric"),
        "performance": performance,
        "selection_criterion": evaluation.get("selection_rule"),
        "random_seed": seed,
        "intended_use": "Apoio à decisão; requer revisão humana antes de uso em produção.",
        "meets_success_threshold": evaluation.get("meets_success_threshold"),
    }


def _build_audit_log(
    context: dict[str, Any], sections: dict[str, Any]
) -> list[dict[str, Any]]:
    events = context.get("agent_events")
    if events:
        return [
            {
                "agent_name": e.get("agent_name"),
                "event_type": e.get("event_type"),
                "rationale": e.get("rationale"),
            }
            for e in events
        ]
    # Sem eventos persistidos: registra ao menos quais etapas estavam presentes.
    present = [name for name, value in sections.items() if value]
    return [{"stage": name, "status": "presente"} for name in present]


# ---------------------------------------------------------------------------
# Montagem do Markdown
# ---------------------------------------------------------------------------

def _build_markdown(
    *,
    experiment_name: str,
    sections: dict[str, Any],
    explainability: dict[str, Any] | None,
    risks: list[str],
    limitations: list[str],
    audit_log: list[dict[str, Any]],
    model_card: dict[str, Any],
    seed: Any,
) -> str:
    problem = sections["problem"] or {}
    profile = sections["profile"] or {}
    cleaning = sections["cleaning"] or {}
    features = sections["features"] or {}
    split = sections["split"] or {}
    training = sections["training"] or {}
    autoencoder = sections["autoencoder"] or {}
    evaluation = sections["evaluation"] or {}

    L: list[str] = [f"# Relatório Técnico — {experiment_name}", ""]

    # 1. Resumo executivo
    L += ["## 1. Resumo executivo", ""]
    best = evaluation.get("best_model")
    if best:
        L.append(
            f"Tarefa de **{problem.get('task_type', _NA)}** sobre o alvo "
            f"`{problem.get('target_column', _NA)}`. Modelo recomendado: **{best}** "
            f"(critério '{evaluation.get('selection_rule', _NA)}')."
        )
    else:
        L.append("Relatório parcial: avaliação de modelos ainda não disponível.")
    L.append("")

    # 2. Definição da tarefa
    L += ["## 2. Definição da tarefa", ""]
    if problem:
        L += [
            f"- **Variável-alvo:** `{problem.get('target_column', _NA)}`",
            f"- **Tipo de tarefa:** {problem.get('task_type', _NA)}",
            f"- **Métrica primária:** {problem.get('primary_metric', _NA)}",
            f"- **Critério de sucesso:** {problem.get('success_threshold', _NA)}",
            f"- **Estratégia de validação:** {problem.get('split_strategy', _NA)}",
        ]
    else:
        L.append(_NA)
    L.append("")

    # 3. Perfil dos dados
    L += ["## 3. Perfil dos dados", ""]
    if profile:
        dup = profile.get("duplicates", {})
        L += [
            f"- Linhas × colunas: **{profile.get('n_rows', _NA)} × {profile.get('n_cols', _NA)}**",
            f"- Duplicatas exatas: {dup.get('n_duplicate_rows', _NA)} "
            f"({dup.get('pct_duplicate_rows', _NA)}%)",
        ]
        target = profile.get("target") or {}
        if target.get("kind") == "classification":
            L.append(
                f"- Alvo: {target.get('n_classes', _NA)} classes, "
                f"razão de desbalanceamento {target.get('imbalance_ratio', _NA)}"
            )
    else:
        L.append(_NA)
    L.append("")

    # 4. Limpeza e engenharia de atributos
    L += ["## 4. Limpeza e engenharia de atributos", ""]
    if cleaning:
        ops = _count_ops(cleaning.get("actions", []))
        L.append("**Limpeza** — ações: " + (ops or "nenhuma"))
    if features:
        groups = features.get("column_groups", {})
        parts = [f"{len(c)} {g}" for g, c in groups.items() if c]
        L.append(f"**Atributos** — grupos: {', '.join(parts) or _NA}; "
                 f"{features.get('n_output_features', _NA)} feature(s) de saída.")
    if not cleaning and not features:
        L.append(_NA)
    L.append("")

    # 5. Estratégia de validação
    L += ["## 5. Estratégia de validação", ""]
    if split:
        traits = [t for t, on in (
            ("estratificado", split.get("stratified")),
            ("agrupado", split.get("grouped")),
            ("temporal", split.get("temporal")),
        ) if on]
        L.append(
            f"- Estratégia: **{split.get('split_strategy', _NA)}**, "
            f"{split.get('n_splits', _NA)} fold(s) (seed {split.get('random_seed', _NA)})."
        )
        if traits:
            L.append(f"- Modo: {', '.join(traits)}.")
        if split.get("no_contamination_verified"):
            L.append("- Disjunção treino/teste verificada (sem contaminação).")
    else:
        L.append(_NA)
    for note in features.get("leakage_notes", [])[:3]:
        L.append(f"- _Leakage:_ {note}")
    L.append("")

    # 6. Modelos testados
    L += ["## 6. Modelos testados", ""]
    results = training.get("results", [])
    if results:
        for r in results:
            status = "" if r.get("status") == "ok" else f" ({r.get('status')})"
            L.append(f"- `{r.get('model_name')}` — {r.get('model_family', _NA)}{status}")
    else:
        L.append(_NA)
    opt = sections["optimization"] or {}
    if opt.get("ranking"):
        L.append("")
        L.append(f"Otimização (Optuna): melhor = `{opt['ranking'][0]['model_name']}` "
                 f"({opt.get('primary_metric')}={opt['ranking'][0]['best_value']}).")
    L.append("")

    # 7. Resultados por métrica e por fold
    L += ["## 7. Resultados por métrica e por fold", ""]
    ranking = evaluation.get("ranking", [])
    if ranking:
        metric = evaluation.get("primary_metric", "métrica")
        L += [
            f"| Modelo | {metric} (média) | Desvio | IC | Folds | Tempo (s) |",
            "|--------|------------------|--------|----|-------|-----------|",
        ]
        for e in ranking:
            ci = (f"[{e['ci_low']}, {e['ci_high']}]"
                  if e.get("ci_low") is not None else _NA)
            L.append(
                f"| {e['model_name']} | {e['primary_mean']} | {e.get('primary_std', _NA)} | "
                f"{ci} | {e.get('n_folds', _NA)} | {e.get('mean_fit_seconds', _NA)} |"
            )
    else:
        L.append(_NA)
    L.append("")

    # 8. Autoencoder vs baseline
    L += ["## 8. Autoencoder vs. baseline", ""]
    if autoencoder:
        L.append(f"- Aplicação: **{autoencoder.get('use_case', _NA)}**")
        L.append(f"- {autoencoder.get('verdict', _NA)}")
    else:
        L.append("Autoencoder não utilizado neste experimento.")
    L.append("")

    # 9. Modelo recomendado
    L += ["## 9. Modelo recomendado", ""]
    if best:
        L.append(f"**{best}** — {evaluation.get('selection_reason', _NA)}")
        meets = evaluation.get("meets_success_threshold")
        if meets is not None:
            L.append(f"- Critério de sucesso: {'atingido' if meets else 'NÃO atingido'}.")
        if evaluation.get("significance_note"):
            L.append(f"- {evaluation['significance_note']}")
        if explainability:
            L.append(f"- **Explicabilidade** ({explainability['method']}), principais atributos:")
            for f in explainability["top_features"][:10]:
                L.append(f"  - `{f['feature']}`: {f['importance']}")
    else:
        L.append(_NA)
    L.append("")

    # 10. Riscos, vieses e restrições de uso
    L += ["## 10. Riscos, vieses e restrições de uso", ""]
    if risks:
        L += [f"- ⚠️ {r}" for r in risks]
    if limitations:
        L += [f"- {x}" for x in limitations]
    if not risks and not limitations:
        L.append("Nenhum risco ou limitação crítica registrada automaticamente.")
    L.append("")

    # 11. Próximos passos
    L += ["## 11. Próximos passos", ""]
    L += [f"- {s}" for s in _next_steps(evaluation, split)]
    L.append("")

    # Appendix: model card + auditoria
    L += ["---", "", "## Model card (resumido)", ""]
    for k, v in model_card.items():
        L.append(f"- **{k}:** {v}")
    L += ["", "## Trilha de auditoria", ""]
    for entry in audit_log:
        if "rationale" in entry:
            L.append(f"- **{entry.get('agent_name')}** ({entry.get('event_type')}): "
                     f"{entry.get('rationale')}")
        else:
            L.append(f"- {entry.get('stage')}: {entry.get('status')}")

    L += ["", "---", f"*Reprodutibilidade:* seed `{seed}`; versões: {_versions()}."]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_ops(actions: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for a in actions:
        op = a.get("operation", "?")
        counts[op] = counts.get(op, 0) + 1
    return ", ".join(f"{n}× {op}" for op, n in sorted(counts.items()))


def _next_steps(evaluation: dict[str, Any], split: dict[str, Any]) -> list[str]:
    steps: list[str] = []
    if evaluation.get("meets_success_threshold") is False:
        steps.append("Coletar mais dados ou criar novos atributos: o critério de sucesso não foi atingido.")
    if (split.get("n_samples") or 10**9) < 200:
        steps.append("Ampliar a base: o tamanho atual limita a confiança das estimativas.")
    steps.append("Monitorar drift em produção e definir gatilho de retreinamento.")
    steps.append("Revisar manualmente a decisão e os riscos antes da implantação.")
    return steps


def _versions() -> str:
    libs = []
    for pkg in ("scikit-learn", "pandas", "numpy", "torch"):
        try:
            libs.append(f"{pkg} {version(pkg)}")
        except PackageNotFoundError:
            continue
    return ", ".join(libs) or _NA


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


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
