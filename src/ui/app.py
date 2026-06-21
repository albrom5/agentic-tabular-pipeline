"""Interface Streamlit do pipeline agentivo (seção 12 do documento de apoio).

Cobre os requisitos de interface conversando com a **API real** (FastAPI):
cadastro de experimento, upload/seleção da base, escolha da variável-alvo e do
tipo de tarefa, visualização do perfil e dos alertas de qualidade persistidos,
revisão das decisões dos agentes, disparo do pipeline, painel de resultados com
ranking/métricas/recomendação e acesso ao histórico de eventos agentivos
gravado no PostgreSQL (RNF03).

A comunicação é isolada em :class:`~src.ui.api_client.ApiClient` (``API_URL``).
O upload é salvo em ``data/raw/uploads/`` — volume compartilhado com a API no
``docker-compose`` —, de modo que o backend leia a base pelo caminho.

Execução:
    # requer a API no ar (uvicorn src.api.main:app) e DATABASE_URL configurado
    streamlit run src/ui/app.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml

from src.agents.data_profile_agent import DataProfileAgent
from src.ui.api_client import ApiClient, ApiError

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
UPLOAD_DIR = RAW_DIR / "uploads"
MODEL_ZOO_PATH = REPO_ROOT / "configs" / "model_zoo.yaml"

TASK_TYPES = ["classification", "regression", "anomaly"]
SPLIT_STRATEGIES = [
    "holdout", "kfold", "stratified_kfold", "group_kfold", "time_split",
]
METRICS_BY_TASK = {
    "classification": ["macro_f1", "balanced_accuracy", "roc_auc", "f1", "accuracy"],
    "regression": ["rmse", "mae", "r2", "mape"],
    "anomaly": ["roc_auc", "average_precision", "precision_at_k"],
}

_CSS = """
<style>
    .block-container { padding-top: 2.2rem; max-width: 1280px; }
    div[data-testid="stMetric"] {
        background: #1c2330; border: 1px solid #2c3445;
        border-radius: 12px; padding: 14px 16px;
    }
    .badge {
        display:inline-block; padding:2px 10px; border-radius:999px;
        font-size:0.75rem; font-weight:600; margin-left:6px;
    }
    .badge-ok    { background:#0f5132; color:#d1e7dd; }
    .badge-run   { background:#664d03; color:#fff3cd; }
    .badge-draft { background:#41464b; color:#e2e3e5; }
    .badge-fail  { background:#5c1f25; color:#f5c2c7; }
    .pill {
        background:#1c2330; border:1px solid #2c3445; border-radius:10px;
        padding:10px 14px; margin-bottom:8px;
    }
</style>
"""

_STATUS_BADGE = {
    "completed": "badge-ok",
    "running": "badge-run",
    "created": "badge-draft",
    "failed": "badge-fail",
}


# ---------------------------------------------------------------------------
# Infraestrutura
# ---------------------------------------------------------------------------

@st.cache_resource
def _client() -> ApiClient:
    return ApiClient()


def _public_api_url() -> str:
    """URL da API acessível pelo navegador (links que abrem no cliente).

    No docker-compose, ``API_URL`` aponta para o hostname interno da rede
    (``http://api:8000``), que o navegador do usuário não resolve. ``API_PUBLIC_URL``
    fornece o endereço externo (padrão ``http://localhost:8000``).
    """
    return os.environ.get("API_PUBLIC_URL", "http://localhost:8000").rstrip("/")


def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("selected_experiment_id", None)
    ss.setdefault("preview_df", None)
    ss.setdefault("preview_source_uri", None)
    # Tokens de acesso por experimento (capability tokens), escopo da sessão.
    ss.setdefault("tokens", {})


@st.cache_data(show_spinner=False)
def _model_options(task_type: str) -> list[str]:
    """Modelos aplicáveis à tarefa, lidos de configs/model_zoo.yaml."""
    try:
        zoo = yaml.safe_load(MODEL_ZOO_PATH.read_text(encoding="utf-8")) or {}
    except OSError:
        return []
    section = zoo.get(task_type, {}) or {}
    return sorted(section.keys())


def _example_datasets() -> list[str]:
    files = sorted(RAW_DIR.glob("*.csv")) + sorted(RAW_DIR.glob("*.parquet"))
    return [str(p.relative_to(REPO_ROOT)) for p in files]


# ---------------------------------------------------------------------------
# Sidebar — seleção de experimento e ações globais
# ---------------------------------------------------------------------------

def _sidebar(api: ApiClient) -> dict[str, Any] | None:
    with st.sidebar:
        st.markdown("### 🧪 Agentic Tabular Pipeline")
        st.caption("MAQ020 · interface conectada à API")

        online = api.health()
        if online:
            st.markdown(":green[● API online] "
                        f"<small>`{api.base_url}`</small>", unsafe_allow_html=True)
            # O link abre no navegador do usuário, que não resolve o hostname
            # interno da rede Docker (``api``). Usa a URL pública da API.
            st.link_button("🛠️ Painel admin", f"{_public_api_url()}/admin", width="stretch",
                           help="Visão administrativa de todos os experimentos, "
                                "execuções e eventos (estilo Django Admin).")
        else:
            st.markdown(":red[● API offline] "
                        f"<small>`{api.base_url}`</small>", unsafe_allow_html=True)
            st.warning("Inicie a API: `uvicorn src.api.main:app` (e configure `DATABASE_URL`).")
            return None

        st.divider()
        if st.button("➕ Novo experimento", width="stretch"):
            st.session_state.selected_experiment_id = None

        # Acesso por token: sem lista aberta — o usuário informa o token recebido
        # na criação para reabrir um experimento (confidencialidade).
        with st.form("access_form"):
            token_in = st.text_input(
                "Token do experimento", type="password",
                help="Informe o token gerado na criação para reabrir os dados "
                     "daquele experimento.",
            )
            if st.form_submit_button("🔓 Acessar", width="stretch") and token_in.strip():
                try:
                    exp = api.resolve_experiment(token_in.strip())
                except ApiError:
                    st.error("Token inválido — nenhum experimento corresponde.")
                else:
                    st.session_state.tokens[exp["id"]] = token_in.strip()
                    st.session_state.selected_experiment_id = exp["id"]
                    st.toast("Experimento carregado.", icon="🔓")

        eid = st.session_state.selected_experiment_id
        if not eid:
            st.info("Crie um experimento na aba **Experimento** ou informe um "
                    "**token** acima para reabrir um existente.")
            return None

        try:
            experiment = api.get_experiment(eid)
        except ApiError as exc:
            st.error(str(exc))
            return None

        badge = _STATUS_BADGE.get(experiment["status"], "badge-draft")
        st.markdown(
            f"**Status:** <span class='badge {badge}'>{experiment['status']}</span>",
            unsafe_allow_html=True,
        )
        st.caption(f"ID `{experiment['id'][:8]}…` · {experiment.get('created_at', '')}")

        st.divider()
        c1, c2 = st.columns(2)
        if c1.button("🔁 Reexecutar", width="stretch",
                     help="RF15 — reexecução reprodutível com a mesma seed."):
            try:
                api.run_experiment(experiment["id"])
                st.toast("Execução disparada.", icon="▶️")
            except ApiError as exc:
                st.error(str(exc))
        if c2.button("↻ Atualizar", width="stretch"):
            st.rerun()

        report = api.get_report(experiment["id"])
        if report and report.get("content"):
            st.download_button(
                "⬇️ Exportar relatório (.md)", data=report["content"],
                file_name=f"{experiment['name']}_report.md", mime="text/markdown",
                width="stretch",
            )
        return experiment


# ---------------------------------------------------------------------------
# Aba 1 — Cadastro do experimento (RF01/RF02)
# ---------------------------------------------------------------------------

def _tab_experiment(api: ApiClient) -> None:
    st.subheader("Cadastrar experimento")
    st.caption("Defina a tarefa, selecione a base e a variável-alvo (RF01/RF02). "
               "Ao salvar, o experimento é criado e o pipeline disparado na API.")

    col_form, col_data = st.columns([3, 2], gap="large")

    # ---- base de dados ----
    with col_data:
        st.markdown("##### Base de dados")
        source = st.radio(
            "Origem da base", ["Base de exemplo", "Upload", "PostgreSQL"],
            horizontal=True, label_visibility="collapsed",
        )
        source_uri: str | None = None
        if source == "Base de exemplo":
            options = _example_datasets()
            if options:
                source_uri = st.selectbox("Dataset disponível (data/raw)", options)
            else:
                st.warning("Nenhum arquivo em data/raw. Gere a base: "
                           "`python -m scripts.generate_demo_dataset`.")
        elif source == "Upload":
            uploaded = st.file_uploader("Arraste um CSV ou Parquet", type=["csv", "parquet"])
            if uploaded is not None:
                source_uri = _persist_upload(uploaded)
        else:
            st.text_input("URI PostgreSQL", placeholder="postgresql://…/schema.tabela",
                          disabled=True)
            st.caption("Conector PostgreSQL é bônus no MVP (RF02).")

        df = _load_preview(source_uri)
        if df is not None:
            st.session_state.preview_df = df
            st.session_state.preview_source_uri = source_uri
            st.success(f"Base carregada: {df.shape[0]} linhas × {df.shape[1]} colunas.")
            with st.expander("Pré-visualizar amostra"):
                st.dataframe(df.head(15), width="stretch")

    # ---- configuração ----
    df = st.session_state.preview_df
    cols = list(df.columns) if df is not None else []
    with col_form:
        st.markdown("##### Configuração do estudo")
        with st.form("experiment_form"):
            name = st.text_input("Nome do experimento", value="experimento_credit")
            c1, c2 = st.columns(2)
            task_type = c1.selectbox("Tipo de tarefa", TASK_TYPES)
            target = c2.selectbox(
                "Variável-alvo", cols or ["—"],
                index=(len(cols) - 1) if cols else 0,
                disabled=(task_type == "anomaly" or not cols),
                help="Opcional para detecção de anomalias não supervisionada.",
            )

            c3, c4 = st.columns(2)
            primary_metric = c3.selectbox("Métrica primária", METRICS_BY_TASK[task_type])
            success = c4.number_input("Critério de sucesso (≥)", 0.0, 1.0, 0.70, 0.01)
            seed = c3.number_input("Seed", 0, 10_000, 42)
            budget = c4.number_input("Budget de treino (min)", 1, 240, 20)

            c5, c6 = st.columns(2)
            split = c5.selectbox("Estratégia de split", SPLIT_STRATEGIES, index=2)
            n_splits = c6.number_input("Nº de folds", 2, 10, 5)

            models = st.multiselect(
                "Modelos a treinar (model zoo)", _model_options(task_type),
                default=_model_options(task_type)[:5],
            )
            id_col = st.selectbox("Coluna de id (opcional)", ["—"] + cols)
            ae_enabled = st.checkbox("Treinar autoencoder tabular (RF10)", value=True)

            submitted = st.form_submit_button(
                "🚀 Criar e executar pipeline", type="primary", width="stretch"
            )

        if submitted:
            _submit_experiment(
                api, name=name, task_type=task_type, target=target if cols else None,
                primary_metric=primary_metric, success=success, seed=int(seed),
                budget=int(budget), split=split, n_splits=int(n_splits),
                models=models, id_col=None if id_col == "—" else id_col,
                ae_enabled=ae_enabled,
            )


def _submit_experiment(api: ApiClient, **f: Any) -> None:
    source_uri = st.session_state.preview_source_uri
    if not source_uri:
        st.error("Selecione ou faça upload de uma base antes de executar.")
        return
    config = _build_config(source_uri=source_uri, **f)
    try:
        created = api.create_experiment(
            name=f["name"], task_type=f["task_type"], target_column=f["target"],
            primary_metric=f["primary_metric"], config=config,
        )
        eid = created["experiment_id"]
        token = created.get("access_token", "")
        api.run_experiment(eid)
    except ApiError as exc:
        st.error(str(exc))
        return
    st.session_state.tokens[eid] = token
    st.session_state.selected_experiment_id = eid
    st.success(f"Experimento criado (`{eid[:8]}…`) e pipeline em execução. "
               "Acompanhe o status na barra lateral e veja **Resultados** ao concluir.")
    if token:
        st.warning(
            "🔑 **Guarde o token de acesso deste experimento — ele é exibido "
            "apenas uma vez.** Sem ele, não será possível reabrir estes dados.",
        )
        st.code(token, language=None)
    st.toast("Experimento criado e execução disparada.", icon="🚀")


def _build_config(*, source_uri: str, name: str, task_type: str, target: str | None,
                  primary_metric: str, success: float, seed: int, budget: int,
                  split: str, n_splits: int, models: list[str], id_col: str | None,
                  ae_enabled: bool) -> dict[str, Any]:
    source_type = "parquet" if source_uri.endswith(".parquet") else "csv"
    return {
        "experiment": {
            "name": name, "task_type": task_type, "target_column": target,
            "primary_metric": primary_metric, "random_seed": seed,
            "success_threshold": success,
        },
        "data": {
            "source_type": source_type, "source_uri": source_uri,
            "id_column": id_col, "time_column": None, "group_column": None,
        },
        "validation": {"split_strategy": split, "n_splits": n_splits, "test_size": 0.2},
        "preprocessing": {
            "numeric_imputation": "median", "categorical_imputation": "most_frequent",
            "categorical_encoding": "one_hot", "scaling": "standard",
            "rare_category_threshold": 0.01,
        },
        "models": {"include": models, "time_budget_seconds": budget * 60},
        "autoencoder": {
            "enabled": ae_enabled, "latent_dim": 8, "epochs": 30,
            "batch_size": 64, "use_case": "latent_features",
        },
        "storage": {"postgres_uri_env": "DATABASE_URL", "artifact_dir": "artifacts/"},
    }


def _persist_upload(uploaded: Any) -> str | None:
    """Salva o upload no volume compartilhado e devolve o caminho relativo."""
    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        dest = UPLOAD_DIR / uploaded.name
        dest.write_bytes(uploaded.getbuffer())
        return str(dest.relative_to(REPO_ROOT))
    except OSError as exc:  # pragma: no cover - feedback de UI
        st.error(f"Falha ao salvar o upload: {exc}")
        return None


def _load_preview(source_uri: str | None) -> pd.DataFrame | None:
    if not source_uri:
        return None
    path = REPO_ROOT / source_uri
    try:
        if source_uri.endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - feedback de UI
        st.error(f"Falha ao ler a base: {exc}")
        return None


# ---------------------------------------------------------------------------
# Aba 2 — Perfil e qualidade (RF03/RF04)
# ---------------------------------------------------------------------------

def _tab_profile(api: ApiClient, experiment: dict[str, Any] | None) -> None:
    st.subheader("Perfil dos dados e alertas de qualidade")
    st.caption("Perfil (RF03) e relatório de qualidade (RF04/RF06) persistidos pela API. "
               "Antes da execução, mostra uma pré-visualização local da base.")

    profile = None
    quality = None
    if experiment is not None:
        dataset = api.get_profile(experiment["id"])
        if dataset:
            profile = dataset.get("profile_json")
            quality = dataset.get("quality_report_json")

    if profile is None:
        df = st.session_state.preview_df
        if df is None:
            st.info("Selecione uma base na aba **Experimento** ou execute o pipeline.")
            return
        st.caption("⚠️ Pré-visualização local (Agente de Perfilamento) — ainda não persistida.")
        profile = _local_profile(df)

    _render_profile(profile)
    if quality:
        _render_quality(quality)


@st.cache_data(show_spinner=False)
def _local_profile(df: pd.DataFrame) -> dict[str, Any]:
    ctx: dict[str, Any] = {"dataframe": df}
    if "default" in df.columns:
        ctx["target_column"] = "default"
        ctx["task_type"] = "classification"
    return DataProfileAgent(event_sink=None).run(ctx).output


def _render_profile(profile: dict[str, Any]) -> None:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Linhas", f"{profile.get('n_rows', 0):,}".replace(",", "."))
    m2.metric("Colunas", profile.get("n_cols", len(profile.get("columns", []))))
    dups = (profile.get("duplicates") or {}).get("n_duplicate_rows", "—")
    m3.metric("Duplicatas", dups)
    total_missing = sum(c.get("n_missing", 0) for c in profile.get("columns", []))
    m4.metric("Células faltantes", f"{total_missing:,}".replace(",", "."))
    if profile.get("content_hash"):
        st.caption(f"Hash de conteúdo: `{profile['content_hash'][:24]}…` (RNF01).")

    if profile.get("columns"):
        st.markdown("##### Perfil por coluna")
        rows = [
            {
                "coluna": c.get("name"),
                "tipo inferido": c.get("inferred_type"),
                "faltantes %": c.get("pct_missing"),
                "únicos": c.get("n_unique"),
                "papel": c.get("role") or ("alvo" if c.get("is_target") else ""),
            }
            for c in profile["columns"]
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    target = profile.get("target")
    if target and target.get("kind") == "classification":
        st.markdown("##### Distribuição do alvo")
        dist = pd.DataFrame(target["class_distribution"]).set_index("label")["count"]
        st.bar_chart(dist)
        st.caption(f"Razão de desbalanceamento: **{target.get('imbalance_ratio')}**.")


def _render_quality(q: dict[str, Any]) -> None:
    st.divider()
    st.markdown("##### Alertas de qualidade (persistidos)")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Duplicatas", q.get("n_duplicate_rows", 0))
    a2.metric("Colunas com faltantes", len(q.get("missing", {})))
    a3.metric("Colunas constantes", len(q.get("constant_columns", [])))
    a4.metric("Faltantes no alvo", q.get("target_missing", 0))

    if q.get("missing"):
        st.markdown("**Valores faltantes**")
        st.dataframe(
            pd.DataFrame(
                [{"coluna": k, "n": v["n"], "%": v["pct"]} for k, v in q["missing"].items()]
            ),
            width="stretch", hide_index=True,
        )
    if q.get("rare_categories"):
        st.markdown("**Categorias raras**")
        for col, info in q["rare_categories"].items():
            st.markdown(f"<div class='pill'><code>{col}</code> — {info['n_rare']} categoria(s) "
                        f"rara(s): {', '.join(info['categories'])}</div>",
                        unsafe_allow_html=True)
    if q.get("outliers"):
        st.markdown("**Outliers (IQR)**")
        st.dataframe(
            pd.DataFrame(
                [{"coluna": k, "n": v["n"], "%": v["pct"]} for k, v in q["outliers"].items()]
            ),
            width="stretch", hide_index=True,
        )


# ---------------------------------------------------------------------------
# Aba 3 — Decisões dos agentes (RNF03)
# ---------------------------------------------------------------------------

def _tab_actions(api: ApiClient, experiment: dict[str, Any] | None) -> None:
    st.subheader("Decisões dos agentes")
    st.caption("Ações de limpeza e engenharia de atributos efetivamente aplicadas, "
               "extraídas dos eventos agentivos persistidos (auditável — RNF03).")

    if experiment is None:
        st.info("Selecione um experimento.")
        return
    events = _safe_events(api, experiment["id"])
    relevant = [e for e in events
                if e["event_type"] in {"cleaning_decision", "feature_engineering"}]
    if not relevant:
        st.info("Execute o pipeline para registrar as decisões dos agentes.")
        return

    for ev in relevant:
        st.markdown(f"**{ev['agent_name']}** · `{ev['event_type']}`")
        out = ev.get("output_json") or {}
        actions = out.get("actions") or out.get("transformations") or []
        if actions:
            st.dataframe(pd.DataFrame(actions), width="stretch", hide_index=True)
        if out.get("warnings"):
            for w in out["warnings"]:
                st.markdown(f"<div class='pill'>⚠️ {w}</div>", unsafe_allow_html=True)
        if ev.get("rationale"):
            st.caption(f"💡 {ev['rationale']}")
        st.divider()


# ---------------------------------------------------------------------------
# Aba 4 — Resultados (RF11/RF12)
# ---------------------------------------------------------------------------

def _tab_results(api: ApiClient, experiment: dict[str, Any] | None) -> None:
    st.subheader("Ranking de modelos e recomendação")
    if experiment is None:
        st.info("Selecione um experimento.")
        return
    if experiment["status"] != "completed":
        st.info(f"Status atual: **{experiment['status']}**. "
                "Os resultados aparecem quando a execução conclui (use ↻ Atualizar).")
        if experiment["status"] != "running":
            return

    report = api.get_report(experiment["id"])
    summary = (report or {}).get("summary_json") or {}
    runs = _safe_runs(api, experiment["id"])
    run = runs[-1] if runs else None
    metrics = (run or {}).get("metrics_json") or {}
    ranking = summary.get("ranking") or (
        api.ranking(run["id"], experiment["id"]) if run else []
    )

    if not ranking:
        st.info("Ranking ainda não disponível.")
        return

    best = metrics.get("best_model") or summary.get("best_model")
    best_entry = next((e for e in ranking if e["model_name"] == best), ranking[0])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Melhor modelo", best)
    mean = best_entry.get("primary_mean", best_entry.get("mean"))
    std = best_entry.get("primary_std", best_entry.get("std", 0.0)) or 0.0
    m2.metric(f"{metrics.get('primary_metric', 'métrica')} (médio)",
              f"{mean:.3f}", f"±{std:.3f}")
    meets = metrics.get("meets_success_threshold")
    m3.metric("Critério de sucesso",
              "—" if meets is None else ("atingido" if meets else "não atingido"))
    m4.metric("Modelos avaliados", len(ranking))

    if summary.get("selection_reason"):
        st.markdown("##### Recomendação final")
        st.success(summary["selection_reason"])

    st.markdown("##### Ranking de modelos (RF11)")
    st.dataframe(_ranking_table(ranking), width="stretch", hide_index=True)

    expl = summary.get("explainability")
    if expl and expl.get("top_features"):
        st.markdown(f"##### Importância de atributos — {expl.get('method')} (RF12)")
        imp = pd.DataFrame(expl["top_features"]).set_index("feature")["importance"]
        st.bar_chart(imp)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        if summary.get("risks"):
            st.markdown("##### Riscos")
            for r in summary["risks"]:
                st.markdown(f"<div class='pill'>⚠️ {r}</div>", unsafe_allow_html=True)
    with c2:
        if summary.get("limitations"):
            st.markdown("##### Limitações")
            for limit in summary["limitations"]:
                st.markdown(f"<div class='pill'>{limit}</div>", unsafe_allow_html=True)

    if report and report.get("content"):
        with st.expander("📄 Relatório técnico completo (RF14)"):
            st.markdown(report["content"])


def _ranking_table(ranking: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for e in ranking:
        rows.append({
            "modelo": e["model_name"],
            "média": round(e.get("primary_mean", e.get("mean", 0.0)), 4),
            "desvio": round(e.get("primary_std", e.get("std", 0.0)) or 0.0, 4),
            "folds": e.get("n_folds", e.get("folds")),
            "tempo fit (s)": round(e.get("mean_fit_seconds") or 0.0, 3),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aba 5 — Histórico agentivo (RNF03, seção 23)
# ---------------------------------------------------------------------------

def _tab_history(api: ApiClient, experiment: dict[str, Any] | None) -> None:
    st.subheader("Histórico de eventos agentivos")
    if experiment is None:
        st.info("Selecione um experimento.")
        return
    st.caption(f"Trilha de auditoria de **{experiment['name']}** "
               "(tabela `agent_events`, RNF03).")

    events = _safe_events(api, experiment["id"])
    if not events:
        st.info("Sem eventos ainda. Execute o pipeline.")
        return

    agents = ["todos"] + sorted({e["agent_name"] for e in events})
    c1, c2 = st.columns([2, 3])
    chosen = c1.selectbox("Filtrar por agente", agents)
    search = c2.text_input("Buscar na justificativa", placeholder="ex.: leakage, fold, mediana…")

    filtered = [
        e for e in events
        if (chosen == "todos" or e["agent_name"] == chosen)
        and (not search or search.lower() in (e.get("rationale") or "").lower())
    ]
    table = pd.DataFrame([
        {
            "data/hora": e.get("created_at"),
            "agente": e["agent_name"],
            "tipo de evento": e["event_type"],
            "justificativa": e.get("rationale"),
        }
        for e in filtered
    ])
    st.dataframe(
        table, width="stretch", hide_index=True,
        column_config={"justificativa": st.column_config.TextColumn(
            "justificativa", width="large")},
    )
    st.caption(f"{len(filtered)} de {len(events)} eventos.")

    if filtered:
        with st.expander("Inspecionar evento (JSONB)"):
            idx = st.number_input("Evento nº", 0, len(filtered) - 1, 0)
            st.json(filtered[int(idx)])


# ---------------------------------------------------------------------------
# Helpers de chamadas tolerantes a falha
# ---------------------------------------------------------------------------

def _safe_events(api: ApiClient, experiment_id: str) -> list[dict[str, Any]]:
    try:
        return api.list_events(experiment_id)
    except ApiError as exc:
        st.error(str(exc))
        return []


def _safe_runs(api: ApiClient, experiment_id: str) -> list[dict[str, Any]]:
    try:
        return api.list_runs(experiment_id)
    except ApiError as exc:
        st.error(str(exc))
        return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Agentic Tabular Pipeline", page_icon="🧪",
        layout="wide", initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)
    _init_state()
    api = _client()
    # O cliente é um singleton (cache_resource); ressincroniza os tokens desta
    # sessão para autorizar as chamadas mesmo após reinício do processo.
    for _eid, _tok in st.session_state.tokens.items():
        api.register_token(_eid, _tok)

    st.title("Agentic Tabular Pipeline")
    st.caption("Sistema agentivo open source para dados tabulares — MAQ020")

    experiment = _sidebar(api)

    tab_setup, tab_profile, tab_actions, tab_results, tab_history = st.tabs(
        ["⚙️ Experimento", "📊 Perfil & Qualidade", "🤖 Decisões dos agentes",
         "🏆 Resultados", "📜 Histórico agentivo"]
    )
    with tab_setup:
        _tab_experiment(api)
    with tab_profile:
        _tab_profile(api, experiment)
    with tab_actions:
        _tab_actions(api, experiment)
    with tab_results:
        _tab_results(api, experiment)
    with tab_history:
        _tab_history(api, experiment)


if __name__ == "__main__":
    main()
