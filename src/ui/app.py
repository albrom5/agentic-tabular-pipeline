"""Interface Streamlit do pipeline agentivo (seção 12 do documento de apoio).

Cobre os oito requisitos de interface: cadastro de experimento, upload/seleção
da base, escolha da variável-alvo e do tipo de tarefa, visualização do perfil e
dos alertas de qualidade, revisão das ações propostas pelos agentes (aceitar ou
ajustar), botão para executar o pipeline, painel de resultados com ranking,
métricas, gráficos e recomendação, e acesso ao histórico de eventos agentivos.

Esta é uma versão de *protótipo de UI*: campos e botões operam sobre dados mock
(``src/ui/mock_data.py``) para validar aparência e usabilidade. O único componente
real exercitado é o Agente de Ingestão e Perfilamento, aplicado à base carregada.

Execução:
    streamlit run src/ui/app.py
"""

from __future__ import annotations

import io
from typing import Any

import pandas as pd
import streamlit as st

from src.agents.data_profile_agent import DataProfileAgent
from src.ui import mock_data as mock

st.set_page_config(
    page_title="Agentic Tabular Pipeline",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Estilo
# ---------------------------------------------------------------------------

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
    .sev-alta   { color:#ff6b6b; font-weight:700; }
    .sev-média  { color:#ffd166; font-weight:700; }
    .sev-baixa  { color:#9aa0a6; font-weight:700; }
    .pill {
        background:#1c2330; border:1px solid #2c3445; border-radius:10px;
        padding:10px 14px; margin-bottom:8px;
    }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

_STATUS_BADGE = {
    "concluído": "badge-ok",
    "em execução": "badge-run",
    "rascunho": "badge-draft",
}


# ---------------------------------------------------------------------------
# Estado de sessão
# ---------------------------------------------------------------------------

def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("dataframe", None)
    ss.setdefault("dataset_name", None)
    ss.setdefault("experiment_config", None)
    ss.setdefault("pipeline_ran", False)
    ss.setdefault("accepted_actions", {})


# ---------------------------------------------------------------------------
# Sidebar — seleção de experimento e ações globais
# ---------------------------------------------------------------------------

def _sidebar() -> dict[str, Any]:
    with st.sidebar:
        st.markdown("### 🧪 Agentic Tabular Pipeline")
        st.caption("MAQ020 · protótipo de interface (dados mock)")
        st.divider()

        names = [e["name"] for e in mock.EXPERIMENTS]
        selected = st.selectbox("Experimento", names, index=0)
        experiment = next(e for e in mock.EXPERIMENTS if e["name"] == selected)

        badge_cls = _STATUS_BADGE.get(experiment["status"], "badge-draft")
        st.markdown(
            f"**Status:** <span class='badge {badge_cls}'>{experiment['status']}</span>",
            unsafe_allow_html=True,
        )
        st.caption(f"ID `{experiment['id'][:8]}…` · criado em {experiment['created_at']}")

        st.divider()
        st.button("➕ Novo experimento", width="stretch")
        st.button("🔁 Reexecutar com mesma seed", width="stretch",
                  help="RF15 — reexecução reprodutível (mock).")
        st.button("⬇️ Exportar relatório técnico", width="stretch",
                  help="RF14 — relatório em Markdown/PDF (mock).")

        st.divider()
        st.caption("Pipeline: 1 Problema · 2 Perfil · 3 Limpeza · 4 Features · "
                   "5 Split · 6 Model Zoo · 7 Autoencoder · 8 Avaliação · 9 Relatório")
    return experiment


# ---------------------------------------------------------------------------
# Aba 1 — Cadastro do experimento (itens 12.1, 12.2, 12.3)
# ---------------------------------------------------------------------------

def _tab_experiment(experiment: dict[str, Any]) -> None:
    st.subheader("Cadastrar experimento")
    st.caption("Defina a tarefa, selecione a base e a variável-alvo (RF01/RF02).")

    col_form, col_data = st.columns([3, 2], gap="large")

    with col_data:
        st.markdown("##### Base de dados")
        source = st.radio(
            "Origem da base", ["Upload", "Base de exemplo", "PostgreSQL"],
            horizontal=True, label_visibility="collapsed",
        )
        uploaded = None
        if source == "Upload":
            uploaded = st.file_uploader("Arraste um CSV ou Parquet", type=["csv", "parquet"])
        elif source == "Base de exemplo":
            st.selectbox("Dataset disponível", mock.SAMPLE_DATASETS, index=0)
        else:
            st.text_input("URI PostgreSQL", placeholder="postgresql://…/schema.tabela",
                          disabled=True)
            st.caption("Conector PostgreSQL é bônus no MVP (RF02).")

        df = _resolve_dataframe(uploaded)
        st.session_state.dataframe = df
        cols = list(df.columns)
        st.success(f"Base carregada: {df.shape[0]} linhas × {df.shape[1]} colunas.")
        with st.expander("Pré-visualizar amostra"):
            st.dataframe(df.head(15), width="stretch")

    with col_form:
        st.markdown("##### Configuração do estudo")
        with st.form("experiment_form"):
            name = st.text_input("Nome do experimento", value=experiment["name"])
            c1, c2 = st.columns(2)
            task_type = c1.selectbox(
                "Tipo de tarefa", mock.TASK_TYPES,
                index=mock.TASK_TYPES.index(experiment["task_type"]),
            )
            target_default = experiment["target_column"] or (cols[-1] if cols else "")
            target_index = cols.index(target_default) if target_default in cols else len(cols) - 1
            target = c2.selectbox(
                "Variável-alvo", cols,
                index=max(target_index, 0),
                disabled=(task_type == "anomaly"),
                help="Opcional para detecção de anomalias não supervisionada.",
            )

            metrics = mock.METRICS_BY_TASK[task_type]
            c3, c4 = st.columns(2)
            primary_metric = c3.selectbox("Métrica primária", metrics)
            success = c4.number_input("Critério de sucesso (≥)", 0.0, 1.0, 0.70, 0.01)

            c5, c6 = st.columns(2)
            split = c5.selectbox("Estratégia de split", mock.SPLIT_STRATEGIES, index=2)
            budget = c6.number_input("Budget de treino (min)", 1, 240, 20)

            constraints = st.text_area(
                "Restrições adicionais (JSON livre)",
                value='{"interpretabilidade": "alta", "max_fit_seconds": 60}',
                height=70,
            )
            submitted = st.form_submit_button("💾 Salvar configuração", type="primary",
                                              width="stretch")

        if submitted:
            st.session_state.experiment_config = {
                "name": name, "task_type": task_type, "target_column": target,
                "primary_metric": primary_metric, "success_threshold": success,
                "split_strategy": split, "budget_minutes": budget,
                "constraints": constraints,
            }
            st.session_state.dataset_name = name
            st.toast("Configuração salva.", icon="✅")

        st.divider()
        st.markdown("##### Executar pipeline")
        st.caption("Dispara os 9 agentes em sequência (mock).")
        if st.button("▶️ Executar pipeline", type="primary", width="stretch"):
            _run_pipeline_mock()


def _resolve_dataframe(uploaded: Any) -> pd.DataFrame:
    if uploaded is not None:
        try:
            if uploaded.name.endswith(".parquet"):
                return pd.read_parquet(uploaded)
            return pd.read_csv(uploaded)
        except Exception as exc:  # pragma: no cover - feedback de UI
            st.error(f"Falha ao ler o arquivo: {exc}")
    if st.session_state.dataframe is not None:
        return st.session_state.dataframe
    return mock.sample_dataframe()


def _run_pipeline_mock() -> None:
    steps = [
        "Formulação do problema", "Ingestão e perfilamento", "Qualidade e limpeza",
        "Engenharia de atributos", "Split e validação", "Treino do model zoo",
        "Autoencoder tabular", "Avaliação e seleção", "Relatório e auditoria",
    ]
    bar = st.progress(0.0, text="Iniciando pipeline…")
    for i, step in enumerate(steps, start=1):
        bar.progress(i / len(steps), text=f"Agente {i}/9 — {step}")
    bar.empty()
    st.session_state.pipeline_ran = True
    st.success("Pipeline executado. Veja os resultados na aba **Resultados**.")
    st.balloons()


# ---------------------------------------------------------------------------
# Aba 2 — Perfil e qualidade (item 12.4)
# ---------------------------------------------------------------------------

def _tab_profile() -> None:
    st.subheader("Perfil dos dados e alertas de qualidade")
    st.caption("Perfil gerado pelo Agente de Ingestão e Perfilamento (RF03) "
               "e alertas de qualidade (RF04/RF06).")

    df = st.session_state.dataframe
    if df is None:
        df = mock.sample_dataframe()

    config = st.session_state.experiment_config or {}
    profile = _run_profile_agent(df, config)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Linhas", f"{profile['n_rows']:,}".replace(",", "."))
    m2.metric("Colunas", profile["n_cols"])
    m3.metric("Duplicatas", profile["duplicates"]["n_duplicate_rows"])
    total_missing = sum(c["n_missing"] for c in profile["columns"])
    m4.metric("Células faltantes", f"{total_missing:,}".replace(",", "."))
    st.caption(f"Hash de conteúdo: `{profile['content_hash'][:24]}…` — versionamento de dados (RNF01).")

    st.markdown("##### Perfil por coluna")
    st.dataframe(_columns_table(profile), width="stretch", hide_index=True)

    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        st.markdown("##### Distribuição do alvo")
        target = profile.get("target")
        if target and target.get("kind") == "classification":
            dist = pd.DataFrame(target["class_distribution"]).set_index("label")["count"]
            st.bar_chart(dist)
            st.caption(f"Razão de desbalanceamento: **{target.get('imbalance_ratio')}** "
                       f"(maioria/minoria).")
        elif target and target.get("kind") == "regression":
            st.bar_chart(df[target["name"]].dropna())
        else:
            st.info("Sem variável-alvo definida (modo não supervisionado).")

    with col_right:
        st.markdown("##### Correlações fortes (|r| ≥ 0.95)")
        if profile["high_correlations"]:
            st.dataframe(pd.DataFrame(profile["high_correlations"]),
                         width="stretch", hide_index=True)
        else:
            st.success("Nenhum par de atributos fortemente correlacionado.")

    st.divider()
    st.markdown("##### Alertas de qualidade")
    report = mock.quality_report()
    s = report["summary"]
    a1, a2, a3 = st.columns(3)
    a1.metric("Severidade alta", s["alta"])
    a2.metric("Severidade média", s["média"])
    a3.metric("Severidade baixa", s["baixa"])
    for alert in report["alerts"]:
        sev = alert["severity"]
        st.markdown(
            f"<div class='pill'><span class='sev-{sev}'>● {sev.upper()}</span> "
            f"&nbsp;<b>{alert['rule']}</b> · <code>{alert['column']}</code><br>"
            f"{alert['message']}<br>"
            f"<small>💡 {alert['suggestion']}</small></div>",
            unsafe_allow_html=True,
        )


@st.cache_data(show_spinner=False)
def _run_profile_agent(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    ctx: dict[str, Any] = {"dataframe": df}
    if config.get("target_column") in df.columns:
        ctx["target_column"] = config["target_column"]
        ctx["task_type"] = config.get("task_type")
    elif "default" in df.columns:
        ctx["target_column"] = "default"
        ctx["task_type"] = "classification"
    return DataProfileAgent(event_sink=None).run(ctx).output


def _columns_table(profile: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for c in profile["columns"]:
        rows.append(
            {
                "coluna": c["name"],
                "tipo inferido": c["inferred_type"],
                "faltantes %": c["pct_missing"],
                "únicos": c["n_unique"],
                "papel": c.get("role") or ("alvo" if c["is_target"] else ""),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aba 3 — Ações propostas pelos agentes (item 12.5)
# ---------------------------------------------------------------------------

def _tab_actions() -> None:
    st.subheader("Ações propostas pelos agentes")
    st.caption("Revise, aceite ou ajuste cada decisão antes de aplicar (RNF03 — auditável).")

    actions = mock.proposed_actions()
    accepted = st.session_state.accepted_actions

    with st.form("actions_form"):
        for i, act in enumerate(actions):
            c_check, c_body = st.columns([1, 11])
            key = f"action_{i}"
            checked = c_check.checkbox(
                "aceitar", value=act["default_accepted"], key=key,
                label_visibility="collapsed",
            )
            accepted[key] = checked
            flag = "⚠️ requer atenção" if not act["default_accepted"] else ""
            c_body.markdown(
                f"**{act['action']}** &nbsp; <small>· {act['agent']} {flag}</small><br>"
                f"<small>{act['reason']}</small>",
                unsafe_allow_html=True,
            )
        applied = st.form_submit_button("✅ Aplicar ações selecionadas", type="primary")

    if applied:
        n = sum(1 for v in accepted.values() if v)
        st.success(f"{n} de {len(actions)} ações aplicadas e registradas como eventos agentivos.")


# ---------------------------------------------------------------------------
# Aba 4 — Resultados (item 12.7)
# ---------------------------------------------------------------------------

def _tab_results(experiment: dict[str, Any]) -> None:
    st.subheader("Ranking de modelos e recomendação")

    if not (st.session_state.pipeline_ran or experiment["status"] == "concluído"):
        st.info("Execute o pipeline na aba **Experimento** para gerar os resultados.")
        return

    rec = mock.recommendation()
    ranking = mock.model_ranking()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Melhor modelo", rec["model"])
    m2.metric(f"{rec['primary_metric']} (médio)", f"{rec['score']:.3f}", f"±{rec['std']:.3f}")
    m3.metric("Critério de sucesso", f"≥ {rec['threshold']:.2f}",
              "atingido" if rec["passes"] else "não atingido")
    m4.metric("Modelos avaliados", len(ranking))

    st.markdown("##### Recomendação final")
    st.success(rec["text"])

    st.markdown("##### Ranking de modelos (RF11)")
    st.caption("Métrica primária por fold: média, desvio e tempo de treino.")
    st.dataframe(
        ranking,
        width="stretch", hide_index=True,
        column_config={
            "macro_f1": st.column_config.ProgressColumn(
                "macro_f1", min_value=0.0, max_value=1.0, format="%.3f"),
            "roc_auc": st.column_config.NumberColumn("roc_auc", format="%.3f"),
            "tempo_fit_s": st.column_config.NumberColumn("tempo fit (s)", format="%.1f s"),
        },
    )

    g1, g2 = st.columns(2, gap="large")
    with g1:
        st.markdown("##### Métrica por fold (top 3)")
        st.line_chart(mock.fold_scores())
    with g2:
        st.markdown("##### Curva ROC (modelo recomendado)")
        st.line_chart(mock.roc_curve().set_index("taxa de falsos positivos"))

    g3, g4 = st.columns(2, gap="large")
    with g3:
        st.markdown("##### Matriz de confusão")
        st.dataframe(_styled_confusion(mock.confusion_matrix()), width="stretch")
    with g4:
        st.markdown("##### Importância de atributos (RF12)")
        st.bar_chart(mock.feature_importance())


# ---------------------------------------------------------------------------
# Aba 5 — Histórico agentivo (item 12.8)
# ---------------------------------------------------------------------------

def _tab_history(experiment: dict[str, Any]) -> None:
    st.subheader("Histórico de eventos agentivos")
    st.caption(f"Trilha de auditoria do experimento **{experiment['name']}** "
               f"(`agent_events`, RNF03).")

    events = mock.agent_events()
    agents = ["todos"] + sorted({e["agent_name"] for e in events})
    c1, c2 = st.columns([2, 3])
    chosen = c1.selectbox("Filtrar por agente", agents)
    search = c2.text_input("Buscar na justificativa", placeholder="ex.: leakage, fold, mediana…")

    filtered = [
        e for e in events
        if (chosen == "todos" or e["agent_name"] == chosen)
        and (not search or search.lower() in e["rationale"].lower())
    ]

    st.dataframe(
        pd.DataFrame(filtered),
        width="stretch", hide_index=True,
        column_config={
            "timestamp": "data/hora",
            "agent_name": "agente",
            "event_type": "tipo de evento",
            "rationale": st.column_config.TextColumn("justificativa", width="large"),
        },
    )

    st.caption(f"{len(filtered)} de {len(events)} eventos.")
    with st.expander("Exemplo de evento persistido (JSONB)"):
        st.json(
            {
                "experiment_id": experiment["id"],
                "agent_name": "Agente de Qualidade e Limpeza",
                "event_type": "cleaning_decision",
                "input_json": {"missing_values": {"income": 0.23}, "duplicates": 12},
                "output_json": {
                    "actions": [
                        {"column": "income", "operation": "median_imputation",
                         "reason": "23% de faltantes; variável numérica relevante"},
                        {"operation": "drop_duplicates", "rows": 12, "reason": "duplicatas exatas"},
                    ],
                    "warnings": ["avaliar viés na variável income"],
                },
                "rationale": "Imputação pela mediana como baseline robusto.",
            }
        )
    st.download_button(
        "⬇️ Baixar eventos (JSON)", data=_events_json(filtered),
        file_name="agent_events.json", mime="application/json",
    )


def _styled_confusion(cm: pd.DataFrame):
    """Gradiente azul manual (sem dependência de matplotlib)."""
    vmax = float(cm.to_numpy().max()) or 1.0

    def _shade(val: Any) -> str:
        alpha = 0.12 + 0.78 * (float(val) / vmax)
        return f"background-color: rgba(46, 134, 222, {alpha:.2f}); color: #f0f4ff;"

    return cm.style.map(_shade).format("{:d}")


def _events_json(events: list[dict[str, Any]]) -> bytes:
    buf = io.StringIO()
    pd.DataFrame(events).to_json(buf, orient="records", force_ascii=False, indent=2)
    return buf.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _init_state()
    st.title("Agentic Tabular Pipeline")
    st.caption("Sistema agentivo open source para dados tabulares — MAQ020")

    experiment = _sidebar()

    tab_setup, tab_profile, tab_actions, tab_results, tab_history = st.tabs(
        ["⚙️ Experimento", "📊 Perfil & Qualidade", "🤖 Ações dos agentes",
         "🏆 Resultados", "📜 Histórico agentivo"]
    )
    with tab_setup:
        _tab_experiment(experiment)
    with tab_profile:
        _tab_profile()
    with tab_actions:
        _tab_actions()
    with tab_results:
        _tab_results(experiment)
    with tab_history:
        _tab_history(experiment)


if __name__ == "__main__":
    main()
