"""Interface Streamlit do pipeline.

Atende aos requisitos de interface da seção 12: cadastrar experimento, selecionar
a base e a variável-alvo, visualizar o perfil dos dados e os alertas de qualidade,
acompanhar as ações propostas pelos agentes, executar o pipeline e inspecionar o
ranking de modelos e o histórico de eventos.

Execução:
    streamlit run src/ui/app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Agentic Tabular Pipeline", layout="wide")


def main() -> None:
    st.title("Agentic Tabular Pipeline")
    st.caption("Sistema agentivo open source para dados tabulares — MAQ020")

    tab_setup, tab_profile, tab_results, tab_history = st.tabs(
        ["Experimento", "Perfil & Qualidade", "Resultados", "Histórico agentivo"]
    )

    with tab_setup:
        st.subheader("Cadastrar experimento")
        st.info("TODO: formulário de configuração + upload/seleção da base (RF01/RF02).")

    with tab_profile:
        st.subheader("Perfil dos dados e alertas de qualidade")
        st.info("TODO: exibir profile_json e quality_report_json (RF03/RF04).")

    with tab_results:
        st.subheader("Ranking de modelos e recomendação")
        st.info("TODO: ranking, métricas, gráficos e comparação com baseline (RF11).")

    with tab_history:
        st.subheader("Histórico de eventos agentivos")
        st.info("TODO: tabela de agent_events do experimento selecionado (RNF03).")


if __name__ == "__main__":
    main()
