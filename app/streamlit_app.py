"""Occasion Intelligence: Pizza + Guaraná.

    uv run streamlit run app/streamlit_app.py

A API precisa estar no ar (`uv run uvicorn occasion.api.app:app`); o script
`app/rodar.sh` sobe as duas.
"""

import streamlit as st

st.set_page_config(
    page_title="Occasion Intelligence · Pizza + Guaraná",
    page_icon="🍕",
    layout="wide",
)

paginas = st.navigation(
    [
        st.Page("paginas/simulador.py", title="Simulador", icon="🎯", default=True),
        st.Page("paginas/mapa.py", title="Mapa de Oportunidade", icon="🗺️"),
        st.Page("paginas/modelo.py", title="Modelo e validação", icon="📊"),
    ]
)
paginas.run()
