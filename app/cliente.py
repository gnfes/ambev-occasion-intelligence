"""Cliente HTTP da API do Occasion Score.

O app não importa o modelo nem o engine: toda previsão passa pela API, como
na arquitetura do case (coleta, modelo, regras e interface em camadas
separadas). O endereço vem de `OCCASION_API_URL` (padrão: localhost:8000).
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("OCCASION_API_URL", "http://localhost:8000").rstrip("/")


class ErroApi(RuntimeError):
    def __init__(self, mensagem: str, status_code: int | None = None) -> None:
        super().__init__(mensagem)
        self.status_code = status_code


def _pedir(metodo: str, rota: str, **kwargs) -> dict | list:
    try:
        r = httpx.request(metodo, f"{API_URL}{rota}", timeout=30, **kwargs)
    except httpx.HTTPError as erro:
        raise ErroApi(
            f"Não foi possível falar com a API em {API_URL} ({erro.__class__.__name__}). "
            "Suba a API com `uv run uvicorn occasion.api.app:app` ou ajuste OCCASION_API_URL."
        ) from erro
    if r.status_code >= 400:
        raise ErroApi(f"A API respondeu {r.status_code}: {r.text}", status_code=r.status_code)
    return r.json()


@st.cache_data(ttl=300, show_spinner=False)
def opcoes() -> dict:
    return _pedir("GET", "/opcoes")


@st.cache_data(ttl=300, show_spinner=False)
def modelo() -> dict:
    return _pedir("GET", "/model")


@st.cache_data(ttl=300, show_spinner=False)
def mapa() -> list[dict]:
    return _pedir("GET", "/mapa")


@st.cache_data(ttl=300, show_spinner=False)
def radar() -> dict:
    return _pedir("GET", "/radar")


@st.cache_data(ttl=300, show_spinner=False)
def radar_uf(sigla: str) -> dict:
    return _pedir("GET", f"/radar/{sigla}")


@st.cache_data(ttl=3600, max_entries=2000, show_spinner=False)
def prever(entrada: tuple) -> dict:
    """POST /predict. A entrada vem como tupla de pares para o cache do Streamlit."""
    return _pedir("POST", "/predict", json=dict(entrada))
