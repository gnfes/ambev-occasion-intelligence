"""App Streamlit: cada página roda sem erro com a API real (TestClient no lugar do HTTP)."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from occasion.api.app import app
from occasion.decisao import engine

APP = Path(__file__).parents[1] / "app"
sys.path.insert(0, str(APP))

import cliente
import rotulos


@pytest.fixture(scope="module")
def api():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def via_testclient(api, monkeypatch):
    def pedir(metodo, rota, **kwargs):
        r = api.request(metodo, rota, **kwargs)
        if r.status_code >= 400:
            raise cliente.ErroApi(r.text)
        return r.json()

    monkeypatch.setattr(cliente, "_pedir", pedir)
    for f in (cliente.opcoes, cliente.modelo, cliente.mapa, cliente.prever):
        f.clear()


@pytest.mark.parametrize("pagina", ["simulador", "mapa", "modelo"])
def test_pagina_roda_sem_erro(pagina):
    at = AppTest.from_file(str(APP / "paginas" / f"{pagina}.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]


def test_simulador_mostra_o_score_da_api(api):
    at = AppTest.from_file(str(APP / "paginas" / "simulador.py"), default_timeout=60).run()
    at.sidebar.selectbox[1].set_value("almoco").run()  # turno
    score = at.metric[0].value
    esperado = api.post(
        "/predict",
        json={
            "turno": "almoco",
            "dia_semana": 6,
            "regiao": "Sudeste",
            "tipo_area": "capital",
            "faixa_idade": "25-29",
            "sexo": "homem",
            "renda_sm": 5.0,
        },
    ).json()["occasion_score"]
    assert score == f"{esperado:.2f} / 100"


def test_api_fora_do_ar_mostra_erro_e_para(monkeypatch):
    def fora(*a, **k):
        raise cliente.ErroApi("API fora do ar")

    monkeypatch.setattr(cliente, "_pedir", fora)
    at = AppTest.from_file(str(APP / "paginas" / "simulador.py"), default_timeout=60).run()
    assert not at.exception
    assert "API fora do ar" in at.error[0].value


@pytest.mark.parametrize("renda", [0, 2, 2.1, 4, 9.5, 10, 15, 20, 20.5, 80])
def test_classe_do_app_igual_a_do_engine(renda):
    assert rotulos.classe_de_renda(renda) == engine.classe_de_renda(renda)
