"""Trend Radar: calendário, clima, tendência, regra do alerta e API."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from occasion.api.app import app
from occasion.radar import radar


def _feriados():
    return pd.DataFrame(
        {
            "data": pd.to_datetime(["2026-02-16", "2026-02-17", "2026-11-20", "2026-12-25"]),
            "nome": ["Carnaval", "Carnaval", "Dia da Consciência Negra", "Natal"],
            "tipo": "national",
        }
    )


def test_calendario_marca_a_semana_de_domingo_a_sabado():
    semanas = pd.Series(pd.to_datetime(["2026-02-15", "2026-07-05", "2026-11-15", "2026-12-20"]))
    cal = radar.calendario_semanal(semanas, _feriados()).set_index("semana")
    assert cal.loc["2026-02-15", "carnaval"] == 1
    assert cal.loc["2026-02-15", "feriado"] == 0  # Carnaval tem variável própria
    assert cal.loc["2026-07-05", "dia_da_pizza"] == 1  # 10/07 cai na sexta
    assert cal.loc["2026-11-15", "feriado"] == 1
    assert cal.loc["2026-12-20", "natal_ano_novo"] == 1
    assert cal.loc["2026-12-20", "feriado"] == 0
    # 15 a 21/11: nenhum dia entre 1 e 10.
    assert cal.loc["2026-11-15", "inicio_mes"] == 0


def test_clima_semanal_mede_anomalia_contra_2016_2020():
    datas = pd.date_range("2016-01-03", "2021-12-31", freq="D")
    diario = pd.DataFrame(
        {
            "data": datas,
            "sigla": "SP",
            "temperature_2m_mean": 20.0,
            "precipitation_sum": 2.0,
            "origem": "historico",
        }
    )
    diario.loc[diario["data"] >= "2021-06-06", "temperature_2m_mean"] = 23.0
    sem = radar.clima_semanal(diario)
    depois = sem[sem["semana"] >= "2021-06-06"]
    assert np.allclose(depois["anom_temp"].dropna(), 3.0)
    assert np.allclose(depois["anom_chuva"].dropna(), 0.0)


def _serie(valores):
    return pd.DataFrame(
        {"semana": pd.date_range("2021-01-03", periods=len(valores), freq="7D"), "indice": valores}
    )


def test_tendencia_sobe_quando_passa_da_faixa_de_ruido():
    rng = np.random.default_rng(0)
    base = 50 + rng.normal(0, 2, 160)
    assert radar.tendencia(_serie(base))["status"] == "estável"
    alta = base.copy()
    alta[-4:] = 80
    t = radar.tendencia(_serie(alta))
    assert t["status"] == "subindo"
    assert t["variacao_anual_pct"] > t["faixa_ruido_pct"]
    baixa = base.copy()
    baixa[-4:] = 25
    assert radar.tendencia(_serie(baixa))["status"] == "caindo"


@pytest.mark.parametrize(
    ("status", "alto", "nivel"),
    [
        ("subindo", 3, "oportunidade"),
        ("subindo", 0, "observar"),
        ("caindo", 5, "atencao"),
        ("estável", 5, "estavel"),
    ],
)
def test_regra_do_alerta(status, alto, nivel):
    assert radar.regra_alerta(status, alto)[0] == nivel


def test_ajuste_recupera_coeficiente_conhecido():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(size=500)})
    df["desvio"] = 5 + 3 * df["x"] + rng.normal(0, 0.1, 500)
    ajuste = radar.ajustar(df, ["x"])
    assert ajuste.coef == pytest.approx([5, 3], abs=0.05)


@pytest.fixture(scope="module")
def cliente():
    with TestClient(app) as c:
        yield c


def test_api_radar_tem_as_27_ufs(cliente):
    r = cliente.get("/radar").json()
    assert len(r["alertas"]) == 27
    assert {a["nivel"] for a in r["alertas"]} <= {
        "oportunidade",
        "observar",
        "atencao",
        "estavel",
        "sem_volume",
    }
    assert r["validacao"]["modelos"][0]["modelo"].startswith("baseline")


def test_api_radar_uf_traz_serie(cliente):
    r = cliente.get("/radar/sp").json()
    assert r["alerta"]["sigla"] == "SP"
    assert len(r["serie"]) > 200
    assert cliente.get("/radar/XX").status_code == 404


def test_uf_sem_volume_fica_fora_do_painel():
    semanas = pd.date_range("2021-01-03", periods=10, freq="7D")
    serie = pd.concat(
        [
            pd.DataFrame(
                {"semana": semanas, "sigla": s, "termo": "pizza", "indice": v, "parcial": False}
            )
            for s, v in [("SP", [50] * 10), ("RR", [0, 0, 0, 40, 50, 60, 0, 55, 45, 50])]
        ]
    )
    assert radar.sem_volume(serie) == ["RR"]


def test_tendencia_sem_ano_anterior_medivel_volta_none():
    valores = np.full(160, 50.0)
    valores[-56:-52] = 0  # mesmas 4 semanas do ano anterior zeradas
    assert radar.tendencia(_serie(valores)) is None
