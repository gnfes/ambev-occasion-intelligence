"""API: contrato de entrada e saída, sobre o modelo versionado."""

import pytest
from fastapi.testclient import TestClient

from occasion.api.app import app
from occasion.decisao import engine
from occasion.modelagem import final
from occasion.modelagem import incerteza as inc

ENTRADA = {
    "turno": "noite",
    "dia_semana": 6,
    "regiao": "Sudeste",
    "tipo_area": "capital",
    "faixa_idade": "25-29",
    "sexo": "mulher",
    "renda_sm": 5,
    "canal": "delivery",
}


@pytest.fixture(scope="module")
def cliente():
    with TestClient(app) as c:
        yield c


def test_health_e_model(cliente):
    assert cliente.get("/health").json() == {"status": "ok", "versao_modelo": final.VERSAO}
    meta = cliente.get("/model").json()
    assert meta["features"] == [
        "turno",
        "dia_semana",
        "regiao",
        "tipo_area",
        "faixa_idade",
        "sexo",
        "renda_pct",
    ]
    assert meta["validacao_temporal"]["modelo"]["pr_auc"] > 0


def test_predict_segue_o_contrato_e_o_engine(cliente):
    r = cliente.post("/predict", json=ENTRADA)
    assert r.status_code == 200
    corpo = r.json()
    for campo in (
        "occasion_score",
        "classification",
        "top_factors",
        "recommendation_category",
        "recommendation_reason",
        "model_metadata",
    ):
        assert campo in corpo
    from occasion.api import app as api

    modelo = final.carregar()
    replicas = api._ler_incerteza(api._caminho_incerteza(), modelo)
    esperado = engine.decidir(engine.Contexto(**ENTRADA), modelo, incerteza=replicas).como_dict()
    assert corpo == esperado
    # Capital tem lacuna de associação na tabela gravada: o sinal chega à resposta.
    assert "tipo de área = capital" in {x["segmento"] for x in corpo["association"]["lacuna"]}
    assert len(corpo["top_factors"]) == 3


def test_canal_e_opcional(cliente):
    sem_canal = {k: v for k, v in ENTRADA.items() if k != "canal"}
    assert cliente.post("/predict", json=sem_canal).status_code == 200


@pytest.mark.parametrize(
    ("campo", "valor"),
    [
        ("turno", "madrugada"),
        ("dia_semana", 8),
        ("dia_semana", 6.0),
        ("dia_semana", "6"),
        ("regiao", "Sul "),
        ("faixa_idade", "25-34"),
        ("sexo", "outro"),
        ("renda_sm", -1),
        ("renda_sm", True),
        ("renda_sm", "5"),
        ("canal", "ifood"),
        ("extra", 1),
    ],
)
def test_entrada_invalida_e_422(cliente, campo, valor):
    r = cliente.post("/predict", json={**ENTRADA, campo: valor})
    assert r.status_code == 422


def test_versao_diferente_e_recusada(cliente):
    r = cliente.post("/predict", json={**ENTRADA, "versao_modelo": "0.9.0"})
    assert r.status_code == 409
    ok = cliente.post("/predict", json={**ENTRADA, "versao_modelo": final.VERSAO})
    assert ok.status_code == 200


def test_opcoes_batem_com_o_engine(cliente):
    o = cliente.get("/opcoes").json()
    assert o["regiao"] == engine.REGIOES
    assert [d["valor"] for d in o["dia_semana"]] == list(engine.DIAS)


def test_valores_de_opcoes_sao_aceitos_no_predict(cliente):
    o = cliente.get("/opcoes").json()
    for d in o["dia_semana"]:
        assert (
            cliente.post("/predict", json={**ENTRADA, "dia_semana": d["valor"]}).status_code == 200
        )
    for campo in ("turno", "regiao", "tipo_area", "faixa_idade", "sexo", "canal"):
        for v in o[campo]:
            assert cliente.post("/predict", json={**ENTRADA, campo: v}).status_code == 200


def test_erro_do_engine_vira_422(cliente, monkeypatch):
    def recusa(*a, **k):
        raise ValueError("tipo_area recusado pelo engine")

    monkeypatch.setattr(engine, "Contexto", recusa)
    r = cliente.post("/predict", json=ENTRADA)
    assert r.status_code == 422
    assert "tipo_area" in r.json()["detail"]


def test_mapa_tem_os_319_municipios_em_ordem(cliente):
    pontos = cliente.get("/mapa").json()
    assert len(pontos) == 319
    assert [p["prioridade"] for p in pontos] == sorted(p["prioridade"] for p in pontos)
    assert all(-34 < p["lat"] < 6 and -74 < p["lon"] < -34 for p in pontos)
    monitorar = [p for p in pontos if p["categoria"] == "monitorar"]
    assert monitorar and all(p["canal"] is None for p in monitorar)


def test_predict_traz_incerteza_fontes_e_limites(cliente):
    corpo = cliente.post("/predict", json=ENTRADA).json()
    u = corpo["uncertainty"]
    assert u["disponivel"] is True
    lo, hi = u["occasion_score_ic95"]
    assert lo < corpo["occasion_score"] < hi
    assert corpo["best_moment"]["occasion_score_ic95"] is not None
    assert {f["id"] for f in corpo["sources"]} >= {"score", "intervalo"}
    assert corpo["limitations"]
    meta = cliente.get("/model").json()
    assert meta["incerteza"]["versao_modelo"] == final.VERSAO
    assert meta["incerteza"]["replicas"] == inc.N_REPLICAS


def test_replicas_de_outra_versao_sao_ignoradas(tmp_path, monkeypatch):
    from occasion.api import app as api

    velho = inc.carregar()
    velho.metadata = {**velho.metadata, "versao_modelo": "0.0.0"}
    monkeypatch.setenv("OCCASION_INCERTEZA_PATH", str(velho.salvar(tmp_path / "v.npz")))
    with TestClient(api.app) as c:
        u = c.post("/predict", json=ENTRADA).json()["uncertainty"]
    assert u["disponivel"] is False
