"""Perfil de criador: escolhido por categoria e região, com fonte em cada número."""

import pytest

from occasion.decisao import criadores, fontes
from occasion.decisao import engine as e
from occasion.modelagem import final


@pytest.fixture(scope="module")
def modelo():
    return final.carregar()


@pytest.mark.parametrize(
    ("categoria", "regiao", "esperado"),
    [
        ("conversao_associacao", "Norte", "comida_regional"),
        ("conversao_associacao", "Nordeste", "comida_regional"),
        ("conversao_associacao", "Sudeste", "review_urbano"),
        ("conversao_associacao", "Centro-Oeste", "review_urbano"),
        ("awareness_associacao", "Sul", "entretenimento_alcance"),
        ("criar_frequencia", "Nordeste", "comida_dia_a_dia"),
        ("ativacao_delivery", "Sudeste", None),
        ("acao_pizzaria", "Sul", None),
        ("combo_em_casa", "Norte", None),
        ("sem_ativacao", "Sul", None),
    ],
)
def test_perfil_por_categoria_e_regiao(categoria, regiao, esperado):
    assert criadores.chave_do_perfil(categoria, regiao) == esperado
    p = criadores.perfil(categoria, regiao)
    assert (p["perfil"] if p else None) == esperado


def test_toda_fonte_citada_existe_e_tem_link():
    dados = criadores.base()
    ids = set(dados["fontes"])
    for f in dados["fontes"].values():
        assert f["url"].startswith("https://") and f["ano"] and f["natureza"]
    for p in dados["perfis"].values():
        citadas = [x["fonte"] for x in p["por_que"] + p["exemplos"] + [p["custo_por_post"]]]
        assert {c for c in citadas if c} <= ids
        assert p["exemplos"] or p["sem_exemplo"]


def test_perfil_resolve_a_fonte_e_avisa_o_limite():
    p = criadores.perfil("awareness_associacao", "Sul")
    assert p["exemplos"][0]["fonte"]["url"].startswith("https://")
    assert "sem checagem de contrato" in p["natureza"]
    assert "não é orçamento" in p["custo_por_post"]["natureza"]


def test_conversao_no_nordeste_leva_perfil_regional_e_fonte(modelo):
    c = e.Contexto("noite", 6, "Nordeste", "capital", "25-29", "homem", 8, canal="delivery")
    r = e.decidir(c, modelo)
    assert r.rule == 3
    assert r.activation_plan["criador"]["perfil"] == "comida_regional"
    assert "criadores" in {f["id"] for f in r.sources}


def test_delivery_sem_sinal_fica_sem_criador(modelo):
    c = e.Contexto("noite", 6, "Sudeste", "interior_urbano", "25-29", "homem", 8, canal="delivery")
    r = e.decidir(c, modelo)
    assert r.recommendation_category == e.Categoria.ATIVACAO_DELIVERY
    assert r.activation_plan["criador"] is None
    assert "criadores" not in {f["id"] for f in r.sources}


def test_fonte_de_criadores_registrada():
    assert "criadores" in fontes.FONTES
