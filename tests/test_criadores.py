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
    assert criadores.chave_do_perfil(categoria, regiao, "capital") == esperado
    p = criadores.perfil(categoria, regiao, "capital")
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
    p = criadores.perfil("awareness_associacao", "Sul", "capital")
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


def test_conversao_rural_fora_do_norte_nordeste_fica_sem_perfil():
    assert criadores.perfil("conversao_associacao", "Sudeste", "rural") is None
    assert criadores.chave_do_perfil("conversao_associacao", "Sudeste", "interior_urbano") == (
        "review_urbano"
    )


@pytest.mark.parametrize(
    "estrago",
    [
        lambda d: d["perfis"]["comida_regional"].__setitem__("por_que", ["texto solto"]),
        lambda d: d.__setitem__("perfis", []),
        lambda d: d["fontes"]["youpix_2026"].pop("titulo"),
        lambda d: d["perfis"]["review_urbano"]["custo_por_post"].pop("faixa"),
        lambda d: d["perfis"]["review_urbano"]["exemplos"][0].pop("regiao"),
    ],
)
def test_validar_pega_o_que_quebraria_a_saida(estrago):
    dados = criadores.base()
    estrago(dados)
    with pytest.raises((ValueError, TypeError)):
        criadores.validar(dados)


def test_exemplo_nacional_nao_cita_outra_regiao():
    p = criadores.perfil("awareness_associacao", "Nordeste", "capital")
    assert all("Sul" not in e["nota"] for e in p["exemplos"])


def test_base_devolve_copia():
    criadores.base()["perfis"]["comida_regional"]["exemplos"].append({"nome": "x"})
    assert criadores.base()["perfis"]["comida_regional"]["exemplos"] == []


def test_exemplo_de_outra_regiao_nao_aparece():
    sudeste = criadores.perfil("conversao_associacao", "Sudeste", "capital")
    sul = criadores.perfil("conversao_associacao", "Sul", "capital")
    assert sudeste["exemplos"] and sudeste["sem_exemplo"] is None
    assert sul["exemplos"] == [] and "no Sul" in sul["sem_exemplo"]


def test_arquivo_invalido_tira_o_criador_sem_derrubar(tmp_path, monkeypatch, modelo):
    ruim = tmp_path / "perfis.json"
    ruim.write_text('{"fontes": {}, "perfis": {"x": {"nome": "x"}},}', encoding="utf-8")
    monkeypatch.setattr(criadores, "ARQUIVO", ruim)
    assert criadores.perfil("awareness_associacao", "Sul", "capital") is None
    c = e.Contexto("noite", 6, "Sul", "capital", "25-29", "homem", 8)
    assert e.decidir(c, modelo).activation_plan["criador"] is None


def test_fonte_inexistente_invalida_o_arquivo():
    dados = criadores.base()
    quebrado = {**dados, "fontes": {k: v for k, v in dados["fontes"].items() if k != "brino_2025"}}
    with pytest.raises(ValueError, match="brino_2025"):
        criadores.validar(quebrado)


def test_resposta_nao_altera_o_cache():
    p = criadores.perfil("awareness_associacao", "Sul", "capital")
    p["por_que"][0]["fonte"]["titulo"] = "alterado"
    assert (
        criadores.perfil("awareness_associacao", "Sul", "capital")["por_que"][0]["fonte"]["titulo"]
        != "alterado"
    )
