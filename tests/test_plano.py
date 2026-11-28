"""Plano de ativação: cada campo sai de um dado medido, e as hipóteses vêm marcadas."""

import pytest

from occasion.decisao import associacao, plano
from occasion.decisao import engine as e
from occasion.modelagem import final


@pytest.fixture(scope="module")
def modelo():
    return final.carregar()


@pytest.fixture(scope="module")
def tabela():
    return associacao.carregar()


def _ctx(turno="noite", dia=6, regiao="Nordeste", area="interior_urbano", **kw):
    return e.Contexto(turno, dia, regiao, area, kw.get("faixa", "25-29"), "mulher",
                      kw.get("renda", 6), canal=kw.get("canal"))  # fmt: skip


def test_capital_lista_so_capitais_da_regiao():
    r = plano.cidades("Sudeste", "capital")
    assert [c["municipio"] for c in r["lista"]] == ["São Paulo", "Belo Horizonte", "Vitória"]


def test_fora_da_capital_nao_lista_capital_e_diz_o_limite_do_mapa():
    r = plano.cidades("Nordeste", "interior_urbano")
    capitais = {nome for nome, *_ in plano.CAPITAIS.values()}
    assert r["lista"] and not {c["municipio"] for c in r["lista"]} & capitais
    assert "não separa região metropolitana de interior" in r["criterio"]


def test_rural_fica_sem_cidades():
    assert plano.cidades("Norte", "rural")["lista"] == []


@pytest.mark.parametrize(
    ("regiao", "esperado"),
    [("Sudeste", "consumo em casa"), ("Sul", "consumo em casa"), ("Nordeste", "casa e salão")],
)
def test_canal_sem_informar_segue_o_ic_de_consumo_em_casa(tabela, regiao, esperado):
    assert plano.canal(None, regiao, tabela)["valor"].startswith(esperado)


def test_canal_informado_vence(tabela):
    assert plano.canal("pizzaria", "Sudeste", tabela)["base"] == "canal informado no contexto"


def test_sem_ativacao_nao_tem_plano(modelo):
    r = e.decidir(_ctx(turno="almoco", dia=2, regiao="Sudeste", area="resto_rm"), modelo)
    assert r.recommendation_category == e.Categoria.SEM_ATIVACAO
    assert r.activation_plan is None


def test_plano_da_conversao(modelo):
    r = e.decidir(_ctx(), modelo)
    p = r.activation_plan
    assert r.rule == 3
    assert p["para_quem"]["segmento_do_sinal"] == ["no Nordeste"]
    assert p["quando"]["horario"]["janela"] == "19h a 21h"
    assert p["quando"]["proximas_4_semanas"]["uf"] == p["onde"]["cidades"][0]["uf"]
    assert p["mecanica"]["natureza"].startswith("hipótese")
    assert p["medicao"]["meta"] is None
    assert p["medicao"]["linha_de_base_pct"] == 16.3
    # O canal da recomendação é o do plano (Nordeste sem maioria clara em casa).
    assert r.recommendation_channel == p["canal"]["valor"]
    assert {"mapa", "inicio_19_21", "radar"} <= {f["id"] for f in r.sources}


def test_horario_so_a_noite(modelo):
    r = e.decidir(_ctx(turno="almoco", dia=6, regiao="Sudeste", area="capital"), modelo)
    assert r.rule == 5  # age na noite de semana, então o horário aparece
    assert r.activation_plan["quando"]["turno"] == "noite"
    assert r.activation_plan["quando"]["horario"] is not None


def test_fora_de_20_a_44_nao_usa_numeros_da_pof(modelo):
    r = e.decidir(_ctx(regiao="Sul", area="capital", faixa="60+", renda=8), modelo)
    p = r.activation_plan
    assert p["quando"]["horario"] is None
    assert p["medicao"]["linha_de_base_pct"] is None


def test_tabela_tem_o_horario_de_inicio(tabela):
    pico = tabela["brasil"]["inicio_19h_21h"]
    assert 60 < pico["pct"] < 80 and pico["ic95"][0] < pico["pct"] < pico["ic95"][1]


def test_janela_vencida_some():
    import datetime as dt

    assert plano.janela("SP", hoje=dt.date(2027, 3, 1)) is None


def test_awareness_comeca_por_criadores_mesmo_com_canal(modelo):
    r = e.decidir(_ctx(regiao="Sul", area="capital", canal="delivery"), modelo)
    assert r.rule == 2
    assert r.recommendation_channel == r.activation_plan["canal"]["valor"]
    assert r.recommendation_channel.startswith("criadores")
