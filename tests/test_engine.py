"""Decision Engine: regras isoladas e de ponta a ponta com o modelo versionado."""

import numpy as np
import pytest

from occasion.decisao import engine as e
from occasion.modelagem import final


@pytest.fixture(scope="module")
def modelo():
    return final.carregar()


@pytest.mark.parametrize(
    ("faixa", "turno", "classe", "melhor", "esperado"),
    [
        ("baixa", "almoco", "C", "alta", (1, e.Categoria.SEM_ATIVACAO)),
        ("alta", "noite", "A", "alta", (2, e.Categoria.CONVERSAO)),
        ("alta", "almoco", "E", "alta", (2, e.Categoria.CONVERSAO)),
        ("media", "noite", "A", "alta", (3, e.Categoria.CRIAR_FREQUENCIA)),
        ("baixa", "noite", "E", "alta", (3, e.Categoria.CRIAR_FREQUENCIA)),
        ("media", "noite", "D", "media", (4, e.Categoria.AWARENESS)),
        ("media", "tarde", "B", "media", (4, e.Categoria.AWARENESS)),
        ("media", "noite", "A", "media", (5, e.Categoria.SEM_ATIVACAO)),
        ("media", "noite", "E", "media", (5, e.Categoria.SEM_ATIVACAO)),
        ("baixa", "noite", "C", "media", (5, e.Categoria.SEM_ATIVACAO)),
    ],
)
def test_regras_em_ordem(faixa, turno, classe, melhor, esperado):
    assert e.regra(faixa, turno, classe, melhor) == esperado


@pytest.mark.parametrize(
    ("renda", "classe"), [(0, "E"), (2, "E"), (2.1, "D"), (4, "D"), (9, "C"), (15, "B"), (21, "A")]
)
def test_classe_de_renda(renda, classe):
    assert e.classe_de_renda(renda) == classe


def test_contexto_recusa_valores_invalidos():
    with pytest.raises(ValueError, match="turno") as erro:
        e.Contexto("madrugada", 8, "Sudeste", "capital", "25-29", "homem", 3)
    assert "dia_semana" in str(erro.value)
    with pytest.raises(ValueError, match="canal"):
        e.Contexto("noite", 6, "Sudeste", "capital", "25-29", "homem", 3, canal="ifood")


def test_sabado_a_noite_na_capital_e_conversao_pelo_canal(modelo):
    c = e.Contexto("noite", 6, "Sudeste", "capital", "25-29", "homem", 8, canal="delivery")
    r = e.decidir(c, modelo)
    assert r.classification == "alta"
    assert r.recommendation_category == e.Categoria.CONVERSAO
    assert r.recommendation_channel == "app de delivery"
    assert r.top_factors[0]["feature"] == "turno"
    assert r.recommendation_reason.startswith("Regra 2.")


def test_contrato_de_saida(modelo):
    c = e.Contexto("noite", 2, "Nordeste", "interior_urbano", "35-39", "mulher", 3)
    saida = e.decidir(c, modelo).como_dict()
    for chave in (
        "occasion_score",
        "classification",
        "top_factors",
        "recommendation_category",
        "recommendation_reason",
        "model_metadata",
    ):
        assert chave in saida
    assert 0 <= saida["occasion_score"] <= 100
    assert saida["model_metadata"]["versao"] == final.VERSAO
    assert len(saida["top_factors"]) == 3


def test_melhor_momento_e_sabado_a_noite(modelo):
    # O modelo é aditivo no logit: o melhor dia e turno não dependem do perfil.
    for regiao in e.REGIOES:
        c = e.Contexto("almoco", 3, regiao, "rural", "60+", "mulher", 1)
        m = e.melhor_momento(modelo, c)
        assert (m["dia"], m["turno"]) == ("sábado", "noite")


def test_dia_fraco_com_perfil_forte_vira_criar_frequencia(modelo):
    c = e.Contexto("noite", 2, "Nordeste", "interior_urbano", "35-39", "mulher", 3)
    r = e.decidir(c, modelo)
    assert r.classification != "alta"
    assert r.best_moment["classification"] == "alta"
    assert r.recommendation_category == e.Categoria.CRIAR_FREQUENCIA


def test_fatores_mostram_dia_e_sexo_por_extenso(modelo):
    c = e.Contexto("noite", 6, "Sul", "capital", "20-24", "mulher", 5)
    r = e.decidir(c, modelo, n_fatores=7)
    valores = {f["feature"]: f["valor"] for f in r.top_factors}
    assert valores["dia_semana"] == "sábado"
    assert valores["sexo"] == "mulher"


def test_renda_vira_percentil_crescente(modelo):
    p = modelo.renda_pct([0.5, 2, 5, 20])
    assert list(p) == sorted(p)
    assert 0 <= p[0] and p[-1] <= 100


@pytest.mark.parametrize("dia", [6.0, True])
def test_contexto_recusa_dia_que_nao_e_inteiro(dia):
    with pytest.raises(ValueError, match="dia_semana"):
        e.Contexto("noite", dia, "Sudeste", "capital", "25-29", "homem", 3)


@pytest.mark.parametrize("renda", ["8", None, True, float("nan"), -1])
def test_contexto_recusa_renda_invalida_com_value_error(renda):
    with pytest.raises(ValueError, match="renda_sm"):
        e.Contexto("noite", 6, "Sudeste", "capital", "25-29", "homem", renda)


def test_mensagem_da_regra_3_fala_do_momento_recomendado(modelo):
    c = e.Contexto("noite", 2, "Nordeste", "interior_urbano", "35-39", "mulher", 3)
    r = e.decidir(c, modelo)
    assert r.rule == 3
    assert "fim de semana" in r.messages[0]


def test_regra_5_na_faixa_media_nao_diz_score_baixo(modelo, monkeypatch):
    # Força o melhor momento fora da faixa alta para isolar a regra 5.
    falso = {"dia_semana": 6, "dia": "sábado", "turno": "noite", "occasion_score": 0.5}
    monkeypatch.setattr(e, "melhor_momento", lambda m, c, *_: {**falso, "classification": "media"})
    monkeypatch.setattr(modelo, "faixa", lambda s: np.array(["media"] * len(s)))
    c = e.Contexto("noite", 2, "Sudeste", "capital", "25-29", "homem", 30)
    r = e.decidir(c, modelo)
    assert (r.rule, r.classification) == (5, "media")
    assert "fora das classes B a D" in r.recommendation_reason
