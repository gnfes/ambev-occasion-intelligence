"""Decision Engine: regras isoladas e de ponta a ponta com o modelo versionado."""

import numpy as np
import pytest

from occasion.decisao import engine as e
from occasion.modelagem import final


@pytest.fixture(scope="module")
def modelo():
    return final.carregar()


C = e.Categoria


@pytest.mark.parametrize(
    ("faixa", "turno", "classe", "melhor", "presenca", "lacuna", "canal", "esperado"),
    [
        ("baixa", "almoco", "C", "alta", True, True, None, (1, C.SEM_ATIVACAO)),
        # Faixa alta: baixa presença vence a lacuna (decisão de 27/11).
        ("alta", "noite", "B", "alta", True, True, "delivery", (2, C.AWARENESS)),
        ("alta", "noite", "C", "alta", False, True, "delivery", (3, C.CONVERSAO_ASSOCIACAO)),
        ("alta", "noite", "C", "alta", False, False, "delivery", (4, C.ATIVACAO_DELIVERY)),
        ("alta", "noite", "C", "alta", False, False, "pizzaria", (4, C.ACAO_PIZZARIA)),
        ("alta", "noite", "C", "alta", False, False, "em_casa", (4, C.COMBO_EM_CASA)),
        ("alta", "almoco", "E", "alta", False, False, None, (4, C.COMBO_EM_CASA)),
        ("media", "noite", "A", "alta", True, False, None, (5, C.CRIAR_FREQUENCIA)),
        ("baixa", "noite", "E", "alta", False, False, None, (5, C.CRIAR_FREQUENCIA)),
        ("media", "noite", "D", "media", False, False, None, (6, C.AWARENESS)),
        ("media", "noite", "A", "media", True, False, None, (6, C.AWARENESS)),
        ("media", "noite", "A", "media", False, True, None, (6, C.AWARENESS)),
        ("media", "noite", "A", "media", False, False, None, (7, C.SEM_ATIVACAO)),
        ("media", "noite", "E", "media", False, False, None, (7, C.SEM_ATIVACAO)),
        ("baixa", "noite", "C", "media", True, True, None, (7, C.SEM_ATIVACAO)),
    ],
)
def test_regras_em_ordem(faixa, turno, classe, melhor, presenca, lacuna, canal, esperado):
    assert e.regra(faixa, turno, classe, melhor, presenca, lacuna, canal) == esperado


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


TABELA = {
    "brasil": {"refri_geral": {"pct": 25.3}, "em_casa": {"pct": 65.9}},
    "segmentos": [
        {
            "dimensao": d,
            "valor": v,
            "refri_geral": {"pct": g, "ic95": [g - 4, g + 4], "n": 900},
            "pizza": {"pct": p, "ic95": [p - 8, p + 8], "n": 120},
            "lacuna_pp": {"valor": round(g - p, 1), "ic95": [lo, hi]},
            "lacuna": lac,
            "baixa_presenca": baixa,
        }
        for d, v, g, p, lo, hi, lac, baixa in [
            ("regiao", "Nordeste", 30.2, 16.3, 5.8, 20.7, True, False),
            ("regiao", "Sul", 17.6, 21.9, -19.0, 7.4, False, True),
            ("tipo_area", "capital", 20.9, 11.6, 1.7, 15.5, True, False),
            ("classe", "B", 14.2, 21.8, -25.4, 7.1, False, True),
        ]
    ],
}


def test_sabado_a_noite_muda_de_categoria_com_o_contexto(modelo):
    base = {
        "turno": "noite",
        "dia_semana": 6,
        "faixa_idade": "25-29",
        "sexo": "homem",
        "renda_sm": 8,
    }
    casos = {
        ("Sul", "capital", None): (2, C.AWARENESS),
        ("Nordeste", "interior_urbano", "delivery"): (3, C.CONVERSAO_ASSOCIACAO),
        ("Sudeste", "interior_urbano", "delivery"): (4, C.ATIVACAO_DELIVERY),
        ("Sudeste", "interior_urbano", "pizzaria"): (4, C.ACAO_PIZZARIA),
        ("Sudeste", "interior_urbano", None): (4, C.COMBO_EM_CASA),
    }
    for (regiao, area, canal), esperado in casos.items():
        c = e.Contexto(regiao=regiao, tipo_area=area, canal=canal, **base)
        r = e.decidir(c, modelo, tabela_associacao=TABELA)
        assert r.classification == "alta"
        assert (r.rule, r.recommendation_category) == esperado, (regiao, area, canal)


def test_lacuna_aparece_na_justificativa_e_nas_fontes(modelo):
    c = e.Contexto("noite", 6, "Nordeste", "capital", "25-29", "homem", 8, canal="delivery")
    r = e.decidir(c, modelo, tabela_associacao=TABELA)
    assert r.rule == 3
    assert r.recommendation_channel == "app de delivery"
    assert r.recommendation_reason.startswith("Regra 3.")
    assert "No Nordeste, o Guaraná está em 30,2%" in r.recommendation_reason
    assert {s["segmento"] for s in r.association["lacuna"]} == {
        "região = Nordeste",
        "tipo de área = capital",
    }
    assert "associacao" in {f["id"] for f in r.sources}


def test_sem_tabela_de_associacao_o_canal_decide(modelo):
    c = e.Contexto("noite", 6, "Sul", "capital", "25-29", "homem", 8, canal="delivery")
    r = e.decidir(c, modelo, tabela_associacao={})
    assert (r.rule, r.recommendation_category) == (4, C.ATIVACAO_DELIVERY)
    assert r.association["disponivel"] is False


def test_sem_canal_cita_consumo_em_casa(modelo):
    c = e.Contexto("noite", 6, "Sudeste", "interior_urbano", "25-29", "homem", 8)
    r = e.decidir(c, modelo, tabela_associacao=TABELA)
    assert r.recommendation_category == C.COMBO_EM_CASA
    assert "65,9% das ocasiões são em casa" in r.recommendation_reason
    assert "em_casa_66" in {f["id"] for f in r.sources}


def test_sem_tabela_nao_cita_fonte_de_em_casa(modelo):
    c = e.Contexto("noite", 6, "Sudeste", "interior_urbano", "25-29", "homem", 8)
    r = e.decidir(c, modelo, tabela_associacao={})
    assert r.recommendation_category == C.COMBO_EM_CASA
    assert "em_casa_66" not in {f["id"] for f in r.sources}


def test_fora_de_20_a_44_nao_usa_associacao(modelo):
    # A associação foi medida só em 20 a 44 anos; o Sul não vira awareness para 60+.
    c = e.Contexto("noite", 6, "Sul", "capital", "60+", "homem", 8, canal="delivery")
    r = e.decidir(c, modelo, tabela_associacao=TABELA)
    assert r.association["disponivel"] is False
    assert r.association["lacuna"] == r.association["baixa_presenca"] == []
    assert r.rule != 2


def test_texto_do_segmento_por_sexo():
    assert e._segmento_texto("sexo", "homem") == "entre homens"
    assert e._segmento_texto("sexo", "mulher") == "entre mulheres"


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
    r = e.decidir(c, modelo, tabela_associacao=TABELA)
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


def test_regra_5_age_na_melhor_noite_de_semana(modelo):
    # A frequência se cria numa noite de segunda a quinta; a mensagem acompanha esse momento.
    c = e.Contexto("noite", 2, "Nordeste", "interior_urbano", "35-39", "mulher", 3)
    r = e.decidir(c, modelo, tabela_associacao=TABELA)
    assert r.rule == 5
    nds = r.activation_plan["quando"]["noite_de_semana"]
    assert nds["dia_semana"] in (1, 2, 3, 4)
    assert r.activation_plan["quando"]["dia_semana"] == nds["dia_semana"]
    assert "meio da semana" in r.messages[0]
    assert f"A ação vai para {nds['dia']} à noite" in r.recommendation_reason


def test_regra_7_na_faixa_media_nao_diz_score_baixo(modelo, monkeypatch):
    # Força o melhor momento fora da faixa alta para isolar a regra 7.
    falso = {"dia_semana": 6, "dia": "sábado", "turno": "noite", "occasion_score": 0.5}
    monkeypatch.setattr(e, "melhor_momento", lambda m, c, *_: {**falso, "classification": "media"})
    monkeypatch.setattr(modelo, "faixa", lambda s: np.array(["media"] * len(s)))
    c = e.Contexto("noite", 2, "Sudeste", "interior_urbano", "25-29", "homem", 30)
    r = e.decidir(c, modelo, tabela_associacao=TABELA)
    assert (r.rule, r.classification) == (7, "media")
    assert "fora das classes B a D" in r.recommendation_reason


def test_sem_intervalo_nao_cita_a_fonte_do_intervalo(modelo):
    c = e.Contexto("noite", 6, "Sudeste", "interior_urbano", "25-29", "homem", 8)
    r = e.decidir(c, modelo, tabela_associacao=TABELA, incerteza=None)
    assert "intervalo" not in {f["id"] for f in r.sources}
    assert all("O intervalo mede" not in x for x in r.limitations)
