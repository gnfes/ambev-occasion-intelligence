"""Associação do Guaraná por segmento: critérios e a tabela gravada que o Engine lê."""

import pytest

from occasion.decisao import associacao as a


def _seg(dim, valor, geral_ic, lacuna_ic, n_pizza):
    return {
        "dimensao": dim,
        "valor": valor,
        "refri_geral": {"pct": sum(geral_ic) / 2, "ic95": list(geral_ic), "n": 500},
        "pizza": {"pct": 20.0, "ic95": [10.0, 30.0], "n": n_pizza},
        "lacuna_pp": {"valor": sum(lacuna_ic) / 2, "ic95": list(lacuna_ic)},
    }


@pytest.mark.parametrize(
    ("seg", "lacuna", "baixa"),
    [
        (_seg("regiao", "X", (26, 34), (5.0, 20.0), 147), True, False),
        # Lacuna com poucas ocasiões de pizza não conta.
        (_seg("tipo_area", "rural", (25, 35), (2.0, 30.0), 33), False, False),
        # IC da lacuna cruza o zero.
        (_seg("regiao", "Y", (20, 30), (-15.0, 16.0), 95), False, False),
        # IC do refrigerante em geral inteiro abaixo do IC nacional (22,8 a 27,8).
        (_seg("classe", "B", (9.9, 18.9), (-25.0, 7.0), 65), False, True),
        # Sobrepõe o IC nacional: não é baixa presença.
        (_seg("tipo_area", "capital", (16.9, 25.5), (1.7, 15.5), 163), True, False),
        # Excluído por decisão de 27/11, mesmo passando no critério.
        (_seg("faixa_idade", "30-34", (18, 26), (0.8, 17.6), 82), False, False),
    ],
)
def test_criterios(seg, lacuna, baixa):
    r = a.classificar(seg, brasil_refri_ic95_min=22.8)
    assert (r["lacuna"], r["baixa_presenca"]) == (lacuna, baixa)


def test_sinais_procura_cada_dimensao_do_contexto():
    tabela = {
        "segmentos": [
            {
                **_seg("regiao", "Norte", (24, 43), (12, 39), 63),
                "lacuna": True,
                "baixa_presenca": False,
            },
            {**_seg("classe", "B", (9, 19), (-25, 7), 65), "lacuna": False, "baixa_presenca": True},
        ]
    }
    s = a.sinais(tabela, {"regiao": "Norte", "classe": "B", "sexo": "homem"})
    assert [x["valor"] for x in s["lacuna"]] == ["Norte"]
    assert [x["valor"] for x in s["baixa_presenca"]] == ["B"]
    assert a.sinais(None, {"regiao": "Norte"}) == {"lacuna": [], "baixa_presenca": []}


def test_tabela_gravada_tem_os_segmentos_aprovados():
    tabela = a.carregar()
    assert tabela is not None
    lacuna = {(s["dimensao"], s["valor"]) for s in tabela["segmentos"] if s["lacuna"]}
    baixa = {(s["dimensao"], s["valor"]) for s in tabela["segmentos"] if s["baixa_presenca"]}
    assert lacuna == {("regiao", "Norte"), ("regiao", "Nordeste"), ("tipo_area", "capital")}
    assert baixa == {("regiao", "Sul"), ("classe", "B")}
    assert tabela["brasil"]["pizza"]["n"] == 444
