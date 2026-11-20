"""Tabela de modelagem: regras com dados sintéticos e números com os reais."""

import numpy as np
import pandas as pd
import pytest

from occasion.ingestao import codigos, trends
from occasion.ingestao.pof import caminho_parquet
from occasion.modelagem.tabela import (
    COLUNAS,
    TURNOS,
    carregar_areas,
    construir,
    percentil_ponderado,
    resumo,
    turno_da_hora,
)

PIZZA, GUARANA, COCA, ARROZ = "8500903", "8200301", "8200102", "6300101"


def _item(pessoa: str, dia: int, hora: int, cod: str, **campos) -> dict:
    base = {
        "edicao": "2017-18",
        "id_pessoa": pessoa,
        "id_domicilio": pessoa[:-3],
        "id_setor": pessoa[:-5],
        "estrato": 3501,
        "uf": 35,
        "regiao": "Sudeste",
        "dia_registro": dia,
        "dia_semana": 6 if dia == 1 else 3,
        "hora": hora,
        "fora_de_casa": False,
        "cod_alimento": cod,
        "peso": 100.0,
        "renda": 954.0,
        "idade": 30,
        "sexo": 1,
        "n_moradores": 2,
        "tem_crianca": False,
    }
    base.update(campos)
    for nome, conjunto in codigos.FLAGS.items():
        base[nome] = cod in conjunto
    return base


A, B = "35000012304102", "35000012304103"


@pytest.fixture
def consumo() -> pd.DataFrame:
    linhas = [
        # A, dia 1 (sábado): pizza 19h + coca 20h -> noite positiva
        _item(A, 1, 19, PIZZA, fora_de_casa=True),
        _item(A, 1, 20, COCA),
        # A, dia 1: pizza 10h + guaraná 11h -> conta na manhã, turno da pizza
        _item(A, 1, 10, PIZZA),
        _item(A, 1, 11, GUARANA),
        # A, dia 2 (quarta): pizza 19h + coca 21h -> 2 horas, não é par
        _item(A, 2, 19, PIZZA),
        _item(A, 2, 21, COCA),
        # B, dia 1: só arroz no almoço e refri sem pizza à noite
        _item(B, 1, 12, ARROZ, peso=300.0, renda=4770.0, idade=19, sexo=2),
        _item(B, 1, 20, COCA, peso=300.0, renda=4770.0, idade=19, sexo=2),
    ]
    return pd.DataFrame(linhas).astype({"hora": "Int8", "dia_semana": "Int8", "uf": "Int8"})


@pytest.fixture
def areas() -> pd.DataFrame:
    return pd.DataFrame(
        {"edicao": ["2017-18"], "uf": [35], "estrato": [3501], "tipo_area": ["capital"]}
    )


@pytest.fixture
def posicao() -> pd.DataFrame:
    return pd.DataFrame(
        {"edicao": ["2017-18", "2008-09"], "uf": [35, 35], "trends_pizza_rank_uf": [4, 1]}
    ).astype({"trends_pizza_rank_uf": "Int8"})


def _linha(tabela: pd.DataFrame, pessoa: str, dia: int, turno: str) -> pd.Series:
    sel = tabela[
        (tabela["id_pessoa"] == pessoa)
        & (tabela["dia_registro"] == dia)
        & (tabela["turno"] == turno)
    ]
    assert len(sel) == 1
    return sel.iloc[0]


def test_turnos_cobrem_as_24_horas():
    turnos = turno_da_hora(pd.Series(range(24)))
    assert turnos.notna().all()
    assert turnos.tolist() == ["manha"] * 11 + ["almoco"] * 4 + ["tarde"] * 3 + ["noite"] * 6


def test_quatro_turnos_por_pessoa_dia(consumo, posicao, areas):
    tabela = construir(consumo, posicao, areas)
    assert list(tabela.columns) == COLUNAS
    assert len(tabela) == 3 * len(TURNOS)
    assert (tabela.groupby(["id_pessoa", "dia_registro"]).size() == 4).all()


def test_alvo_ate_uma_hora_no_turno_da_pizza(consumo, posicao, areas):
    tabela = construir(consumo, posicao, areas)
    assert _linha(tabela, A, 1, "noite")["alvo_pizza_refri"]
    assert _linha(tabela, A, 1, "manha")["alvo_pizza_refri"]
    assert not _linha(tabela, A, 1, "almoco")["alvo_pizza_refri"]
    assert not _linha(tabela, A, 2, "noite")["alvo_pizza_refri"]
    assert _linha(tabela, A, 2, "noite")["teve_pizza"]
    assert not tabela.loc[tabela["id_pessoa"] == B, "alvo_pizza_refri"].any()
    assert tabela["alvo_pizza_refri"].sum() == 2


def test_auxiliares_de_guarana_e_local(consumo, posicao, areas):
    tabela = construir(consumo, posicao, areas)
    assert _linha(tabela, A, 1, "manha")["teve_guarana"]
    assert not _linha(tabela, A, 1, "noite")["teve_guarana"]
    assert _linha(tabela, A, 1, "noite")["fora_de_casa"]
    assert not _linha(tabela, A, 1, "manha")["fora_de_casa"]
    # sem pizza no turno, o local fica vazio
    assert pd.isna(_linha(tabela, B, 1, "noite")["fora_de_casa"])


def test_local_desconhecido_so_vira_falso_sem_duvida(consumo, posicao, areas):
    # A, dia 2, noite: mais uma pizza em casa e outra de local desconhecido
    extra = pd.DataFrame(
        [
            _item(A, 2, 20, PIZZA, fora_de_casa=False),
            _item(A, 2, 20, PIZZA, fora_de_casa=pd.NA),
        ]
    ).astype({"hora": "Int8", "dia_semana": "Int8", "uf": "Int8", "fora_de_casa": "boolean"})
    consumo = pd.concat([consumo.astype({"fora_de_casa": "boolean"}), extra], ignore_index=True)
    tabela = construir(consumo, posicao, areas)
    # dia 2 à noite: pizzas em casa e uma de local desconhecido -> NA
    assert pd.isna(_linha(tabela, A, 2, "noite")["fora_de_casa"])
    # dia 1 à noite: a pizza fora de casa decide
    assert _linha(tabela, A, 1, "noite")["fora_de_casa"]


def test_sem_trends_da_edicao_e_erro(consumo, posicao, areas):
    with pytest.raises(ValueError, match="sem posição do Trends"):
        construir(consumo, posicao[posicao["edicao"] != "2017-18"], areas)


def test_atributos_e_calendario(consumo, posicao, areas):
    tabela = construir(consumo, posicao, areas)
    a = _linha(tabela, A, 1, "noite")
    assert a["dia_semana"] == 6
    assert a["tipo_area"] == "capital"
    assert a["trends_pizza_rank_uf"] == 4
    assert a["renda_sm"] == pytest.approx(1.0)
    assert a["faixa_idade"] == "30-34"
    assert _linha(tabela, B, 1, "noite")["faixa_idade"] == "10-19"
    assert a["peso_final"] == 100.0 and a["id_setor"] == "350000123"


def test_fator_dia_iguala_os_dias_da_semana(consumo, posicao, areas):
    tabela = construir(consumo, posicao, areas)
    dias = tabela.drop_duplicates(["id_pessoa", "dia_registro"])
    peso = dias["peso_final"] * dias["fator_dia"]
    parcela = peso.groupby(dias["dia_semana"]).sum() / peso.sum()
    assert parcela.to_numpy() == pytest.approx([0.5, 0.5])


def test_percentil_ponderado_usa_ponto_medio():
    valor = pd.Series([10.0, 20.0, 20.0, 30.0])
    peso = pd.Series([1.0, 1.0, 1.0, 1.0])
    assert percentil_ponderado(valor, peso).tolist() == pytest.approx([12.5, 50.0, 50.0, 87.5])


def test_posicao_trends_ignora_volume_insuficiente(tmp_path):
    arquivo = tmp_path / "trends.csv"
    pd.DataFrame(
        {
            "edicao": ["2008-09"] * 3 + ["2017-18"],
            "janela": ["x"] * 4,
            "uf": [35, 33, 53, 35],
            "sigla": ["SP", "RJ", "DF", "SP"],
            "pizza": [90, 95, 0, 75],
        }
    ).to_csv(arquivo, index=False)
    posicao = trends.posicao_uf(arquivo).set_index(["edicao", "uf"])["trends_pizza_rank_uf"]
    assert posicao[("2008-09", 33)] == 1
    assert posicao[("2008-09", 35)] == 2
    assert pd.isna(posicao[("2008-09", 53)])
    assert posicao[("2017-18", 35)] == 1


def test_trends_versionado_tem_as_27_ufs_por_edicao():
    df = pd.read_csv(trends.ARQUIVO)
    assert df.groupby("edicao")["uf"].nunique().tolist() == [27, 27]
    assert set(df["uf"]) == set(trends.UF_CODIGOS.values())


# Números conferidos nos microdados em 16/11 (analises/tabela_modelagem).
ESPERADO = {
    "2008_2009": {"linhas": 267_612, "pessoas_dia": 66_903, "positivos": 668, "pct": 1.34},
    "2017_2018": {"linhas": 340_072, "pessoas_dia": 85_018, "positivos": 811, "pct": 1.18},
}


@pytest.mark.dados
@pytest.mark.parametrize("edicao", list(ESPERADO))
def test_tabela_real_reproduz_a_conferencia(edicao):
    caminho = caminho_parquet(edicao)
    if not caminho.exists():
        pytest.skip(f"{caminho.name} não existe; rode a ingestão antes")
    tabela = construir(pd.read_parquet(caminho), trends.posicao_uf())
    numeros = resumo(tabela)
    esperado = ESPERADO[edicao]
    assert numeros["linhas"] == esperado["linhas"]
    assert numeros["pessoas_dia"] == esperado["pessoas_dia"]
    assert numeros["positivos"] == esperado["positivos"]
    assert numeros["pct_dias_positivos"] == pytest.approx(esperado["pct"], abs=0.01)
    # chaves e features sem buracos (Trends só falta em UFs de volume baixo)
    obrigatorias = [c for c in COLUNAS if c not in {"fora_de_casa", "trends_pizza_rank_uf"}]
    assert tabela[obrigatorias].notna().all().all()
    assert (tabela.groupby(["id_pessoa", "dia_registro"]).size() == 4).all()
    # o alvo só existe em turno com pizza
    assert not (tabela["alvo_pizza_refri"] & ~tabela["teve_pizza"]).any()
    dias = tabela.drop_duplicates(["id_pessoa", "dia_registro"])
    parcela = (dias["peso_final"] * dias["fator_dia"]).groupby(dias["dia_semana"]).sum()
    assert np.allclose(parcela / parcela.sum(), 1 / 7)


def test_estrato_sem_tipo_de_area_e_erro(consumo, posicao, areas):
    with pytest.raises(ValueError, match="estratos sem tipo de área"):
        construir(consumo, posicao, areas.assign(estrato=9999))


def test_mapa_de_estratos_versionado_cobre_as_27_ufs():
    areas = carregar_areas()
    assert areas.groupby("edicao")["uf"].nunique().tolist() == [27, 27]
    assert not areas.duplicated(["edicao", "uf", "estrato"]).any()
    assert areas["tipo_area"].notna().all()
