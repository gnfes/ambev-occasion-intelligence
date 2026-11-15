"""Testes com os microdados reais da POF.

Os dados não ficam no git. Para rodar: `uv run python -m occasion.ingestao.pof`
baixa os microdados do IBGE e gera os parquets em data/processed/.
"""

import numpy as np
import pandas as pd
import pytest

from occasion.ingestao import codigos
from occasion.ingestao.layout import layout_sas
from occasion.ingestao.pof import COLUNAS, _programa_sas, caminho_parquet
from occasion.paths import RAW_DIR

pytestmark = pytest.mark.dados

# Números oficiais publicados pelo IBGE para o bloco de consumo alimentar pessoal
# (moradores de 10 anos ou mais com dois dias de registro).
ESPERADO = {
    "2008_2009": {"linhas": 735_814, "pessoas": 34_003, "pizza_dias": 2.24, "pizza_refri": 57.6},
    "2017_2018": {"linhas": 1_175_390, "pessoas": 46_164, "pizza_dias": 2.62, "pizza_refri": 43.4},
}


def _carregar(edicao: str) -> pd.DataFrame:
    caminho = caminho_parquet(edicao)
    if not caminho.exists():
        pytest.skip(f"{caminho.name} não existe; rode a ingestão antes")
    return pd.read_parquet(caminho)


@pytest.fixture(scope="module", params=list(ESPERADO))
def edicao(request):
    return request.param, _carregar(request.param)


def _ocasioes(df: pd.DataFrame) -> pd.DataFrame:
    """Ocasião = pessoa x dia de registro x hora (x refeição, onde existe)."""
    chave = ["id_pessoa", "dia_registro", "hora"]
    if df["refeicao"].notna().any():
        chave.append("refeicao")
    df = df.assign(refri=df["refri_generico"] | df["refri_marca"])
    return df.groupby(chave, dropna=False).agg(
        pizza=("pizza", "max"), refri=("refri", "max"), peso=("peso", "first")
    )


def test_colunas_harmonizadas(edicao):
    _, df = edicao
    assert list(df.columns) == COLUNAS


def test_linhas_e_pessoas_batem_com_o_ibge(edicao):
    nome, df = edicao
    assert len(df) == ESPERADO[nome]["linhas"]
    assert df["id_pessoa"].nunique() == ESPERADO[nome]["pessoas"]


def test_pesos_positivos_e_constantes_por_pessoa(edicao):
    _, df = edicao
    assert (df["peso"] > 0).all()
    assert (df.groupby("id_pessoa")["peso"].nunique() == 1).all()


def test_campos_basicos_preenchidos(edicao):
    _, df = edicao
    for coluna in ["uf", "regiao", "situacao", "dia_registro", "dia_semana", "hora", "idade", "sexo"]:
        assert df[coluna].notna().all(), coluna
    assert df["hora"].between(0, 23).all()
    assert df["dia_semana"].between(1, 7).all()
    assert set(df["dia_registro"].unique()) == {1, 2}


def test_codigos_de_pizza_presentes(edicao):
    _, df = edicao
    presentes = set(df.loc[df["pizza"], "cod_alimento"])
    assert {"8500903", "8500914", "8500915", "8500916", "8500917"} <= presentes
    assert presentes <= set(codigos.PIZZA)
    assert not (df["pizza"] & df["pizza_doce"]).any()


def test_pizza_em_dias_pessoa_ponderados(edicao):
    nome, df = edicao
    dias = df.groupby(["id_pessoa", "dia_registro"]).agg(pizza=("pizza", "max"), peso=("peso", "first"))
    taxa = 100 * np.average(dias["pizza"], weights=dias["peso"])
    assert taxa == pytest.approx(ESPERADO[nome]["pizza_dias"], abs=0.01)


def test_pizza_com_refrigerante_na_mesma_ocasiao(edicao):
    nome, df = edicao
    com_pizza = _ocasioes(df).query("pizza")
    taxa = 100 * np.average(com_pizza["refri"], weights=com_pizza["peso"])
    assert taxa == pytest.approx(ESPERADO[nome]["pizza_refri"], abs=0.1)


def test_2017_lista_generica_reproduz_32_7():
    df = _carregar("2017_2018")
    df = df.assign(refri_marca=False)  # só a lista proposta
    com_pizza = _ocasioes(df).query("pizza")
    taxa = 100 * np.average(com_pizza["refri"], weights=com_pizza["peso"])
    assert taxa == pytest.approx(32.7, abs=0.1)


def test_2008_tem_dia_da_semana_mas_nao_tipo_de_refeicao():
    df = _carregar("2008_2009")
    assert df["dia_semana"].notna().all()
    assert df["refeicao"].isna().all()
    assert df["fora_de_casa"].notna().all()


def test_layout_2017_bate_com_posicoes_validadas():
    sas = _programa_sas("2017_2018", RAW_DIR)
    if not sas.exists():
        pytest.skip("leitor SAS de 2017-18 não baixado")
    campos = {c.nome: c.colspec for c in layout_sas(sas, "CONSUMO_ALIMENTAR")}
    assert campos["UF"] == (0, 2)
    assert campos["COD_UPA"] == (7, 16)
    assert campos["QUADRO"] == (21, 23)
    assert campos["V9001"] == (31, 38)
    assert campos["V9015"] == (38, 41)
    assert campos["V9017"] == (43, 44)
    assert campos["V9018"] == (44, 45)
    assert campos["PESO_FINAL"] == (503, 518)
    assert campos["RENDA_TOTAL"] == (518, 528)
    assert campos["DIA_SEMANA"] == (528, 543)
    assert campos["DIA_ATIPICO"] == (543, 544)
    morador = {c.nome: c.colspec for c in layout_sas(sas, "MORADOR.txt")}
    assert morador["V0403"] == (32, 35)  # idade
    assert morador["V0404"] == (35, 36)  # sexo
