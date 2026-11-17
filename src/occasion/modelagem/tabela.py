"""Tabela de modelagem do Occasion Score.

Uma linha por pessoa × dia de registro × turno. Toda pessoa-dia gera os
quatro turnos, comendo ou não neles, para o denominador não depender do
padrão de refeições. O alvo é pizza com refrigerante (genérico ∪ marca) com
até 1 hora de diferença; um par que cruza a fronteira do turno conta no turno
da pizza. As decisões estão no documento "Tabela de modelagem do Occasion
Score" (17/11/2026).

Grava `data/processed/tabela_modelagem_<edicao>.parquet`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from occasion.ingestao import trends
from occasion.ingestao.pof import caminho_parquet
from occasion.paths import PROCESSED_DIR

EDICOES = {"2008_2009": "2008-09", "2017_2018": "2017-18"}

# Limite superior de cada turno, em horas cheias. A madrugada (0h a 4h) fica
# na manhã: são 2 ocasiões de pizza nessa faixa em cada edição.
TURNOS = {"manha": 10, "almoco": 14, "tarde": 17, "noite": 23}
TIPO_TURNO = pd.CategoricalDtype(list(TURNOS), ordered=True)

# Pizza e refri contam como juntos com até esta diferença de horas.
JANELA_PAR_HORAS = 1

# Faixas usadas na validação da POF (analises/validacao_pof).
FAIXAS_IDADE = {
    "10-19": 19,
    "20-24": 24,
    "25-29": 29,
    "30-34": 34,
    "35-39": 39,
    "40-44": 44,
    "45-59": 59,
    "60+": 200,
}
TIPO_FAIXA = pd.CategoricalDtype(list(FAIXAS_IDADE), ordered=True)

# Salário mínimo na data de referência da renda de cada edição: R$ 415 em
# janeiro de 2009 (Lei 11.709/2008; R$ 465 só a partir de fevereiro) e R$ 954
# em janeiro de 2018 (Decreto 9.255/2017).
SALARIO_MINIMO = {"2008-09": 415.0, "2017-18": 954.0}

DOMINGO, QUARTA, SABADO = 7, 3, 6

COLUNAS = [
    # chaves
    "edicao",
    "id_pessoa",
    "id_domicilio",
    "id_setor",
    "estrato",
    "dia_registro",
    "turno",
    # alvo
    "alvo_pizza_refri",
    # auxiliares (nunca features)
    "teve_pizza",
    "teve_guarana",
    "fora_de_casa",
    "uf",
    # pesos
    "peso_final",
    "fator_dia",
    # features
    "dia_semana",
    "fim_de_semana",
    "quarta",
    "regiao",
    "situacao",
    "trends_pizza_rank_uf",
    "faixa_idade",
    "sexo",
    "renda_pct",
    "renda_sm",
    "n_moradores",
    "tem_crianca",
]

_PESSOA_DIA = ["id_pessoa", "dia_registro"]


def turno_da_hora(hora: pd.Series) -> pd.Series:
    """Turno de cada hora (0 a 23); hora ausente fica sem turno."""
    limites = [-1, *TURNOS.values()]
    return pd.cut(hora.astype("float"), limites, labels=list(TURNOS)).astype(TIPO_TURNO)


def _pares(consumo: pd.DataFrame, bebida: pd.Series) -> pd.DataFrame:
    """Turnos (no horário da pizza) em que a bebida veio até 1 hora da pizza."""
    pizza = consumo.loc[consumo["pizza"], [*_PESSOA_DIA, "hora"]].dropna()
    outra = consumo.loc[bebida, [*_PESSOA_DIA, "hora"]].dropna()
    pares = pizza.merge(outra, on=_PESSOA_DIA, suffixes=("", "_bebida"))
    perto = (pares["hora"].astype(int) - pares["hora_bebida"].astype(int)).abs()
    pares = pares[perto <= JANELA_PAR_HORAS]
    return pares.assign(turno=turno_da_hora(pares["hora"]))[[*_PESSOA_DIA, "turno"]]


def _marca(base: pd.DataFrame, turnos: pd.DataFrame, nome: str) -> pd.DataFrame:
    chave = [*_PESSOA_DIA, "turno"]
    marcados = turnos[chave].drop_duplicates().assign(**{nome: True})
    base = base.merge(marcados, on=chave, how="left", validate="one_to_one")
    base[nome] = base[nome].fillna(False).astype(bool)
    return base


def percentil_ponderado(valor: pd.Series, peso: pd.Series) -> pd.Series:
    """Percentil (0 a 100) de cada valor na distribuição ponderada.

    Usa o ponto médio do peso dos empates: quem tem a renda mediana fica com
    50, e valores iguais recebem o mesmo percentil.
    """
    por_valor = peso.groupby(valor).sum().sort_index()
    meio = por_valor.cumsum() - por_valor / 2
    return valor.map(100 * meio / por_valor.sum())


def _pessoas(consumo: pd.DataFrame, edicao: str) -> pd.DataFrame:
    atributos = [
        "id_pessoa",
        "id_domicilio",
        "id_setor",
        "estrato",
        "uf",
        "regiao",
        "situacao",
        "peso",
        "renda",
        "idade",
        "sexo",
        "n_moradores",
        "tem_crianca",
    ]
    pessoas = consumo[atributos].drop_duplicates("id_pessoa").reset_index(drop=True)
    pessoas["renda_pct"] = percentil_ponderado(pessoas["renda"], pessoas["peso"])
    pessoas["renda_sm"] = pessoas["renda"] / SALARIO_MINIMO[edicao]
    limites = [9, *FAIXAS_IDADE.values()]
    pessoas["faixa_idade"] = pd.cut(
        pessoas["idade"].astype("float"), limites, labels=list(FAIXAS_IDADE)
    ).astype(TIPO_FAIXA)
    return pessoas.rename(columns={"peso": "peso_final"}).drop(columns=["renda", "idade"])


def _fator_dia(dias: pd.DataFrame, pessoas: pd.DataFrame) -> pd.Series:
    """Fator que faz cada dia da semana valer 1/7 do peso das pessoas-dia."""
    peso = dias["id_pessoa"].map(pessoas.set_index("id_pessoa")["peso_final"])
    peso = peso[dias["dia_semana"].notna()]
    parcela = peso.groupby(dias["dia_semana"]).sum() / peso.sum()
    return dias["dia_semana"].map(1 / (7 * parcela)).astype("Float64")


def construir(consumo: pd.DataFrame, posicao_trends: pd.DataFrame) -> pd.DataFrame:
    """Monta a tabela de uma edição a partir do parquet de consumo."""
    if consumo["edicao"].nunique() != 1:
        raise ValueError("o consumo deve ter exatamente uma edição")
    edicao = consumo["edicao"].iloc[0]

    dias = consumo.groupby(_PESSOA_DIA, as_index=False).agg(dia_semana=("dia_semana", "first"))
    pessoas = _pessoas(consumo, edicao)
    dias["fator_dia"] = _fator_dia(dias, pessoas)

    base = dias.merge(pd.Series(list(TURNOS), name="turno", dtype=TIPO_TURNO), how="cross")

    consumo = consumo.assign(turno=turno_da_hora(consumo["hora"]))
    refri = consumo["guarana"] | consumo["refri_generico"] | consumo["refri_marca"]
    base = _marca(base, _pares(consumo, refri), "alvo_pizza_refri")
    base = _marca(base, _pares(consumo, consumo["guarana"]), "teve_guarana")
    base = _marca(base, consumo[consumo["pizza"]], "teve_pizza")

    # Local da pizza no turno (só para a camada de canal): fora se algum item
    # de pizza do turno foi fora de casa; NA se não teve pizza ou o local é
    # desconhecido.
    local = (
        consumo[consumo["pizza"]]
        .groupby([*_PESSOA_DIA, "turno"], observed=True)["fora_de_casa"]
        .max()
        .rename("fora_de_casa")
        .reset_index()
    )
    base = base.merge(local, on=[*_PESSOA_DIA, "turno"], how="left", validate="one_to_one")
    base["fora_de_casa"] = base["fora_de_casa"].astype("boolean")

    base = base.merge(pessoas, on="id_pessoa", how="left", validate="many_to_one")
    base["edicao"] = edicao
    base["fim_de_semana"] = base["dia_semana"].isin([SABADO, DOMINGO]).astype("boolean")
    base["quarta"] = (base["dia_semana"] == QUARTA).astype("boolean")
    base.loc[base["dia_semana"].isna(), ["fim_de_semana", "quarta"]] = pd.NA

    posicao = posicao_trends.loc[posicao_trends["edicao"] == edicao, ["uf", "trends_pizza_rank_uf"]]
    base = base.merge(
        posicao.astype({"uf": base["uf"].dtype}), on="uf", how="left", validate="many_to_one"
    )

    base = base.sort_values([*_PESSOA_DIA, "turno"]).reset_index(drop=True)
    return base[COLUNAS]


def caminho_tabela(edicao: str, destino: Path = PROCESSED_DIR) -> Path:
    return destino / f"tabela_modelagem_{edicao}.parquet"


def processar(
    edicao: str,
    origem: Path = PROCESSED_DIR,
    destino: Path = PROCESSED_DIR,
    arquivo_trends: Path = trends.ARQUIVO,
) -> Path:
    consumo = pd.read_parquet(caminho_parquet(edicao, origem))
    tabela = construir(consumo, trends.posicao_uf(arquivo_trends))
    saida = caminho_tabela(edicao, destino)
    saida.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_parquet(saida, index=False)
    return saida


def resumo(tabela: pd.DataFrame) -> dict[str, float]:
    """Números de conferência: linhas, positivos e taxa ponderada por dia."""
    dias = tabela.groupby(_PESSOA_DIA).agg(
        alvo=("alvo_pizza_refri", "any"), peso=("peso_final", "first")
    )
    return {
        "linhas": len(tabela),
        "pessoas_dia": len(dias),
        "positivos": int(tabela["alvo_pizza_refri"].sum()),
        "pct_dias_positivos": float(100 * np.average(dias["alvo"], weights=dias["peso"])),
    }


def main() -> None:
    for edicao in EDICOES:
        saida = processar(edicao)
        print(f"Tabela {edicao}: {saida} {resumo(pd.read_parquet(saida))}")


if __name__ == "__main__":
    main()
