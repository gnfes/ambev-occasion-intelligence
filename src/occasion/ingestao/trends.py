"""Interesse por "pizza" no Google Trends por UF, na janela de campo de cada POF.

O Google Trends devolve uma amostra, então dois pedidos iguais podem diferir
um pouco. Por isso o resultado fica versionado em
`data/referencia/trends_pizza_uf.csv` e a tabela de modelagem lê esse arquivo;
`baixar()` só é usado para refazer a coleta.

Janelas de campo (fontes):
- 2008-09: coleta de 19/05/2008 a maio de 2009 (Informe DPE abr-jun/2008 do
  IBGE). Usamos até 31/05/2009.
- 2017-18: o IBGE iniciou a coleta em 26/06/2017 (blog do IBRE/FGV) e o
  período de 12 meses vai até 09/07/2018 (Informe IPECE 162, nov/2019).

Valor 0 no Trends quer dizer volume de buscas insuficiente, não interesse
zero; essas UFs ficam sem posição.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from occasion.paths import REFERENCIA_DIR

ARQUIVO = REFERENCIA_DIR / "trends_pizza_uf.csv"

JANELAS = {
    "2008-09": "2008-05-19 2009-05-31",
    "2017-18": "2017-06-26 2018-07-09",
}

# Sigla -> código IBGE da UF.
UF_CODIGOS = {
    "RO": 11, "AC": 12, "AM": 13, "RR": 14, "PA": 15, "AP": 16, "TO": 17,
    "MA": 21, "PI": 22, "CE": 23, "RN": 24, "PB": 25, "PE": 26, "AL": 27, "SE": 28, "BA": 29,
    "MG": 31, "ES": 32, "RJ": 33, "SP": 35,
    "PR": 41, "SC": 42, "RS": 43,
    "MS": 50, "MT": 51, "GO": 52, "DF": 53,
}  # fmt: skip


def baixar(destino: Path = ARQUIVO, pausa: float = 8.0) -> Path:
    """Refaz a coleta no Google Trends e grava o CSV de referência."""
    from pytrends.request import TrendReq

    trends = TrendReq(hl="pt-BR", tz=180)
    partes = []
    for edicao, janela in JANELAS.items():
        trends.build_payload(["pizza"], geo="BR", timeframe=janela)
        uf = trends.interest_by_region(resolution="REGION", inc_low_vol=True, inc_geo_code=True)
        partes.append(
            pd.DataFrame(
                {
                    "edicao": edicao,
                    "janela": janela,
                    "sigla": uf["geoCode"].str.removeprefix("BR-").to_numpy(),
                    "pizza": uf["pizza"].to_numpy(),
                }
            )
        )
        time.sleep(pausa)
    saida = pd.concat(partes, ignore_index=True)
    saida["uf"] = saida["sigla"].map(UF_CODIGOS)
    desconhecidas = saida.loc[saida["uf"].isna(), "sigla"].unique().tolist()
    if desconhecidas:
        raise ValueError(f"códigos de região desconhecidos no Trends: {desconhecidas}")
    saida["uf"] = saida["uf"].astype(int)
    destino.parent.mkdir(parents=True, exist_ok=True)
    saida[["edicao", "janela", "uf", "sigla", "pizza"]].to_csv(destino, index=False)
    return destino


def posicao_uf(caminho: Path = ARQUIVO) -> pd.DataFrame:
    """Posição de cada UF no interesse por pizza dentro da edição.

    1 = maior interesse. Empates ficam com a menor posição. UFs com valor 0
    (volume insuficiente) ficam com NA.
    """
    df = pd.read_csv(caminho)
    valido = df["pizza"].where(df["pizza"] > 0)
    df["trends_pizza_rank_uf"] = (
        valido.groupby(df["edicao"]).rank(ascending=False, method="min").astype("Int8")
    )
    return df[["edicao", "uf", "trends_pizza_rank_uf"]]


def main() -> None:
    print(baixar())


if __name__ == "__main__":
    main()
