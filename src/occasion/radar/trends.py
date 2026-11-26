"""Série semanal do Google Trends por UF (últimos 5 anos).

Cada UF é pedida sozinha ("today 5-y", semanas de domingo a sábado). O índice
vai de 0 a 100 dentro de cada pedido, então só se compara uma UF com ela mesma:
o nível de SP não é comparável ao de PE. Buscas medem interesse, não consumo,
e por isso o Radar nunca as usa como target do Occasion Score.

Termos:
- "pizza": sinal principal. Inclui buscas de receita e de delivery; não dá para
  separar as duas intenções no Trends.
- "guaraná": sinal de marca, lido à parte. A combinação "pizza guaraná" não
  tem volume (no Brasil, 2016-2026, "guaraná" fica perto de 1 quando "pizza"
  está perto de 50 no mesmo pedido), então não existe série da ocasião.

Os pedidos usam a categoria "Comida e bebida" (cat=71) do Trends. Sem ela, a
série de "pizza" pega buscas que não têm a ver com comer: em fevereiro de 2025
a mediana das UFs chegou a 4 vezes o normal por causa de "fantasia de pizza"
no Carnaval (consulta em alta no Trends, BR, 10/02 a 02/03/2025). Com a
categoria, esse pico some e ficam os picos de calendário (Ano-Novo, Carnaval,
Dia da Pizza em 10 de julho).

O Trends devolve uma amostra: dois pedidos iguais podem diferir um pouco. A
coleta fica versionada em `data/referencia/radar_trends_semanal.csv` e o resto
do Radar lê esse arquivo; `baixar()` só refaz a coleta.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from occasion.ingestao.trends import UF_CODIGOS
from occasion.paths import REFERENCIA_DIR

ARQUIVO = REFERENCIA_DIR / "radar_trends_semanal.csv"
TERMOS = ("pizza", "guaraná")
JANELA = "today 5-y"
CATEGORIA = 71  # Comida e bebida


def _pedido(trends, termo: str, geo: str, tentativas: int = 5) -> pd.DataFrame:
    for tentativa in range(tentativas):
        try:
            trends.build_payload([termo], cat=CATEGORIA, geo=geo, timeframe=JANELA)
            return trends.interest_over_time()
        except Exception as erro:  # o pytrends levanta TooManyRequestsError e afins
            if tentativa == tentativas - 1:
                raise
            espera = 30 * (tentativa + 1)
            print(f"{geo} {termo}: {erro.__class__.__name__}, nova tentativa em {espera}s")
            time.sleep(espera)
    raise RuntimeError("inalcançável")


def baixar(destino: Path = ARQUIVO, pausa: float = 6.0, verify: str | bool = True) -> Path:
    """Refaz a coleta (27 UFs x termos) e grava o CSV longo."""
    from pytrends.request import TrendReq

    trends = TrendReq(hl="pt-BR", tz=180, requests_args={"verify": verify})
    partes = []
    for sigla in UF_CODIGOS:
        for termo in TERMOS:
            serie = _pedido(trends, termo, f"BR-{sigla}")
            if serie.empty:
                print(f"{sigla} {termo}: sem dados")
                continue
            partes.append(
                pd.DataFrame(
                    {
                        "semana": serie.index.date,
                        "sigla": sigla,
                        "termo": termo,
                        "indice": serie[termo].to_numpy(),
                        "parcial": serie["isPartial"].astype(bool).to_numpy(),
                    }
                )
            )
            print(f"{sigla} {termo}: {len(serie)} semanas")
            time.sleep(pausa)
    saida = pd.concat(partes, ignore_index=True)
    saida.insert(2, "uf", saida["sigla"].map(UF_CODIGOS).astype(int))
    destino.parent.mkdir(parents=True, exist_ok=True)
    saida.to_csv(destino, index=False)
    return destino


def ler(caminho: Path = ARQUIVO) -> pd.DataFrame:
    df = pd.read_csv(caminho, parse_dates=["semana"])
    return df


def main() -> None:
    import os

    verify: str | bool = os.environ.get("REQUESTS_CA_BUNDLE") or True
    print(baixar(verify=verify))


if __name__ == "__main__":
    main()
