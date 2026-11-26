"""Feriados nacionais (BrasilAPI), marcados por semana.

A BrasilAPI (https://brasilapi.com.br/docs#tag/Feriados-Nacionais) devolve só
feriados nacionais, incluindo Carnaval e Corpus Christi como pontos
facultativos. Feriados estaduais e municipais ficam de fora: é uma limitação
documentada do Radar.

A coleta fica versionada em `data/referencia/radar_feriados.csv`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from occasion.paths import REFERENCIA_DIR

ARQUIVO = REFERENCIA_DIR / "radar_feriados.csv"
URL = "https://brasilapi.com.br/api/feriados/v1/{ano}"
INICIO = 2016


def baixar(destino: Path = ARQUIVO) -> Path:
    """Baixa de 2016 até o ano que vem (a próxima janela do Radar pode virar o ano)."""
    import datetime as dt

    import httpx

    partes = []
    with httpx.Client() as cliente:
        for ano in range(INICIO, dt.date.today().year + 2):
            resposta = cliente.get(URL.format(ano=ano), timeout=60)
            resposta.raise_for_status()
            partes.append(pd.DataFrame(resposta.json()))
    saida = pd.concat(partes, ignore_index=True).rename(
        columns={"date": "data", "name": "nome", "type": "tipo"}
    )
    destino.parent.mkdir(parents=True, exist_ok=True)
    saida[["data", "nome", "tipo"]].to_csv(destino, index=False)
    return destino


def ler(caminho: Path = ARQUIVO) -> pd.DataFrame:
    return pd.read_csv(caminho, parse_dates=["data"])


def main() -> None:
    print(baixar())


if __name__ == "__main__":
    main()
