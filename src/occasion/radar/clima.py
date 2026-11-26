"""Clima diário das 27 capitais (Open-Meteo), agregado por semana.

O Radar usa o clima da capital como clima da UF. Isso é uma hipótese: em UFs
grandes (AM, PA, MT, MG, BA) o interior pode ter outro regime de chuva. A
capital foi escolhida porque concentra a maior parte das buscas e das
ocasiões estimadas no Mapa de Oportunidade.

Fontes:
- histórico: Open-Meteo Historical Weather API (reanálise ERA5/ERA5-Land,
  https://open-meteo.com/en/docs/historical-weather-api), desde 2016 para ter
  uma climatologia de 10 anos por semana do ano;
- semanas recentes e próximos 16 dias: Open-Meteo Forecast API
  (https://open-meteo.com/en/docs), com `past_days` para cobrir o atraso de
  ~5 dias do arquivo histórico.

Licença: dados Open-Meteo sob CC BY 4.0. Coordenadas das sedes municipais
arredondadas a 0,01 grau (a grade do Open-Meteo tem ~10 km).

A coleta fica versionada em `data/referencia/radar_clima_diario.csv`.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from occasion.paths import REFERENCIA_DIR

ARQUIVO = REFERENCIA_DIR / "radar_clima_diario.csv"
INICIO_HISTORICO = "2016-01-01"
VARIAVEIS = ("temperature_2m_max", "temperature_2m_mean", "precipitation_sum")

# Sigla -> (capital, latitude, longitude) da sede municipal.
CAPITAIS = {
    "RO": ("Porto Velho", -8.76, -63.90), "AC": ("Rio Branco", -9.97, -67.81),
    "AM": ("Manaus", -3.12, -60.02), "RR": ("Boa Vista", 2.82, -60.67),
    "PA": ("Belém", -1.46, -48.49), "AP": ("Macapá", 0.03, -51.07),
    "TO": ("Palmas", -10.18, -48.33), "MA": ("São Luís", -2.53, -44.30),
    "PI": ("Teresina", -5.09, -42.80), "CE": ("Fortaleza", -3.73, -38.53),
    "RN": ("Natal", -5.79, -35.21), "PB": ("João Pessoa", -7.12, -34.86),
    "PE": ("Recife", -8.05, -34.88), "AL": ("Maceió", -9.67, -35.74),
    "SE": ("Aracaju", -10.91, -37.07), "BA": ("Salvador", -12.97, -38.50),
    "MG": ("Belo Horizonte", -19.92, -43.94), "ES": ("Vitória", -20.32, -40.34),
    "RJ": ("Rio de Janeiro", -22.91, -43.17), "SP": ("São Paulo", -23.55, -46.63),
    "PR": ("Curitiba", -25.43, -49.27), "SC": ("Florianópolis", -27.60, -48.55),
    "RS": ("Porto Alegre", -30.03, -51.23), "MS": ("Campo Grande", -20.47, -54.62),
    "MT": ("Cuiabá", -15.60, -56.10), "GO": ("Goiânia", -16.68, -49.25),
    "DF": ("Brasília", -15.79, -47.88),
}  # fmt: skip

URL_HISTORICO = "https://archive-api.open-meteo.com/v1/archive"
URL_PREVISAO = "https://api.open-meteo.com/v1/forecast"


def _pedir(cliente, url: str, params: dict, tentativas: int = 8) -> pd.DataFrame:
    """Um pedido por capital, com novas tentativas em 429 (limite por IP)."""
    import time

    import httpx

    partes = []
    for sigla, (_, lat, lon) in CAPITAIS.items():
        consulta = {
            "latitude": lat,
            "longitude": lon,
            "daily": ",".join(VARIAVEIS),
            "timezone": "America/Sao_Paulo",
        } | params
        for tentativa in range(tentativas):
            try:
                resposta = cliente.get(url, params=consulta, timeout=120)
            except httpx.TransportError:
                if tentativa == tentativas - 1:
                    raise
                time.sleep(10 * (tentativa + 1))
                continue
            if resposta.status_code != 429 or tentativa == tentativas - 1:
                break
            time.sleep(10 * (tentativa + 1))
        resposta.raise_for_status()
        diario = pd.DataFrame(resposta.json()["daily"]).rename(columns={"time": "data"})
        diario.insert(1, "sigla", sigla)
        partes.append(diario)
        time.sleep(1)
    return pd.concat(partes, ignore_index=True)


def baixar(destino: Path = ARQUIVO, hoje: dt.date | None = None) -> Path:
    """Baixa histórico desde 2016, as últimas semanas e a previsão de 16 dias."""
    import httpx

    hoje = hoje or dt.date.today()
    fim_historico = hoje - dt.timedelta(days=7)
    with httpx.Client() as cliente:
        historico = _pedir(
            cliente,
            URL_HISTORICO,
            {"start_date": INICIO_HISTORICO, "end_date": fim_historico.isoformat()},
        )
        recente = _pedir(cliente, URL_PREVISAO, {"past_days": 14, "forecast_days": 16})
    historico["origem"] = "historico"
    recente["origem"] = ["previsao" if d > hoje.isoformat() else "recente" for d in recente["data"]]
    recente = recente[recente["data"] > fim_historico.isoformat()]
    saida = pd.concat([historico, recente], ignore_index=True)
    saida["coletado_em"] = hoje.isoformat()
    destino.parent.mkdir(parents=True, exist_ok=True)
    saida.to_csv(destino, index=False)
    return destino


def ler(caminho: Path = ARQUIVO) -> pd.DataFrame:
    return pd.read_csv(caminho, parse_dates=["data"])


def main() -> None:
    print(baixar())


if __name__ == "__main__":
    main()
