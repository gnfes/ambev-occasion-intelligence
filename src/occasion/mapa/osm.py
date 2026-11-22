"""Pizzarias do OpenStreetMap por município.

A lista vem de uma consulta ao Overpass API (consulta em `CONSULTA`): todo
elemento com `cuisine` contendo "pizza", mais restaurantes, lanchonetes,
cafés e bares com "pizz" no nome. O servidor do Overpass recusa conexões do
ambiente de nuvem do projeto, então o JSON é baixado à parte e salvo em
`data/raw/osm/pizzarias_osm_br.json` (a data da base OSM fica no próprio
arquivo, em `osm3s.timestamp_osm_base`).

Cada ponto entra no município cujo polígono o contém (malha de distritos do
Censo 2022, dissolvida por município).

Limitação: a cobertura do OSM no Brasil é desigual e tende a ser melhor nas
cidades grandes; não há fonte pública que dê o total real de pizzarias por
município para corrigir a contagem.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from occasion.mapa.censo import CENSO_DIR
from occasion.paths import PROCESSED_DIR, RAW_DIR, REFERENCIA_DIR

OSM_DIR = RAW_DIR / "osm"
ARQUIVO_OSM = OSM_DIR / "pizzarias_osm_br.json"
# Cópia filtrada e versionada (dados © colaboradores do OpenStreetMap, licença ODbL),
# para o mapa rodar sem refazer a consulta.
ARQUIVO_REFERENCIA = REFERENCIA_DIR / "pizzarias_osm_br.csv"
ARQUIVO_MALHA = CENSO_DIR / "BR_distritos_CD2022.gpkg"
ARQUIVO_MUNICIPIOS_GEO = PROCESSED_DIR / "municipios_geo.parquet"

CONSULTA = (
    '[out:json][timeout:900];area["ISO3166-1"="BR"][admin_level=2]->.br;'
    '(nwr["cuisine"~"pizza",i](area.br);'
    'nwr["name"~"pizz",i]["amenity"~"restaurant|fast_food|cafe|bar|pub"](area.br););'
    "out center tags;"
)
AMENIDADES = {"restaurant", "fast_food", "cafe", "bar", "pub", "food_court"}


def carregar(caminho: Path = ARQUIVO_OSM) -> tuple[pd.DataFrame, str]:
    """Pontos das pizzarias e a data da base OSM (do JSON bruto ou da cópia versionada)."""
    if caminho == ARQUIVO_OSM and not caminho.exists():
        df = pd.read_csv(ARQUIVO_REFERENCIA, dtype={"osm_id": str, "timestamp_osm_base": str})
        data = df["timestamp_osm_base"].dropna()
        return df.drop(columns="timestamp_osm_base"), data.iloc[0] if len(data) else ""
    with open(caminho, encoding="utf-8") as f:
        dados = json.load(f)
    linhas = []
    for el in dados["elements"]:
        tags = el.get("tags", {})
        lat = el.get("lat", el.get("center", {}).get("lat"))
        lon = el.get("lon", el.get("center", {}).get("lon"))
        if lat is None or lon is None:
            continue
        linhas.append(
            {
                "osm_id": f"{el['type']}/{el['id']}",
                "lat": lat,
                "lon": lon,
                "nome": tags.get("name"),
                "amenity": tags.get("amenity"),
                "shop": tags.get("shop"),
                "cuisine": tags.get("cuisine"),
            }
        )
    df = pd.DataFrame(linhas).drop_duplicates("osm_id")
    return filtrar(df), dados.get("osm3s", {}).get("timestamp_osm_base", "")


def filtrar(df: pd.DataFrame) -> pd.DataFrame:
    """Fica com estabelecimentos de comida que servem pizza.

    `cuisine` com "pizza" conta em qualquer amenity de comida (ou sem
    amenity, como `shop` de comida pronta); nome com "pizz" só conta em
    restaurante, lanchonete, café ou bar.
    """
    cozinha = df["cuisine"].fillna("").str.contains("pizza", case=False)
    nome = df["nome"].fillna("").str.contains("pizz", case=False)
    comida = df["amenity"].isin(AMENIDADES)
    sem_amenity = df["amenity"].isna()
    return df[(cozinha & (comida | sem_amenity)) | (nome & comida)].reset_index(drop=True)


def municipios_geo(malha: Path = ARQUIVO_MALHA):
    """Polígonos dos municípios (dissolve dos distritos), em WGS84, com cache."""
    import geopandas as gpd  # grupo opcional "mapa"

    if ARQUIVO_MUNICIPIOS_GEO.exists():
        return gpd.read_parquet(ARQUIVO_MUNICIPIOS_GEO)
    dist = gpd.read_file(malha, columns=["CD_MUN"])
    mun = dist.dissolve("CD_MUN").reset_index().to_crs(4326)
    mun["CD_MUN"] = mun["CD_MUN"].astype(int)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    mun.to_parquet(ARQUIVO_MUNICIPIOS_GEO)
    return mun


def por_municipio(pontos: pd.DataFrame, municipios) -> pd.DataFrame:
    """Pontos com o código do município que os contém (fora do Brasil: sem código)."""
    import geopandas as gpd  # grupo opcional "mapa"

    gdf = gpd.GeoDataFrame(
        pontos, geometry=gpd.points_from_xy(pontos["lon"], pontos["lat"]), crs=4326
    )
    junto = gpd.sjoin(gdf, municipios[["CD_MUN", "geometry"]], how="left", predicate="within")
    junto = junto.drop_duplicates("osm_id")
    return pd.DataFrame(junto.drop(columns=["geometry", "index_right"]))


def contagem(pontos_mun: pd.DataFrame) -> pd.Series:
    return pontos_mun.dropna(subset=["CD_MUN"]).groupby("CD_MUN").size().rename("pizzarias_osm")


def salvar_referencia(caminho: Path = ARQUIVO_OSM) -> Path:
    """Grava a cópia filtrada e versionada a partir do JSON bruto."""
    if not caminho.exists():
        raise FileNotFoundError(caminho)
    df, data = carregar(caminho)
    df.assign(timestamp_osm_base=data).to_csv(ARQUIVO_REFERENCIA, index=False)
    return ARQUIVO_REFERENCIA
