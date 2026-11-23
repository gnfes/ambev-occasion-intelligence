"""Mapa de Oportunidade: municípios em prioridade, pela API."""

import json
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

import cliente

st.title("Mapa de Oportunidade")
st.caption(
    "319 municípios com 100 mil habitantes ou mais (Censo 2022). Potencial = ocasiões de pizza "
    "com refrigerante esperadas pelo Occasion Score no público de 20 a 44 anos; oferta = "
    "pizzarias ativas no CNPJ (Receita Federal, 11/2026) por 100 mil habitantes. A prioridade é "
    "uma regra documentada aplicada ao território, não uma nova previsão."
)

try:
    df = pd.DataFrame(cliente.mapa())
except cliente.ErroApi as erro:
    st.error(str(erro))
    st.stop()

CORES = {
    "conversao_pizzaria_bairro": [200, 16, 46],
    "conversao_delivery_criadores": [242, 142, 43],
    "awareness_criadores": [78, 121, 167],
    "monitorar": [160, 160, 160],
}
CURTOS = {
    "Conversão com a pizzaria do bairro (combo pizza + Guaraná)": "Pizzaria do bairro",
    "Conversão no delivery e com criadores": "Delivery e criadores",
    "Awareness e associação com criadores": "Awareness",
    "Monitorar": "Monitorar",
}
categorias = df.drop_duplicates("categoria").set_index("categoria")["recomendacao"].to_dict()

f1, f2 = st.columns([3, 2])
escolha = f1.multiselect(
    "Recomendação",
    list(categorias),
    default=list(categorias),
    format_func=lambda c: CURTOS.get(categorias[c], categorias[c]),
)
ufs = f2.multiselect("UF", sorted(df["uf"].unique()))
vis = df[df["categoria"].isin(escolha)]
if ufs:
    vis = vis[vis["uf"].isin(ufs)]

st.caption("Municípios por recomendação (todo o universo):")
resumo = df.groupby("recomendacao", sort=False).agg(
    municipios=("cd_mun", "size"), ocasioes_semana=("ocasioes_semana", "sum")
)
cols = st.columns(len(resumo))
for col, (nome, linha) in zip(cols, resumo.iterrows(), strict=True):
    col.metric(CURTOS.get(nome, nome), linha["municipios"], help=nome)

vis = vis.assign(
    cor=vis["categoria"].map(CORES),
    raio=(vis["ocasioes_semana"] ** 0.5) * 90,
    ocasioes_fmt=vis["ocasioes_semana"].map(lambda v: f"{v:,.0f}".replace(",", ".")),
)
# Contorno das UFs (malha do IBGE, qualidade mínima), para o mapa não depender de tiles.
UFS_GEOJSON = Path(__file__).parents[2] / "data" / "referencia" / "ufs_ibge_minima.geojson"
contorno = json.loads(UFS_GEOJSON.read_text(encoding="utf-8"))

st.pydeck_chart(
    pdk.Deck(
        layers=[
            pdk.Layer(
                "GeoJsonLayer",
                contorno,
                stroked=True,
                filled=True,
                get_fill_color=[240, 240, 236],
                get_line_color=[150, 150, 150],
                line_width_min_pixels=1,
            ),
            pdk.Layer(
                "ScatterplotLayer",
                vis,
                get_position=["lon", "lat"],
                get_fill_color="cor",
                get_radius="raio",
                radius_min_pixels=3,
                opacity=0.75,
                pickable=True,
            ),
        ],
        initial_view_state=pdk.ViewState(latitude=-15.5, longitude=-52, zoom=3.4),
        tooltip={
            "html": "<b>{prioridade}º · {municipio} ({uf})</b><br/>{recomendacao}<br/>"
            "{ocasioes_fmt} ocasiões por semana · {ocasioes_por_mil} por mil<br/>"
            "{pizzarias_100k} pizzarias por 100 mil hab.",
            "style": {"maxWidth": "320px"},
        },
        map_style=None,
    ),
    height=520,
)
st.caption("Tamanho do ponto: ocasiões esperadas por semana. Cor: recomendação.")

st.subheader("Ranking")
st.dataframe(
    vis[
        [
            "prioridade",
            "municipio",
            "uf",
            "recomendacao",
            "canal",
            "ocasioes_semana",
            "ocasioes_por_mil",
            "pizzarias_100k",
            "pct_classe_bd",
            "perto_do_corte",
            "justificativa",
        ]
    ].rename(
        columns={
            "ocasioes_semana": "ocasiões/semana",
            "ocasioes_por_mil": "ocasiões por mil (20-44)",
            "pizzarias_100k": "pizzarias por 100 mil",
            "pct_classe_bd": "% classes B-D",
            "perto_do_corte": "perto do corte",
        }
    ),
    hide_index=True,
    use_container_width=True,
    column_config={
        "ocasiões/semana": st.column_config.NumberColumn(format="%d"),
        "justificativa": st.column_config.TextColumn(width="large"),
    },
)
