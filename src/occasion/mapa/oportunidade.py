"""Mapa de Oportunidade: prioriza municípios para ativação de Pizza + Guaraná.

Cruza o Occasion Score aplicado ao território (`score.py`, Censo 2022) com a
oferta de pizzarias no OpenStreetMap (`osm.py`). A prioridade é uma regra
documentada, a mesma lógica do Decision Engine levada ao município, e não uma
previsão nova: sem histórico de ações e resultados por cidade, nada aqui
afirma que uma ativação vende mais que outra.

Universo: municípios com 100 mil habitantes ou mais (Censo 2022). Abaixo
disso, a contagem do OSM é pequena demais para separar oferta alta de baixa.

Regras (a primeira que vale decide):

1. Potencial alto e oferta alta → conversão com a pizzaria do bairro.
2. Potencial alto e oferta baixa → conversão no delivery e com criadores.
3. Potencial médio e público majoritariamente B a D → awareness com criadores.
4. Resto → monitorar.

- Potencial: taxa de ocasiões esperadas por mil pessoas de 20 a 44 anos por
  semana. Alto = taxa igual ou acima da média nacional; médio = de metade da
  média até a média; baixo = abaixo de metade.
- Oferta: pizzarias do OSM por 100 mil habitantes. Alta = igual ou acima da
  mediana do universo. A régua é relativa porque a cobertura do OSM varia; o
  número absoluto não é o total real de pizzarias.
- Ordem dentro de cada categoria: volume de ocasiões esperadas por semana.
  Volume, e não taxa, porque a taxa põe no topo as capitais de renda alta
  (classe A), e as classes B a D somam 86% das ocasiões (POF 2017-18).

    uv run --group mapa python -m occasion.mapa.oportunidade
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd

from occasion.mapa import censo, osm, score
from occasion.paths import PROCESSED_DIR

POPULACAO_MINIMA = 100_000
FRACAO_MEDIO = 0.5  # potencial médio a partir de metade da taxa nacional
MAIORIA_BD = 50.0  # % do público nas classes B a D
# Município "perto do corte": taxa a menos de 15% de um limite de potencial.
MARGEM_CORTE = 0.15
# Previsto ÷ observado por tipo de área na validação temporal (treino 2008-09,
# teste 2017-18; avaliacao_occasion_score.md, tabela "Segmentos"). Só para a
# análise de sensibilidade: o ranking principal usa o modelo sem ajuste.
PREVISTO_SOBRE_OBS = {"capital": 1.266, "resto_rm": 0.682, "interior_urbano": 0.931, "rural": 1.271}
ARQUIVO_RANKING = PROCESSED_DIR / "mapa_oportunidade.csv"
ARQUIVO_RELATORIO = PROCESSED_DIR / "mapa_oportunidade.md"
ARQUIVO_PONTOS = PROCESSED_DIR / "mapa_oportunidade.json"


class Categoria(StrEnum):
    PIZZARIA = "conversao_pizzaria_bairro"
    DELIVERY = "conversao_delivery_criadores"
    AWARENESS = "awareness_criadores"
    MONITORAR = "monitorar"


NOMES = {
    Categoria.PIZZARIA: "Conversão com a pizzaria do bairro (combo pizza + Guaraná)",
    Categoria.DELIVERY: "Conversão no delivery e com criadores",
    Categoria.AWARENESS: "Awareness e associação com criadores",
    Categoria.MONITORAR: "Monitorar",
}
CANAL = {
    Categoria.PIZZARIA: "pizzarias do bairro (parceria local), depois criadores locais",
    Categoria.DELIVERY: "apps de delivery e criadores de conteúdo",
    Categoria.AWARENESS: "criadores de conteúdo do segmento",
    Categoria.MONITORAR: None,
}


def nivel_potencial(taxa: pd.Series, taxa_nacional: float) -> pd.Series:
    return pd.Series(
        np.where(
            taxa >= taxa_nacional,
            "alto",
            np.where(taxa >= FRACAO_MEDIO * taxa_nacional, "medio", "baixo"),
        ),
        index=taxa.index,
    )


def regra(potencial: str, oferta_alta: bool, pct_classe_bd: float) -> tuple[int, Categoria]:
    """As 4 regras, em ordem. Separada para ser testada sem dados."""
    if potencial == "alto" and oferta_alta:
        return 1, Categoria.PIZZARIA
    if potencial == "alto":
        return 2, Categoria.DELIVERY
    if potencial == "medio" and pct_classe_bd >= MAIORIA_BD:
        return 3, Categoria.AWARENESS
    return 4, Categoria.MONITORAR


def _justificativa(r: pd.Series, taxa_nacional: float, mediana_oferta: float) -> str:
    taxa = f"{r['ocasioes_por_mil']:.0f} ocasiões por mil pessoas de 20 a 44 por semana"
    rel = r["ocasioes_por_mil"] / taxa_nacional
    oferta = f"{r['pizzarias_100k']:.1f} pizzarias no OSM por 100 mil habitantes"
    textos = {
        1: f"Potencial alto ({taxa}, {rel:.1f}× a média nacional) e oferta alta ({oferta}, "
        f"mediana {mediana_oferta:.1f}): a ocasião já acontece e há pizzarias para parceria.",
        2: f"Potencial alto ({taxa}, {rel:.1f}× a média nacional), mas oferta baixa no OSM "
        f"({oferta}, mediana {mediana_oferta:.1f}): converter onde o pedido acontece "
        "(delivery) e com criadores. A oferta baixa pode ser falta de mapeamento no OSM.",
        3: f"Potencial médio ({taxa}, {rel:.1f}× a média nacional) com "
        f"{r['pct_classe_bd']:.0f}% do público nas classes B a D: construir associação.",
        4: f"Potencial {r['potencial']} ({taxa}, {rel:.1f}× a média nacional)"
        + (
            f" e só {r['pct_classe_bd']:.0f}% do público nas classes B a D."
            if r["potencial"] == "medio"
            else "."
        ),
    }
    return textos[int(r["regra"])]


def priorizar(municipios: pd.DataFrame, taxa_nacional: float) -> pd.DataFrame:
    """Aplica as regras ao universo e ordena por categoria e volume."""
    df = municipios[municipios["populacao"] >= POPULACAO_MINIMA].copy()
    df["pizzarias_100k"] = 1e5 * df["pizzarias_osm"] / df["populacao"]
    mediana_oferta = float(df["pizzarias_100k"].median())
    df["potencial"] = nivel_potencial(df["ocasioes_por_mil"], taxa_nacional)
    df["oferta_alta"] = df["pizzarias_100k"] >= mediana_oferta
    rel = df["ocasioes_por_mil"] / taxa_nacional
    df["perto_do_corte"] = (rel - 1).abs().lt(MARGEM_CORTE) | (rel / FRACAO_MEDIO - 1).abs().lt(
        MARGEM_CORTE
    )
    regras = [
        regra(p, o, b)
        for p, o, b in zip(df["potencial"], df["oferta_alta"], df["pct_classe_bd"], strict=True)
    ]
    df["regra"] = [n for n, _ in regras]
    df["categoria"] = [c.value for _, c in regras]
    df["recomendacao"] = [NOMES[c] for _, c in regras]
    df["canal"] = [CANAL[c] for _, c in regras]
    df["justificativa"] = df.apply(_justificativa, axis=1, args=(taxa_nacional, mediana_oferta))
    df = df.sort_values(["regra", "ocasioes_semana"], ascending=[True, False])
    df["prioridade"] = np.arange(1, len(df) + 1)
    df.attrs["mediana_oferta"] = mediana_oferta
    return df.reset_index(drop=True)


def montar() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Junta perfil, score e OSM. Devolve (todos os municípios, universo priorizado, meta)."""
    perfil = pd.read_parquet(censo.ARQUIVO_PERFIL)
    mun = score.pontuar(perfil)
    pop = censo.populacao()
    mun = mun.merge(pop[["CD_MUN", "populacao", "renda_media_responsavel"]], on="CD_MUN")
    pontos, data_osm = osm.carregar()
    pontos_mun = osm.por_municipio(pontos, osm.municipios_geo())
    mun["pizzarias_osm"] = mun["CD_MUN"].map(osm.contagem(pontos_mun)).fillna(0).astype(int)
    taxa_nacional = 1000 * mun["ocasioes_semana"].sum() / mun["publico_20_44"].sum()
    ranking = priorizar(mun, taxa_nacional)
    meta = {
        "taxa_nacional": taxa_nacional,
        "data_osm": data_osm,
        "pizzarias_osm_total": len(pontos),
        "pizzarias_osm_fora": int(pontos_mun["CD_MUN"].isna().sum()),
        "mediana_oferta": ranking.attrs["mediana_oferta"],
    }
    return mun, ranking, meta


# ------------------------------------------------------------------- relatório


def _md(df: pd.DataFrame) -> str:
    cab = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    linhas = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([cab, sep, *linhas])


def checagens(mun: pd.DataFrame, ranking: pd.DataFrame) -> dict:
    """Números que testam o mapa contra dados que o modelo não viu."""
    uni = mun[mun["populacao"] >= POPULACAO_MINIMA].copy()
    uni["pizzarias_100k"] = 1e5 * uni["pizzarias_osm"] / uni["populacao"]
    spearman = uni[["ocasioes_por_mil", "pizzarias_100k"]].corr(method="spearman").iloc[0, 1]
    por_regiao = (
        mun.groupby("regiao")
        .apply(
            lambda d: pd.Series(
                {
                    "publico_20_44_mi": d["publico_20_44"].sum() / 1e6,
                    "ocasioes_por_mil": 1000
                    * d["ocasioes_semana"].sum()
                    / d["publico_20_44"].sum(),
                    "pizzarias_100k": 1e5 * d["pizzarias_osm"].sum() / d["populacao"].sum(),
                }
            ),
            include_groups=False,
        )
        .round(1)
        .reset_index()
    )
    rpr = por_regiao[["ocasioes_por_mil", "pizzarias_100k"]].corr(method="spearman").iloc[0, 1]
    return {"spearman_municipios": spearman, "por_regiao": por_regiao, "spearman_regioes": rpr}


def sensibilidade(ranking: pd.DataFrame, taxa_nacional: float) -> dict:
    """Quanto o potencial muda se o viés por tipo de área da validação for corrigido."""
    ajustada = ranking["ocasioes_por_mil"] / ranking["tipo_area"].map(PREVISTO_SOBRE_OBS)
    nivel = nivel_potencial(ajustada, taxa_nacional)
    cruz = pd.crosstab(ranking["potencial"], nivel).reset_index()
    cruz.columns.name = None
    mudam = ranking.loc[nivel != ranking["potencial"], ["municipio", "tipo_area"]]
    spearman = pd.Series(ranking["ocasioes_por_mil"]).corr(ajustada, method="spearman")
    return {"cruzamento": cruz, "mudam": mudam, "spearman": spearman}


def relatorio(mun: pd.DataFrame, ranking: pd.DataFrame, meta: dict) -> str:
    chk = checagens(mun, ranking)
    sens = sensibilidade(ranking, meta["taxa_nacional"])
    tipos_mudam = sens["mudam"]["tipo_area"].value_counts().to_dict()
    resumo = (
        ranking.groupby(["regra", "recomendacao"])
        .agg(
            municipios=("CD_MUN", "size"),
            publico_20_44_mi=("publico_20_44", lambda s: round(s.sum() / 1e6, 1)),
            ocasioes_semana_mil=("ocasioes_semana", lambda s: round(s.sum() / 1e3)),
        )
        .reset_index()
    )
    cols = [
        "prioridade",
        "municipio",
        "uf",
        "recomendacao",
        "ocasioes_semana",
        "ocasioes_por_mil",
        "pizzarias_100k",
        "pct_classe_bd",
    ]
    top = ranking[cols].head(30).copy()
    top["ocasioes_semana"] = top["ocasioes_semana"].round(-2).astype(int)
    for c in ["ocasioes_por_mil", "pizzarias_100k", "pct_classe_bd"]:
        top[c] = top[c].round(1)
    por_cat = []
    for (n, nome), g in ranking.groupby(["regra", "recomendacao"]):
        nomes = ", ".join(
            f"{m} ({u})"
            for m, u in zip(g["municipio"].head(10), g["uf"].map(censo.UFS).head(10), strict=True)
        )
        por_cat.append(f"- **Regra {n}, {nome}:** {nomes}")
    total = mun["publico_20_44"].sum()
    return f"""# Mapa de Oportunidade

Universo: {len(ranking)} municípios com {POPULACAO_MINIMA:,} habitantes ou mais (Censo 2022), com
{ranking["publico_20_44"].sum() / total:.0%} do público de 20 a 44 anos do país. Base do OSM:
{meta["data_osm"]} ({meta["pizzarias_osm_total"]:,} pizzarias no Brasil, {meta["pizzarias_osm_fora"]} fora de
qualquer município).

Taxa nacional esperada: {meta["taxa_nacional"]:.1f} ocasiões de pizza com refrigerante por mil pessoas
de 20 a 44 anos por semana. Mediana da oferta no universo: {meta["mediana_oferta"]:.1f} pizzarias no OSM
por 100 mil habitantes.

## Resumo por categoria

{_md(resumo)}

## Primeiros de cada categoria

{chr(10).join(por_cat)}

## Ranking (30 primeiros)

{_md(top)}

## Checagem contra dado independente

O potencial vem só do modelo (POF) e do Censo; a oferta vem do OSM. Se o
potencial fizer sentido, onde ele é maior deveria haver mais pizzarias.

- Correlação de Spearman entre a taxa esperada e as pizzarias por 100 mil, nos
  {len(ranking)} municípios: {chk["spearman_municipios"]:.2f}.
- Por região (Spearman {chk["spearman_regioes"]:.2f}, só 5 pontos):

{_md(chk["por_regiao"])}

## Sensibilidade ao viés por tipo de área

Na validação temporal, o modelo superestimou as capitais (previsto ÷ observado
1,27) e subestimou o resto das regiões metropolitanas (0,68). Dividindo a taxa
de cada município por esse fator, a ordem quase não muda (Spearman
{sens["spearman"]:.2f}), mas {len(sens["mudam"])} municípios trocam de nível de potencial
({tipos_mudam}). Linhas = potencial do mapa; colunas = potencial ajustado:

{_md(sens["cruzamento"])}

O ranking principal fica sem ajuste: os fatores vêm do modelo da validação
(treinado só em 2008-09), não do modelo final, e o do resto da RM se apoia em
129 ocasiões. {int(ranking["perto_do_corte"].sum())} municípios estão a menos de
{MARGEM_CORTE:.0%} de um corte de potencial e ficam marcados em `perto_do_corte`.

## Limitações

- O modelo foi treinado na POF 2008-09 e 2017-18 e validado de uma edição
  para a outra; aplicá-lo à população de 2022 supõe que o padrão se manteve
  (hipótese, sem dado; a POF 2024-25 ainda não saiu).
- Renda: a distribuição de cada município é a da UF (microdados do Censo),
  reescalada pela renda média do responsável no município. A forma da
  distribuição dentro do município é hipótese (sem dado).
- Idade: o Censo por município agrupa 30 a 39 e 40 a 49 anos; a divisão usa
  a proporção da UF nos microdados.
- OSM: a base tem {meta["pizzarias_osm_total"]:,} pizzarias no Brasil, enquanto só as
  pizzarias independentes eram 33,6 mil em 2024 segundo o iFood (número de
  comunicado do iFood lido em resumo de busca em 11/11/2026; a página é
  bloqueada neste ambiente). O OSM cobre uma parte pequena e desigual da
  oferta, então a oferta é comparada com a mediana, e não usada como total
  real. A correlação fraca com o potencial (acima) pode vir dessa cobertura.
- Guaraná: a fatia de guaraná nas ocasiões é 21% sem diferença entre
  segmentos (POF 2017-18), então não muda a prioridade entre municípios.
"""


def pontos_mapa(ranking: pd.DataFrame, municipios_geo) -> list[dict]:
    """Centroide e atributos de cada município do universo, para o mapa."""
    geo = municipios_geo.set_index("CD_MUN")
    cent = geo.loc[ranking["CD_MUN"], "geometry"].to_crs(5880).centroid.to_crs(4326)
    out = []
    for (_, r), p in zip(ranking.iterrows(), cent, strict=True):
        out.append(
            {
                "cd_mun": int(r["CD_MUN"]),
                "municipio": r["municipio"],
                "uf": censo.UFS[int(r["uf"])],
                "lat": round(p.y, 4),
                "lon": round(p.x, 4),
                "prioridade": int(r["prioridade"]),
                "categoria": r["categoria"],
                "recomendacao": r["recomendacao"],
                "canal": r["canal"] if isinstance(r["canal"], str) else None,
                "ocasioes_semana": round(float(r["ocasioes_semana"])),
                "ocasioes_por_mil": round(float(r["ocasioes_por_mil"]), 1),
                "pizzarias_osm": int(r["pizzarias_osm"]),
                "pizzarias_100k": round(float(r["pizzarias_100k"]), 1),
                "pct_classe_bd": round(float(r["pct_classe_bd"])),
                "populacao": int(r["populacao"]),
                "perto_do_corte": bool(r["perto_do_corte"]),
                "justificativa": r["justificativa"],
            }
        )
    return out


def main(destino: Path = PROCESSED_DIR) -> None:
    mun, ranking, meta = montar()
    destino.mkdir(parents=True, exist_ok=True)
    saida = ranking.copy()
    saida["uf"] = saida["uf"].map(censo.UFS)
    saida.to_csv(destino / ARQUIVO_RANKING.name, index=False)
    (destino / ARQUIVO_RELATORIO.name).write_text(relatorio(mun, ranking, meta), encoding="utf-8")
    pontos = pontos_mapa(ranking, osm.municipios_geo())
    (destino / ARQUIVO_PONTOS.name).write_text(
        json.dumps({"meta": meta, "municipios": pontos}, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(ranking.groupby("recomendacao").size())


if __name__ == "__main__":
    main()
