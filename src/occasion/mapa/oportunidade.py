"""Mapa de Oportunidade: prioriza municípios para ativação de Pizza + Guaraná.

Cruza o Occasion Score aplicado ao território (`score.py`, Censo 2022) com a
oferta de pizzarias no CNPJ da Receita (`cnpj.py`); o OpenStreetMap
(`osm.py`) fica como checagem. A prioridade é uma regra
documentada, a mesma lógica do Decision Engine levada ao município, e não uma
previsão nova: sem histórico de ações e resultados por cidade, nada aqui
afirma que uma ativação vende mais que outra.

Universo: municípios com 100 mil habitantes ou mais (Censo 2022), que somam
58% do público de 20 a 44 anos (escopo aprovado do mapa).

Regras (a primeira que vale decide):

1. Potencial alto e oferta alta → conversão com a pizzaria do bairro.
2. Potencial alto e oferta baixa → conversão no delivery e com criadores.
3. Potencial médio e 50% ou mais do público nas classes B a D → awareness com criadores.
4. Resto → monitorar.

- Potencial: taxa de ocasiões esperadas por mil pessoas de 20 a 44 anos por
  semana. Alto = taxa igual ou acima da média nacional; médio = de metade da
  média até a média; baixo = abaixo de metade.
- Oferta: pizzarias ativas no CNPJ por 100 mil habitantes. Alta = igual ou
  acima da mediana do universo. A contagem é um piso (só pizzarias com
  "pizz" no nome fantasia, ver `cnpj.py`), por isso a régua é a mediana e
  não um número absoluto.
- Potencial com correção regional: as ocasiões esperadas do Nordeste e do
  Centro-Oeste são divididas pelo previsto ÷ observado do modelo final na
  região (`vies.py`), as únicas regiões em que o IC95 exclui 1.
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

from occasion.mapa import censo, cnpj, osm, score
from occasion.paths import PROCESSED_DIR

POPULACAO_MINIMA = 100_000
FRACAO_MEDIO = 0.5  # potencial médio a partir de metade da taxa nacional
MAIORIA_BD = 50.0  # % do público nas classes B a D
# Município "perto do corte": taxa a menos de 15% de um limite de potencial.
MARGEM_CORTE = 0.15
# Previsto ÷ observado do modelo final em 2017-18, fora da amostra, com IC95 por
# bootstrap de setores (`vies.py`, data/processed/vies_occasion_score.csv).
# Corrigido no ranking: só as regiões cujo IC95 exclui 1.
CORRECAO_REGIAO = {"Nordeste": 0.675, "Centro-Oeste": 1.421}
# Por tipo de área o IC95 inclui 1 nos três tipos urbanos: só sensibilidade.
PREVISTO_SOBRE_OBS = {"capital": 1.23, "resto_rm": 0.791, "interior_urbano": 0.893, "rural": 1.626}
ARQUIVO_VIES = PROCESSED_DIR / "vies_occasion_score.csv"
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
    oferta = f"{r['pizzarias_100k']:.1f} pizzarias ativas no CNPJ por 100 mil habitantes"
    textos = {
        1: f"Potencial alto ({taxa}, {rel:.1f}× a média nacional) e oferta alta ({oferta}, "
        f"mediana {mediana_oferta:.1f}): a ocasião já acontece e há pizzarias para parceria.",
        2: f"Potencial alto ({taxa}, {rel:.1f}× a média nacional), mas oferta baixa de "
        f"pizzarias ({oferta}, mediana {mediana_oferta:.1f}): converter onde o pedido "
        "acontece (delivery) e com criadores.",
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
    df["pizzarias_100k"] = 1e5 * df["pizzarias_cnpj"] / df["populacao"]
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


def corrigir_regiao(mun: pd.DataFrame) -> pd.DataFrame:
    """Divide o esperado pelo previsto ÷ observado da região (só NE e CO).

    Vale para as ocasiões da semana e para o score de sábado à noite;
    `pct_faixa_alta` (fatia do público na faixa alta do score) fica sem
    correção e é a única coluna do score nessa situação.
    """
    fator = mun["regiao"].map(CORRECAO_REGIAO).fillna(1.0)
    mun = mun.copy()
    mun["ocasioes_semana_sem_correcao"] = mun["ocasioes_semana"]
    mun["ocasioes_semana"] = mun["ocasioes_semana"] / fator
    if "score_sabado_noite" in mun:
        mun["score_sabado_noite"] = mun["score_sabado_noite"] / fator
    mun["ocasioes_por_mil"] = 1000 * mun["ocasioes_semana"] / mun["publico_20_44"]
    return mun


def montar() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Junta perfil, score, CNPJ e OSM. Devolve (todos os municípios, universo priorizado, meta)."""
    perfil = pd.read_parquet(censo.ARQUIVO_PERFIL)
    mun = corrigir_regiao(score.pontuar(perfil))
    pop = censo.populacao()
    mun = mun.merge(pop[["CD_MUN", "populacao", "renda_media_responsavel"]], on="CD_MUN")
    pontos, data_osm = osm.carregar()
    pontos_mun = osm.por_municipio(pontos, osm.municipios_geo())
    mun["pizzarias_osm"] = mun["CD_MUN"].map(osm.contagem(pontos_mun)).fillna(0).astype(int)
    rf = pd.read_csv(cnpj.ARQUIVO, index_col="CD_MUN")
    faltam = mun.loc[
        (mun["populacao"] >= POPULACAO_MINIMA) & ~mun["CD_MUN"].isin(rf.index), "municipio"
    ]
    if len(faltam):
        raise ValueError(f"municípios do universo sem contagem no CNPJ: {list(faltam)}")
    mun["pizzarias_cnpj"] = mun["CD_MUN"].map(rf["pizzarias_cnpj"]).fillna(0).astype(int)
    mun["restaurantes_cnpj"] = mun["CD_MUN"].map(rf["estabelecimentos_cnpj"]).fillna(0).astype(int)
    taxa_nacional = 1000 * mun["ocasioes_semana"].sum() / mun["publico_20_44"].sum()
    ranking = priorizar(mun, taxa_nacional)
    meta = {
        "taxa_nacional": taxa_nacional,
        "data_osm": data_osm,
        "pizzarias_osm_total": len(pontos),
        "pizzarias_osm_fora": int(pontos_mun["CD_MUN"].isna().sum()),
        "mediana_oferta": ranking.attrs["mediana_oferta"],
        "mes_cnpj": str(rf["mes_receita"].iloc[0]),
        "pizzarias_cnpj_total": int(rf["pizzarias_cnpj"].sum()),
    }
    return mun, ranking, meta


# ------------------------------------------------------------------- relatório


def _mil(n: float) -> str:
    """Inteiro com ponto de milhar, como no resto do relatório em português."""
    return f"{n:,.0f}".replace(",", ".")


def _fator(vr: pd.DataFrame, regiao: str) -> str:
    return f"{vr.loc[regiao, 'previsto_sobre_obs']:.2f}".replace(".", ",")


def _ic(vr: pd.DataFrame, regiao: str) -> str:
    baixo, alto = vr.loc[regiao, ["ic95_baixo", "ic95_alto"]]
    return f"{baixo:.2f} a {alto:.2f}".replace(".", ",")


def _md(df: pd.DataFrame) -> str:
    cab = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    linhas = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([cab, sep, *linhas])


def checagens(mun: pd.DataFrame, ranking: pd.DataFrame) -> dict:
    """Números que testam o mapa contra dados que o modelo não viu."""
    uni = mun[mun["populacao"] >= POPULACAO_MINIMA].copy()
    uni["pizzarias_100k"] = 1e5 * uni["pizzarias_cnpj"] / uni["populacao"]
    uni["osm_100k"] = 1e5 * uni["pizzarias_osm"] / uni["populacao"]
    uni["sem_correcao"] = 1000 * uni["ocasioes_semana_sem_correcao"] / uni["publico_20_44"]
    sp = uni[["ocasioes_por_mil", "sem_correcao", "pizzarias_100k", "osm_100k"]].corr(
        method="spearman"
    )
    por_regiao = (
        mun.groupby("regiao")
        .apply(
            lambda d: pd.Series(
                {
                    "publico_20_44_mi": d["publico_20_44"].sum() / 1e6,
                    "ocasioes_por_mil": 1000
                    * d["ocasioes_semana"].sum()
                    / d["publico_20_44"].sum(),
                    "pizzarias_cnpj_100k": 1e5 * d["pizzarias_cnpj"].sum() / d["populacao"].sum(),
                }
            ),
            include_groups=False,
        )
        .round(1)
        .reset_index()
    )
    rpr = por_regiao[["ocasioes_por_mil", "pizzarias_cnpj_100k"]].corr(method="spearman")
    return {
        "spearman_municipios": sp.loc["ocasioes_por_mil", "pizzarias_100k"],
        "spearman_sem_correcao": sp.loc["sem_correcao", "pizzarias_100k"],
        "spearman_osm_cnpj": sp.loc["osm_100k", "pizzarias_100k"],
        "por_regiao": por_regiao,
        "spearman_regioes": rpr.iloc[0, 1],
    }


def sensibilidade(ranking: pd.DataFrame, taxa_nacional: float) -> dict:
    """Quanto o potencial muda se o viés por tipo de área for corrigido no lugar do regional.

    Os fatores por tipo de área foram medidos no modelo sem correção, então
    entram sobre a taxa sem a correção regional (corrigir os dois somaria
    vieses que se sobrepõem).
    """
    sem_correcao = 1000 * ranking["ocasioes_semana_sem_correcao"] / ranking["publico_20_44"]
    ajustada = sem_correcao / ranking["tipo_area"].map(PREVISTO_SOBRE_OBS)
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
    top["uf"] = top["uf"].map(censo.UFS)
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
    vies = pd.read_csv(ARQUIVO_VIES)
    vr = vies[vies["grupo"] == "regiao"].set_index("valor")
    mudam_regiao = ranking.loc[
        ranking["regiao"].isin(list(CORRECAO_REGIAO)), ["ocasioes_semana_sem_correcao"]
    ]
    return f"""# Mapa de Oportunidade

Universo: {len(ranking)} municípios com {_mil(POPULACAO_MINIMA)} habitantes ou mais (Censo 2022), com
{ranking["publico_20_44"].sum() / total:.0%} do público de 20 a 44 anos do país. Oferta: CNPJ da Receita
Federal ({meta["mes_cnpj"]}), {_mil(meta["pizzarias_cnpj_total"])} pizzarias ativas nos municípios ligados ao IBGE pela regra de
`cnpj.py`. Checagem: OpenStreetMap de {meta["data_osm"]} ({_mil(meta["pizzarias_osm_total"])} pizzarias).

Taxa nacional esperada (com a correção regional): {meta["taxa_nacional"]:.1f} ocasiões de pizza com
refrigerante por mil pessoas de 20 a 44 anos por semana. Mediana da oferta no universo:
{meta["mediana_oferta"]:.1f} pizzarias ativas no CNPJ por 100 mil habitantes.

## Resumo por categoria

{_md(resumo)}

## Primeiros de cada categoria

{chr(10).join(por_cat)}

## Ranking (30 primeiros)

{_md(top)}

## Correção regional do potencial

O modelo final, medido fora da amostra em 2017-18 (`vies.py`), subestima o
Nordeste (previsto ÷ observado {CORRECAO_REGIAO["Nordeste"]:.2f}) e superestima o
Centro-Oeste ({CORRECAO_REGIAO["Centro-Oeste"]:.2f}); são as duas regiões em que o IC95
por bootstrap de setores exclui 1. As ocasiões esperadas desses
{len(mudam_regiao)} municípios do universo são divididas pelo fator. Norte ({_fator(vr, "Norte")}),
Sudeste ({_fator(vr, "Sudeste")}) e Sul ({_fator(vr, "Sul")}) ficam sem ajuste porque o IC95 inclui 1.

## Checagem contra dado independente

O potencial vem só do modelo (POF) e do Censo; a oferta vem do CNPJ. Se o
potencial fizer sentido, onde ele é maior deveria haver mais pizzarias.

- Correlação de Spearman entre a taxa esperada e as pizzarias do CNPJ por
  100 mil, nos {len(ranking)} municípios: {chk["spearman_municipios"]:.2f}
  (sem a correção regional: {chk["spearman_sem_correcao"]:.2f}).
- Pizzarias do OSM contra as do CNPJ, por 100 mil: Spearman
  {chk["spearman_osm_cnpj"]:.2f}. O OSM mede sobretudo quanto a cidade foi mapeada
  (ver `osm.py`), por isso saiu da regra.
- Por região (Spearman {chk["spearman_regioes"]:.2f}, só 5 pontos):

{_md(chk["por_regiao"])}

## Sensibilidade ao viés por tipo de área

Por tipo de área o viés do modelo final não é significativo nas áreas urbanas
(IC95 inclui 1: capital {PREVISTO_SOBRE_OBS["capital"]:.2f}, resto da RM
{PREVISTO_SOBRE_OBS["resto_rm"]:.2f}, interior urbano {PREVISTO_SOBRE_OBS["interior_urbano"]:.2f}), então o
ranking não corrige. Se a correção fosse por tipo de área em vez de por
região (sobre a taxa sem correção regional), a ordem teria Spearman
{sens["spearman"]:.2f} com a do mapa e {len(sens["mudam"])} municípios trocariam de nível de potencial
({tipos_mudam}). Linhas = potencial do mapa; colunas = potencial ajustado:

{_md(sens["cruzamento"])}

{int(ranking["perto_do_corte"].sum())} municípios estão a menos de {MARGEM_CORTE:.0%} de um corte de
potencial e ficam marcados em `perto_do_corte`.

## Limitações

- O modelo foi treinado na POF 2008-09 e 2017-18 e validado de uma edição
  para a outra; aplicá-lo à população de 2022 supõe que o padrão se manteve
  (hipótese, sem dado; a POF 2024-25 ainda não saiu).
- A correção regional usa fatores medidos em 2017-18 com {int(vr.loc["Nordeste", "positivos"])} ocasiões no
  Nordeste e {int(vr.loc["Centro-Oeste", "positivos"])} no Centro-Oeste; os intervalos são largos
  (Nordeste {_ic(vr, "Nordeste")}; Centro-Oeste {_ic(vr, "Centro-Oeste")}).
- Renda: a distribuição de cada município é a da UF (microdados do Censo),
  reescalada pela renda média do responsável no município. A forma da
  distribuição dentro do município é hipótese (sem dado).
- Idade: o Censo por município agrupa 30 a 39 e 40 a 49 anos; a divisão usa
  a proporção da UF nos microdados.
- CNPJ: não existe CNAE só de pizzaria. A regra (ativa, CNAE de
  restaurante, lanchonete ou comida para consumo em casa, "pizz" no nome
  fantasia) deixa de fora quem não tem "pizz" no nome, então a contagem é um
  piso: {_mil(meta["pizzarias_cnpj_total"])} contra 33,6 mil pizzarias independentes em 2024
  segundo o iFood (comunicado lido em resumo de busca em 11/11/2026; a
  página é bloqueada neste ambiente). Também fica de fora quem vende sem
  CNPJ, o que pesa mais onde a informalidade é maior (hipótese, sem medida
  por município aqui).
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
                "pizzarias_cnpj": int(r["pizzarias_cnpj"]),
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
