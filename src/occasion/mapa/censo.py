"""Perfil dos municípios a partir do Censo Demográfico 2022 (IBGE).

Para aplicar o Occasion Score a um município, o modelo precisa saber quem mora
lá nas mesmas features do treino: faixa de idade, sexo, tipo de área (capital,
resto da região metropolitana, interior urbano ou rural), região e percentil
de renda do domicílio. O Censo público não traz tudo isso cruzado por
município, então este módulo monta o perfil em três passos, cada um com a sua
fonte:

1. Idade × sexo por município (agregados por município). O Censo agrupa 30 a
   39 e 40 a 49 anos; a divisão em faixas de 5 anos usa a proporção da UF nos
   microdados da amostra.
2. Parte urbana e rural de cada grupo de idade × sexo (agregados por setor,
   somados pela situação do setor).
3. Renda: o Censo por município só traz a média e a mediana da renda do
   responsável. A distribuição usada é a da UF (microdados, renda do
   domicílio das pessoas de 20 a 44 anos, separada em urbana e rural),
   reescalada pela razão entre a renda média do responsável no município e na
   UF. Que a forma da distribuição do município seja a da UF é hipótese (sem
   dado). A renda vira percentil na distribuição nacional do Censo, porque o
   modelo usa o percentil (a renda da POF inclui renda não monetária e não é
   comparável em reais).

Os arquivos ficam em `data/raw/censo_2022/` e são baixados do FTP do IBGE.

    uv run --group mapa python -m occasion.mapa.censo
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from occasion.ingestao.download import baixar_arquivo
from occasion.paths import PROCESSED_DIR, RAW_DIR

CENSO_DIR = RAW_DIR / "censo_2022"
MICRO_DIR = CENSO_DIR / "microdados"
ARQUIVO_PERFIL = PROCESSED_DIR / "perfil_municipios.parquet"
ARQUIVO_CDF = PROCESSED_DIR / "renda_censo_cdf.parquet"

FTP = "https://ftp.ibge.gov.br/Censos/Censo_Demografico_2022"
ARQUIVOS = {
    "Agregados_por_municipios_demografia_BR.csv": (
        f"{FTP}/Agregados_por_Setores_Censitarios/Agregados_por_Municipio_csv/"
        "Agregados_por_municipios_demografia_BR.zip"
    ),
    "Agregados_por_municipios_basico_BR.csv": (
        f"{FTP}/Agregados_por_Setores_Censitarios/Agregados_por_Municipio_csv/"
        "Agregados_por_municipios_basico_BR_20260520.zip"
    ),
    "Agregados_por_municipios_renda_responsavel_BR.csv": (
        f"{FTP}/Agregados_por_Setores_Censitarios_Rendimento_do_Responsavel/"
        "Agregados_por_municipios_renda_responsavel_BR_20260508_csv.zip"
    ),
    "Agregados_por_setores_demografia_BR.csv": (
        f"{FTP}/Agregados_por_Setores_Censitarios/Agregados_por_Setor_csv/"
        "Agregados_por_setores_demografia_BR.zip"
    ),
    "Agregados_por_setores_basico_BR.csv": (
        f"{FTP}/Agregados_por_Setores_Censitarios/Agregados_por_Setor_csv/"
        "Agregados_por_setores_basico_BR_20260520.zip"
    ),
    "Municípios_BR_Recortes_Territoriais_2022.xlsx": (
        f"{FTP}/Microdados_e_Areas_de_Ponderacao/Documentacao/"
        "Divis%C3%A3o%20territorial/Munic%C3%ADpios_BR_Recortes_Territoriais_2022.xlsx"
    ),
    "BR_distritos_CD2022.gpkg": (
        f"{FTP}/Agregados_por_Setores_Censitarios/malha_com_atributos/distritos/gpkg/BR/"
        "BR_distritos_CD2022.gpkg"
    ),
}
MICRODADOS = f"{FTP}/Microdados_e_Areas_de_Ponderacao/Microdados_de_acesso_Publico/csv"
UFS = {
    11: "RO", 12: "AC", 13: "AM", 14: "RR", 15: "PA", 16: "AP", 17: "TO",
    21: "MA", 22: "PI", 23: "CE", 24: "RN", 25: "PB", 26: "PE", 27: "AL", 28: "SE", 29: "BA",
    31: "MG", 32: "ES", 33: "RJ", 35: "SP", 41: "PR", 42: "SC", 43: "RS",
    50: "MS", 51: "MT", 52: "GO", 53: "DF",
}  # fmt: skip
REGIAO_UF = {
    1: "Norte",
    2: "Nordeste",
    3: "Sudeste",
    4: "Sul",
    5: "Centro-Oeste",
}
# Capitais (nome no IBGE) por UF.
CAPITAIS = {
    11: "Porto Velho", 12: "Rio Branco", 13: "Manaus", 14: "Boa Vista", 15: "Belém",
    16: "Macapá", 17: "Palmas", 21: "São Luís", 22: "Teresina", 23: "Fortaleza", 24: "Natal",
    25: "João Pessoa", 26: "Recife", 27: "Maceió", 28: "Aracaju", 29: "Salvador",
    31: "Belo Horizonte", 32: "Vitória", 33: "Rio de Janeiro", 35: "São Paulo", 41: "Curitiba",
    42: "Florianópolis", 43: "Porto Alegre", 50: "Campo Grande", 51: "Cuiabá", 52: "Goiânia",
    53: "Brasília",
}  # fmt: skip
# UFs em que a POF 2017-18 tem estrato "resto da RM" (data/referencia/estratos_pof.csv).
# Nas demais, o entorno da capital entra como interior urbano, como na POF.
UFS_COM_RESTO_RM = {13, 15, 16, 21, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 35, 41, 42, 43, 51, 52}

# Faixas do modelo no público da Ambev (20 a 44 anos).
FAIXAS_ALVO = ["20-24", "25-29", "30-34", "35-39", "40-44"]
# Colunas dos agregados (dicionário de dados do Censo 2022, tema Demografia).
COLS_IDADE = {
    "homem": {"20-24": "V01013", "25-29": "V01014", "30-39": "V01015", "40-49": "V01016"},
    "mulher": {"20-24": "V01024", "25-29": "V01025", "30-39": "V01026", "40-49": "V01027"},
}
# Pontos da distribuição de renda de cada UF (quantis do meio de cada vintil).
N_QUANTIS = 20


# --------------------------------------------------------------------- download


def baixar(destino: Path = CENSO_DIR) -> None:
    """Baixa os agregados, a malha e os microdados (só Pessoas e Domicílios)."""
    destino.mkdir(parents=True, exist_ok=True)
    for nome, url in ARQUIVOS.items():
        if (destino / nome).exists():
            continue
        if url.endswith(".zip"):
            zip_local = baixar_arquivo(url, destino / Path(url).name)
            with zipfile.ZipFile(zip_local) as z:
                z.extract(nome, destino)
            zip_local.unlink()
        else:
            baixar_arquivo(url, destino / nome)
    micro = destino / "microdados"
    micro.mkdir(exist_ok=True)
    for cod, sigla in UFS.items():
        if (micro / f"Pessoas_{cod}_publico.csv").exists():
            continue
        zip_local = baixar_arquivo(f"{MICRODADOS}/{cod}_{sigla}.zip", micro / f"{cod}_{sigla}.zip")
        with zipfile.ZipFile(zip_local) as z:
            for membro in z.namelist():
                if membro.startswith(("Pessoas_", "Domicilios_")):
                    z.extract(membro, micro)
        zip_local.unlink()


def _ler_agregado(caminho: Path) -> pd.DataFrame:
    """Lê um CSV dos agregados: separador `;`, decimal com vírgula, `X` = sigilo."""
    df = pd.read_csv(caminho, sep=";", encoding="latin-1", dtype=str)
    for col in df.columns:
        if col.upper().startswith("V"):
            df[col] = pd.to_numeric(
                df[col].str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
                errors="coerce",
            )
    return df


# ------------------------------------------------------------------ microdados


def _microdados_uf(cod_uf: int, micro: Path = MICRO_DIR) -> pd.DataFrame:
    """Pessoas da amostra com a renda do domicílio (R$ de julho de 2022)."""
    pes = pd.read_csv(
        micro / f"Pessoas_{cod_uf}_publico.csv",
        sep=";",
        usecols=["P0100", "P0110", "P0140", "P0150", "P0180"],
        dtype={"P0100": "int64", "P0110": "float64", "P0140": "int8", "P0150": "int8"},
    )
    dom = pd.read_csv(
        micro / f"Domicilios_{cod_uf}_publico.csv",
        sep=";",
        usecols=["D0100", "D0350"],
        dtype={"D0100": "int64"},
    )
    df = pes.merge(dom, left_on="P0100", right_on="D0100", how="left")
    df["uf"] = cod_uf
    df["urbana"] = df["P0140"] == 1
    df["sexo"] = df["P0150"].map({1: "homem", 2: "mulher"})
    df["idade_cat"] = pd.to_numeric(df["P0180"], errors="coerce")
    df["renda"] = pd.to_numeric(df["D0350"], errors="coerce")
    return df[["uf", "P0110", "urbana", "sexo", "idade_cat", "renda"]].rename(
        columns={"P0110": "peso"}
    )


# Categorias de P0180: 05 = 20 a 24 anos, ..., 09 = 40 a 44, 10 = 45 a 49.
CAT_FAIXA = {5: "20-24", 6: "25-29", 7: "30-34", 8: "35-39", 9: "40-44", 10: "45-49"}


def resumo_microdados(micro: Path = MICRO_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Três tabelas tiradas dos microdados, UF por UF.

    - `divisao`: por UF e sexo, a parte de 30-34 em 30-39 e de 40-44 em 40-49.
    - `quantis`: por UF e situação, os quantis da renda do domicílio das
      pessoas de 20 a 44 anos (`N_QUANTIS` pontos) e a renda média.
    - `cdf`: a distribuição nacional da renda do domicílio por pessoa de 10
      anos ou mais (pontos para interpolar o percentil), como na POF.
    """
    divisao, quantis, nacional = [], [], []
    for cod in UFS:
        df = _microdados_uf(cod, micro)
        faixa = df["idade_cat"].map(CAT_FAIXA)
        for sexo, g in df.assign(faixa=faixa).groupby("sexo"):
            peso = g.groupby("faixa")["peso"].sum()
            divisao.append(
                {
                    "uf": cod,
                    "sexo": sexo,
                    "p_30_34": peso.get("30-34", 0) / (peso.get("30-34", 0) + peso.get("35-39", 0)),
                    "p_40_44": peso.get("40-44", 0) / (peso.get("40-44", 0) + peso.get("45-49", 0)),
                }
            )
        alvo = df[df["idade_cat"].between(5, 9) & df["renda"].notna()]
        for urbana, g in alvo.groupby("urbana"):
            q = _quantis_ponderados(g["renda"].to_numpy(), g["peso"].to_numpy())
            linha = {
                "uf": cod,
                "urbana": urbana,
                "media_renda": np.average(g["renda"], weights=g["peso"]),
            }
            linha |= {f"q{i:02d}": v for i, v in enumerate(q)}
            quantis.append(linha)
        dez_mais = df[(df["idade_cat"] >= 3) & df["renda"].notna()]
        nacional.append(dez_mais[["renda", "peso"]])
    cdf = _cdf(pd.concat(nacional, ignore_index=True))
    return pd.DataFrame(divisao), pd.DataFrame(quantis), cdf


def _quantis_ponderados(valor: np.ndarray, peso: np.ndarray, n: int = N_QUANTIS) -> np.ndarray:
    ordem = np.argsort(valor)
    v, w = valor[ordem], peso[ordem]
    acum = (np.cumsum(w) - w / 2) / w.sum()
    alvos = (np.arange(n) + 0.5) / n
    return np.interp(alvos, acum, v)


def _cdf(df: pd.DataFrame) -> pd.DataFrame:
    """Percentil (0 a 100) de cada valor de renda, ponto médio dos empates (como na POF)."""
    por_valor = df.groupby("renda")["peso"].sum().sort_index()
    meio = por_valor.cumsum() - por_valor / 2
    return pd.DataFrame(
        {"renda": por_valor.index.to_numpy(), "pct": (100 * meio / por_valor.sum()).to_numpy()}
    )


def percentil_renda(renda: np.ndarray, cdf: pd.DataFrame) -> np.ndarray:
    return np.interp(renda, cdf["renda"].to_numpy(), cdf["pct"].to_numpy())


# ------------------------------------------------------------------ território


def tipo_area_urbana(recortes: pd.DataFrame) -> pd.Series:
    """Tipo de área da parte urbana de cada município, na definição da POF.

    Capital; resto da região metropolitana da capital (só nas UFs em que a
    POF 2017-18 tem esse estrato); senão, interior urbano.
    """
    rec = recortes.assign(uf=recortes["CD_MUN"] // 100000)
    capital = rec.apply(lambda r: CAPITAIS.get(r["uf"]) == r["NM_MUN"], axis=1)
    if capital.sum() != len(CAPITAIS):
        raise ValueError(f"esperava {len(CAPITAIS)} capitais, achei {capital.sum()}")
    rm_capital = rec.loc[capital].set_index("uf")["COD_CATMETROPOL"]
    mesma_rm = rec["COD_CATMETROPOL"].notna() & (
        rec["COD_CATMETROPOL"] == rec["uf"].map(rm_capital)
    )
    tipo = np.where(
        capital,
        "capital",
        np.where(mesma_rm & rec["uf"].isin(UFS_COM_RESTO_RM), "resto_rm", "interior_urbano"),
    )
    return pd.Series(tipo, index=rec["CD_MUN"], name="tipo_area_urbana")


def _parte_urbana(pasta: Path) -> pd.DataFrame:
    """Parte urbana de cada grupo de idade × sexo do município (soma dos setores)."""
    cols = [c for s in COLS_IDADE.values() for c in s.values()]
    dem = _ler_agregado(pasta / "Agregados_por_setores_demografia_BR.csv")
    dem = dem.rename(columns={"CD_setor": "CD_SETOR"})[["CD_SETOR", *cols]]
    bas = pd.read_csv(
        pasta / "Agregados_por_setores_basico_BR.csv",
        sep=";",
        encoding="latin-1",
        usecols=["CD_SETOR", "CD_SIT", "CD_MUN"],
        dtype=str,
    )
    df = dem.merge(bas, on="CD_SETOR", how="inner")
    # Situação 1 a 3 = urbana (dicionário dos agregados por setor).
    df["urbana"] = df["CD_SIT"].isin(["1", "2", "3"])
    soma = df.groupby(["CD_MUN", "urbana"])[cols].sum().unstack("urbana", fill_value=0)
    urb = soma.xs(True, axis=1, level="urbana") / (
        soma.xs(True, axis=1, level="urbana") + soma.xs(False, axis=1, level="urbana")
    )
    urb.index = urb.index.astype(int)
    return urb


def perfil_municipios(
    pasta: Path = CENSO_DIR,
    divisao: pd.DataFrame | None = None,
    quantis: pd.DataFrame | None = None,
    cdf: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Uma linha por município × situação × sexo × faixa × ponto de renda.

    `pessoas` é a população estimada da célula (cada ponto de renda leva
    1/`N_QUANTIS` do grupo); as features ficam prontas para o modelo.
    """
    if divisao is None or quantis is None or cdf is None:
        divisao, quantis, cdf = resumo_microdados(pasta / "microdados")
    dem = _ler_agregado(pasta / "Agregados_por_municipios_demografia_BR.csv")
    dem["CD_MUN"] = dem["CD_MUN"].astype(int)
    bas = _ler_agregado(pasta / "Agregados_por_municipios_basico_BR.csv")
    bas["CD_MUN"] = bas["CD_MUN"].astype(int)
    renda = _ler_agregado(pasta / "Agregados_por_municipios_renda_responsavel_BR.csv")
    renda["CD_MUN"] = renda["CD_MUN"].astype(int)
    recortes = pd.read_excel(pasta / "Municípios_BR_Recortes_Territoriais_2022.xlsx")
    tipo_urb = tipo_area_urbana(recortes)
    urb = _parte_urbana(pasta)

    # Razão de renda média do responsável: município / UF (média ponderada pelos responsáveis).
    renda["uf"] = renda["CD_MUN"] // 100000
    renda["massa"] = renda["V06004"] * renda["V06001"]
    media_uf = renda.groupby("uf")["massa"].sum() / renda.groupby("uf")["V06001"].sum()
    renda["razao_renda"] = renda["V06004"] / renda["uf"].map(media_uf)
    razao = renda.set_index("CD_MUN")["razao_renda"]

    div = divisao.set_index(["uf", "sexo"])
    q = quantis.set_index(["uf", "urbana"])
    qcols = [f"q{i:02d}" for i in range(N_QUANTIS)]
    linhas = []
    for _, m in dem.iterrows():
        mun = int(m["CD_MUN"])
        uf = mun // 100000
        for sexo, cols in COLS_IDADE.items():
            d = div.loc[(uf, sexo)]
            pop = {
                "20-24": m[cols["20-24"]],
                "25-29": m[cols["25-29"]],
                "30-34": m[cols["30-39"]] * d["p_30_34"],
                "35-39": m[cols["30-39"]] * (1 - d["p_30_34"]),
                "40-44": m[cols["40-49"]] * d["p_40_44"],
            }
            fonte = {
                "20-24": "20-24",
                "25-29": "25-29",
                "30-34": "30-39",
                "35-39": "30-39",
                "40-44": "40-49",
            }
            for faixa, n in pop.items():
                if not n or np.isnan(n):
                    continue
                col_fonte = cols[fonte[faixa]]
                p_urb = urb.at[mun, col_fonte] if mun in urb.index else np.nan
                if np.isnan(p_urb):
                    p_urb = 1.0
                for urbana, parte in ((True, p_urb), (False, 1 - p_urb)):
                    if parte <= 0:
                        continue
                    linhas.append((mun, uf, urbana, sexo, faixa, n * parte))
    cel = pd.DataFrame(
        linhas, columns=["CD_MUN", "uf", "urbana", "sexo", "faixa_idade", "pessoas_grupo"]
    )
    cel["tipo_area"] = np.where(cel["urbana"], cel["CD_MUN"].map(tipo_urb), "rural")
    cel["regiao"] = (cel["uf"] // 10).map(REGIAO_UF)
    cel["razao_renda"] = cel["CD_MUN"].map(razao).fillna(1.0)

    # Quantis da UF na situação, reescalados pelo município, viram percentil nacional.
    base = q.reindex(pd.MultiIndex.from_frame(cel[["uf", "urbana"]]))[qcols].to_numpy()
    renda_mun = base * cel["razao_renda"].to_numpy()[:, None]
    pct = percentil_renda(renda_mun.ravel(), cdf).reshape(renda_mun.shape)
    longo = cel.loc[cel.index.repeat(N_QUANTIS)].reset_index(drop=True)
    longo["renda_pct"] = pct.ravel()
    longo["renda_mensal"] = renda_mun.ravel()
    longo["pessoas"] = longo["pessoas_grupo"] / N_QUANTIS
    nomes = bas.set_index("CD_MUN")["NM_MUN"]
    longo["municipio"] = longo["CD_MUN"].map(nomes)
    return longo.drop(columns=["pessoas_grupo"])


def populacao(pasta: Path = CENSO_DIR) -> pd.DataFrame:
    """População total, nome, UF e renda média do responsável por município."""
    bas = _ler_agregado(pasta / "Agregados_por_municipios_basico_BR.csv")
    renda = _ler_agregado(pasta / "Agregados_por_municipios_renda_responsavel_BR.csv")
    df = bas[["CD_MUN", "NM_MUN", "CD_UF", "v0001"]].merge(
        renda[["CD_MUN", "V06004"]], on="CD_MUN", how="left"
    )
    df = df.rename(columns={"v0001": "populacao", "V06004": "renda_media_responsavel"})
    df["CD_MUN"] = df["CD_MUN"].astype(int)
    df["uf"] = df["CD_UF"].astype(int).map(UFS)
    return df.drop(columns=["CD_UF"])


def main() -> None:
    baixar()
    divisao, quantis, cdf = resumo_microdados()
    perfil = perfil_municipios(divisao=divisao, quantis=quantis, cdf=cdf)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    perfil.to_parquet(ARQUIVO_PERFIL, index=False)
    cdf.to_parquet(ARQUIVO_CDF, index=False)
    buf = io.StringIO()
    perfil.groupby("tipo_area")["pessoas"].sum().to_string(buf)
    print(
        f"{perfil['CD_MUN'].nunique()} municípios, {perfil['pessoas'].sum():,.0f} pessoas de 20 a 44"
    )
    print(buf.getvalue())


if __name__ == "__main__":
    main()
