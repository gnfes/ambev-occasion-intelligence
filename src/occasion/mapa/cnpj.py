"""Pizzarias por município no CNPJ aberto da Receita Federal.

O OSM depende de alguém ter mapeado a cidade; o CNPJ lista todos os
estabelecimentos registrados. Não existe CNAE só de pizzaria, então a
identificação é uma regra (explícita, não um dado):

- estabelecimento ativo (situação cadastral 02);
- CNAE principal de restaurante (5611-2/01), lanchonete (5611-2/03) ou
  alimentos preparados para consumo domiciliar (5620-1/04, onde se
  registram muitas pizzarias só de entrega);
- "PIZZ" no nome fantasia.

Fica de fora a pizzaria registrada sem "pizz" no nome fantasia (ou sem nome
fantasia), então a contagem é um piso. O município do CNPJ usa o código da
Receita (TOM); a ligação com o código do IBGE é feita por nome e UF.

Fonte: Receita Federal, Dados Abertos do CNPJ, pasta mensal (compartilhamento
público em arquivos.receitafederal.gov.br). Os zips ficam em
`data/raw/cnpj/` e a contagem por município, pequena, em
`data/processed/pizzarias_cnpj.csv`.

    uv run python -m occasion.mapa.cnpj
"""

from __future__ import annotations

import io
import re
import unicodedata
import zipfile
from pathlib import Path

import pandas as pd

from occasion.mapa import censo
from occasion.paths import PROCESSED_DIR, RAW_DIR

CNPJ_DIR = RAW_DIR / "cnpj"
ARQUIVO = PROCESSED_DIR / "pizzarias_cnpj.csv"
COMPARTILHAMENTO = "YggdBLfdninEJX9"  # token público do compartilhamento da Receita
WEBDAV = "https://arquivos.receitafederal.gov.br/public.php/webdav"
MES = "2026-11"

ATIVA = "02"
CNAES = {
    "5611201": "restaurante",
    "5611203": "lanchonete",
    "5620104": "consumo_domiciliar",
}
# Colunas do layout de Estabelecimentos (sem cabeçalho, separador ";").
COL_NOME_FANTASIA = 4
COL_SITUACAO = 5
COL_CNAE = 11
COL_UF = 19
COL_MUNICIPIO = 20
COLUNAS = [COL_NOME_FANTASIA, COL_SITUACAO, COL_CNAE, COL_UF, COL_MUNICIPIO]

# Grafias da Receita que não batem com o IBGE depois de normalizar (nome
# normalizado da Receita → nome normalizado do IBGE). Nenhum tem 100 mil
# habitantes; entram para a contagem ficar completa. A chave vale para todas
# as UFs, então só entra nome que não existe em outra UF.
NOMES_RECEITA = {
    "FORTALEZA DO TABOCAO": "TABOCAO",
    "ELDORADO DOS CARAJAS": "ELDORADO DO CARAJAS",
    "SANTA ISABEL DO PARA": "SANTA IZABEL DO PARA",
    "ASSU": "ACU",
    "BOA SAUDE": "JANUARIO CICCO",
    "MUQUEM DE SAO FRANCISCO": "MUQUEM DO SAO FRANCISCO",
    "AMPARO DE SAO FRANCISCO": "AMPARO DO SAO FRANCISCO",
    "AMPARO DA SERRA": "AMPARO DO SERRA",
    "BRASOPOLIS": "BRAZOPOLIS",
    "PARATI": "PARATY",
    "BALNEARIO DE PICARRAS": "BALNEARIO PICARRAS",
    "SANTANA DO LIVRAMENTO": "SANT ANA DO LIVRAMENTO",
    "SANTO ANTONIO DO LEVERGER": "SANTO ANTONIO DE LEVERGER",
    "COUTO DE MAGALHAES": "COUTO MAGALHAES",
    "SAO VALERIO DA NATIVIDADE": "SAO VALERIO",
}


def normalizar(nome: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", str(nome)).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Z0-9]+", " ", sem_acento.upper()).strip()


def _ler_zip(caminho: Path, **kw) -> pd.DataFrame:
    with zipfile.ZipFile(caminho) as z:
        membro = z.namelist()[0]
        with z.open(membro) as f:
            return pd.read_csv(
                io.TextIOWrapper(f, encoding="latin-1"),
                sep=";",
                header=None,
                dtype=str,
                **kw,
            )


def contar_arquivo(caminho: Path) -> pd.DataFrame:
    """Restaurantes ativos e pizzarias por município (código TOM) e CNAE em um zip."""
    partes = []
    with zipfile.ZipFile(caminho) as z, z.open(z.namelist()[0]) as f:
        leitor = pd.read_csv(
            io.TextIOWrapper(f, encoding="latin-1"),
            sep=";",
            header=None,
            dtype=str,
            usecols=COLUNAS,
            chunksize=1_000_000,
        )
        for bloco in leitor:
            b = bloco[(bloco[COL_SITUACAO] == ATIVA) & bloco[COL_CNAE].isin(CNAES)]
            pizza = b[COL_NOME_FANTASIA].fillna("").str.upper().str.contains("PIZZ")
            partes.append(
                b.assign(pizzaria=pizza)
                .groupby([COL_UF, COL_MUNICIPIO, COL_CNAE])["pizzaria"]
                .agg(estabelecimentos="size", pizzarias="sum")
                .reset_index()
            )
    return pd.concat(partes, ignore_index=True)


def municipios_receita() -> pd.DataFrame:
    """Tabela de municípios da Receita: código TOM e nome (sem UF)."""
    rec = _ler_zip(CNPJ_DIR / "Municipios.zip").rename(columns={0: "tom", 1: "nome"})
    return rec


def ligar_ibge(contagem: pd.DataFrame, rec: pd.DataFrame, ibge: pd.DataFrame) -> pd.DataFrame:
    """Põe o código IBGE na contagem por município (código TOM + UF)."""
    ib = ibge.assign(
        chave=ibge["municipio"].map(normalizar),
        uf_sigla=ibge["uf"],
    )
    chave = rec["nome"].map(normalizar)
    rec = rec.assign(chave=chave.map(lambda n: NOMES_RECEITA.get(n, n)))
    if ib.duplicated(["uf_sigla", "chave"]).any():
        raise ValueError("nome normalizado repetido na mesma UF: a ligação dobraria contagens")
    c = contagem.rename(columns={COL_UF: "uf_sigla", COL_MUNICIPIO: "tom"}).merge(
        rec[["tom", "chave"]], on="tom", how="left"
    )
    c = c.merge(ib[["CD_MUN", "uf_sigla", "chave"]], on=["uf_sigla", "chave"], how="left")
    c.attrs["sem_ibge"] = c[c["CD_MUN"].isna()][["uf_sigla", "tom", "chave"]].drop_duplicates()
    return c


def main() -> None:
    zips = sorted(CNPJ_DIR.glob("Estabelecimentos*.zip"))
    if len(zips) != 10:
        raise FileNotFoundError(f"esperava 10 zips de Estabelecimentos em {CNPJ_DIR}")
    contagem = (
        pd.concat([contar_arquivo(z) for z in zips], ignore_index=True)
        .groupby([COL_UF, COL_MUNICIPIO, COL_CNAE], as_index=False)[
            ["estabelecimentos", "pizzarias"]
        ]
        .sum()
    )
    ibge = censo.populacao().rename(columns={"NM_MUN": "municipio"})
    ibge = ibge[["CD_MUN", "municipio", "uf"]]
    c = ligar_ibge(contagem, municipios_receita(), ibge)
    sem = c.attrs["sem_ibge"]
    print(f"municípios da Receita sem código IBGE: {len(sem)}")
    print(sem.head(30).to_string(index=False))
    por_mun = c.dropna(subset=["CD_MUN"]).pivot_table(
        index="CD_MUN",
        columns=COL_CNAE,
        values=["estabelecimentos", "pizzarias"],
        aggfunc="sum",
        fill_value=0,
    )
    por_mun.columns = [f"{v}_{CNAES[k]}" for v, k in por_mun.columns]
    por_mun["estabelecimentos_cnpj"] = por_mun.filter(like="estabelecimentos_").sum(axis=1)
    por_mun["pizzarias_cnpj"] = por_mun.filter(like="pizzarias_").sum(axis=1)
    por_mun.index = por_mun.index.astype(int)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    por_mun.assign(mes_receita=MES).to_csv(ARQUIVO)
    print(por_mun[["estabelecimentos_cnpj", "pizzarias_cnpj"]].sum())


if __name__ == "__main__":
    main()
