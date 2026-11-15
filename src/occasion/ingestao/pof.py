"""Ingestão do consumo alimentar pessoal das POFs 2008-09 e 2017-18.

Gera, para cada edição, uma tabela em nível de item consumido com colunas
harmonizadas e grava em `data/processed/pof_<edicao>_consumo.parquet`.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd

from occasion.ingestao import codigos
from occasion.ingestao.download import baixar_edicao, pasta_edicao
from occasion.ingestao.layout import layout_sas, ler_fwf
from occasion.paths import PROCESSED_DIR, RAW_DIR

REGIOES = {1: "Norte", 2: "Nordeste", 3: "Sudeste", 4: "Sul", 5: "Centro-Oeste"}

# 1 = segunda ... 7 = domingo, como no dicionário de 2008-09.
DIAS_SEMANA = {
    "segunda-feira": 1,
    "terca-feira": 2,
    "quarta-feira": 3,
    "quinta-feira": 4,
    "sexta-feira": 5,
    "sabado": 6,
    "domingo": 7,
}

COLUNAS = [
    "edicao",
    "id_pessoa",
    "uf",
    "regiao",
    "situacao",
    "estrato",
    "dia_registro",
    "dia_semana",
    "hora",
    "fora_de_casa",
    "local_refeicao",
    "refeicao",
    "dia_atipico",
    "cod_alimento",
    "qtd_g",
    "peso",
    "renda",
    "idade",
    "sexo",
    *codigos.FLAGS,
]


def _programa_sas(edicao: str, raiz: Path) -> Path:
    return pasta_edicao(edicao, raiz) / "leitura" / "SAS" / "Leitura dos Microdados - SAS.txt"


def _dados(edicao: str, raiz: Path) -> Path:
    return pasta_edicao(edicao, raiz) / "dados"


def _sem_acento(texto: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c)
    ).lower()


def _flags(df: pd.DataFrame) -> pd.DataFrame:
    for nome, conjunto in codigos.FLAGS.items():
        df[nome] = df["cod_alimento"].isin(conjunto)
    return df


def _finalizar(df: pd.DataFrame) -> pd.DataFrame:
    df["regiao"] = (df["uf"] // 10).map(REGIOES)
    df["situacao"] = df["situacao"].map({1: "urbano", 2: "rural"})
    df = _flags(df)
    tipos = {
        "uf": "Int8",
        "estrato": "Int16",
        "dia_registro": "Int8",
        "dia_semana": "Int8",
        "hora": "Int8",
        "fora_de_casa": "boolean",
        "local_refeicao": "Int8",
        "refeicao": "Int8",
        "dia_atipico": "boolean",
        "idade": "Int16",
        "sexo": "Int8",
    }
    return df[COLUNAS].astype(tipos).reset_index(drop=True)


def ler_2017_2018(raiz: Path = RAW_DIR) -> pd.DataFrame:
    edicao = "2017_2018"
    sas = _programa_sas(edicao, raiz)
    consumo = ler_fwf(
        _dados(edicao, raiz) / "CONSUMO_ALIMENTAR.txt",
        layout_sas(sas, "CONSUMO_ALIMENTAR"),
        colunas=[
            "UF", "ESTRATO_POF", "TIPO_SITUACAO_REG", "COD_UPA", "NUM_DOM", "NUM_UC",
            "COD_INFORMANTE", "QUADRO", "V9001", "V9015", "V9017", "V9018", "QTD",
            "PESO_FINAL", "RENDA_TOTAL", "DIA_SEMANA", "DIA_ATIPICO",
        ],
        texto={"V9001"},
    )
    morador = ler_fwf(
        _dados(edicao, raiz) / "MORADOR.txt",
        layout_sas(sas, "MORADOR.txt"),
        colunas=["COD_UPA", "NUM_DOM", "NUM_UC", "COD_INFORMANTE", "V0403", "V0404"],
    )
    chave = ["COD_UPA", "NUM_DOM", "NUM_UC", "COD_INFORMANTE"]
    df = consumo.merge(morador, on=chave, how="left", validate="many_to_one")
    df["edicao"] = "2017-18"
    df["id_pessoa"] = (
        df["COD_UPA"].astype("int64").astype(str).str.zfill(9)
        + df["NUM_DOM"].astype(int).astype(str).str.zfill(2)
        + df["NUM_UC"].astype(int).astype(str)
        + df["COD_INFORMANTE"].astype(int).astype(str).str.zfill(2)
    )
    df["uf"] = df["UF"]
    df["situacao"] = df["TIPO_SITUACAO_REG"]
    df["estrato"] = df["ESTRATO_POF"]
    df["dia_registro"] = df["QUADRO"].map({72: 1, 73: 2})
    dias = {v: DIAS_SEMANA.get(_sem_acento(v)) for v in df["DIA_SEMANA"].dropna().unique()}
    df["dia_semana"] = df["DIA_SEMANA"].map(dias)
    df["hora"] = pd.to_numeric(df["V9015"].str.rstrip("Hh"), errors="coerce")
    df["local_refeicao"] = df["V9018"]
    # 1 = em casa ou levado de casa; 2 a 7 = fora de casa; 9 = não determinado.
    df["fora_de_casa"] = df["V9018"].map({1: False, **{v: True for v in range(2, 8)}})
    df["refeicao"] = df["V9017"]
    df["dia_atipico"] = df["DIA_ATIPICO"].map({1: True, 2: False})
    df["cod_alimento"] = df["V9001"]
    df["qtd_g"] = df["QTD"]
    df["peso"] = df["PESO_FINAL"]
    df["renda"] = df["RENDA_TOTAL"]
    df["idade"] = df["V0403"]
    df["sexo"] = df["V0404"]
    return _finalizar(df)


def ler_2008_2009(raiz: Path = RAW_DIR) -> pd.DataFrame:
    edicao = "2008_2009"
    sas = _programa_sas(edicao, raiz)
    consumo = ler_fwf(
        _dados(edicao, raiz) / "T_CONSUMO_S.txt",
        layout_sas(sas, "T_CONSUMO_S"),
        colunas=[
            "COD_UF", "NUM_SEQ", "NUM_DV", "NUM_DOM", "NUM_UC", "COD_INFORMANTE",
            "ESTRATO_POF", "PESO_FINAL", "NUM_QUADRO", "V9018", "V9015", "V9001",
            "RENDA_TOTAL", "QTD_FINAL", "DIA_SEMANA",
        ],
        texto={"V9001"},
    )
    morador = ler_fwf(
        _dados(edicao, raiz) / "T_MORADOR_S.txt",
        layout_sas(sas, "T_MORADOR_S"),
        colunas=[
            "COD_UF", "NUM_SEQ", "NUM_DV", "NUM_DOM", "NUM_UC", "COD_INFORMANTE",
            "IDADE_ANOS", "V0405", "TIPO_SITUACAO_REG",
        ],
    )
    chave = ["COD_UF", "NUM_SEQ", "NUM_DV", "NUM_DOM", "NUM_UC", "COD_INFORMANTE"]
    df = consumo.merge(morador, on=chave, how="left", validate="many_to_one")
    df["edicao"] = "2008-09"
    df["id_pessoa"] = (
        df["COD_UF"].astype(int).astype(str).str.zfill(2)
        + df["NUM_SEQ"].astype(int).astype(str).str.zfill(3)
        + df["NUM_DV"].astype(int).astype(str)
        + df["NUM_DOM"].astype(int).astype(str).str.zfill(2)
        + df["NUM_UC"].astype(int).astype(str)
        + df["COD_INFORMANTE"].astype(int).astype(str).str.zfill(2)
    )
    df["uf"] = df["COD_UF"]
    df["situacao"] = df["TIPO_SITUACAO_REG"]
    df["estrato"] = df["ESTRATO_POF"]
    df["dia_registro"] = df["NUM_QUADRO"].map({71: 1, 72: 2})
    df["dia_semana"] = df["DIA_SEMANA"]
    df["hora"] = df["V9015"]
    # V9018 em 2008-09 é a "fonte do alimento": 1 dentro e 2 fora do domicílio.
    df["fora_de_casa"] = df["V9018"].map({1: False, 2: True})
    df["local_refeicao"] = pd.NA
    df["refeicao"] = pd.NA
    df["dia_atipico"] = pd.NA
    df["cod_alimento"] = df["V9001"]
    df["qtd_g"] = df["QTD_FINAL"]
    df["peso"] = df["PESO_FINAL"]
    df["renda"] = df["RENDA_TOTAL"]
    df["idade"] = df["IDADE_ANOS"]
    df["sexo"] = df["V0405"]
    return _finalizar(df)


LEITORES = {"2008_2009": ler_2008_2009, "2017_2018": ler_2017_2018}


def caminho_parquet(edicao: str, destino: Path = PROCESSED_DIR) -> Path:
    return destino / f"pof_{edicao}_consumo.parquet"


def processar(
    edicao: str,
    raiz: Path = RAW_DIR,
    destino: Path = PROCESSED_DIR,
    baixar: bool = True,
) -> Path:
    if baixar:
        baixar_edicao(edicao, raiz)
    df = LEITORES[edicao](raiz)
    saida = caminho_parquet(edicao, destino)
    saida.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(saida, index=False)
    return saida


def main() -> None:
    for edicao in LEITORES:
        print(f"POF {edicao}: {processar(edicao)}")


if __name__ == "__main__":
    main()
