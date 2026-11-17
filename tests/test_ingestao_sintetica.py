"""Ingestão de ponta a ponta com microdados sintéticos.

Monta, num diretório temporário, a mesma estrutura de pastas da POF real
(programa SAS + arquivos fixed-width) com poucas linhas inventadas e roda os
leitores das duas edições. Não depende dos dados do IBGE, então roda no CI.
"""

from pathlib import Path

import pandas as pd
import pytest

from occasion.ingestao.download import pasta_edicao
from occasion.ingestao.pof import COLUNAS, caminho_parquet, processar

PIZZA, GUARANA, ARROZ = "8500903", "8200301", "6300101"


def _sas(blocos: dict[str, list[tuple[str, int, bool, int]]]) -> str:
    """Programa SAS no formato do IBGE: um DATA/INFILE/INPUT por arquivo."""
    partes = []
    for arquivo, campos in blocos.items():
        linhas, pos = [], 1
        for nome, largura, texto, decimais in campos:
            formato = f"{'$' if texto else ''}{largura}.{decimais or ''}"
            linhas.append(f"@{pos:<4} {nome:<20} {formato}")
            pos += largura
        partes.append(
            f'DATA {arquivo};\nINFILE "...\\Dados\\{arquivo}.txt" LRECL=600;\n'
            f"INPUT\n" + "\n".join(linhas) + " ;\nrun;\n"
        )
    return "/* Leitura dos Microdados - sintético */\n" + "\n".join(partes)


def _linha(campos: list[tuple[str, int, bool, int]], valores: dict[str, object]) -> str:
    saida = ""
    for nome, largura, texto, _ in campos:
        valor = str(valores.get(nome, ""))
        assert len(valor) <= largura, nome
        saida += valor.ljust(largura) if texto else valor.rjust(largura)
    return saida


def _gravar(raiz: Path, edicao: str, blocos: dict, dados: dict[str, list[dict]]) -> None:
    base = pasta_edicao(edicao, raiz)
    sas = base / "leitura" / "SAS" / "Leitura dos Microdados - SAS.txt"
    sas.parent.mkdir(parents=True)
    sas.write_text(_sas(blocos), encoding="latin-1")
    (base / "dados").mkdir()
    for arquivo, linhas in dados.items():
        texto = "\n".join(_linha(blocos[arquivo], v) for v in linhas) + "\n"
        (base / "dados" / f"{arquivo}.txt").write_text(texto, encoding="latin-1")


# (nome, largura, texto, decimais)
CONSUMO_2017 = [
    ("UF", 2, False, 0),
    ("ESTRATO_POF", 4, False, 0),
    ("TIPO_SITUACAO_REG", 1, False, 0),
    ("COD_UPA", 9, False, 0),
    ("NUM_DOM", 2, False, 0),
    ("NUM_UC", 1, False, 0),
    ("COD_INFORMANTE", 2, False, 0),
    ("QUADRO", 2, False, 0),
    ("V9001", 7, True, 0),
    ("V9015", 3, True, 0),
    ("V9017", 1, False, 0),
    ("V9018", 1, False, 0),
    ("QTD", 8, False, 2),
    ("PESO_FINAL", 15, False, 8),
    ("RENDA_TOTAL", 10, False, 2),
    ("DIA_SEMANA", 15, True, 0),
    ("DIA_ATIPICO", 1, False, 0),
]
MORADOR_2017 = [
    ("COD_UPA", 9, False, 0),
    ("NUM_DOM", 2, False, 0),
    ("NUM_UC", 1, False, 0),
    ("COD_INFORMANTE", 2, False, 0),
    ("V0403", 3, False, 0),
    ("V0404", 1, False, 0),
]
CONSUMO_2008 = [
    ("COD_UF", 2, False, 0),
    ("NUM_SEQ", 3, False, 0),
    ("NUM_DV", 1, False, 0),
    ("NUM_DOM", 2, False, 0),
    ("NUM_UC", 1, False, 0),
    ("COD_INFORMANTE", 2, False, 0),
    ("ESTRATO_POF", 4, False, 0),
    ("PESO_FINAL", 14, False, 8),
    ("NUM_QUADRO", 2, False, 0),
    ("V9018", 1, False, 0),
    ("V9015", 2, False, 0),
    ("V9001", 7, True, 0),
    ("RENDA_TOTAL", 10, False, 2),
    ("QTD_FINAL", 8, False, 2),
    ("DIA_SEMANA", 1, False, 0),
]
MORADOR_2008 = [
    ("COD_UF", 2, False, 0),
    ("NUM_SEQ", 3, False, 0),
    ("NUM_DV", 1, False, 0),
    ("NUM_DOM", 2, False, 0),
    ("NUM_UC", 1, False, 0),
    ("COD_INFORMANTE", 2, False, 0),
    ("IDADE_ANOS", 3, False, 0),
    ("V0405", 1, False, 0),
    ("TIPO_SITUACAO_REG", 1, False, 0),
]


# Duas pessoas no mesmo domicílio: só o informante muda, então um erro na
# chave do merge com o morador troca idade e sexo entre elas.
P1_2017 = {"COD_UPA": 350000123, "NUM_DOM": 4, "NUM_UC": 1, "COD_INFORMANTE": 2}
P2_2017 = {**P1_2017, "COD_INFORMANTE": 3}
SEM_MORADOR_2017 = {**P1_2017, "COD_INFORMANTE": 9}


def _consumo_2017(pessoa: dict, **campos) -> dict:
    padrao = {
        "UF": 35,
        "ESTRATO_POF": 3501,
        "TIPO_SITUACAO_REG": 1,
        "QUADRO": 72,
        "V9001": PIZZA,
        "V9015": "20H",
        "V9017": 5,
        "V9018": 3,
        "QTD": "25000",
        "PESO_FINAL": "123456789012345",
        "RENDA_TOTAL": "1234567",
        "DIA_SEMANA": "Sábado",
        "DIA_ATIPICO": 2,
    }
    return {**pessoa, **padrao, **campos}


P1_2008 = {"COD_UF": 29, "NUM_SEQ": 7, "NUM_DV": 3, "NUM_DOM": 1, "NUM_UC": 1, "COD_INFORMANTE": 1}
P2_2008 = {**P1_2008, "COD_INFORMANTE": 2}


def _consumo_2008(pessoa: dict, **campos) -> dict:
    padrao = {
        "ESTRATO_POF": 2901,
        "PESO_FINAL": "50000000000",
        "NUM_QUADRO": 71,
        "V9018": 2,
        "V9015": 19,
        "V9001": PIZZA,
        "RENDA_TOTAL": "250000",
        "QTD_FINAL": "20000",
        "DIA_SEMANA": 5,
    }
    return {**pessoa, **padrao, **campos}


@pytest.fixture
def raiz_2017(tmp_path: Path) -> Path:
    raiz = tmp_path / "raw"
    _gravar(
        raiz,
        "2017_2018",
        {"CONSUMO_ALIMENTAR": CONSUMO_2017, "MORADOR": MORADOR_2017},
        {
            "CONSUMO_ALIMENTAR": [
                _consumo_2017(P1_2017),
                _consumo_2017(P1_2017, V9001=GUARANA, QTD="35000"),
                _consumo_2017(
                    P1_2017,
                    QUADRO=73,
                    V9001=ARROZ,
                    V9015="12H",
                    V9017=3,
                    V9018=1,
                    QTD="15000",
                    DIA_SEMANA="Terça-feira",
                    DIA_ATIPICO=1,
                ),
                # local não determinado, hora sem "H" e dia da semana ilegível
                _consumo_2017(P2_2017, V9015="7", V9018=9, DIA_SEMANA="feriado"),
                # hora em branco e pessoa sem registro no MORADOR
                _consumo_2017(SEM_MORADOR_2017, V9015=""),
            ],
            "MORADOR": [
                {**P1_2017, "V0403": 31, "V0404": 2},
                {**P2_2017, "V0403": 58, "V0404": 1},
            ],
        },
    )
    return raiz


@pytest.fixture
def raiz_2008(tmp_path: Path) -> Path:
    raiz = tmp_path / "raw"
    _gravar(
        raiz,
        "2008_2009",
        {"T_CONSUMO_S": CONSUMO_2008, "T_MORADOR_S": MORADOR_2008},
        {
            "T_CONSUMO_S": [
                _consumo_2008(P1_2008),
                _consumo_2008(P1_2008, V9001=GUARANA, QTD_FINAL="30000"),
                _consumo_2008(P2_2008, NUM_QUADRO=72, V9018=1, V9015=7, V9001=ARROZ, DIA_SEMANA=6),
            ],
            "T_MORADOR_S": [
                {**P1_2008, "IDADE_ANOS": 44, "V0405": 1, "TIPO_SITUACAO_REG": 2},
                {**P2_2008, "IDADE_ANOS": 12, "V0405": 2, "TIPO_SITUACAO_REG": 2},
            ],
        },
    )
    return raiz


def test_2017_2018_harmoniza_campos(raiz_2017, tmp_path):
    saida = processar("2017_2018", raiz=raiz_2017, destino=tmp_path / "proc", baixar=False)
    assert saida == caminho_parquet("2017_2018", tmp_path / "proc")
    df = pd.read_parquet(saida)
    assert list(df.columns) == COLUNAS
    assert df["id_pessoa"].tolist() == [
        "35000012304102",
        "35000012304102",
        "35000012304102",
        "35000012304103",
        "35000012304109",
    ]
    assert (df["edicao"] == "2017-18").all()
    assert (df["uf"] == 35).all() and (df["estrato"] == 3501).all()
    assert (df["regiao"] == "Sudeste").all() and (df["situacao"] == "urbano").all()
    assert df["dia_registro"].tolist() == [1, 1, 2, 1, 1]
    assert df["cod_alimento"].tolist() == [PIZZA, GUARANA, ARROZ, PIZZA, PIZZA]
    assert df["qtd_g"].tolist() == pytest.approx([250.0, 350.0, 150.0, 250.0, 250.0])
    assert df["peso"].iloc[0] == pytest.approx(1234567.89012345)
    assert df["renda"].iloc[0] == pytest.approx(12345.67)
    assert df["pizza"].tolist() == [True, False, False, True, True]
    assert df["guarana"].tolist() == [False, True, False, False, False]
    assert df["refri_generico"].tolist() == [False, True, False, False, False]


def test_2017_2018_casos_de_borda(raiz_2017, tmp_path):
    df = pd.read_parquet(processar("2017_2018", raiz=raiz_2017, destino=tmp_path, baixar=False))
    # acentos e maiúsculas ignorados; texto desconhecido vira NA
    assert df["dia_semana"].tolist() == [6, 6, 2, pd.NA, 6]
    assert df["hora"].tolist() == [20, 20, 12, 7, pd.NA]
    # 9 = local não determinado
    assert df["fora_de_casa"].tolist() == [True, True, False, pd.NA, True]
    assert df["local_refeicao"].tolist() == [3, 3, 1, 9, 3]
    assert df["refeicao"].tolist() == [5, 5, 3, 5, 5]
    assert df["dia_atipico"].tolist() == [False, False, True, False, False]
    # cada pessoa com o próprio morador; quem não tem MORADOR fica, com NA
    assert df["idade"].tolist() == [31, 31, 31, 58, pd.NA]
    assert df["sexo"].tolist() == [2, 2, 2, 1, pd.NA]


def test_2008_2009_harmoniza_campos(raiz_2008, tmp_path):
    df = pd.read_parquet(processar("2008_2009", raiz=raiz_2008, destino=tmp_path, baixar=False))
    assert list(df.columns) == COLUNAS
    assert df["id_pessoa"].tolist() == ["29007301101", "29007301101", "29007301102"]
    assert (df["edicao"] == "2008-09").all()
    assert (df["uf"] == 29).all() and (df["estrato"] == 2901).all()
    assert (df["regiao"] == "Nordeste").all() and (df["situacao"] == "rural").all()
    assert df["dia_registro"].tolist() == [1, 1, 2]
    assert df["dia_semana"].tolist() == [5, 5, 6]
    assert df["hora"].tolist() == [19, 19, 7]
    assert df["fora_de_casa"].tolist() == [True, True, False]
    assert df["cod_alimento"].tolist() == [PIZZA, GUARANA, ARROZ]
    assert df["qtd_g"].tolist() == pytest.approx([200.0, 300.0, 200.0])
    assert df["peso"].tolist() == pytest.approx([500.0] * 3)
    assert df["renda"].tolist() == pytest.approx([2500.0] * 3)
    assert df["pizza"].tolist() == [True, False, False]
    assert df["guarana"].tolist() == [False, True, False]
    assert df["refri_generico"].tolist() == [False, True, False]
    # campos que só existem em 2017-18
    assert df["refeicao"].isna().all() and df["local_refeicao"].isna().all()
    assert df["dia_atipico"].isna().all()
    assert df["idade"].tolist() == [44, 44, 12] and df["sexo"].tolist() == [1, 1, 2]
