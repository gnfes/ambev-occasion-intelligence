from pathlib import Path

import pytest

from occasion.ingestao.layout import layout_sas, ler_fwf

SAS = """\
DATA EXEMPLO;
INFILE "...\\EXEMPLO.txt";
INPUT
@01  UF        2.   /* UNIDADE DA FEDERAÇÃO */
@03  COD       $7.
@10  HORA      $3.
@13  PESO      6.2
@19  RENDA     7.2 ;
run;
"""


@pytest.fixture
def programa(tmp_path: Path) -> Path:
    caminho = tmp_path / "leitura.sas"
    caminho.write_text(SAS, encoding="latin-1")
    return caminho


def test_layout_le_posicoes_e_tipos(programa):
    campos = {c.nome: c for c in layout_sas(programa, "EXEMPLO.txt")}
    assert list(campos) == ["UF", "COD", "HORA", "PESO", "RENDA"]
    assert campos["COD"].colspec == (2, 9) and campos["COD"].texto
    assert campos["PESO"].decimais == 2 and not campos["PESO"].texto


def test_ler_fwf_aplica_decimais_implicitos(programa, tmp_path):
    dados = tmp_path / "EXEMPLO.txt"
    dados.write_text("358500903 7H01234512.50\n", encoding="latin-1")
    df = ler_fwf(dados, layout_sas(programa, "EXEMPLO.txt"), texto={"COD"})
    linha = df.iloc[0]
    assert linha["UF"] == 35
    assert linha["COD"] == "8500903"
    assert linha["HORA"] == "7H"
    assert linha["PESO"] == pytest.approx(123.45)  # sem ponto: decimais implícitos
    assert linha["RENDA"] == pytest.approx(12.5)  # com ponto: valor como está


def test_layout_falha_para_arquivo_inexistente(programa):
    with pytest.raises(KeyError):
        layout_sas(programa, "OUTRO.txt")
