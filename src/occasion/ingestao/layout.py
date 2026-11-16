"""Leitura dos layouts fixed-width a partir do programa SAS oficial do IBGE.

As posições não ficam escritas no código: vêm do arquivo
`Leitura dos Microdados - SAS.txt` que acompanha cada edição da POF.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_CAMPO = re.compile(r"@\s*(\d+)\s+([A-Za-z_][\w,]*)\s+(\$?)(\d+)\.(\d*)")


@dataclass(frozen=True)
class Campo:
    nome: str
    inicio: int  # posição inicial 0-based
    largura: int
    decimais: int
    texto: bool

    @property
    def colspec(self) -> tuple[int, int]:
        return (self.inicio, self.inicio + self.largura)


def _ler_texto(caminho: Path) -> str:
    bruto = caminho.read_bytes()
    try:
        texto = bruto.decode("utf-8")
    except UnicodeDecodeError:
        texto = bruto.decode("latin-1")
    return texto.replace("\r\n", "\n").replace("\r", "\n")


def layout_sas(programa_sas: Path, arquivo: str) -> list[Campo]:
    """Extrai os campos do bloco INPUT que lê `arquivo` no programa SAS."""
    # Comentários saem antes de tudo: um nome de arquivo ou um ";" dentro de
    # um comentário não pode definir onde o bloco começa ou termina.
    texto = re.sub(r"/\*.*?\*/", " ", _ler_texto(programa_sas), flags=re.S)
    nome = re.escape(re.sub(r"\.txt$", "", arquivo, flags=re.I))
    infile = re.search(rf'INFILE\s+"(?:[^"]*[\\/])?{nome}(?:\.txt)?"', texto, flags=re.I)
    if infile is None:
        raise KeyError(f"{arquivo} não aparece num INFILE de {programa_sas}")
    entrada = re.compile(r"\bINPUT\b", flags=re.I).search(texto, infile.end())
    if entrada is None:
        raise ValueError(f"Bloco INPUT de {arquivo} não encontrado")
    fim = texto.find(";", entrada.end())
    bloco = texto[entrada.end() : fim + 1 if fim >= 0 else len(texto)]
    campos = [
        Campo(
            nome=m.group(2).replace(",", ""),
            inicio=int(m.group(1)) - 1,
            largura=int(m.group(4)),
            decimais=int(m.group(5) or 0),
            texto=bool(m.group(3)),
        )
        for m in _CAMPO.finditer(bloco)
    ]
    if not campos:
        raise ValueError(f"Nenhum campo encontrado para {arquivo}")
    return campos


def ler_fwf(
    caminho: Path,
    campos: list[Campo],
    colunas: list[str] | None = None,
    texto: set[str] | None = None,
) -> pd.DataFrame:
    """Lê o arquivo fixed-width com os campos do layout oficial.

    Campos numéricos com decimais implícitos (sem ponto no arquivo) são
    divididos por 10**decimais, como no SAS; valores com ponto explícito
    ficam como estão. `texto` força colunas a ficarem como string.
    """
    texto = set(texto or ())
    selecionados = [c for c in campos if colunas is None or c.nome in colunas]
    if colunas is not None:
        faltando = set(colunas) - {c.nome for c in selecionados}
        if faltando:
            raise KeyError(f"Campos ausentes no layout: {sorted(faltando)}")
    df = pd.read_fwf(
        caminho,
        colspecs=[c.colspec for c in selecionados],
        names=[c.nome for c in selecionados],
        dtype=str,
        header=None,
        encoding="latin-1",
    )
    for c in selecionados:
        serie = df[c.nome].str.strip()
        if c.texto or c.nome in texto:
            df[c.nome] = serie.replace("", pd.NA)
            continue
        num = pd.to_numeric(serie, errors="coerce")
        if c.decimais:
            implicito = serie.notna() & ~serie.str.contains(".", regex=False, na=False)
            num = num.where(~implicito, num / 10**c.decimais)
        df[c.nome] = num
    return df
