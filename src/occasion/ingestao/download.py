"""Download idempotente dos microdados oficiais da POF no FTP do IBGE."""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

from occasion.paths import RAW_DIR

FTP_BASE = "https://ftp.ibge.gov.br/Orcamentos_Familiares"

# Pacotes oficiais de cada edição (nomes com a data de publicação do IBGE).
PACOTES: dict[str, dict[str, str]] = {
    "2017_2018": {
        "pasta": "Pesquisa_de_Orcamentos_Familiares_2017_2018/Microdados",
        "dados": "Dados_20230713.zip",
        "documentacao": "Documentacao_20230713.zip",
        "leitura": "Programas_de_Leitura_20230713.zip",
        "tradutores": "Tradutores_20230713.zip",
    },
    "2008_2009": {
        "pasta": "Pesquisa_de_Orcamentos_Familiares_2008_2009/Microdados",
        "dados": "Dados_20231009.zip",
        "documentacao": "Documentacao_20231009.zip",
        "leitura": "Programas_de_Leitura_20231009.zip",
        "tradutores": "Tradutores_das_Tabelas_20231009.zip",
    },
}


def pasta_edicao(edicao: str, raiz: Path = RAW_DIR) -> Path:
    return raiz / f"pof_{edicao}"


def baixar_arquivo(url: str, destino: Path) -> Path:
    """Baixa `url` para `destino`.

    O download vai para um arquivo `.part`, o tamanho é conferido com o
    Content-Length da própria resposta e só então o arquivo é renomeado, então
    um download interrompido nunca é confundido com um arquivo pronto.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_suffix(destino.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp, open(parcial, "wb") as out:
        esperado = resp.headers.get("Content-Length")
        shutil.copyfileobj(resp, out, length=1 << 20)
    if esperado is not None and parcial.stat().st_size != int(esperado):
        parcial.unlink()
        raise OSError(f"Download incompleto de {url}")
    parcial.replace(destino)
    return destino


def _extrair(zip_local: Path, destino: Path) -> None:
    """Extrai o zip corrigindo os nomes acentuados.

    Os zips do IBGE não marcam os nomes como UTF-8, então o `zipfile` os lê
    como cp437 e "Aquisição" vira "Aquisiç╞o". Os nomes foram gravados em
    cp850, que é o que se usa para decodificá-los aqui.
    """
    with zipfile.ZipFile(zip_local) as zf:
        for info in zf.infolist():
            if not info.flag_bits & 0x800:
                info.filename = info.filename.encode("cp437").decode("cp850")
            zf.extract(info, destino)


def _assinatura(zip_local: Path) -> str:
    return f"{zip_local.name} {zip_local.stat().st_size}"


def baixar_edicao(edicao: str, raiz: Path = RAW_DIR, extrair: bool = True) -> Path:
    """Baixa e extrai os pacotes de uma edição em `data/raw/pof_<edicao>/`.

    Idempotente: um zip só é baixado de novo se não existir ou não for um zip
    válido, e só é extraído de novo se a marca `.extraido` da pasta não
    corresponder ao zip atual (nome e tamanho).
    """
    info = PACOTES[edicao]
    destino_base = pasta_edicao(edicao, raiz)
    for chave in ("dados", "documentacao", "leitura", "tradutores"):
        nome = info[chave]
        zip_local = destino_base / "zips" / nome
        if not zip_local.exists() or not zipfile.is_zipfile(zip_local):
            baixar_arquivo(f"{FTP_BASE}/{info['pasta']}/{nome}", zip_local)
        if extrair:
            marca = destino_base / chave / ".extraido"
            if not marca.exists() or marca.read_text().strip() != _assinatura(zip_local):
                _extrair(zip_local, destino_base / chave)
                marca.write_text(_assinatura(zip_local))
    return destino_base


def main() -> None:
    for edicao in PACOTES:
        print(f"POF {edicao}: {baixar_edicao(edicao)}")


if __name__ == "__main__":
    main()
