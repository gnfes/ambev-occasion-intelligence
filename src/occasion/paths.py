"""Caminhos do projeto.

Os dados ficam em `data/` na raiz do repositório. Com o pacote instalado fora
do modo editável (por exemplo numa imagem Docker), essa raiz não existe; nesse
caso, aponte `OCCASION_DATA_DIR` para a pasta de dados.
"""

import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("OCCASION_DATA_DIR", RAIZ / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
