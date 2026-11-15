"""Caminhos do projeto."""

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DATA_DIR = RAIZ / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
