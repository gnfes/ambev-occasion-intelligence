"""Perfil de criador de conteúdo do plano de ativação.

Aprovado em 27/11/2026. O Engine não escolhe pessoas: escolhe um perfil (nicho, porte e
plataforma) pela categoria e pela região da recomendação. Os perfis, os números de mercado e
os exemplos ficam em `data/referencia/perfis_criadores.json`, cada um com a sua fonte.

- Conversão no Norte ou no Nordeste → comida regional (micro, Instagram).
- Conversão nas outras regiões (capitais) → review de comida urbano (micro a médio).
- Awareness (Sul, classe B ou faixa média com sinal) → entretenimento e humor com alcance.
- Criar frequência → comida do dia a dia (nano a micro, série semanal).
- Delivery, pizzaria e combo em casa → sem criador (o canal é o próprio ponto de venda).

Os números de mercado vêm de matérias que citam as pesquisas, e os exemplos não tiveram
contrato nem disponibilidade checados; a saída diz isso.
"""

from __future__ import annotations

import json
from functools import lru_cache

from occasion.paths import REFERENCIA_DIR

ARQUIVO = REFERENCIA_DIR / "perfis_criadores.json"
REGIOES_REGIONAIS = {"Norte", "Nordeste"}
POR_CATEGORIA = {
    "awareness_associacao": "entretenimento_alcance",
    "criar_frequencia": "comida_dia_a_dia",
}
AVISO = (
    "números de mercado de fontes secundárias; exemplos sem checagem de contrato, "
    "exclusividade ou disponibilidade"
)


def _versao() -> int | None:
    try:
        return ARQUIVO.stat().st_mtime_ns
    except FileNotFoundError:
        return None


@lru_cache(maxsize=1)
def _ler(versao: int | None) -> dict:
    if versao is None:
        return {}
    return json.loads(ARQUIVO.read_text(encoding="utf-8"))


def base() -> dict:
    """Arquivo de perfis, relido quando muda (como as outras tabelas de referência)."""
    return _ler(_versao())


def chave_do_perfil(categoria: str, regiao: str) -> str | None:
    if categoria == "conversao_associacao":
        return "comida_regional" if regiao in REGIOES_REGIONAIS else "review_urbano"
    return POR_CATEGORIA.get(categoria)


def _com_fonte(item: dict, fontes: dict) -> dict:
    f = fontes.get(item.get("fonte")) if item.get("fonte") else None
    return {**item, "fonte": f}


def perfil(categoria: str, regiao: str) -> dict | None:
    """Perfil de criador para a recomendação; None quando a categoria não usa criador."""
    chave = chave_do_perfil(categoria, regiao)
    dados = base()
    p = dados.get("perfis", {}).get(chave) if chave else None
    if p is None:
        return None
    fontes = dados.get("fontes", {})
    return {
        "perfil": chave,
        "nome": p["nome"],
        "nicho": p["nicho"],
        "porte": p["porte"],
        "plataforma": p["plataforma"],
        "por_que": [_com_fonte(x, fontes) for x in p["por_que"]],
        "custo_por_post": _com_fonte(p["custo_por_post"], fontes),
        "exemplos": [_com_fonte(x, fontes) for x in p["exemplos"]],
        "sem_exemplo": p.get("sem_exemplo"),
        "levantado_em": dados.get("levantado_em"),
        "natureza": AVISO,
    }
