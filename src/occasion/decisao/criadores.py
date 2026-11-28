"""Perfil de criador de conteúdo do plano de ativação.

Aprovado em 27/11/2026. O Engine não escolhe pessoas: escolhe um perfil (nicho, porte e
plataforma) pela categoria e pela região da recomendação. Os perfis, os números de mercado e
os exemplos ficam em `data/referencia/perfis_criadores.json`, cada um com a sua fonte.

- Conversão no Norte ou no Nordeste → comida regional (micro, Instagram).
- Conversão nas outras regiões → review de comida urbano (micro a médio). Hoje ela só
  dispara pela lacuna das capitais; área rural fica sem perfil.
- Awareness (Sul, classe B ou faixa média com sinal) → entretenimento e humor com alcance.
- Criar frequência → comida do dia a dia (nano a micro, série semanal).
- Delivery, pizzaria e combo em casa → sem criador (o canal é o próprio ponto de venda).

Os números de mercado vêm de matérias que citam as pesquisas, e os exemplos não tiveram
contrato nem disponibilidade checados; a saída diz isso. Um exemplo com região só aparece
quando a recomendação é nessa região. Um arquivo inválido (JSON quebrado, campo faltando ou
fonte citada que não existe) não derruba o plano: o plano sai sem criador e o erro vai para o
log.
"""

from __future__ import annotations

import copy
import json
import logging
from functools import lru_cache

from occasion.paths import REFERENCIA_DIR

log = logging.getLogger(__name__)
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
    except OSError:
        return None


def _exige(obj, campos: tuple[str, ...], onde: str) -> None:
    if not isinstance(obj, dict):
        raise TypeError(f"{onde}: esperava um objeto")
    for campo in campos:
        if not obj.get(campo):
            raise ValueError(f"{onde}: falta {campo}")


def validar(dados: dict) -> None:
    """Falha se faltar um campo que a saída usa ou se um número citar uma fonte inexistente."""
    _exige(dados, ("fontes", "perfis"), "arquivo")
    fontes = dados["fontes"]
    for chave, f in fontes.items():
        _exige(f, ("titulo", "ano", "url", "natureza"), f"fonte {chave}")
    for chave, p in dados["perfis"].items():
        _exige(p, ("nome", "nicho", "porte", "plataforma", "por_que", "custo_por_post"), chave)
        if not isinstance(p.get("exemplos"), list):
            raise TypeError(f"perfil {chave}: exemplos deve ser uma lista")
        for x in p["por_que"]:
            _exige(x, ("texto",), f"perfil {chave}, por_que")
        _exige(p["custo_por_post"], ("faixa", "natureza"), f"perfil {chave}, custo_por_post")
        for x in p["exemplos"]:
            _exige(x, ("nome", "nota"), f"perfil {chave}, exemplo")
            if "regiao" not in x:
                raise ValueError(f"perfil {chave}: exemplo {x['nome']!r} sem a chave regiao")
        for item in [*p["por_que"], *p["exemplos"], p["custo_por_post"]]:
            if item.get("fonte") is not None and item["fonte"] not in fontes:
                raise ValueError(f"perfil {chave}: fonte {item['fonte']!r} não existe")
        if not p["exemplos"] and not p.get("sem_exemplo"):
            raise ValueError(f"perfil {chave}: sem exemplo e sem sem_exemplo")


@lru_cache(maxsize=1)
def _ler(versao: int | None) -> dict:
    if versao is None:
        log.warning("perfis_criadores.json não encontrado, plano sem criador")
        return {}
    try:
        dados = json.loads(ARQUIVO.read_text(encoding="utf-8"))
        validar(dados)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as erro:
        log.warning("perfis_criadores.json inválido, plano sem criador: %s", erro)
        return {}
    return dados


def base() -> dict:
    """Cópia do arquivo de perfis, relido quando muda (como as outras tabelas de referência)."""
    return copy.deepcopy(_ler(_versao()))


def chave_do_perfil(categoria: str, regiao: str, tipo_area: str) -> str | None:
    if categoria == "conversao_associacao":
        if regiao in REGIOES_REGIONAIS:
            return "comida_regional"
        # Hoje, fora do Norte e do Nordeste a conversão só vem da lacuna das capitais. Se a
        # tabela de associação mudar e outra área disparar, o review urbano segue valendo
        # para área urbana; a rural fica sem perfil (o mapa não tem as cidades dela).
        return None if tipo_area == "rural" else "review_urbano"
    return POR_CATEGORIA.get(categoria)


def _com_fonte(item: dict, fontes: dict) -> dict:
    f = fontes[item["fonte"]] if item.get("fonte") else None
    return {**item, "fonte": f}


def perfil(categoria: str, regiao: str, tipo_area: str) -> dict | None:
    """Perfil de criador para a recomendação; None quando a categoria não usa criador."""
    chave = chave_do_perfil(categoria, regiao, tipo_area)
    dados = base()
    p = dados.get("perfis", {}).get(chave) if chave else None
    if p is None:
        return None
    fontes = dados["fontes"]
    exemplos = [x for x in p["exemplos"] if x.get("regiao") in (None, regiao)]
    sem_exemplo = p.get("sem_exemplo")
    if p["exemplos"] and not exemplos:
        sem_exemplo = f"nenhum exemplo com fonte no {regiao}; a busca fica com a agência"
    return {
        "perfil": chave,
        "nome": p["nome"],
        "nicho": p["nicho"],
        "porte": p["porte"],
        "plataforma": p["plataforma"],
        "por_que": [_com_fonte(x, fontes) for x in p["por_que"]],
        "custo_por_post": _com_fonte(p["custo_por_post"], fontes),
        "exemplos": [_com_fonte(x, fontes) for x in exemplos],
        "sem_exemplo": None if exemplos else sem_exemplo,
        "levantado_em": dados.get("levantado_em"),
        "natureza": AVISO,
    }
