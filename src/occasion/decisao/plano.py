"""Plano de ativação: a recomendação do Engine com público, lugar, momento, canal e medição.

Aprovado em 27/11/2026. Cada campo sai de um dado já medido no projeto; o que é hipótese
vem marcado como tal. O plano não muda a regra nem a categoria: só detalha a decisão.

- Para quem: o perfil, o segmento que disparou a regra e o percentil na classe.
- Onde: as 3 cidades de maior prioridade do Mapa de Oportunidade na região (Censo 2022 e
  CNPJ). Tipo de área capital → capitais da região; região metropolitana ou interior →
  cidades que não são capital (o mapa não separa as duas). Área rural → sem lista (o mapa
  só tem municípios com 100 mil habitantes ou mais).
- Quando: o momento do contexto; na regra 5, a melhor noite de segunda a quinta do perfil
  (ou o melhor momento, se essa noite estiver na faixa baixa); à noite, a janela de 19h a
  21h, em que começa a maior parte das ocasiões (POF, nacional); e as próximas 4 semanas da
  UF da primeira cidade, com feriados (Trend Radar, informativo; some quando a janela passa).
- Canal (é também o `recommendation_channel`): awareness e criar frequência começam por
  criadores; nas outras categorias, o canal informado. Sem canal, "em casa" só onde o IC95 da
  fatia em casa na região fica inteiro acima de 50%; nas outras regiões, casa e salão.
- Mecânica e mensagem: uma por categoria, marcadas como hipótese (sem dado de efeito).
- Medição: o indicador e a linha de base do segmento na POF. Sem meta: não há dado de efeito
  para defini-la.
"""

from __future__ import annotations

import datetime as dt
import json
from functools import lru_cache

import pandas as pd

from occasion.paths import REFERENCIA_DIR
from occasion.radar.clima import CAPITAIS

ARQUIVO_MAPA = REFERENCIA_DIR / "mapa_oportunidade.csv"
ARQUIVO_RADAR = REFERENCIA_DIR / "radar_alertas.json"
N_CIDADES = 3
REGIAO_POR_CODIGO = {"1": "Norte", "2": "Nordeste", "3": "Sudeste", "4": "Sul", "5": "Centro-Oeste"}
CANAL_PLANO = {
    "delivery": "app de delivery",
    "pizzaria": "pizzaria do bairro (salão e balcão)",
    "em_casa": "consumo em casa (delivery e retirada)",
}
HIPOTESE = "hipótese: sem dado de efeito"

MECANICA = {
    "conversao_associacao": (
        (
            "Combo pizza + Guaraná no canal indicado, nas cidades listadas, ligando o Guaraná "
            "que o segmento já toma no dia a dia à pizza."
        ),
        '"O Guaraná que você já toma no dia a dia, agora na pizza"',
    ),
    "awareness_associacao": (
        (
            "Conteúdo com criadores que falam com o segmento, mostrando pizza com Guaraná, "
            "antes de qualquer oferta de preço."
        ),
        '"Coisa Nossa": pizza com Guaraná como ritual da casa',
    ),
    "ativacao_delivery": (
        "Combo pizza + Guaraná em destaque no app de delivery, no horário indicado.",
        '"Pediu pizza? O Guaraná já vem junto"',
    ),
    "combo_em_casa": (
        "Combo pizza + Guaraná para pedir ou retirar, com a garrafa de 1 litro ou mais.",
        '"Pizza em casa pede Guaraná"',
    ),
    "acao_pizzaria": (
        "Guaraná no cardápio e no combo da pizzaria do bairro, com material no salão e no balcão.",
        '"Na pizzaria do bairro, o par é Guaraná"',
    ),
    "criar_frequencia": (
        (
            "Criadores online chamando para a pizza com Guaraná na noite de semana indicada; "
            "depois, a mesma oferta na pizzaria do bairro."
        ),
        '"Pizza não é só sábado"',
    ),
}


MECANICA_SEM_NOITE_DE_SEMANA = (
    (
        "Criadores online chamando para a pizza com Guaraná no melhor momento do perfil (as "
        "noites de semana dele ficam na faixa baixa); depois, a mesma oferta na pizzaria do bairro."
    ),
    '"Pizza boa pede Guaraná"',
)


def _versao(caminho) -> int | None:
    try:
        return caminho.stat().st_mtime_ns
    except FileNotFoundError:
        return None


def _mapa() -> pd.DataFrame | None:
    """Mapa de Oportunidade, relido quando o arquivo muda (como a tabela de associação)."""
    return _ler_mapa(_versao(ARQUIVO_MAPA))


def _radar() -> dict:
    return _ler_radar(_versao(ARQUIVO_RADAR))


@lru_cache(maxsize=1)
def _ler_mapa(versao: int | None) -> pd.DataFrame | None:
    if versao is None:
        return None
    m = pd.read_csv(ARQUIVO_MAPA)
    m["regiao"] = m["cd_mun"].astype(str).str[0].map(REGIAO_POR_CODIGO)
    capitais = {(uf, nome) for uf, (nome, *_) in CAPITAIS.items()}
    m["capital"] = [
        (uf, nome) in capitais for uf, nome in zip(m["uf"], m["municipio"], strict=True)
    ]
    return m


@lru_cache(maxsize=1)
def _ler_radar(versao: int | None) -> dict:
    if versao is None:
        return {}
    alertas = json.loads(ARQUIVO_RADAR.read_text(encoding="utf-8")).get("alertas", [])
    return {a["sigla"]: a for a in alertas}


def cidades(regiao: str, tipo_area: str, n: int = N_CIDADES) -> dict:
    m = _mapa()
    if tipo_area == "rural":
        return {"lista": [], "criterio": "o mapa só tem municípios com 100 mil habitantes ou mais"}
    if m is None:
        return {"lista": [], "criterio": "mapa de oportunidade indisponível"}
    capital = tipo_area == "capital"
    sel = m[(m["regiao"] == regiao) & (m["capital"] == capital)].sort_values("prioridade")
    criterio = (
        f"capitais do {regiao} por prioridade no Mapa de Oportunidade"
        if capital
        else f"cidades do {regiao} fora das capitais, por prioridade no Mapa de Oportunidade "
        "(o mapa não separa região metropolitana de interior)"
    )
    lista = [
        {
            "municipio": r.municipio,
            "uf": r.uf,
            "prioridade_nacional": int(r.prioridade),
            "ocasioes_semana": round(r.ocasioes_semana),
            "pizzarias_por_100k": float(r.pizzarias_100k),
        }
        for r in sel.head(n).itertuples()
    ]
    return {"lista": lista, "criterio": criterio}


def janela(uf: str | None, hoje: dt.date | None = None) -> dict | None:
    """Próximas 4 semanas do Radar; None se o arquivo já passou dessa janela."""
    a = _radar().get(uf) if uf else None
    pj = a.get("proxima_janela") if a else None
    if not pj or not pj.get("semanas"):
        return None
    hoje = hoje or dt.date.today()
    # Só as semanas que ainda não terminaram (semana = domingo a sábado).
    semanas = [s for s in pj["semanas"] if dt.date.fromisoformat(s) + dt.timedelta(days=6) >= hoje]
    if not semanas:
        return None
    fim = dt.date.fromisoformat(semanas[-1]) + dt.timedelta(days=6)
    inteira = len(semanas) == len(pj["semanas"])
    # Desvio e feriados foram calculados para as 4 semanas: numa janela encurtada saem.
    return {
        "uf": uf,
        "de": semanas[0],
        "ate": fim.isoformat(),
        "feriados": list(pj.get("feriados", [])) if inteira else [],
        "desvio_previsto_busca_pct": pj.get("desvio_previsto_pct") if inteira else None,
        "natureza": "informativo: o Trend Radar não superou o baseline fora do tempo",
    }


def _segmento(tabela: dict | None, dimensao: str, valor: str) -> dict | None:
    for s in (tabela or {}).get("segmentos", []):
        if s["dimensao"] == dimensao and s["valor"] == valor:
            return s
    return None


def canal(canal_informado: str | None, regiao: str, tabela: dict | None) -> dict:
    if canal_informado:
        return {"valor": CANAL_PLANO[canal_informado], "base": "canal informado no contexto"}
    seg = _segmento(tabela, "regiao", regiao)
    casa = seg.get("em_casa") if seg else None
    if not casa:
        return {"valor": "casa e salão", "base": "sem dado de consumo em casa para este público"}
    texto = (
        f"no {regiao}, {casa['pct']:.1f}% das ocasiões de pizza com refrigerante são em casa "
        f"(IC95 {casa['ic95'][0]:.1f} a {casa['ic95'][1]:.1f}; delivery incluído)"
    ).replace(".", ",")
    if casa["ic95"][0] > 50:
        return {"valor": CANAL_PLANO["em_casa"], "base": texto, "pct_em_casa": casa["pct"]}
    return {
        "valor": "casa e salão (delivery e pizzaria do bairro)",
        "base": texto + ", sem maioria clara",
        "pct_em_casa": casa["pct"],
    }


CANAL_CRIADORES = {
    "awareness_associacao": "criadores de conteúdo do segmento",
    "criar_frequencia": "criadores de conteúdo (online), depois pizzaria do bairro (local)",
}


def _canal_da_categoria(categoria: str, contexto, tabela: dict | None) -> dict:
    """Canal do plano, que é também o `recommendation_channel` da resposta."""
    if categoria in CANAL_CRIADORES:
        return {
            "valor": CANAL_CRIADORES[categoria],
            "base": "a categoria constrói associação antes da oferta: começa por conteúdo",
        }
    c = canal(contexto.canal, contexto.regiao, tabela)
    if categoria == "combo_em_casa" and not contexto.canal:
        # Regra 4 sem canal: a categoria já é consumo em casa; o dado da região vai como base.
        base = c["base"]
        if c["valor"] != CANAL_PLANO["em_casa"]:
            base += "; informe o canal se a ação for em outro lugar"
        return {**c, "valor": CANAL_PLANO["em_casa"], "base": base}
    return c


def medicao(categoria: str, contexto, lacuna: list[dict], presenca: list[dict], tabela) -> dict:
    """Indicador e linha de base na POF; a meta fica vazia (sem dado de efeito)."""
    if presenca:
        s = presenca[0]
        return {
            "indicador": "fatia do Guaraná no refrigerante em geral do segmento",
            "segmento": s["segmento_texto"],
            "linha_de_base_pct": s["guarana_refri_geral_pct"],
            "ic95": s["guarana_refri_geral_ic95"],
            "meta": None,
        }
    if lacuna:
        s = lacuna[0]
        return {
            "indicador": "fatia do Guaraná nas ocasiões de pizza com refrigerante do segmento",
            "segmento": s["segmento_texto"],
            "linha_de_base_pct": s["guarana_na_pizza_pct"],
            "ic95": s["guarana_na_pizza_ic95"],
            "meta": None,
        }
    seg = _segmento(tabela, "regiao", contexto.regiao)
    if seg is None:
        return {"indicador": "fatia do Guaraná nas ocasiões de pizza com refrigerante",
                "segmento": None, "linha_de_base_pct": None, "ic95": None, "meta": None}  # fmt: skip
    return {
        "indicador": "fatia do Guaraná nas ocasiões de pizza com refrigerante da região",
        "segmento": f"no {contexto.regiao}",
        "linha_de_base_pct": seg["pizza"]["pct"],
        "ic95": seg["pizza"]["ic95"],
        "ocasioes_na_amostra": seg["pizza"]["n"],
        "meta": None,
    }


def montar(
    *,
    categoria: str,
    contexto,
    perfil_texto: str,
    percentil: float | None,
    lacuna: list[dict],
    presenca: list[dict],
    tabela: dict | None,
    momento: dict,
    noite_de_semana: dict | None,
    medido: bool,
) -> dict | None:
    """Plano da recomendação; None quando a categoria é sem ativação.

    `momento` = {"dia_semana", "dia", "turno"} em que se age (na regra 5, a melhor noite de
    segunda a quinta, ou o melhor momento se essa noite for fraca).
    `medido` = a faixa de idade está no público em que a associação foi medida (20 a 44).
    """
    if categoria not in MECANICA:
        return None
    tab = tabela if medido else None
    onde = cidades(contexto.regiao, contexto.tipo_area)
    uf = onde["lista"][0]["uf"] if onde["lista"] else None
    brasil = (tab or {}).get("brasil", {})
    pico = brasil.get("inicio_19h_21h")
    horario = None
    if momento["turno"] == "noite" and pico:
        horario = {"janela": "19h a 21h", "pct_ocasioes_comecam": pico["pct"], "ic95": pico["ic95"],
                   "base": "início das ocasiões de pizza com refrigerante, Brasil, 20 a 44 anos"}  # fmt: skip
    gatilho = [s["segmento_texto"] for s in (presenca or lacuna)]
    mecanica, mensagem = MECANICA[categoria]
    if categoria == "criar_frequencia" and noite_de_semana is None:
        # Regra 5 com todas as noites de semana na faixa baixa: a ação vai para o melhor
        # momento do perfil, e o texto não pode falar em noite de semana.
        mecanica, mensagem = MECANICA_SEM_NOITE_DE_SEMANA
    return {
        "para_quem": {
            "perfil": perfil_texto,
            "segmento_do_sinal": gatilho,
            "percentil_na_classe": percentil,
        },
        "onde": {"cidades": onde["lista"], "criterio": onde["criterio"]},
        "quando": {
            "dia": momento["dia"],
            "dia_semana": momento["dia_semana"],
            "turno": momento["turno"],
            "horario": horario,
            "noite_de_semana": noite_de_semana,
            "proximas_4_semanas": janela(uf),
        },
        "canal": _canal_da_categoria(categoria, contexto, tab),
        "mecanica": {"texto": mecanica, "mensagem": mensagem, "natureza": HIPOTESE},
        "medicao": medicao(categoria, contexto, lacuna, presenca, tab),
    }
