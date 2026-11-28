"""Fonte e ano de cada número da resposta, e o que o modelo não sabe.

Cada número que aparece na justificativa do Decision Engine tem aqui a sua
fonte, o ano do dado e um link. Os números da POF foram calculados pelo
projeto e estão no relatório de avaliação; o da Worldpanel é de fonte
secundária (notícia que cita o estudo), marcado como tal.
"""

from __future__ import annotations

REPO = "https://github.com/gnfes/ambev-occasion-intelligence/blob/develop"
POF_FTP = "https://ftp.ibge.gov.br/Orcamentos_Familiares"
POF_2017 = f"{POF_FTP}/Pesquisa_de_Orcamentos_Familiares_2017_2018/Microdados/"
AVALIACAO = f"{REPO}/docs/relatorios/avaliacao_occasion_score.md"
VIES = f"{REPO}/docs/relatorios/vies_por_grupo.md"

_POF_CALCULO = "POF 2017-18 (IBGE, microdados), cálculo do projeto"

FONTES: dict[str, dict] = {
    "score": {
        "numero": "Occasion Score e faixas",
        "fonte": "Modelo do projeto sobre a POF 2008-09 e 2017-18 (IBGE, microdados), "
        "com a correção por grupo",
        "ano": "2008-09 e 2017-18",
        "url": f"{POF_FTP}/",
        "detalhe": VIES,
    },
    "intervalo": {
        "numero": "Intervalo de confiança (IC95) e estabilidade da faixa",
        "fonte": "Bootstrap por setor censitário sobre a POF 2008-09 e 2017-18, cálculo do projeto",
        "ano": "2008-09 e 2017-18",
        "url": f"{POF_FTP}/",
        "detalhe": None,
    },
    "media_populacao": {
        "numero": "Média da população (taxa observada de pizza com refrigerante por "
        "pessoa-turno, cada dia da semana com peso 1/7)",
        "fonte": _POF_CALCULO,
        "ano": "2017-18",
        "url": POF_2017,
        "detalhe": AVALIACAO,
    },
    "noite_93": {
        "numero": "93% das ocasiões de pizza com refrigerante são à noite",
        "fonte": _POF_CALCULO,
        "ano": "2017-18",
        "url": POF_2017,
        "detalhe": AVALIACAO,
    },
    "classes_86": {
        "numero": "Classes B a D somam 86% das ocasiões; a classe A, 5%",
        "fonte": _POF_CALCULO,
        "ano": "2017-18",
        "url": POF_2017,
        "detalhe": AVALIACAO,
    },
    "associacao": {
        "numero": "Fatia do Guaraná no refrigerante em geral e na pizza por segmento (lacuna de "
        "associação e baixa presença da marca)",
        "fonte": _POF_CALCULO + "; bootstrap por setor, critérios em decisao/associacao.py",
        "ano": "2017-18",
        "url": POF_2017,
        "detalhe": f"{REPO}/data/referencia/associacao_guarana.json",
    },
    "em_casa_66": {
        "numero": "66% das ocasiões de pizza com refrigerante são em casa (delivery incluído)",
        "fonte": _POF_CALCULO,
        "ano": "2017-18",
        "url": POF_2017,
        "detalhe": f"{REPO}/data/referencia/associacao_guarana.json",
    },
    "mapa": {
        "numero": "Cidades por prioridade: potencial do Occasion Score na população de 20 a 44 "
        "anos e pizzarias ativas por 100 mil habitantes",
        "fonte": "Censo 2022 (IBGE) e CNPJ (Receita Federal), cálculo do projeto",
        "ano": "2022 e 2026",
        "url": f"{REPO}/docs/relatorios/mapa_oportunidade.md",
        "detalhe": f"{REPO}/data/referencia/mapa_oportunidade.csv",
    },
    "inicio_19_21": {
        "numero": "70% das ocasiões de pizza com refrigerante começam entre 19h e 21h",
        "fonte": _POF_CALCULO,
        "ano": "2017-18",
        "url": POF_2017,
        "detalhe": f"{REPO}/data/referencia/associacao_guarana.json",
    },
    "radar": {
        "numero": "Próximas 4 semanas da UF, com feriados e desvio previsto da busca por pizza",
        "fonte": "Google Trends, feriados nacionais e Open-Meteo; Trend Radar do projeto "
        "(informativo, sem ganho sobre o baseline)",
        "ano": "2026",
        "url": f"{REPO}/docs/relatorios/trend_radar.md",
        "detalhe": f"{REPO}/data/referencia/radar_alertas.json",
    },
    "zero_worldpanel": {
        "numero": "Refrigerante zero foi de 8,8% para 14,4% do valor da categoria",
        "fonte": "Worldpanel by Numerator, Consumer Insights 3º tri 2025, citado por "
        "Mercado & Consumo em 16/12/2025 (fonte secundária)",
        "ano": "2023 a 2025",
        "url": "https://mercadoeconsumo.com.br/16/12/2025/economia/"
        "brasileiros-optam-por-habitos-mais-saudaveis-e-buscam-alimentos-com-menos-acucar/",
        "detalhe": None,
    },
}

# Números citados por regra (além do score, do intervalo e da média, sempre presentes).
POR_REGRA: dict[int, list[str]] = {
    1: ["noite_93"],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
    7: [],
}

LIMITES: list[str] = [
    (
        "Os dados são da POF 2017-18, a mais recente com microdados de consumo; o hábito pode ter "
        "mudado desde então, e o intervalo de confiança não cobre essa mudança."
    ),
    (
        "A POF não tem a data do consumo, só o dia da semana e a hora: clima, feriados e época do "
        "ano não entram no score (o Trend Radar olha para isso com dados de busca)."
    ),
    "Sem preço, promoção ou ponto de venda: o score não mede o efeito de uma oferta.",
    "A POF não separa delivery: comida pedida em casa conta como consumo em casa.",
    (
        "O alvo é pizza com qualquer refrigerante: o score não diz onde o Guaraná perde para outra "
        "marca. Isso entra no Engine pela associação por segmento, inferida pela fatia do Guaraná "
        "na POF (não é pesquisa de marca), com amostra pequena na pizza."
    ),
    (
        "É associação, não causa: o score diz onde a ocasião acontece, não que uma ação vai criar "
        "consumo novo."
    ),
    (
        "O intervalo mede a variação da amostra da POF (bootstrap por setor). Ele não cobre erro de "
        "especificação do modelo."
    ),
]


def fontes_da_resposta(
    regra: int,
    com_mensagens: bool,
    com_associacao: bool = False,
    cita_em_casa: bool = False,
    plano: dict | None = None,
    com_intervalo: bool = True,
) -> list[dict]:
    """Fonte e ano de cada número que aparece numa recomendação."""
    chaves = ["score", *(["intervalo"] if com_intervalo else []), "media_populacao"]
    chaves += POR_REGRA.get(regra, [])
    if regra in (2, 3, 6) and com_associacao:
        chaves.append("associacao")
    if regra == 6 and not com_associacao:
        chaves.append("classes_86")
    if regra == 4 and cita_em_casa:
        chaves.append("em_casa_66")
    if com_mensagens:
        chaves.append("zero_worldpanel")
    if plano:
        if plano["onde"]["cidades"]:
            chaves.append("mapa")
        if plano["quando"]["horario"]:
            chaves.append("inicio_19_21")
        if plano["quando"]["proximas_4_semanas"]:
            chaves.append("radar")
        if "associacao" not in chaves and plano["medicao"]["linha_de_base_pct"] is not None:
            chaves.append("associacao")
    return [{"id": k, **FONTES[k]} for k in chaves]
