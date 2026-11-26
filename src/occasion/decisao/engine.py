"""Decision Engine do Occasion Score.

Recebe um contexto (momento, lugar e público), calcula o Occasion Score e os
fatores com o modelo final e aplica regras explícitas, em ordem, para
escolher uma categoria de ativação com justificativa. Não é uma segunda
previsão: sem histórico de ações e resultados, a escolha é uma regra
documentada, e o sistema não afirma que uma ação vende mais que outra.

Regras (aprovadas em 21/11; a primeira que vale decide):

1. Faixa baixa fora da noite → sem ativação. 93% das ocasiões de pizza com
   refrigerante são à noite (POF 2017-18).
2. Faixa alta → conversão no momento da pizza (combo pizza + Guaraná). O
   guaraná aparece em só 21% dessas ocasiões (IC95 15% a 28%), sem diferença
   significativa entre segmentos.
3. Faixa média ou baixa, mas o mesmo perfil chega à faixa alta no seu
   melhor momento da semana → criar frequência: criadores online, depois
   pizzaria do bairro. Que a ação crie consumo novo, e não só mude o dia, é
   hipótese (sem dado).
4. Faixa média e classe B a D → awareness e associação com criadores do
   segmento. B a D somam 86% das ocasiões; a classe A, 5%.
5. Resto → sem ativação (monitorar).

Posição na classe (26/11): além da faixa nacional, a saída traz a posição
do score entre as pessoas-turno da mesma classe de renda em 2017-18. A regra
continua a mesma; a posição serve para mirar a expansão B a D sem inflar o
score. A classe D está calibrada (previsto ÷ observado 0,95, IC95 0,71 a
1,35, `vies_por_grupo.md`), mas a ocasião é menos frequente nela, então
poucos perfis da D chegam à faixa alta nacional.

Incerteza (26/11): com as réplicas do bootstrap (`modelagem.incerteza`), a
saída traz o IC95 do score e do melhor momento, a fração das réplicas em
cada faixa, as ocasiões reais do perfil e de cada entrada e os avisos de
poucos dados. A regra continua decidida pelo score pontual; se a faixa está
instável, a justificativa diz isso.

As fontes dos números estão no relatório `avaliacao_occasion_score.md` e no
documento do plano (seção "Decision Engine"); cada resposta lista fonte, ano
e link de cada número (`decisao.fontes`), e o que o modelo não sabe.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd

from occasion.decisao import fontes
from occasion.modelagem import final
from occasion.modelagem import incerteza as inc
from occasion.modelagem.tabela import FAIXAS_IDADE, TURNOS

DIAS = {
    1: "segunda",
    2: "terça",
    3: "quarta",
    4: "quinta",
    5: "sexta",
    6: "sábado",
    7: "domingo",
}
REGIOES = ["Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste"]
TIPOS_AREA = ["capital", "resto_rm", "interior_urbano", "rural"]
SEXOS = {"homem": 1, "mulher": 2}
CANAIS = ["delivery", "pizzaria", "em_casa"]
CLASSES_EXPANSAO = {"B", "C", "D"}
# Topo dentro da classe: mesmo corte da faixa alta nacional (10% maiores).
TOPO_NA_CLASSE = 90.0


class Categoria(StrEnum):
    CONVERSAO = "conversao_no_momento"
    CRIAR_FREQUENCIA = "criar_frequencia"
    AWARENESS = "awareness_associacao"
    SEM_ATIVACAO = "sem_ativacao"


NOMES = {
    Categoria.CONVERSAO: "Conversão no momento da pizza (combo pizza + Guaraná)",
    Categoria.CRIAR_FREQUENCIA: "Criar frequência: criadores online, depois pizzaria do bairro",
    Categoria.AWARENESS: "Awareness e associação com criadores do segmento",
    Categoria.SEM_ATIVACAO: "Sem ativação",
}

CANAL_TEXTO = {
    "delivery": "app de delivery",
    "pizzaria": "pizzaria do bairro",
    "em_casa": "consumo em casa (inclui delivery; a POF não separa delivery)",
}


def classe_de_renda(renda_sm: float) -> str:
    """Classe pela renda mensal do domicílio em salários mínimos (corte usual do IBGE)."""
    for limite, classe in [(2, "E"), (4, "D"), (10, "C"), (20, "B")]:
        if renda_sm <= limite:
            return classe
    return "A"


@dataclass(frozen=True)
class Contexto:
    """Entrada do sistema: momento, lugar, público e canal (opcional)."""

    turno: str
    dia_semana: int
    regiao: str
    tipo_area: str
    faixa_idade: str
    sexo: str
    renda_sm: float
    canal: str | None = None

    def __post_init__(self) -> None:
        erros = []
        if self.turno not in TURNOS:
            erros.append(f"turno deve ser um de {list(TURNOS)}")
        # 6.0 == 6 passaria no `in`, mas vira a categoria "6.0", que o modelo não conhece.
        if type(self.dia_semana) is not int or self.dia_semana not in DIAS:
            erros.append("dia_semana deve ser de 1 (segunda) a 7 (domingo)")
        if self.regiao not in REGIOES:
            erros.append(f"regiao deve ser uma de {REGIOES}")
        if self.tipo_area not in TIPOS_AREA:
            erros.append(f"tipo_area deve ser um de {TIPOS_AREA}")
        if self.faixa_idade not in FAIXAS_IDADE:
            erros.append(f"faixa_idade deve ser uma de {list(FAIXAS_IDADE)}")
        if self.sexo not in SEXOS:
            erros.append(f"sexo deve ser um de {list(SEXOS)}")
        numero = isinstance(self.renda_sm, int | float) and not isinstance(self.renda_sm, bool)
        if not numero or not np.isfinite(self.renda_sm) or self.renda_sm < 0:
            erros.append("renda_sm deve ser um número maior ou igual a zero")
        if self.canal is not None and self.canal not in CANAIS:
            erros.append(f"canal deve ser um de {CANAIS} ou vazio")
        if erros:
            raise ValueError("; ".join(erros))

    @property
    def classe(self) -> str:
        return classe_de_renda(self.renda_sm)


@dataclass
class Recomendacao:
    """Saída no contrato do case, com o detalhe da regra aplicada."""

    occasion_score: float
    classification: str
    top_factors: list[dict]
    recommendation_category: str
    recommendation_name: str
    recommendation_reason: str
    recommendation_channel: str | None
    messages: list[str]
    rule: int
    best_moment: dict
    class_position: dict = field(default_factory=dict)
    uncertainty: dict = field(default_factory=dict)
    sources: list[dict] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    model_metadata: dict = field(default_factory=dict)

    def como_dict(self) -> dict:
        return asdict(self)


def _linhas(modelo: final.OccasionScore, contextos: list[Contexto]) -> pd.DataFrame:
    """Linhas no formato das features do modelo."""
    df = pd.DataFrame(
        {
            "turno": [c.turno for c in contextos],
            "dia_semana": [str(c.dia_semana) for c in contextos],
            "regiao": [c.regiao for c in contextos],
            "tipo_area": [c.tipo_area for c in contextos],
            "faixa_idade": [c.faixa_idade for c in contextos],
            "sexo": [str(SEXOS[c.sexo]) for c in contextos],
            "renda_pct": modelo.renda_pct(np.array([c.renda_sm for c in contextos], dtype=float)),
        }
    )
    for col in df.columns.drop("renda_pct"):
        df[col] = df[col].astype("category")
    return df


def melhor_momento(
    modelo: final.OccasionScore, contexto: Contexto, incerteza: inc.Incerteza | None = None
) -> dict:
    """Dia e turno de maior score para o mesmo perfil (mesmo lugar e público)."""
    grade = [
        Contexto(**{**asdict(contexto), "dia_semana": d, "turno": tr})
        for d in DIAS
        for tr in TURNOS
    ]
    linhas = _linhas(modelo, grade)
    score = modelo.prever(linhas)
    i = int(np.argmax(score))
    faixa = str(modelo.faixa(score[[i]])[0])
    ic = incerteza.intervalo(modelo, linhas.iloc[[i]])[0] if incerteza else None
    return {
        "dia_semana": grade[i].dia_semana,
        "dia": DIAS[grade[i].dia_semana],
        "turno": grade[i].turno,
        "occasion_score": round(100 * float(score[i]), 3),
        "occasion_score_ic95": ic["ic95"] if ic else None,
        "classification": faixa,
        "prob_faixa": ic["prob_faixa"] if ic else None,
        "faixa_estavel": ic["prob_faixa"][faixa] >= inc.ESTABILIDADE_MIN if ic else None,
    }


NOMES_FAIXA = {"alta": "alta", "media": "média", "baixa": "baixa"}
SEXO_TEXTO = {str(v): k for k, v in SEXOS.items()}


def _num(x: float) -> str:
    """Score com casas suficientes para não virar 0,00 na faixa baixa."""
    return f"{x:.2f}" if x >= 0.1 else f"{x:.2g}"


def _valor_texto(feature: str, valor: str) -> str:
    if feature == "dia_semana":
        return DIAS[int(valor)]
    if feature == "sexo":
        return SEXO_TEXTO.get(valor, valor)
    return valor


def avaliar_incerteza(
    modelo: final.OccasionScore,
    incerteza: inc.Incerteza | None,
    linha: pd.DataFrame,
    classe: str,
    faixa: str,
) -> dict:
    """Intervalo, estabilidade da faixa, suporte nos dados e avisos de um contexto."""
    if incerteza is None:
        return {
            "disponivel": False,
            "occasion_score_ic95": None,
            "prob_faixa": None,
            "faixa_estavel": None,
            "poucos_dados": None,
            "avisos": ["Intervalo indisponível: as réplicas do bootstrap não foram encontradas."],
            "suporte": None,
            "metodo": None,
        }
    ic = incerteza.intervalo(modelo, linha)[0]
    sup = incerteza.suporte_de(linha.iloc[0], classe)
    avisos, poucos = [], False
    for e in sup["por_entrada"]:
        if e["ocasioes"] < inc.MIN_OCASIOES_ENTRADA:
            poucos = True
            rotulo = "classe de renda" if e["feature"] == "classe" else final.ROTULOS[e["feature"]]
            avisos.append(
                f"Poucos dados para {rotulo} = {_valor_texto(e['feature'], e['valor'])}: "
                f"{e['ocasioes']} ocasiões nas duas edições."
            )
    lo, hi = ic["ic95"]
    if lo <= 0 or hi / lo > inc.RAZAO_IC_MAX:
        poucos = True
        avisos.append(
            f"Poucos dados para esta combinação: o intervalo vai de {_num(lo)} a {_num(hi)}, e o "
            f"limite superior passa de {inc.RAZAO_IC_MAX:.0f}× o inferior."
        )
    estavel = ic["prob_faixa"][faixa] >= inc.ESTABILIDADE_MIN
    if not estavel:
        avisos.append(
            f"Faixa na fronteira: o score cai na faixa {NOMES_FAIXA[faixa]} em "
            f"{100 * ic['prob_faixa'][faixa]:.0f}% das reamostragens."
        )
    meta = incerteza.metadata
    return {
        "disponivel": True,
        "occasion_score_ic95": ic["ic95"],
        "prob_faixa": ic["prob_faixa"],
        "faixa_estavel": estavel,
        "poucos_dados": poucos,
        "avisos": avisos,
        "suporte": sup,
        "metodo": f"{meta.get('metodo')}; {meta.get('replicas')} réplicas",
    }


def _mensagens(dia_semana: int, categoria: Categoria) -> list[str]:
    """Mensagens do momento em que se vai agir (na regra 3, o melhor momento do perfil)."""
    if categoria == Categoria.SEM_ATIVACAO:
        return []
    momento = "ritual de fim de semana" if dia_semana in (5, 6, 7) else "pausa no meio da semana"
    return [
        f'"Coisa Nossa" como {momento} (hipótese de mensagem, sem dado por segmento)',
        (
            "Guaraná Zero como opção de produto: o zero foi de 8,8% para 14,4% do valor da "
            "categoria entre 2023 e 2025 (Worldpanel); a POF não mede isso por segmento"
        ),
    ]


def _texto_fatores(fatores: list[dict]) -> str:
    partes = [
        f"{f['rotulo']} = {f['valor']} ({f['direcao']} a chance, ×{f['razao_chances']:.2f})"
        for f in fatores
    ]
    return "; ".join(partes)


def regra(faixa: str, turno: str, classe: str, faixa_melhor_momento: str) -> tuple[int, Categoria]:
    """As 5 regras, em ordem. Separada para ser testada sem o modelo."""
    if faixa == "baixa" and turno != "noite":
        return 1, Categoria.SEM_ATIVACAO
    if faixa == "alta":
        return 2, Categoria.CONVERSAO
    if faixa_melhor_momento == "alta":
        return 3, Categoria.CRIAR_FREQUENCIA
    if faixa == "media" and classe in CLASSES_EXPANSAO:
        return 4, Categoria.AWARENESS
    return 5, Categoria.SEM_ATIVACAO


def posicao_na_classe(modelo: final.OccasionScore, score: float, classe: str) -> dict:
    """Percentil do score entre as pessoas-turno da mesma classe (2017-18)."""
    pct = float(modelo.percentil_na_classe(np.array([score]), [classe])[0])
    if not np.isfinite(pct):
        return {"classe": classe, "percentil_na_classe": None, "topo_10_na_classe": None}
    pct = round(pct, 1)
    return {
        "classe": classe,
        "percentil_na_classe": pct,
        "topo_10_na_classe": pct >= TOPO_NA_CLASSE,
    }


def decidir(
    contexto: Contexto,
    modelo: final.OccasionScore,
    n_fatores: int = 3,
    incerteza: inc.Incerteza | None = None,
) -> Recomendacao:
    linha = _linhas(modelo, [contexto])
    score = float(modelo.prever(linha)[0])
    faixa = str(modelo.faixa(np.array([score]))[0])
    fatores = modelo.fatores(linha, n=n_fatores)[0]
    for f in fatores:  # valores legíveis no lugar dos códigos do modelo
        if f["feature"] == "dia_semana":
            f["valor"] = DIAS[contexto.dia_semana]
        elif f["feature"] == "sexo":
            f["valor"] = contexto.sexo
    melhor = melhor_momento(modelo, contexto, incerteza)
    incerto = avaliar_incerteza(modelo, incerteza, linha, contexto.classe, faixa)
    n, categoria = regra(faixa, contexto.turno, contexto.classe, melhor["classification"])

    canal = None
    if categoria == Categoria.CONVERSAO:
        canal = CANAL_TEXTO[contexto.canal or "em_casa"]
    elif categoria == Categoria.CRIAR_FREQUENCIA:
        canal = "criadores de conteúdo (online), depois pizzaria do bairro (local)"
    elif categoria == Categoria.AWARENESS:
        canal = "criadores de conteúdo do segmento"

    momento = f"{DIAS[contexto.dia_semana]}, {contexto.turno}"
    ic = incerto["occasion_score_ic95"]
    texto_ic = f", IC95 {_num(ic[0])} a {_num(ic[1])}" if ic else ""
    base = (
        f"Occasion Score {100 * score:.2f} de 100{texto_ic} (faixa {faixa}) para {momento}. "
        f"Fatores: {_texto_fatores(fatores)}."
    )
    if incerto["faixa_estavel"] is False:
        base += (
            f" Atenção: a faixa {NOMES_FAIXA[faixa]} aparece em só "
            f"{100 * incerto['prob_faixa'][faixa]:.0f}% das reamostragens; a regra usa o score "
            "pontual."
        )
    if n == 3 and melhor["faixa_estavel"] is False:
        base += (
            f" Atenção: o melhor momento fica na faixa alta em só "
            f"{100 * melhor['prob_faixa']['alta']:.0f}% das reamostragens; a regra 3 usa o score "
            "pontual."
        )
    motivos = {
        1: "Faixa baixa fora da noite: 93% das ocasiões de pizza com refrigerante são à noite.",
        2: "Faixa alta: a ocasião é provável, e o guaraná aparece em só 21% delas "
        "(IC95 15% a 28%, POF 2017-18), sem segmento que difira disso.",
        3: f"O momento pedido é fraco, mas o mesmo perfil chega à faixa alta em "
        f"{melhor['dia']}, {melhor['turno']} (score {melhor['occasion_score']:.2f}). Há afinidade "
        "para criar frequência; que a ação gere consumo novo é hipótese (sem dado).",
        4: f"Faixa média na classe {contexto.classe}: as classes B a D somam 86% das ocasiões "
        "(POF 2017-18), e a Ambev quer expandir além da classe A.",
        5: (
            f"Faixa média na classe {contexto.classe}, fora das classes B a D, e sem momento da "
            "semana em faixa alta para este perfil."
            if faixa == "media"
            else "Score baixo e sem momento da semana em faixa alta para este perfil."
        ),
    }
    mensagens = _mensagens(
        melhor["dia_semana"] if categoria == Categoria.CRIAR_FREQUENCIA else contexto.dia_semana,
        categoria,
    )
    return Recomendacao(
        occasion_score=round(100 * score, 3),
        classification=faixa,
        top_factors=fatores,
        recommendation_category=categoria.value,
        recommendation_name=NOMES[categoria],
        recommendation_reason=f"Regra {n}. {motivos[n]} {base}",
        recommendation_channel=canal,
        messages=mensagens,
        rule=n,
        best_moment=melhor,
        class_position=posicao_na_classe(modelo, score, contexto.classe),
        uncertainty=incerto,
        sources=fontes.fontes_da_resposta(n, com_mensagens=bool(mensagens)),
        limitations=list(fontes.LIMITES),
        model_metadata={
            k: modelo.metadata.get(k) for k in ("modelo", "versao", "treinado_em", "features")
        },
    )
