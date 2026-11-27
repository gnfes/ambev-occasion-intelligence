"""Decision Engine do Occasion Score.

Recebe um contexto (momento, lugar e público), calcula o Occasion Score e os
fatores com o modelo final e aplica regras explícitas, em ordem, para
escolher uma categoria de ativação com justificativa. Não é uma segunda
previsão: sem histórico de ações e resultados, a escolha é uma regra
documentada, e o sistema não afirma que uma ação vende mais que outra.

Regras (aprovadas em 21/11 e revistas em 27/11 para seguir a seção 7 do
TAPI: a categoria depende do score, do contexto e da associação à marca no
segmento, e não só da faixa; a primeira regra que vale decide):

1. Faixa baixa fora da noite → sem ativação. 93% das ocasiões de pizza com
   refrigerante são à noite (POF 2017-18).
2. Faixa alta + baixa presença da marca no segmento → awareness e construção
   de associação, com criadores do segmento. O Guaraná já é fraco no
   refrigerante em geral ali (`associacao`).
3. Faixa alta + lacuna de associação no segmento → conversão: levar o
   Guaraná para a pizza, pelo canal informado. O Guaraná é forte no
   refrigerante do dia a dia e some na pizza (`associacao`).
4. Faixa alta, sem sinal de associação → o canal decide: delivery →
   ativação via delivery; pizzaria → ação na pizzaria; em casa ou sem canal →
   combo para consumo em casa (66% das ocasiões são em casa, com delivery
   incluído, POF 2017-18).
5. Faixa média ou baixa, mas o mesmo perfil chega à faixa alta no seu
   melhor momento da semana → criar frequência: criadores online, depois
   pizzaria do bairro. Que a ação crie consumo novo, e não só mude o dia, é
   hipótese (sem dado).
6. Faixa média + (baixa presença, lacuna ou classe B a D) → awareness e
   associação com criadores do segmento. B a D somam 86% das ocasiões.
7. Resto → sem ativação (monitorar).

Quando o segmento tem baixa presença e lacuna ao mesmo tempo, vale a baixa
presença (regra 2 antes da 3), decisão de 27/11.

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
from functools import lru_cache

import numpy as np
import pandas as pd

from occasion.decisao import associacao, fontes, plano
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
    CONVERSAO_ASSOCIACAO = "conversao_associacao"
    ATIVACAO_DELIVERY = "ativacao_delivery"
    COMBO_EM_CASA = "combo_em_casa"
    ACAO_PIZZARIA = "acao_pizzaria"
    CRIAR_FREQUENCIA = "criar_frequencia"
    AWARENESS = "awareness_associacao"
    SEM_ATIVACAO = "sem_ativacao"


NOMES = {
    Categoria.CONVERSAO_ASSOCIACAO: "Conversão: levar o Guaraná para a pizza",
    Categoria.ATIVACAO_DELIVERY: "Ativação via delivery (combo pizza + Guaraná no pedido)",
    Categoria.COMBO_EM_CASA: "Combo pizza + Guaraná para consumo em casa",
    Categoria.ACAO_PIZZARIA: "Ação na pizzaria do bairro (pizza + Guaraná no salão)",
    Categoria.CRIAR_FREQUENCIA: "Criar frequência: criadores online, depois pizzaria do bairro",
    Categoria.AWARENESS: "Awareness e construção de associação com criadores do segmento",
    Categoria.SEM_ATIVACAO: "Sem ativação",
}
# Regra 4: o canal informado escolhe a categoria (sem canal, consumo em casa).
POR_CANAL = {
    "delivery": Categoria.ATIVACAO_DELIVERY,
    "pizzaria": Categoria.ACAO_PIZZARIA,
    "em_casa": Categoria.COMBO_EM_CASA,
    None: Categoria.COMBO_EM_CASA,
}
ROTULO_SEGMENTO = {
    "regiao": "região",
    "tipo_area": "tipo de área",
    "classe": "classe",
    "faixa_idade": "faixa de idade",
    "sexo": "sexo",
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
    association: dict = field(default_factory=dict)
    activation_plan: dict | None = None
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
        "faixa_estavel": len(set(ic["faixas_ic95"])) == 1 if ic else None,
    }


def noite_de_semana(modelo: final.OccasionScore, contexto: Contexto) -> dict:
    """Noite de segunda a quinta de maior score para o mesmo perfil (regra 5, frequência)."""
    grade = [
        Contexto(**{**asdict(contexto), "dia_semana": d, "turno": "noite"}) for d in (1, 2, 3, 4)
    ]
    score = modelo.prever(_linhas(modelo, grade))
    i = int(np.argmax(score))
    return {
        "dia_semana": grade[i].dia_semana,
        "dia": DIAS[grade[i].dia_semana],
        "occasion_score": round(100 * float(score[i]), 3),
        "classification": str(modelo.faixa(score[[i]])[0]),
    }


AREA_PERFIL = {
    "capital": "capital",
    "resto_rm": "região metropolitana",
    "interior_urbano": "interior urbano",
    "rural": "área rural",
}


def _perfil_texto(c: Contexto) -> str:
    return (
        f"{c.sexo}, {c.faixa_idade.replace('-', ' a ')} anos, classe {c.classe}, {c.regiao}, "
        f"{AREA_PERFIL.get(c.tipo_area, c.tipo_area)}"
    )


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
    f_lo, f_hi = ic["faixas_ic95"]
    estavel = f_lo == f_hi
    if not estavel:
        avisos.append(
            f"Faixa incerta: o intervalo vai de {_num(lo)} (faixa {NOMES_FAIXA[f_lo]}) a "
            f"{_num(hi)} (faixa {NOMES_FAIXA[f_hi]}); o score fica na faixa "
            f"{NOMES_FAIXA[faixa]} em {100 * ic['prob_faixa'][faixa]:.0f}% das reamostragens."
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
    """Mensagens do momento em que se vai agir (na regra 5, a melhor noite de semana)."""
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


def regra(
    faixa: str,
    turno: str,
    classe: str,
    faixa_melhor_momento: str,
    baixa_presenca: bool = False,
    lacuna: bool = False,
    canal: str | None = None,
) -> tuple[int, Categoria]:
    """As 7 regras, em ordem. Separada para ser testada sem o modelo."""
    if faixa == "baixa" and turno != "noite":
        return 1, Categoria.SEM_ATIVACAO
    if faixa == "alta" and baixa_presenca:
        return 2, Categoria.AWARENESS
    if faixa == "alta" and lacuna:
        return 3, Categoria.CONVERSAO_ASSOCIACAO
    if faixa == "alta":
        return 4, POR_CANAL[canal]
    if faixa_melhor_momento == "alta":
        return 5, Categoria.CRIAR_FREQUENCIA
    if faixa == "media" and (baixa_presenca or lacuna or classe in CLASSES_EXPANSAO):
        return 6, Categoria.AWARENESS
    return 7, Categoria.SEM_ATIVACAO


def _associacao_padrao() -> dict | None:
    """Tabela gravada, relida quando o arquivo muda (sem reiniciar a API)."""
    try:
        versao = associacao.ARQUIVO.stat().st_mtime_ns
    except FileNotFoundError:
        versao = None
    return _carregar_associacao(versao)


@lru_cache(maxsize=1)
def _carregar_associacao(versao: int | None) -> dict | None:
    return associacao.carregar() if versao is not None else None


def _segmentos(contexto: Contexto) -> dict[str, str]:
    return {
        "regiao": contexto.regiao,
        "tipo_area": contexto.tipo_area,
        "classe": contexto.classe,
        "faixa_idade": contexto.faixa_idade,
        "sexo": contexto.sexo,
    }


AREA_TEXTO = {
    "capital": "nas capitais",
    "resto_rm": "no resto das regiões metropolitanas",
    "interior_urbano": "no interior urbano",
    "rural": "na área rural",
}


def _segmento_texto(dimensao: str, valor: str) -> str:
    """Segmento em português corrido ("no Nordeste", "na classe B")."""
    if dimensao == "regiao":
        return f"no {valor}"
    if dimensao == "tipo_area":
        return AREA_TEXTO.get(valor, valor)
    if dimensao == "classe":
        return f"na classe {valor}"
    if dimensao == "faixa_idade":
        return f"na faixa de {valor.replace('-', ' a ')} anos"
    if dimensao == "sexo":
        return {"homem": "entre homens", "mulher": "entre mulheres"}.get(valor, f"sexo {valor}")
    return f"{dimensao} = {valor}"


def _resumo_segmento(s: dict) -> dict:
    return {
        "segmento": f"{ROTULO_SEGMENTO[s['dimensao']]} = {s['valor']}",
        "segmento_texto": _segmento_texto(s["dimensao"], s["valor"]),
        "guarana_refri_geral_pct": s["refri_geral"]["pct"],
        "guarana_refri_geral_ic95": s["refri_geral"]["ic95"],
        "guarana_na_pizza_pct": s["pizza"]["pct"],
        "guarana_na_pizza_ic95": s["pizza"]["ic95"],
        "lacuna_pp": s["lacuna_pp"]["valor"],
        "lacuna_ic95": s["lacuna_pp"]["ic95"],
        "ocasioes_pizza": s["pizza"]["n"],
    }


def _br(x: float) -> str:
    return f"{x:.1f}".replace(".", ",")


def _texto_lacuna(segs: list[dict]) -> str:
    return ". ".join(
        f"{s['segmento_texto'][0].upper() + s['segmento_texto'][1:]}, o Guaraná está em {_br(s['guarana_refri_geral_pct'])}% do "
        f"refrigerante em geral e em {_br(s['guarana_na_pizza_pct'])}% das ocasiões de pizza, "
        f"lacuna de {_br(s['lacuna_pp'])} pontos (IC95 {_br(s['lacuna_ic95'][0])} a "
        f"{_br(s['lacuna_ic95'][1])})"
        for s in segs
    )


def _texto_presenca(segs: list[dict], brasil: float | None) -> str:
    ref = f", contra {_br(brasil)}% no Brasil" if brasil is not None else ""
    return ". ".join(
        f"{s['segmento_texto'][0].upper() + s['segmento_texto'][1:]}, o Guaraná está em {_br(s['guarana_refri_geral_pct'])}% do "
        f"refrigerante em geral (IC95 {_br(s['guarana_refri_geral_ic95'][0])} a "
        f"{_br(s['guarana_refri_geral_ic95'][1])}){ref}"
        for s in segs
    )


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
    tabela_associacao: dict | None = None,
) -> Recomendacao:
    """Score, fatores e categoria de ativação de um contexto.

    `tabela_associacao` None usa a tabela gravada (`associacao.ARQUIVO`); um dict vazio
    desliga os sinais de associação (regras 2, 3 e o critério de associação da 6).
    """
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
    tabela = _associacao_padrao() if tabela_associacao is None else tabela_associacao
    # Os sinais foram medidos em 20 a 44 anos; fora disso o Engine não os usa.
    medido = contexto.faixa_idade in associacao.FAIXAS_MEDIDAS
    sinal = associacao.sinais(tabela if medido else None, _segmentos(contexto))
    lacuna = [_resumo_segmento(s) for s in sinal["lacuna"]]
    presenca = [_resumo_segmento(s) for s in sinal["baixa_presenca"]]
    brasil = tabela["brasil"] if tabela and medido else None
    n, categoria = regra(
        faixa,
        contexto.turno,
        contexto.classe,
        melhor["classification"],
        baixa_presenca=bool(presenca),
        lacuna=bool(lacuna),
        canal=contexto.canal,
    )

    posicao = posicao_na_classe(modelo, score, contexto.classe)
    # Regra 5: a frequência se cria numa noite de semana (a melhor de segunda a quinta).
    nds = noite_de_semana(modelo, contexto) if categoria == Categoria.CRIAR_FREQUENCIA else None
    if nds and nds["classification"] == "baixa":
        nds = None  # noite de semana fraca: a ação fica no melhor momento do perfil
    agir = (
        {"dia_semana": nds["dia_semana"], "dia": nds["dia"], "turno": "noite"}
        if nds
        else {"dia_semana": melhor["dia_semana"], "dia": melhor["dia"], "turno": melhor["turno"]}
        if categoria == Categoria.CRIAR_FREQUENCIA
        else {
            "dia_semana": contexto.dia_semana,
            "dia": DIAS[contexto.dia_semana],
            "turno": contexto.turno,
        }
    )
    plano_ativacao = plano.montar(
        categoria=categoria.value,
        contexto=contexto,
        perfil_texto=_perfil_texto(contexto),
        percentil=posicao.get("percentil_na_classe"),
        lacuna=lacuna,
        presenca=presenca,
        tabela=tabela,
        momento=agir,
        noite_de_semana=nds,
        medido=medido,
    )
    canal = plano_ativacao["canal"]["valor"] if plano_ativacao else None

    momento = f"{DIAS[contexto.dia_semana]}, {contexto.turno}"
    ic = incerto["occasion_score_ic95"]
    texto_ic = f", IC95 {_num(ic[0])} a {_num(ic[1])}" if ic else ""
    base = (
        f"Occasion Score {100 * score:.2f} de 100{texto_ic} (faixa {faixa}) para {momento}. "
        f"Fatores: {_texto_fatores(fatores)}."
    )
    if incerto["faixa_estavel"] is False:
        base += (
            f" Atenção: o intervalo cruza um corte de faixa (faixa {NOMES_FAIXA[faixa]} em "
            f"{100 * incerto['prob_faixa'][faixa]:.0f}% das reamostragens); a regra usa o score "
            "pontual."
        )
    if n == 5 and melhor["faixa_estavel"] is False:
        base += (
            f" Atenção: o intervalo do melhor momento cruza o corte da faixa alta (alta em "
            f"{100 * melhor['prob_faixa']['alta']:.0f}% das reamostragens); a regra 5 usa o "
            "score pontual."
        )
    ref_br = brasil["refri_geral"]["pct"] if brasil else None
    casa = brasil["em_casa"]["pct"] if brasil else None
    motivos = {
        1: "Faixa baixa fora da noite: 93% das ocasiões de pizza com refrigerante são à noite.",
        2: "Faixa alta num segmento onde o Guaraná já é fraco no refrigerante em geral. "
        f"{_texto_presenca(presenca, ref_br)}. Antes de converter na pizza, é preciso presença e "
        "associação.",
        3: "Faixa alta num segmento onde o Guaraná é forte no refrigerante do dia a dia, mas some "
        f"na pizza. {_texto_lacuna(lacuna)}. A marca já está na casa; falta ligá-la à pizza.",
        4: "Faixa alta, sem sinal de lacuna nem de baixa presença do Guaraná no segmento: "
        + (
            f"o canal informado ({plano.CANAL_PLANO[contexto.canal]}) decide a ação."
            if contexto.canal
            else "sem canal informado, a ação é para consumo em casa"
            + (f" ({_br(casa)}% das ocasiões são em casa, com delivery incluído)." if casa else ".")
        ),
        5: f"O momento pedido é fraco, mas o mesmo perfil chega à faixa alta em "
        f"{melhor['dia']}, {melhor['turno']} (score {melhor['occasion_score']:.2f}). Há afinidade "
        "para criar frequência; que a ação gere consumo novo é hipótese (sem dado)."
        + (
            f" A ação vai para {nds['dia']} à noite, a melhor noite de segunda a quinta deste "
            f"perfil (score {nds['occasion_score']:.2f}, faixa "
            f"{NOMES_FAIXA[nds['classification']]})."
            if nds
            else ""
        ),
        6: "Faixa média: "
        + (
            f"Guaraná fraco no refrigerante em geral. {_texto_presenca(presenca, ref_br)}."
            if presenca
            else f"lacuna de associação. {_texto_lacuna(lacuna)}."
            if lacuna
            else f"classe {contexto.classe}, e as classes B a D somam 86% das ocasiões "
            "(POF 2017-18); a Ambev quer expandir além da classe A."
        ),
        7: (
            f"Faixa média na classe {contexto.classe}, fora das classes B a D, sem sinal de "
            "associação e sem momento da semana em faixa alta para este perfil."
            if faixa == "media"
            else "Score baixo e sem momento da semana em faixa alta para este perfil."
        ),
    }
    mensagens = _mensagens(agir["dia_semana"], categoria)
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
        association={
            "disponivel": bool(tabela) and medido,
            "lacuna": lacuna,
            "baixa_presenca": presenca,
            "brasil_guarana_refri_geral_pct": ref_br,
        },
        activation_plan=plano_ativacao,
        class_position=posicao,
        uncertainty=incerto,
        sources=fontes.fontes_da_resposta(
            n,
            com_mensagens=bool(mensagens),
            com_associacao=bool(lacuna or presenca),
            cita_em_casa=contexto.canal is None and casa is not None,
            plano=plano_ativacao,
        ),
        limitations=list(fontes.LIMITES),
        model_metadata={
            k: modelo.metadata.get(k) for k in ("modelo", "versao", "treinado_em", "features")
        },
    )
