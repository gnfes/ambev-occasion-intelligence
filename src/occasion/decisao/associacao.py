"""Associação do Guaraná à pizza por segmento, para o Decision Engine.

Mede, no público de 20 a 44 anos da POF 2017-18, duas fatias do Guaraná:

- no refrigerante em geral: ocasiões de refrigerante (pessoa × dia × hora)
  em que houve Guaraná;
- na pizza: ocasiões de pizza com refrigerante (mesma definição do alvo do
  modelo: refrigerante até 1 hora da pizza, no turno da pizza) em que o
  refrigerante foi Guaraná.

Daí saem duas leituras por segmento (aprovadas em 27/11/2026):

- **lacuna de associação**: o Guaraná é forte no refrigerante do dia a dia e
  some na pizza. Critério: IC95 da diferença (geral − pizza) acima de zero e
  pelo menos `MIN_OCASIOES_PIZZA` ocasiões de pizza no segmento;
- **baixa presença da marca**: o Guaraná já é fraco no refrigerante em geral.
  Critério: IC95 da fatia no refrigerante em geral do segmento inteiro abaixo
  do IC95 nacional, sem sobreposição. (A proposta de 27/11 comparava com a
  estimativa pontual nacional; as capitais ficavam no fio, com o limite
  superior entre 25,2 e 25,5 conforme a reamostragem, contra 25,3. Sem
  sobreposição, o resultado não depende do sorteio e os segmentos aprovados
  são os mesmos: Sul e classe B.)

A associação é inferida pela fatia do Guaraná, não medida por pesquisa de
marca. Os IC95 vêm de bootstrap por setor censitário, com os pesos amostrais
e cada dia da semana valendo 1/7. Os critérios ficam fixos aqui, sem ajuste
depois de ver o resultado. A faixa de 30 a 34 anos passa no limite (IC95 da
lacuna começa em 0,6 ponto) e fica de fora por decisão de 27/11: são 17
comparações, e uma pode dar positivo por acaso.

Grava `data/referencia/associacao_guarana.json`, lido pelo Engine.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from occasion.ingestao.pof import caminho_parquet
from occasion.modelagem import tabela as T
from occasion.paths import REFERENCIA_DIR

ARQUIVO = REFERENCIA_DIR / "associacao_guarana.json"
EDICAO = "2017_2018"
IDADE_MIN, IDADE_MAX = 20, 44
# Faixas do modelo dentro do público medido; fora delas o Engine não usa os sinais.
FAIXAS_MEDIDAS = ("20-24", "25-29", "30-34", "35-39", "40-44")
MIN_OCASIOES_PIZZA = 60
REPLICAS = 1000
SEMENTE = 2026
DIMENSOES = ["regiao", "tipo_area", "classe", "faixa_idade", "sexo"]
# Segmentos que passam no critério só no limite e ficam de fora (decisão de 27/11/2026).
EXCLUIDOS = {("faixa_idade", "30-34")}
# Cortes de classe pela renda do domicílio em salários mínimos (mesmos do Engine).
CORTES_CLASSE = [(2, "E"), (4, "D"), (10, "C"), (20, "B")]
_PESSOA_DIA = ["id_pessoa", "dia_registro"]
# Janela de início da ocasião citada no plano de ativação (hora cheia, inclusive).
HORARIO_PICO = (19, 21)


def _classe(renda_sm: pd.Series) -> pd.Series:
    def uma(x: float) -> str | float:
        if pd.isna(x):  # renda ausente não vira classe A
            return np.nan
        for limite, classe in CORTES_CLASSE:
            if x <= limite:
                return classe
        return "A"

    return renda_sm.map(uma)


def _preparar(consumo: pd.DataFrame, areas: pd.DataFrame) -> pd.DataFrame:
    """Itens de consumo do público, com peso por dia, segmento e turno."""
    dias = consumo.groupby(_PESSOA_DIA, as_index=False).agg(dia_semana=("dia_semana", "first"))
    pessoas = T._pessoas(consumo, "2017-18")
    fator = T._fator_dia(dias, pessoas)
    dias["w_dia"] = fator.to_numpy()
    c = consumo.merge(dias[[*_PESSOA_DIA, "w_dia"]], on=_PESSOA_DIA, how="left")
    c = c[(c["idade"] >= IDADE_MIN) & (c["idade"] <= IDADE_MAX)].copy()
    c["w"] = c["peso"] * c["w_dia"].astype(float)
    c["classe"] = _classe(c["renda"] / T.SALARIO_MINIMO["2017-18"])
    limites = [9, *T.FAIXAS_IDADE.values()]
    c["faixa_idade"] = pd.cut(
        c["idade"].astype(float), limites, labels=list(T.FAIXAS_IDADE)
    ).astype(object)
    c["sexo"] = c["sexo"].map({1: "homem", 2: "mulher"})
    c = T._tipo_area(c, areas, "2017-18")
    c["tipo_area"] = c["tipo_area"].astype(object)
    c["turno"] = T.turno_da_hora(c["hora"])
    return c


def ocasioes_refri(c: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por pessoa × dia × hora com refrigerante; `guarana` se houve Guaraná."""
    refri = c["guarana"] | c["refri_generico"] | c["refri_marca"]
    seg = ["w", "id_setor", *DIMENSOES]
    return (
        c[refri]
        .dropna(subset=["hora"])
        .groupby([*_PESSOA_DIA, "hora"])
        .agg(guarana=("guarana", "max"), **{k: (k, "first") for k in seg})
        .reset_index()
    )


def ocasioes_pizza(c: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por ocasião de pizza com refrigerante (definição do alvo do modelo)."""
    refri = c["guarana"] | c["refri_generico"] | c["refri_marca"]
    pizza = c.loc[c["pizza"]].dropna(subset=["hora"])
    bebida = c.loc[refri, [*_PESSOA_DIA, "hora", "guarana"]].dropna(subset=["hora"])
    pares = pizza.merge(bebida, on=_PESSOA_DIA, suffixes=("", "_bebida"))
    perto = (pares["hora"].astype(int) - pares["hora_bebida"].astype(int)).abs()
    pares = pares[perto <= T.JANELA_PAR_HORAS]
    seg = ["w", "id_setor", *DIMENSOES]
    return (
        pares.groupby([*_PESSOA_DIA, "turno"], observed=True)
        .agg(
            guarana=("guarana_bebida", "max"),
            em_casa=("local_refeicao", lambda s: bool((s == 1).any())),
            inicio=("hora", "min"),
            **{k: (k, "first") for k in seg},
        )
        .reset_index()
        .assign(inicio_19_21=lambda d: d["inicio"].astype(int).between(*HORARIO_PICO))
    )


def _fatia(d: pd.DataFrame, coluna: str) -> float:
    w = d["w"].to_numpy(float)
    return float(100 * (w * d[coluna].to_numpy(float)).sum() / w.sum()) if w.sum() > 0 else np.nan


def _indices(d: pd.DataFrame, setores: np.ndarray) -> list[np.ndarray]:
    pos = pd.Series(np.arange(len(d))).groupby(d["id_setor"].to_numpy()).apply(np.asarray)
    vazio = np.array([], dtype=int)
    return [pos.get(s, vazio) for s in setores]


def bootstrap(
    refri: pd.DataFrame, pizza: pd.DataFrame, replicas: int = REPLICAS, semente: int = SEMENTE
) -> dict:
    """Fatias e lacuna de um segmento, com IC95 por bootstrap conjunto de setores."""
    rng = np.random.default_rng(semente)
    setores = np.union1d(refri["id_setor"].unique(), pizza["id_setor"].unique())
    ir, ip = _indices(refri, setores), _indices(pizza, setores)
    g, p = _fatia(refri, "guarana"), _fatia(pizza, "guarana")
    casa, pico = _fatia(pizza, "em_casa"), _fatia(pizza, "inicio_19_21")
    bs = []
    for _ in range(replicas):
        amostra = rng.integers(0, len(setores), len(setores))
        r = refri.iloc[np.concatenate([ir[i] for i in amostra])]
        z = pizza.iloc[np.concatenate([ip[i] for i in amostra])]
        bs.append(
            (
                _fatia(r, "guarana"),
                _fatia(z, "guarana"),
                _fatia(z, "em_casa"),
                _fatia(z, "inicio_19_21"),
            )
        )
    bs = np.array(bs)
    lacuna = bs[:, 0] - bs[:, 1]

    def ic(x: np.ndarray) -> list[float]:
        x = x[np.isfinite(x)]
        return [round(float(np.percentile(x, 2.5)), 1), round(float(np.percentile(x, 97.5)), 1)]

    return {
        "refri_geral": {"pct": round(g, 1), "ic95": ic(bs[:, 0]), "n": len(refri)},
        "pizza": {"pct": round(p, 1), "ic95": ic(bs[:, 1]), "n": len(pizza)},
        "lacuna_pp": {"valor": round(g - p, 1), "ic95": ic(lacuna)},
        "em_casa": {"pct": round(casa, 1), "ic95": ic(bs[:, 2])},
        "inicio_19h_21h": {"pct": round(pico, 1), "ic95": ic(bs[:, 3])},
    }


def classificar(seg: dict, brasil_refri_ic95_min: float) -> dict:
    """Aplica os critérios de lacuna e de baixa presença a um segmento já medido."""
    lacuna = seg["lacuna_pp"]["ic95"][0] > 0 and seg["pizza"]["n"] >= MIN_OCASIOES_PIZZA
    baixa = seg["refri_geral"]["ic95"][1] < brasil_refri_ic95_min
    if (seg["dimensao"], seg["valor"]) in EXCLUIDOS:
        lacuna = baixa = False
    return {**seg, "lacuna": bool(lacuna), "baixa_presenca": bool(baixa)}


def calcular(consumo: pd.DataFrame, areas: pd.DataFrame, replicas: int = REPLICAS) -> dict:
    c = _preparar(consumo, areas)
    refri, pizza = ocasioes_refri(c), ocasioes_pizza(c)
    brasil = bootstrap(refri, pizza, replicas)
    segmentos = []
    for dim in DIMENSOES:
        for valor in sorted(set(refri[dim].dropna()) | set(pizza[dim].dropna())):
            med = bootstrap(refri[refri[dim] == valor], pizza[pizza[dim] == valor], replicas)
            seg = {"dimensao": dim, "valor": str(valor), **med}
            segmentos.append(classificar(seg, brasil["refri_geral"]["ic95"][0]))
    return {
        "fonte": "POF 2017-18 (IBGE, microdados de consumo alimentar), cálculo do projeto",
        "publico": f"{IDADE_MIN} a {IDADE_MAX} anos",
        "metodo": (
            "Fatia do Guaraná ponderada (peso amostral, cada dia da semana com 1/7); IC95 por "
            f"bootstrap conjunto de setores censitários ({replicas} réplicas, semente {SEMENTE})."
        ),
        "criterios": {
            "lacuna": f"IC95 de (geral − pizza) acima de zero e ≥ {MIN_OCASIOES_PIZZA} ocasiões "
            "de pizza",
            "baixa_presenca": "IC95 da fatia no refrigerante em geral inteiro abaixo do IC95 "
            "nacional (sem sobreposição)",
            "excluidos": [f"{d} = {v}" for d, v in sorted(EXCLUIDOS)],
        },
        "brasil": brasil,
        "segmentos": segmentos,
    }


def carregar(caminho: Path = ARQUIVO) -> dict | None:
    """Tabela de associação gravada (None se o arquivo não existir)."""
    if not Path(caminho).exists():
        return None
    return json.loads(Path(caminho).read_text(encoding="utf-8"))


def sinais(tabela: dict | None, segmentos: dict[str, str]) -> dict:
    """Segmentos do contexto com lacuna ou baixa presença, com os números de cada um."""
    if not tabela:
        return {"lacuna": [], "baixa_presenca": []}
    por_chave = {(s["dimensao"], s["valor"]): s for s in tabela["segmentos"]}
    achados = [por_chave.get((d, v)) for d, v in segmentos.items()]
    achados = [s for s in achados if s]
    return {
        "lacuna": [s for s in achados if s["lacuna"]],
        "baixa_presenca": [s for s in achados if s["baixa_presenca"]],
    }


def main() -> None:
    consumo = pd.read_parquet(caminho_parquet(EDICAO))
    resultado = calcular(consumo, T.carregar_areas())
    ARQUIVO.write_text(json.dumps(resultado, ensure_ascii=False, indent=1), encoding="utf-8")
    for s in resultado["segmentos"]:
        marca = "lacuna" if s["lacuna"] else "baixa presença" if s["baixa_presenca"] else ""
        print(
            f"{s['dimensao']:12} {s['valor']:16} geral {s['refri_geral']['pct']:5.1f} "
            f"pizza {s['pizza']['pct']:5.1f} (n={s['pizza']['n']:3}) "
            f"lacuna {s['lacuna_pp']['valor']:+5.1f} {s['lacuna_pp']['ic95']} {marca}"
        )
    print(f"gravado em {ARQUIVO}")


if __name__ == "__main__":
    main()
