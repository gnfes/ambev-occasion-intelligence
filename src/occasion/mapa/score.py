"""Occasion Score de cada município, com o modelo final sem retreino.

O modelo é aditivo no logit (sem interação entre o momento e o perfil), então
o logit de uma pessoa num momento é o termo do perfil mais o termo do
momento. Isso permite pontuar os 28 momentos da semana (7 dias × 4 turnos)
de cada célula do perfil sem montar 28 cópias da tabela.

Saídas por município (público de 20 a 44 anos):

- `score_sabado_noite`: score médio no sábado à noite, o melhor momento de
  todos os perfis (avaliacao_occasion_score.md), em pontos de 0 a 100.
- `ocasioes_semana`: soma, nas pessoas e nos 28 momentos, da probabilidade
  de pizza com refrigerante. É o volume esperado de ocasiões por semana se o
  padrão da POF 2017-18 valer em 2022 (hipótese: o modelo foi validado de
  2008-09 para 2017-18, não para 2022).
- `ocasioes_por_mil`: o mesmo volume por mil pessoas de 20 a 44 (a taxa).
- `pct_faixa_alta`: parte do público na faixa alta do score no sábado à noite.
- `pct_classe_bd`: parte do público nas classes B a D pela renda do domicílio
  em salários mínimos de 2022 (R$ 1.212), nos cortes do Decision Engine. A
  renda do Censo é só monetária, então a classe sai um pouco subestimada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from occasion.decisao.engine import DIAS, SEXOS, classe_de_renda
from occasion.modelagem import final
from occasion.modelagem.tabela import TURNOS

SALARIO_MINIMO_2022 = 1212.0  # Lei 14.358/2022, vigente em julho de 2022 (data do Censo)
SABADO, NOITE = 6, "noite"


def _features(df: pd.DataFrame, dia: int, turno: str) -> pd.DataFrame:
    x = pd.DataFrame(
        {
            "turno": turno,
            "dia_semana": str(dia),
            "regiao": df["regiao"].to_numpy(),
            "tipo_area": df["tipo_area"].to_numpy(),
            "faixa_idade": df["faixa_idade"].to_numpy(),
            "sexo": df["sexo"].map({k: str(v) for k, v in SEXOS.items()}).to_numpy(),
            "renda_pct": df["renda_pct"].to_numpy(dtype=float),
        }
    )
    for col in x.columns.drop("renda_pct"):
        x[col] = x[col].astype("category")
    return x


def termos_momento(modelo: final.OccasionScore, perfil: pd.DataFrame) -> pd.Series:
    """Diferença de logit de cada momento em relação ao sábado à noite."""
    uma = perfil.iloc[[0]]
    ref = modelo._logit_bruto(_features(uma, SABADO, NOITE))[0]
    return pd.Series(
        {(d, t): modelo._logit_bruto(_features(uma, d, t))[0] - ref for d in DIAS for t in TURNOS}
    )


def pontuar(perfil: pd.DataFrame, modelo: final.OccasionScore | None = None) -> pd.DataFrame:
    """Agrega o score do perfil (uma linha por célula) em uma linha por município."""
    modelo = modelo or final.carregar()
    x = _features(perfil, SABADO, NOITE)
    base = modelo._logit_bruto(x)
    # Correção por grupo (região, tipo de área): depende só do perfil, não do momento.
    off = modelo.offset(x)
    delta = termos_momento(modelo, perfil).to_numpy()

    def prob(logit_bruto: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(modelo.platt_a * logit_bruto + modelo.platt_b + off)))

    p_sab = prob(base)
    # Soma dos 28 momentos, em blocos para não montar uma matriz de 28 colunas inteira.
    p_semana = np.zeros(len(perfil))
    for d in delta:
        p_semana += prob(base + d)

    classe = pd.Series(perfil["renda_mensal"] / SALARIO_MINIMO_2022).map(classe_de_renda)
    df = perfil.assign(
        w=perfil["pessoas"],
        sab=p_sab * perfil["pessoas"],
        sem=p_semana * perfil["pessoas"],
        alta=(p_sab >= modelo.corte_alta) * perfil["pessoas"],
        bd=classe.isin(["B", "C", "D"]).to_numpy() * perfil["pessoas"],
    )
    g = df.groupby(["CD_MUN", "municipio", "uf", "regiao"], as_index=False)[
        ["w", "sab", "sem", "alta", "bd"]
    ].sum()
    tipo = (
        perfil[perfil["tipo_area"] != "rural"]
        .drop_duplicates("CD_MUN")
        .set_index("CD_MUN")["tipo_area"]
    )
    rural = df[df["tipo_area"] == "rural"].groupby("CD_MUN")["w"].sum()
    return pd.DataFrame(
        {
            "CD_MUN": g["CD_MUN"],
            "municipio": g["municipio"],
            "uf": g["uf"],
            "regiao": g["regiao"],
            "tipo_area": g["CD_MUN"].map(tipo).fillna("rural"),
            "pct_rural": 100 * g["CD_MUN"].map(rural).fillna(0).to_numpy() / g["w"],
            "publico_20_44": g["w"],
            "score_sabado_noite": 100 * g["sab"] / g["w"],
            "ocasioes_semana": g["sem"],
            "ocasioes_por_mil": 1000 * g["sem"] / g["w"],
            "pct_faixa_alta": 100 * g["alta"] / g["w"],
            "pct_classe_bd": 100 * g["bd"] / g["w"],
        }
    )
