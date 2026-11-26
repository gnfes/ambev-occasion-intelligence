"""Correção de viés por grupo do Occasion Score (offset no logit calibrado).

O modelo final, medido fora da amostra em 2017-18, erra o nível de alguns
grupos (por exemplo, subestima o Nordeste). A correção soma um offset ao
logit depois do Platt, por grupo:

    logit_corrigido = a × logit_bruto + b + global + Σ offset[coluna][valor]

Dois passos separados, para que o intervalo de confiança por bootstrap
reestime só os offsets, com os grupos fixos:

1. `selecionar_grupos`: entra o grupo cujo previsto ÷ observado tem IC95
   (bootstrap por setor) que exclui 1, com um mínimo de ocasiões. É a mesma
   regra usada na correção do Mapa de Oportunidade em 22/11.
   Depois, `validar_grupos` tira o grupo cuja correção, estimada numa
   metade dos setores, não melhora a calibração dele na outra metade.
2. `estimar_offsets`: logística ponderada com o logit calibrado como offset
   fixo e uma dummy por grupo selecionado, mais um termo global que mantém
   a média nacional calibrada. Função pura (só arrays e DataFrame).

Classe de renda não é candidata: a classe D está calibrada (previsto ÷
observado 0,95, IC95 0,71 a 1,35, no modelo final fora da amostra,
`docs/relatorios/vies_por_grupo.md`); ela concentra ocasiões perdidas
porque a taxa real é menor, o que é questão da regra de decisão, não do
score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CANDIDATOS = ("regiao", "tipo_area")
MIN_POSITIVOS = 30
N_BOOT = 500
# Penalidade L2 fraca nos offsets dos grupos (não no global), equivalente a
# uma priori N(0, 1) no logit: com pesos normalizados para somar o número de
# linhas, a informação de um grupo é da ordem do número de ocasiões dele
# (centenas), então a penalidade só pesa se o grupo quase não tem ocasiões
# numa réplica do bootstrap, e evita que a estimação divirja.
L2 = 1.0


def razao_por_grupo(
    grupos: pd.Series,
    setor: pd.Series | np.ndarray,
    y: np.ndarray,
    score: np.ndarray,
    w: np.ndarray,
    n_boot: int = N_BOOT,
    semente: int = 7,
) -> pd.DataFrame:
    """Previsto ÷ observado por valor do grupo, com IC95 por bootstrap de setores."""
    df = pd.DataFrame(
        {
            "g": np.asarray(grupos).astype(str),
            "setor": np.asarray(setor),
            "py": score * w,
            "oy": y * w,
            "pos": y,
        }
    )
    por_setor = df.groupby(["setor", "g"])[["py", "oy"]].sum().unstack("g", fill_value=0)
    setores = por_setor.index.to_numpy()
    rng = np.random.default_rng(semente)
    soma = por_setor.sum()
    ponto = soma["py"] / soma["oy"]
    boots = []
    for _ in range(n_boot):
        amostra = por_setor.loc[rng.choice(setores, len(setores), replace=True)].sum()
        boots.append(amostra["py"] / amostra["oy"])
    # Amostra sem nenhum positivo no grupo: razão indefinida, fica fora do IC.
    boots = pd.DataFrame(boots).replace([np.inf, -np.inf], np.nan)
    pos = df.groupby("g")["pos"].sum()
    return pd.DataFrame(
        {
            "valor": ponto.index,
            "positivos": pos.reindex(ponto.index).to_numpy(),
            "previsto_sobre_obs": ponto.to_numpy(),
            "ic95_baixo": boots.quantile(0.025).to_numpy(),
            "ic95_alto": boots.quantile(0.975).to_numpy(),
        }
    )


def selecionar_grupos(
    grupos: pd.DataFrame,
    setor: pd.Series | np.ndarray,
    y: np.ndarray,
    score: np.ndarray,
    w: np.ndarray,
    candidatos: tuple[str, ...] = CANDIDATOS,
    min_positivos: int = MIN_POSITIVOS,
    n_boot: int = N_BOOT,
) -> list[tuple[str, str]]:
    """Grupos (coluna, valor) com IC95 de previsto ÷ observado que exclui 1."""
    selecionados = []
    for col in candidatos:
        r = razao_por_grupo(grupos[col], setor, y, score, w, n_boot=n_boot)
        fora = (r["ic95_baixo"] > 1) | (r["ic95_alto"] < 1)
        ok = fora & (r["positivos"] >= min_positivos)
        selecionados += [(col, str(v)) for v in r.loc[ok, "valor"]]
    return selecionados


def matriz_grupos(grupos: pd.DataFrame, selecionados: list[tuple[str, str]]) -> np.ndarray:
    """Uma coluna 0/1 por grupo selecionado, na ordem de `selecionados`."""
    if not selecionados:
        return np.zeros((len(grupos), 0))
    return np.column_stack(
        [(grupos[c].astype(str) == v).to_numpy(dtype=float) for c, v in selecionados]
    )


def estimar_offsets(
    logit_calibrado: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    grupos: pd.DataFrame,
    selecionados: list[tuple[str, str]],
    l2: float = L2,
    max_iter: int = 50,
) -> dict:
    """Offsets no logit calibrado: {"global": c, coluna: {valor: offset}}.

    Newton (IRLS) numa logística com `logit_calibrado` como offset fixo, um
    intercepto global e uma dummy por grupo selecionado. Pesos normalizados
    para somar o número de linhas (a escala da penalidade `l2` depende disso).
    """
    D = matriz_grupos(grupos, selecionados)
    X = np.column_stack([np.ones(len(y)), D])
    wn = w * len(w) / w.sum()
    pen = np.full(X.shape[1], l2)
    pen[0] = 0.0
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        p = 1 / (1 + np.exp(-(logit_calibrado + X @ beta)))
        grad = X.T @ (wn * (y - p)) - pen * beta
        hess = (X * (wn * p * (1 - p))[:, None]).T @ X + np.diag(pen)
        passo = np.linalg.solve(hess, grad)
        beta += passo
        if np.max(np.abs(passo)) < 1e-8:
            break
    saida: dict = {"global": float(beta[0])}
    for (col, val), b in zip(selecionados, beta[1:], strict=True):
        saida.setdefault(col, {})[val] = float(b)
    return saida


def aplicar(offsets: dict, grupos: pd.DataFrame) -> np.ndarray:
    """Soma dos offsets de cada linha (global + grupos), no logit."""
    total = np.full(len(grupos), float(offsets.get("global", 0.0)))
    for col, valores in offsets.items():
        if col == "global":
            continue
        total += grupos[col].astype(str).map(valores).fillna(0.0).to_numpy(dtype=float)
    return total


def selecionados_de(offsets: dict) -> list[tuple[str, str]]:
    """Lista (coluna, valor) dos grupos com offset, na ordem em que foram gravados."""
    return [(c, v) for c, vals in offsets.items() if c != "global" for v in vals]


def _platt(logit: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression

    cal = LogisticRegression(C=1e6).fit(logit.reshape(-1, 1), y, sample_weight=w)
    return float(cal.coef_[0, 0]), float(cal.intercept_[0])


def cruzar_metades(
    logit_bruto: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    grupos: pd.DataFrame,
    lado: np.ndarray,
    selecionados: list[tuple[str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    """Score antes (Platt) e depois (Platt + offsets dos grupos fixos), fora da amostra.

    Platt e offsets são ajustados numa metade dos setores e aplicados na
    outra, nos dois sentidos. Os grupos são os mesmos nas duas metades.
    """
    antes = np.empty(len(y))
    depois = np.empty(len(y))
    for k in (0, 1):
        aj, ap = lado != k, lado == k
        a, b = _platt(logit_bruto[aj], y[aj], w[aj])
        z = a * logit_bruto + b
        off = estimar_offsets(z[aj], y[aj], w[aj], grupos[aj], selecionados)
        antes[ap] = 1 / (1 + np.exp(-z[ap]))
        depois[ap] = 1 / (1 + np.exp(-(z[ap] + aplicar(off, grupos[ap]))))
    return antes, depois


def validar_grupos(
    logit_bruto: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    grupos: pd.DataFrame,
    lado: np.ndarray,
    selecionados: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Mantém só os grupos cuja correção melhora a calibração fora da amostra.

    Com os grupos fixos, os offsets estimados numa metade são medidos na
    outra. Fica o grupo em que |log(previsto ÷ observado)| cai em relação ao
    Platt sozinho. Um grupo significativo nos dados inteiros que não se
    repete fora da amostra (poucas ocasiões) sai, e os outros são reestimados.
    """
    atuais = list(selecionados)
    while atuais:
        antes, depois = cruzar_metades(logit_bruto, y, w, grupos, lado, atuais)
        piora = []
        for col, val in atuais:
            m = (grupos[col].astype(str) == val).to_numpy()
            obs = (y * w)[m].sum()
            erro_antes = abs(np.log((antes * w)[m].sum() / obs))
            erro_depois = abs(np.log((depois * w)[m].sum() / obs))
            if erro_depois >= erro_antes:
                piora.append((col, val))
        if not piora:
            break
        atuais = [g for g in atuais if g not in piora]
    return atuais
