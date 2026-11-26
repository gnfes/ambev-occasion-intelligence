"""Incerteza do Occasion Score: intervalo de confiança, suporte nos dados e avisos.

Intervalo (IC95): bootstrap por setor censitário, o cluster da amostra da
POF. Em cada réplica, os setores de cada edição são sorteados com reposição
(o peso de cada linha é multiplicado pelo número de vezes que o setor saiu)
e o pipeline do modelo servido é refeito inteiro:

1. logística em 2008-09 + uma metade dos setores de 2017-18, prevendo a
   outra metade (as mesmas metades do modelo servido);
2. Platt nessas previsões fora da amostra;
3. offsets da correção por grupo (`correcao.estimar_offsets`) nos grupos já
   escolhidos pelo modelo servido (seleção fixa, sem descontinuidade);
4. logística final nas duas edições.

A transformação das features (one-hot e splines da renda, com nós
uniformes de 0 a 100) fica a do modelo servido. Cada réplica vira um vetor
`beta` e uma constante `c` no espaço transformado:

    logit = termos @ beta + c,  beta = a × coef + offsets,  c = a × intercepto + b + global

então o intervalo de qualquer contexto sai na hora, sem reajustar nada. As
faixas usam os cortes do modelo servido; a estabilidade da faixa é a fração
das réplicas em que o score cai em cada uma.

O intervalo mede só a variação da amostra da POF. Ele não cobre a mudança
de hábito desde 2017-18 nem erro de especificação do modelo.

Suporte: ocasiões reais (sem peso) nas duas edições para o perfil (região ×
tipo de área × faixa de idade × sexo × classe de renda) e para cada valor de
entrada. O aviso de "poucos dados" vem do valor de entrada com menos de 30
ocasiões ou do intervalo largo (limite superior acima de 3× o inferior);
esses limites são convenção do projeto, não vêm de dado. A contagem do
perfil é só informativa: nenhum perfil completo de 2017-18 chega a 20
ocasiões (2% do peso da população tem perfil com 10 ou mais), porque o
modelo é aditivo e soma o efeito de cada variável em vez de depender da
combinação exata. Um aviso pela contagem do perfil apareceria sempre.

    uv run python -m occasion.modelagem.incerteza
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from occasion.modelagem import correcao, final
from occasion.modelagem import treino as t
from occasion.modelagem.experimentos import classe

ARQUIVO = final.MODELOS_DIR / f"occasion_score_v{final.VERSAO}_incerteza.npz"
N_REPLICAS = 200
SEMENTE = 2026

# Convenções do aviso (não vêm de dado; documentadas no README e na resposta).
MIN_OCASIOES_ENTRADA = 30
RAZAO_IC_MAX = 3.0
ESTABILIDADE_MIN = 0.80

PERFIL = ["regiao", "tipo_area", "faixa_idade", "sexo", "classe"]
ENTRADAS = ["turno", "dia_semana", "regiao", "tipo_area", "faixa_idade", "sexo", "classe"]
FAIXAS = ("alta", "media", "baixa")


# ------------------------------------------------------------------ réplicas


def contagens_setor(tudo: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Quantas vezes o setor de cada linha saiu no sorteio, dentro da própria edição."""
    m = np.zeros(len(tudo), dtype=np.int64)
    for ed in tudo["edicao"].unique():
        idx = np.flatnonzero((tudo["edicao"] == ed).to_numpy())
        codigos, inverso = np.unique(tudo[t.GRUPO].to_numpy()[idx], return_inverse=True)
        sorteio = np.bincount(rng.integers(0, len(codigos), len(codigos)), minlength=len(codigos))
        m[idx] = sorteio[inverso]
    return m


def _peso_treino(pf: np.ndarray, m: np.ndarray, ed: np.ndarray, usar: np.ndarray) -> np.ndarray:
    """`final._peso_treino` nas linhas `usar`, com cada linha repetida `m` vezes."""
    w = np.zeros(len(pf))
    for e in np.unique(ed[usar]):
        s = usar & (ed == e)
        w[s] = pf[s] * m[s] / ((pf[s] * m[s]).sum() / m[s].sum())
    return w


def replica(
    modelo: final.OccasionScore,
    Z: np.ndarray,
    tudo: pd.DataFrame,
    eh_rec: np.ndarray,
    lado: np.ndarray,
    m: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Pipeline do modelo servido numa reamostragem (`m` = repetições por linha).

    `lado` vale para as linhas de 2017-18 (as outras são ignoradas). Com `m`
    igual a 1 em tudo, devolve o próprio modelo servido.
    """
    lr0 = modelo.modelo.est[-1]
    y = tudo[t.ALVO].astype(int).to_numpy()
    pf = tudo["peso_final"].to_numpy(dtype=float)
    pm = tudo["peso_metrica"].to_numpy(dtype=float)
    ed = tudo["edicao"].astype(str).to_numpy()
    tem = m > 0

    def ajustar(usar: np.ndarray):
        w = _peso_treino(pf, m, ed, usar)
        return clone(lr0).fit(Z[usar], y[usar], sample_weight=w[usar])

    logit_oof = np.full(len(tudo), np.nan)
    for k in (0, 1):
        treino = tem & (~eh_rec | (eh_rec & (lado != k)))
        prever = tem & eh_rec & (lado == k)
        lr = ajustar(treino)
        logit_oof[prever] = Z[prever] @ lr.coef_[0] + lr.intercept_[0]

    rec = tem & eh_rec
    a, b = final._platt(logit_oof[rec], y[rec], pm[rec] * m[rec])
    # Os offsets são estimados com as linhas repetidas de fato: a penalidade de
    # `estimar_offsets` depende do número de linhas.
    rep = np.repeat(np.flatnonzero(rec), m[rec])
    selecionados = [tuple(g) for g in modelo.metadata.get("correcao_grupos", {}).get("grupos", [])]
    offsets = (
        correcao.estimar_offsets(
            a * logit_oof[rep] + b,
            y[rep],
            pm[rep],
            tudo.iloc[rep].reset_index(drop=True),
            selecionados,
        )
        if selecionados
        else {}
    )

    lr = ajustar(tem)
    vetor = dataclasses.replace(modelo, offsets=offsets).vetor_offsets()
    beta = a * lr.coef_[0] + vetor
    c = a * lr.intercept_[0] + b + float(offsets.get("global", 0.0))
    return beta, float(c)


def _dados(modelo: final.OccasionScore, antiga: pd.DataFrame, recente: pd.DataFrame):
    tudo = final._juntar(antiga, recente)
    eh_rec = (tudo["edicao"] == recente["edicao"].iloc[0]).to_numpy()
    lado = np.full(len(tudo), -1)
    # Mesmas metades de setores do modelo servido (`final.logit_oof`).
    lado[eh_rec] = t.metades_setor(tudo[eh_rec].reset_index(drop=True))
    return tudo, eh_rec, lado, modelo._termos(tudo)


def replicas(
    modelo: final.OccasionScore,
    antiga: pd.DataFrame,
    recente: pd.DataFrame,
    n: int = N_REPLICAS,
    semente: int = SEMENTE,
) -> tuple[np.ndarray, np.ndarray]:
    """`n` réplicas do bootstrap por setor: (beta [n × termos], c [n])."""
    tudo, eh_rec, lado, Z = _dados(modelo, antiga, recente)
    rng = np.random.default_rng(semente)
    betas, cs = [], []
    for _ in range(n):
        beta, c = replica(modelo, Z, tudo, eh_rec, lado, contagens_setor(tudo, rng))
        betas.append(beta)
        cs.append(c)
    return np.vstack(betas), np.array(cs)


# ------------------------------------------------------------------- suporte


def _com_classe(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(classe=classe(df["renda_sm"]).astype("string"))


def _chave(valores) -> str:
    return "|".join(str(v) for v in valores)


def suporte(antiga: pd.DataFrame, recente: pd.DataFrame) -> dict:
    """Ocasiões e linhas (sem peso) por valor de entrada e por perfil, nas duas edições."""
    df = _com_classe(final._juntar(antiga, recente))
    y = df[t.ALVO].astype(int)
    por_entrada = {}
    for col in ENTRADAS:
        g = y.groupby(df[col].astype("string"), observed=True)
        por_entrada[col] = {
            str(v): {"ocasioes": int(s), "linhas": int(n)}
            for v, s, n in zip(g.sum().index, g.sum(), g.count(), strict=True)
        }
    chaves = df[PERFIL].astype("string").agg("|".join, axis=1)
    g = y.groupby(chaves)
    perfil = {
        str(k): {"ocasioes": int(s), "linhas": int(n)}
        for k, s, n in zip(g.sum().index, g.sum(), g.count(), strict=True)
    }
    return {"por_entrada": por_entrada, "perfil": perfil}


# ------------------------------------------------------------------- serviço


@dataclass
class Incerteza:
    """Réplicas do bootstrap e suporte, ligados a uma versão do modelo."""

    beta: np.ndarray
    c: np.ndarray
    suporte: dict
    metadata: dict = field(default_factory=dict)

    def scores(self, modelo: final.OccasionScore, df: pd.DataFrame) -> np.ndarray:
        """Score (0 a 1) de cada réplica: matriz [réplicas × linhas]."""
        z = modelo._termos(df) @ self.beta.T + self.c
        return (1 / (1 + np.exp(-z))).T

    def intervalo(self, modelo: final.OccasionScore, df: pd.DataFrame) -> list[dict]:
        """IC95 (em 0 a 100) e fração das réplicas em cada faixa, por linha."""
        s = self.scores(modelo, df)
        lo, hi = np.percentile(s, [2.5, 97.5], axis=0)
        saida = []
        for j in range(s.shape[1]):
            faixa = modelo.faixa(s[:, j])
            saida.append(
                {
                    # 4 algarismos significativos: arredondar casas decimais zeraria o
                    # limite inferior na faixa baixa e dispararia o aviso de intervalo largo.
                    "ic95": [_sig(100 * lo[j]), _sig(100 * hi[j])],
                    "prob_faixa": {f: round(float(np.mean(faixa == f)), 3) for f in FAIXAS},
                }
            )
        return saida

    def suporte_de(self, linha: pd.Series, classe_renda: str) -> dict:
        """Ocasiões do perfil e de cada valor de entrada da linha (códigos do modelo)."""
        valores = {**{c: str(linha[c]) for c in ENTRADAS if c != "classe"}, "classe": classe_renda}
        perfil = self.suporte["perfil"].get(_chave(valores[c] for c in PERFIL))
        return {
            "perfil": {
                "chave": {c: valores[c] for c in PERFIL},
                "ocasioes": perfil["ocasioes"] if perfil else 0,
                "linhas": perfil["linhas"] if perfil else 0,
            },
            "por_entrada": [
                {
                    "feature": c,
                    "valor": valores[c],
                    **self.suporte["por_entrada"]
                    .get(c, {})
                    .get(valores[c], {"ocasioes": 0, "linhas": 0}),
                }
                for c in ENTRADAS
            ],
        }

    # ----------------------------------------------------------- persistência

    def salvar(self, caminho: Path = ARQUIVO) -> Path:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            caminho,
            beta=self.beta,
            c=self.c,
            suporte=np.array(json.dumps(self.suporte, ensure_ascii=False)),
            metadata=np.array(json.dumps(self.metadata, ensure_ascii=False)),
        )
        return caminho


def _sig(x: float) -> float:
    return float(f"{float(x):.4g}")


def compativel(incerteza: Incerteza, modelo: final.OccasionScore) -> bool:
    """Réplicas da mesma versão e do mesmo espaço de termos do modelo servido."""
    n_termos = len(modelo.modelo.est[0].get_feature_names_out())
    return (
        incerteza.metadata.get("versao_modelo") == modelo.metadata.get("versao")
        and incerteza.beta.ndim == 2
        and incerteza.beta.shape[1] == n_termos
        and incerteza.c.shape == (incerteza.beta.shape[0],)
    )


def carregar(caminho: Path = ARQUIVO) -> Incerteza:
    with np.load(caminho, allow_pickle=False) as z:
        return Incerteza(
            beta=z["beta"],
            c=z["c"],
            suporte=json.loads(str(z["suporte"])),
            metadata=json.loads(str(z["metadata"])),
        )


def metadata(modelo: final.OccasionScore, n: int, semente: int) -> dict:
    return {
        "versao_modelo": modelo.metadata.get("versao"),
        "gerado_em": date.today().isoformat(),
        "metodo": "bootstrap por setor censitário (cluster da POF), dentro de cada edição; "
        "cada réplica refaz logística, Platt fora da amostra e offsets da correção por grupo "
        "(grupos fixos do modelo servido)",
        "replicas": n,
        "semente": semente,
        "nivel": 0.95,
        "cobre": "variação da amostra da POF (2008-09 e 2017-18)",
        "nao_cobre": "mudança de hábito desde 2017-18 e erro de especificação do modelo",
        "avisos": {
            "min_ocasioes_entrada": MIN_OCASIOES_ENTRADA,
            "razao_ic_max": RAZAO_IC_MAX,
            "estabilidade_min_faixa": ESTABILIDADE_MIN,
            "nota": "limites por convenção do projeto, não estimados de dado; a contagem do "
            "perfil é informativa (nenhum perfil completo chega a 20 ocasiões)",
        },
        "perfil": PERFIL,
    }


def gerar(
    modelo: final.OccasionScore,
    antiga: pd.DataFrame,
    recente: pd.DataFrame,
    n: int = N_REPLICAS,
    semente: int = SEMENTE,
) -> Incerteza:
    beta, c = replicas(modelo, antiga, recente, n=n, semente=semente)
    return Incerteza(beta, c, suporte(antiga, recente), metadata(modelo, n, semente))


def rodar(destino: Path = ARQUIVO) -> Path:
    modelo = final.carregar()
    antiga = t.preparar(t.carregar("2008_2009"))
    recente = t.preparar(t.carregar("2017_2018"))
    return gerar(modelo, antiga, recente).salvar(destino)


def main() -> None:
    print(rodar())


if __name__ == "__main__":
    main()
