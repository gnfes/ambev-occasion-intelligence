"""Treino e avaliação do Occasion Score.

Validação temporal: treina na POF 2008-09 e avalia na POF 2017-18. O
Occasion Score é a probabilidade calibrada (0 a 100). A calibração usa a
própria 2017-18, em validação cruzada por setor censitário (metade dos
setores calibra, a outra metade é avaliada, e depois o contrário), para que
nenhuma linha seja avaliada com uma calibração que a viu.

Pesos:
- treino: `peso_final` (pessoas na proporção da população; sem `fator_dia`);
- métricas: `peso_final * fator_dia` (cada dia da semana vale 1/7).

Incertezas: bootstrap por setor censitário (a unidade da amostra da POF),
pareado entre variantes para comparar modelos na mesma reamostragem.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from occasion.modelagem.tabela import caminho_tabela

ALVO = "alvo_pizza_refri"
GRUPO = "id_setor"

FEATURES = [
    "turno",
    "dia_semana",
    "regiao",
    "tipo_area",
    "trends_pizza_rank_uf",
    "faixa_idade",
    "sexo",
    "renda_pct",
    "n_moradores",
    "tem_crianca",
]
CATEGORICAS = ["turno", "dia_semana", "regiao", "tipo_area", "faixa_idade", "sexo", "dia_grupo"]

# Agrupamento do dia da semana testado se as métricas por dia ficarem instáveis.
DIA_GRUPO = {1: "seg-qui", 2: "seg-qui", 3: "seg-qui", 4: "seg-qui", 5: "sex", 6: "sab", 7: "dom"}

PARAMS_LGBM = {
    "objective": "binary",
    "learning_rate": 0.03,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_jobs": -1,
}


def carregar(edicao: str) -> pd.DataFrame:
    return pd.read_parquet(caminho_tabela(edicao))


def preparar(tabela: pd.DataFrame) -> pd.DataFrame:
    """Colunas derivadas usadas por variantes do modelo.

    - `trends_pct_uf`: posição do Trends dividida pelo número de UFs com
      posição na edição (0 a 1). Em 2008-09 só 22 UFs têm valor; em 2017-18,
      27. A posição crua 22 é a última em 2008-09 e o meio-baixo em 2017-18.
    - `dia_grupo`: seg-qui, sex, sab, dom.
    """
    df = tabela.copy()
    rank = df["trends_pizza_rank_uf"].astype("float")
    n_ufs = rank.groupby(df["edicao"]).transform("max")
    df["trends_pct_uf"] = rank / n_ufs
    df["dia_grupo"] = df["dia_semana"].map(DIA_GRUPO)
    df["tem_crianca"] = df["tem_crianca"].astype("float")
    df["n_moradores"] = df["n_moradores"].astype("float")
    df["renda_pct"] = df["renda_pct"].astype("float")
    for col in CATEGORICAS:
        df[col] = df[col].astype("string").astype("category")
    df["peso_metrica"] = df["peso_final"] * df["fator_dia"].astype("float")
    return df


def _alinhar_categorias(treino: pd.DataFrame, teste: pd.DataFrame, cols: list[str]):
    teste = teste.copy()
    for col in cols:
        if col in CATEGORICAS:
            teste[col] = pd.Categorical(teste[col], categories=treino[col].cat.categories)
    return teste


# ---------------------------------------------------------------- modelos


@dataclass
class Baseline:
    """Frequência histórica ponderada por dia da semana × turno (sem ML)."""

    taxas: pd.Series = field(default_factory=pd.Series)
    geral: float = 0.0

    def fit(self, df: pd.DataFrame) -> Baseline:
        w = df["peso_final"]
        y = df[ALVO].astype(float)
        chave = [df["dia_semana"], df["turno"]]
        self.taxas = (y * w).groupby(chave, observed=True).sum() / w.groupby(
            chave, observed=True
        ).sum()
        self.geral = float(np.average(y, weights=w))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        idx = pd.MultiIndex.from_arrays([df["dia_semana"], df["turno"]])
        return self.taxas.reindex(idx).fillna(self.geral).to_numpy()


def _logistica(features: list[str]):
    cat = [c for c in features if c in CATEGORICAS]
    num = [c for c in features if c not in CATEGORICAS]
    transformadores = [("cat", OneHotEncoder(handle_unknown="ignore", drop=None), cat)]
    if "renda_pct" in num:
        transformadores.append(("renda", SplineTransformer(n_knots=5, degree=3), ["renda_pct"]))
    outros = [c for c in num if c != "renda_pct"]
    if outros:
        transformadores.append(
            (
                "num",
                make_pipeline(SimpleImputer(add_indicator=True), StandardScaler()),
                outros,
            )
        )
    return make_pipeline(
        ColumnTransformer(transformadores), LogisticRegression(C=1.0, max_iter=2000)
    )


@dataclass
class Modelo:
    tipo: str  # "logistica" | "lgbm"
    features: list[str]
    params: dict = field(default_factory=dict)
    n_arvores: int = 300

    def fit(self, df: pd.DataFrame) -> Modelo:
        self._cats = {c: df[c].cat.categories for c in self.features if c in CATEGORICAS}
        X, y = df[self.features], df[ALVO].astype(int)
        w = df["peso_final"] / df["peso_final"].mean()
        if self.tipo == "logistica":
            self.est = _logistica(self.features).fit(X, y, logisticregression__sample_weight=w)
        else:
            self.est = lgb.LGBMClassifier(
                **PARAMS_LGBM, **self.params, n_estimators=self.n_arvores, random_state=42
            ).fit(X, y, sample_weight=w)
        return self

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.features].copy()
        for c, cats in self._cats.items():
            X[c] = pd.Categorical(X[c], categories=cats)
        return X

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.est.predict_proba(self._X(df))[:, 1]


def escolher_lgbm(
    treino: pd.DataFrame,
    features: list[str],
    grade: list[dict] | None = None,
    n_folds: int = 5,
) -> tuple[dict, int, pd.DataFrame]:
    """Escolhe hiperparâmetros por validação cruzada por setor só no treino.

    Devolve os parâmetros, o número de árvores (média das paradas antecipadas)
    e a tabela da grade.
    """
    grade = grade or [
        {"num_leaves": nl, "min_child_samples": mcs} for nl in (7, 15, 31) for mcs in (100, 400)
    ]
    X, y = treino[features], treino[ALVO].astype(int)
    w = (treino["peso_final"] / treino["peso_final"].mean()).to_numpy()
    folds = list(GroupKFold(n_splits=n_folds).split(X, y, treino[GRUPO]))
    linhas = []
    for params in grade:
        aps, its = [], []
        for tr, va in folds:
            m = lgb.LGBMClassifier(**PARAMS_LGBM, **params, n_estimators=2000, random_state=42)
            m.fit(
                X.iloc[tr],
                y.iloc[tr],
                sample_weight=w[tr],
                eval_X=(X.iloc[va],),
                eval_y=(y.iloc[va],),
                eval_sample_weight=[w[va]],
                eval_metric="average_precision",
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            p = m.predict_proba(X.iloc[va])[:, 1]
            aps.append(average_precision_score(y.iloc[va], p, sample_weight=w[va]))
            its.append(m.best_iteration_ or 2000)
        linhas.append({**params, "pr_auc_cv": np.mean(aps), "n_arvores": int(np.mean(its))})
    tab = pd.DataFrame(linhas).sort_values("pr_auc_cv", ascending=False)
    melhor = tab.iloc[0]
    params = {k: int(melhor[k]) for k in grade[0]}
    return params, int(melhor["n_arvores"]), tab


# ------------------------------------------------------------ calibração


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def metades_setor(df: pd.DataFrame, semente: int = 7) -> np.ndarray:
    """0/1 por linha: metade dos setores censitários em cada lado."""
    setores = df[GRUPO].unique()
    rng = np.random.default_rng(semente)
    lado = pd.Series(rng.permutation(len(setores)) % 2, index=setores)
    return df[GRUPO].map(lado).to_numpy()


def calibrar_cruzado(p: np.ndarray, df: pd.DataFrame, lado: np.ndarray) -> np.ndarray:
    """Calibração de Platt (logística no logit) ajustada numa metade e aplicada na outra.

    Platt é monótona, então não muda ordenação dentro de cada metade.
    """
    y = df[ALVO].astype(int).to_numpy()
    w = df["peso_final"].to_numpy()
    saida = np.empty_like(p)
    for k in (0, 1):
        ajuste = lado != k
        cal = LogisticRegression(C=1e6).fit(
            _logit(p[ajuste]).reshape(-1, 1), y[ajuste], sample_weight=w[ajuste]
        )
        saida[~ajuste] = cal.predict_proba(_logit(p[~ajuste]).reshape(-1, 1))[:, 1]
    return saida


# ---------------------------------------------------------------- métricas


def metricas(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> dict[str, float]:
    base = float(np.average(y, weights=w))
    corte = np.quantile(p, 0.9)
    topo = p >= corte
    return {
        "pr_auc": average_precision_score(y, p, sample_weight=w),
        "roc_auc": roc_auc_score(y, p, sample_weight=w),
        "taxa_base": base,
        "lift_top10": float(np.average(y[topo], weights=w[topo]) / base),
        "recall_top10": float((y * w)[topo].sum() / (y * w).sum()),
    }


def metricas_prob(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> dict[str, float]:
    return {
        "brier": brier_score_loss(y, p, sample_weight=w),
        "log_loss": log_loss(y, p, sample_weight=w),
        "media_prevista": float(np.average(p, weights=w)),
        "media_observada": float(np.average(y, weights=w)),
    }


def limiar_f1(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    """Limiar que maximiza o F1 ponderado."""
    ordem = np.argsort(-p)
    ys, ws, ps = y[ordem], w[ordem], p[ordem]
    tp = np.cumsum(ys * ws)
    fp = np.cumsum((1 - ys) * ws)
    total = tp[-1]
    prec = tp / (tp + fp)
    rec = tp / total
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(ps[np.argmax(f1)])


def classificacao_cruzada(
    y: np.ndarray, p: np.ndarray, w: np.ndarray, lado: np.ndarray
) -> dict[str, float]:
    """Precision, recall, F1 e matriz de confusão com limiar escolhido na outra metade."""
    pred = np.zeros_like(y)
    limiares = []
    for k in (0, 1):
        escolha = lado != k
        t = limiar_f1(y[escolha], p[escolha], w[escolha])
        limiares.append(t)
        pred[~escolha] = (p[~escolha] >= t).astype(int)
    tp = float((w * (pred == 1) * (y == 1)).sum())
    fp = float((w * (pred == 1) * (y == 0)).sum())
    fn = float((w * (pred == 0) * (y == 1)).sum())
    tn = float((w * (pred == 0) * (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn)
    return {
        "limiar_medio": float(np.mean(limiares)),
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "vp_pct": 100 * tp / w.sum(),
        "fp_pct": 100 * fp / w.sum(),
        "fn_pct": 100 * fn / w.sum(),
        "vn_pct": 100 * tn / w.sum(),
        "positivos_previstos_pct": 100 * (tp + fp) / w.sum(),
    }


def curva_calibracao(y: np.ndarray, p: np.ndarray, w: np.ndarray, n: int = 10) -> pd.DataFrame:
    """Média prevista e observada por faixa de score (decis ponderados de p)."""
    ordem = np.argsort(p)
    acum = np.cumsum(w[ordem]) / w.sum()
    faixa = np.empty(len(p), dtype=int)
    faixa[ordem] = np.minimum((acum * n).astype(int), n - 1)
    df = pd.DataFrame({"faixa": faixa, "y": y, "p": p, "w": w})
    return df.groupby("faixa").apply(
        lambda g: pd.Series(
            {
                "prevista": np.average(g["p"], weights=g["w"]),
                "observada": np.average(g["y"], weights=g["w"]),
                "positivos": int(g["y"].sum()),
            }
        ),
        include_groups=False,
    )


def bootstrap_setor(
    df: pd.DataFrame,
    previsoes: dict[str, np.ndarray],
    metrica=average_precision_score,
    n: int = 300,
    semente: int = 11,
) -> pd.DataFrame:
    """Reamostra setores com reposição; devolve a métrica de cada variante por réplica.

    A mesma reamostragem é usada para todas as variantes (comparação pareada).
    """
    y = df[ALVO].astype(int).to_numpy()
    w = df["peso_metrica"].to_numpy()
    codigos, setor = np.unique(df[GRUPO].to_numpy(), return_inverse=True)
    linhas_por_setor = pd.Series(np.arange(len(df))).groupby(setor).apply(np.array).to_list()
    rng = np.random.default_rng(semente)
    res = {k: [] for k in previsoes}
    for _ in range(n):
        escolha = rng.integers(0, len(codigos), len(codigos))
        idx = np.concatenate([linhas_por_setor[s] for s in escolha])
        for k, p in previsoes.items():
            res[k].append(metrica(y[idx], p[idx], sample_weight=w[idx]))
    return pd.DataFrame(res)
