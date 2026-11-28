"""Comparação de candidatos para o Occasion Score, com regra de escolha fixada antes.

Todos os candidatos usam as 7 features finais e treinam na POF 2008-09.
Os setores censitários de 2017-18 são divididos em duas metades fixas: a
metade de seleção escolhe o candidato e a metade de avaliação só mede o
escolhido, para o teste não virar critério de escolha.

Por que não escolher pela CV dentro de 2008-09: na ablação, a CV de
2008-09 preferia o LightGBM (PR-AUC 0,065 vs 0,049 da logística), e em
2017-18 ele perdeu (0,027 vs 0,029). A CV dentro da mesma edição não mede a
mudança entre edições, que é o que o modelo enfrenta. A CV de 2008-09 segue
no relatório para mostrar isso.

Regra: vence o candidato com maior PR-AUC na metade de seleção, a menos
que um candidato mais simples não fique significativamente atrás dele
(teste unilateral: o limite inferior do IC95 da diferença "melhor − mais
simples", por bootstrap pareado de setores na metade de seleção, não passa
de zero). Nesse caso fica o mais simples (ordem de simplicidade: a
numeração dos candidatos).

A coluna `pr_auc_cv_2008` do AutoML é a AP interna do FLAML (pesos
`peso_final`, melhor configuração entre as testadas), não a mesma CV dos
outros candidatos: ela tende a ser otimista e aparece só como referência.

Candidatos:
1. logística (C=1), o modelo atual;
2. logística com C escolhido por CV (0,01 a 10);
3. logística com interação turno × dia da semana;
4. LightGBM regularizado (folhas pequenas, L2 forte);
5. Random Forest com folhas grandes;
6. AutoML (FLAML) com orçamento de tempo, escolhendo pela mesma CV por setor.

    uv run --group automl python -m occasion.modelagem.candidatos
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

from occasion.modelagem import treino as t
from occasion.modelagem.experimentos import FEATURES_FINAIS, _ic, _tabela_md
from occasion.paths import PROCESSED_DIR

N_FOLDS = 5
N_BOOTSTRAP = 300
ORCAMENTO_AUTOML_S = 900


def _com_interacao(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["turno_dia"] = (df["turno"].astype(str) + "_" + df["dia_semana"].astype(str)).astype(
        "category"
    )
    return df


class _Sklearn:
    """Envolve um estimador do scikit-learn com as features e pesos do projeto."""

    def __init__(self, fabrica, features: list[str], prep=None):
        self.fabrica, self.features, self.prep = fabrica, features, prep

    def fit(self, df: pd.DataFrame):
        df = self.prep(df) if self.prep else df
        self._cats = {c: df[c].cat.categories for c in self.features if c in self._categoricas()}
        w = df["peso_final"] / df["peso_final"].mean()
        self.est = self.fabrica()
        passo = self.est.steps[-1][0]
        self.est.fit(
            self._X(df), df[t.ALVO].astype(int), **{f"{passo}__sample_weight": w.to_numpy()}
        )
        return self

    def _categoricas(self) -> list[str]:
        return [*t.CATEGORICAS, "turno_dia"]

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.features].copy()
        for c, cats in self._cats.items():
            X[c] = pd.Categorical(X[c], categories=cats).astype(str)
        return X

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        df = self.prep(df) if self.prep else df
        return self.est.predict_proba(self._X(df))[:, 1]


def _pre(features: list[str]):
    cat = [c for c in features if c != "renda_pct"]
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ("renda", SplineTransformer(n_knots=5, degree=3), ["renda_pct"]),
        ]
    )


def logistica(features: list[str], C: float = 1.0, prep=None) -> _Sklearn:
    return _Sklearn(
        lambda: make_pipeline(_pre(features), LogisticRegression(C=C, max_iter=3000)),
        features,
        prep,
    )


def floresta(features: list[str]) -> _Sklearn:
    return _Sklearn(
        lambda: make_pipeline(
            _pre(features),
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=300,
                max_features="sqrt",
                n_jobs=-1,
                random_state=42,
            ),
        ),
        features,
    )


def lgbm_regularizado(features: list[str], n_arvores: int) -> t.Modelo:
    params = {"num_leaves": 7, "min_child_samples": 800, "reg_lambda": 20.0, "max_bin": 63}
    return t.Modelo("lgbm", features, params, n_arvores)


def _cv(treino: pd.DataFrame, fabrica) -> list[float]:
    aps = []
    for tr, va in GroupKFold(n_splits=N_FOLDS).split(treino, groups=treino[t.GRUPO]):
        a, b = treino.iloc[tr], treino.iloc[va]
        p = fabrica().fit(a).predict(b)
        aps.append(
            average_precision_score(b[t.ALVO].astype(int), p, sample_weight=b["peso_metrica"])
        )
    return aps


def _arvores_lgbm(treino: pd.DataFrame, features: list[str]) -> int:
    """Número de árvores do LightGBM regularizado por parada antecipada na CV."""
    _, n, _ = t.escolher_lgbm(
        treino,
        features,
        grade=[{"num_leaves": 7, "min_child_samples": 800, "reg_lambda": 20.0, "max_bin": 63}],
    )
    return n


class _AutoML:
    """FLAML escolhendo modelo e hiperparâmetros por CV agrupada por setor em 2008-09."""

    def __init__(self, features: list[str], orcamento: int = ORCAMENTO_AUTOML_S):
        self.features, self.orcamento = features, orcamento

    def fit(self, df: pd.DataFrame):
        from flaml import AutoML

        self._cats = {c: df[c].cat.categories for c in self.features if c in t.CATEGORICAS}
        self.est = AutoML()
        self.est.fit(
            X_train=self._X(df),
            y_train=df[t.ALVO].astype(int).to_numpy(),
            sample_weight=(df["peso_final"] / df["peso_final"].mean()).to_numpy(),
            task="classification",
            metric="ap",
            eval_method="cv",
            n_splits=N_FOLDS,
            split_type="group",
            groups=df[t.GRUPO].to_numpy(),
            time_budget=self.orcamento,
            seed=42,
            verbose=0,
        )
        return self

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.features].copy()
        for c, cats in self._cats.items():
            X[c] = pd.Categorical(X[c], categories=cats)
        return X

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.est.predict_proba(self._X(df))[:, 1]


def rodar(saida: Path = PROCESSED_DIR, com_automl: bool = True) -> Path:
    if com_automl:
        try:
            from flaml import AutoML  # noqa: F401
        except ImportError as erro:
            raise SystemExit(
                "AutoML precisa do grupo opcional: uv run --group automl python -m "
                "occasion.modelagem.candidatos"
            ) from erro
    tr = t.preparar(t.carregar("2008_2009"))
    te = t.preparar(t.carregar("2017_2018"))
    y = te[t.ALVO].astype(int).to_numpy()
    w = te["peso_metrica"].to_numpy()
    F = FEATURES_FINAIS

    # C da logística por CV (só 2008-09).
    grade_c = {C: np.mean(_cv(tr, lambda C=C: logistica(F, C))) for C in (0.01, 0.1, 1.0, 10.0)}
    melhor_c = max(grade_c, key=grade_c.get)
    print(f"C escolhido: {melhor_c}", flush=True)
    n_arvores = _arvores_lgbm(tr, F)
    print(f"árvores do LightGBM regularizado: {n_arvores}", flush=True)

    candidatos = {
        "1. logística (C=1)": lambda: logistica(F),
        f"2. logística (C={melhor_c} por CV)": lambda: logistica(F, melhor_c),
        "3. logística + turno × dia": lambda: logistica([*F, "turno_dia"], prep=_com_interacao),
        f"4. LightGBM regularizado ({n_arvores} árvores)": lambda: lgbm_regularizado(F, n_arvores),
        "5. Random Forest": lambda: floresta(F),
    }

    linhas, prev = [], {"baseline": t.Baseline().fit(tr).predict(te)}
    for nome, fabrica in candidatos.items():
        aps = _cv(tr, fabrica)
        prev[nome] = fabrica().fit(tr).predict(te)
        print(f"{nome}: CV {np.mean(aps):.4f}", flush=True)
        linhas.append(
            {
                "candidato": nome,
                "pr_auc_cv_2008": np.mean(aps),
                "ep_cv": np.std(aps, ddof=1) / np.sqrt(len(aps)),
            }
        )

    automl_desc = ""
    if com_automl:
        auto = _AutoML(F).fit(tr)
        nome = f"6. AutoML FLAML ({auto.est.best_estimator}; CV = AP interna)"
        prev[nome] = auto.predict(te)
        # A CV do FLAML é a do próprio fit (1 - perda = AP média nos folds por setor).
        linhas.append(
            {
                "candidato": nome,
                # AP interna do FLAML: não comparável à CV dos outros (ver docstring).
                "pr_auc_cv_2008": 1 - auto.est.best_loss,
                "ep_cv": np.nan,
            }
        )
        automl_desc = (
            f"AutoML: {ORCAMENTO_AUTOML_S}s, melhor estimador `{auto.est.best_estimator}`, "
            f"configuração `{ {k: float(round(v, 4)) if isinstance(v, float) else v for k, v in auto.est.best_config.items()} }`.\n"
        )

    tab = pd.DataFrame(linhas).set_index("candidato")
    sel = t.metades_setor(te) == 0
    te_sel, te_ava = te[sel].reset_index(drop=True), te[~sel].reset_index(drop=True)
    bs_sel = t.bootstrap_setor(te_sel, {k: p[sel] for k, p in prev.items()}, n=N_BOOTSTRAP)
    bs_ava = t.bootstrap_setor(te_ava, {k: p[~sel] for k, p in prev.items()}, n=N_BOOTSTRAP)
    ref = "1. logística (C=1)"
    for nome in tab.index:
        tab.loc[nome, "pr_auc_selecao"] = t.metricas(y[sel], prev[nome][sel], w[sel])["pr_auc"]
        m = t.metricas(y[~sel], prev[nome][~sel], w[~sel])
        tab.loc[nome, "pr_auc_avaliacao"] = m["pr_auc"]
        tab.loc[nome, "recall_top10_avaliacao"] = m["recall_top10"]
        tab.loc[nome, "dif_vs_logistica_avaliacao [IC95]"] = _ic(bs_ava[nome] - bs_ava[ref])
        tab.loc[nome, "dif_vs_baseline_avaliacao [IC95]"] = _ic(bs_ava[nome] - bs_ava["baseline"])

    # Regra de escolha (fixada antes): maior PR-AUC na metade de seleção; se a
    # diferença para um mais simples tem IC95 com zero, fica o mais simples.
    melhor = tab["pr_auc_selecao"].idxmax()
    escolhido = next(
        n for n in tab.index if n == melhor or (bs_sel[melhor] - bs_sel[n]).quantile(0.025) <= 0
    )
    tab.loc[escolhido, "pr_auc_2017_completo"] = t.metricas(y, prev[escolhido], w)["pr_auc"]

    partes = [
        "# Occasion Score: comparação de candidatos\n",
        (
            "Treino em 2008-09. Escolha na metade de seleção dos setores de 2017-18; "
            "a metade de avaliação só mede. Regra: maior PR-AUC na seleção; se a "
            "diferença para um candidato mais simples tem IC95 com zero, fica o mais "
            "simples. A CV de 2008-09 aparece só para comparação.\n"
        ),
        "Grade do C da logística (CV 2008-09): "
        + ", ".join(f"C={c}: {v:.4f}" for c, v in grade_c.items())
        + "\n",
        automl_desc,
        _tabela_md(tab),
        f"\nMaior PR-AUC na seleção: **{melhor}**. Escolhido pela regra: **{escolhido}**.\n",
    ]
    relatorio = saida / "candidatos_occasion_score.md"
    relatorio.write_text("\n".join(partes) + "\n", encoding="utf-8")
    return relatorio


def main() -> None:
    print(rodar())


if __name__ == "__main__":
    main()
