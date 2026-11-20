"""Experimentos do Occasion Score: baseline, modelos, ablação, calibração e segmentos.

Roda tudo de uma vez e grava um relatório em Markdown e as previsões de
2017-18 em `data/processed/`. Treino: POF 2008-09. Avaliação: POF 2017-18.

    uv run python -m occasion.modelagem.experimentos
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from occasion.modelagem import treino as t
from occasion.paths import PROCESSED_DIR

# Features finais (decididas pela ablação; ver relatório).
FEATURES_FINAIS = [
    "turno",
    "dia_semana",
    "regiao",
    "tipo_area",
    "faixa_idade",
    "sexo",
    "renda_pct",
]

# Classes por renda mensal da unidade de consumo em salários mínimos (corte
# usual do IBGE: E até 2 SM, D 2-4, C 4-10, B 10-20, A acima de 20).
CLASSES = pd.CategoricalDtype(["E", "D", "C", "B", "A"], ordered=True)

N_BOOTSTRAP = 300


def _sem(*cols: str) -> list[str]:
    return [f for f in t.FEATURES if f not in cols]


VARIANTES = {
    "completo (10 features)": t.FEATURES,
    "sem tem_crianca": _sem("tem_crianca"),
    "sem n_moradores": _sem("n_moradores"),
    "sem os dois": _sem("tem_crianca", "n_moradores"),
    "sem os dois, só região": _sem("tem_crianca", "n_moradores", "trends_pizza_rank_uf"),
    "sem os dois, só Trends": _sem("tem_crianca", "n_moradores", "regiao"),
    "sem os dois, sem geografia": _sem(
        "tem_crianca", "n_moradores", "regiao", "trends_pizza_rank_uf"
    ),
    "Trends normalizado (0-1)": [
        "trends_pct_uf" if f == "trends_pizza_rank_uf" else f for f in t.FEATURES
    ],
    "dia agrupado (seg-qui, sex, sáb, dom)": [
        "dia_grupo" if f == "dia_semana" else f for f in t.FEATURES
    ],
}


def classe(renda_sm: pd.Series) -> pd.Series:
    return pd.cut(renda_sm, [-np.inf, 2, 4, 10, 20, np.inf], labels=list(CLASSES.categories))


def _ic(serie: pd.Series) -> str:
    return f"{serie.mean():+.4f} [{serie.quantile(0.025):+.4f}; {serie.quantile(0.975):+.4f}]"


def _tabela_md(df: pd.DataFrame, casas: int = 4) -> str:
    """Tabela Markdown sem depender do pacote `tabulate`."""
    df = df.reset_index()

    def fmt(v: object) -> str:
        if isinstance(v, float | np.floating):
            return f"{v:.{casas}f}"
        return "" if pd.isna(v) else str(v)

    linhas = ["| " + " | ".join(map(str, df.columns)) + " |"]
    linhas.append("|" + "---|" * len(df.columns))
    linhas += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(linhas)


def segmentos(te: pd.DataFrame, score: np.ndarray, base: np.ndarray) -> pd.DataFrame:
    """Métricas por segmento: ranking dentro do segmento e calibração do segmento."""
    y = te[t.ALVO].astype(int).to_numpy()
    w = te["peso_metrica"].to_numpy()
    cortes = {
        "turno": te["turno"].astype(str),
        "regiao": te["regiao"].astype(str),
        "dia_semana": te["dia_semana"].astype(str),
        "publico": np.where(
            te["faixa_idade"].astype(str).isin(["20-24", "25-29", "30-34", "35-39", "40-44"]),
            "20-44",
            "outras idades",
        ),
        "classe": classe(te["renda_sm"]).astype(str),
    }
    linhas = []
    for nome, grupo in cortes.items():
        for valor in sorted(pd.unique(grupo)):
            s = np.asarray(grupo == valor)
            pos = int(y[s].sum())
            linha = {
                "segmento": nome,
                "valor": valor,
                "positivos": pos,
                "taxa_obs_pct": 100 * np.average(y[s], weights=w[s]),
                "score_medio_pct": 100 * np.average(score[s], weights=w[s]),
            }
            if 0 < pos < s.sum():
                linha["pr_auc_modelo"] = average_precision_score(y[s], score[s], sample_weight=w[s])
                linha["pr_auc_baseline"] = average_precision_score(
                    y[s], base[s], sample_weight=w[s]
                )
            linhas.append(linha)
    return pd.DataFrame(linhas).set_index(["segmento", "valor"])


def canal(te: pd.DataFrame, score: np.ndarray, corte: float) -> pd.DataFrame:
    """Recall por local da pizza (em casa vs fora): só existe para os positivos."""
    pos = te[t.ALVO].astype(bool).to_numpy()
    local = te["fora_de_casa"].map({True: "fora de casa", False: "em casa"}).fillna("desconhecido")
    w = te["peso_metrica"].to_numpy()
    linhas = []
    for valor in ["em casa", "fora de casa", "desconhecido"]:
        s = pos & (local == valor).to_numpy()
        if s.sum():
            linhas.append(
                {
                    "local": valor,
                    "positivos": int(s.sum()),
                    "recall_no_top10": float(w[s & (score >= corte)].sum() / w[s].sum()),
                }
            )
    return pd.DataFrame(linhas).set_index("local")


def rodar(saida: Path = PROCESSED_DIR) -> Path:
    tr = t.preparar(t.carregar("2008_2009"))
    te = t.preparar(t.carregar("2017_2018"))
    y = te[t.ALVO].astype(int).to_numpy()
    w = te["peso_metrica"].to_numpy()
    lado = t.metades_setor(te)
    partes: list[str] = []

    # 1. Hiperparâmetros do LightGBM por CV por setor, só em 2008-09.
    params, n_arvores, grade = t.escolher_lgbm(tr, t.FEATURES)

    # 2. Ablação com os dois modelos.
    prev: dict[str, np.ndarray] = {"baseline": t.Baseline().fit(tr).predict(te)}
    for nome, feats in VARIANTES.items():
        prev[f"logistica | {nome}"] = t.Modelo("logistica", feats).fit(tr).predict(te)
        prev[f"lgbm | {nome}"] = t.Modelo("lgbm", feats, params, n_arvores).fit(tr).predict(te)
    prev["logistica | FINAL"] = t.Modelo("logistica", FEATURES_FINAIS).fit(tr).predict(te)
    prev["lgbm | FINAL"] = t.Modelo("lgbm", FEATURES_FINAIS, params, n_arvores).fit(tr).predict(te)

    bs = t.bootstrap_setor(te, prev, n=N_BOOTSTRAP)
    linhas = []
    for nome, p in prev.items():
        ref = (
            "baseline" if nome == "baseline" else nome.split(" | ")[0] + " | completo (10 features)"
        )
        linhas.append(
            {
                "variante": nome,
                **t.metricas(y, p, w),
                "dif_pr_auc_vs_completo [IC95]": _ic(bs[nome] - bs[ref]),
                "dif_pr_auc_vs_baseline [IC95]": _ic(bs[nome] - bs["baseline"]),
            }
        )
    ablacao = pd.DataFrame(linhas).set_index("variante")

    # 3. Modelo final: logística com as features finais, calibrada (Platt cruzado).
    bruto = prev["logistica | FINAL"]
    score = t.calibrar_cruzado(bruto, te, lado)
    base_cal = t.calibrar_cruzado(prev["baseline"], te, lado)
    prob = pd.DataFrame(
        {
            "baseline calibrado": t.metricas_prob(y, base_cal, w),
            "logística final sem calibrar": t.metricas_prob(y, bruto, w),
            "logística final calibrada": t.metricas_prob(y, score, w),
        }
    ).T.rename_axis("modelo")
    classif = pd.DataFrame(
        {
            "baseline": t.classificacao_cruzada(y, base_cal, w, lado),
            "logística final": t.classificacao_cruzada(y, score, w, lado),
        }
    ).T.rename_axis("modelo")
    curva = t.curva_calibracao(y, score, w)
    seg = segmentos(te, score, prev["baseline"])
    corte_top10 = float(np.quantile(score, 0.9))
    can = canal(te, score, corte_top10)

    # Coeficientes da logística (para conferência; SHAP vem na fase de explicação).
    modelo = t.Modelo("logistica", FEATURES_FINAIS).fit(tr)
    nomes = modelo.est[0].get_feature_names_out()
    coefs = pd.Series(modelo.est[-1].coef_[0], index=nomes).sort_values()

    # Relatório
    partes.append("# Occasion Score: experimentos (treino POF 2008-09, teste POF 2017-18)\n")
    partes.append(
        f"Linhas: treino {len(tr):,} ({int(tr[t.ALVO].sum())} positivos), "
        f"teste {len(te):,} ({int(y.sum())} positivos). Taxa ponderada no teste: "
        f"{100 * np.average(y, weights=w):.3f}% das pessoa-turno.\n"
    )
    partes.append("## Grade do LightGBM (CV por setor em 2008-09)\n")
    partes.append(_tabela_md(grade.set_index(["num_leaves", "min_child_samples"])))
    partes.append(f"\nEscolhido: {params}, {n_arvores} árvores.\n")
    partes.append("## Ablação (ranking em 2017-18; IC95 por bootstrap pareado de setores)\n")
    partes.append(_tabela_md(ablacao))
    partes.append("\n## Probabilidade (Platt cruzado por setor)\n")
    partes.append(_tabela_md(prob, 5))
    partes.append("\n## Classificação (limiar de F1 escolhido na outra metade dos setores)\n")
    partes.append(_tabela_md(classif))
    partes.append("\n## Curva de calibração (decis ponderados do score)\n")
    partes.append(_tabela_md(curva * [100, 100, 1], 3))
    partes.append("\n## Segmentos (modelo final calibrado)\n")
    partes.append(_tabela_md(seg))
    partes.append("\n## Canal: recall no top 10% por local da pizza\n")
    partes.append(_tabela_md(can))
    partes.append("\n## Coeficientes da logística final\n")
    partes.append(_tabela_md(coefs.rename_axis("termo").to_frame("coef"), 3))

    saida.mkdir(parents=True, exist_ok=True)
    relatorio = saida / "experimentos_occasion_score.md"
    relatorio.write_text("\n".join(partes) + "\n")
    te.assign(occasion_score=100 * score, score_baseline=100 * base_cal)[
        ["id_pessoa", "dia_registro", "turno", t.ALVO, "occasion_score", "score_baseline"]
    ].to_parquet(saida / "previsoes_2017_2018.parquet", index=False)
    (saida / "experimentos_resumo.json").write_text(
        json.dumps(
            {
                "features_finais": FEATURES_FINAIS,
                "lgbm_params": params,
                "lgbm_arvores": n_arvores,
                "pr_auc": {k: float(v) for k, v in ablacao["pr_auc"].items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return relatorio


def main() -> None:
    print(rodar())


if __name__ == "__main__":
    main()
