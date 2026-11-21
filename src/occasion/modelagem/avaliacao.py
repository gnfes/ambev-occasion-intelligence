"""Avaliação e interpretação do Occasion Score: importância, faixas, segmentos e erros.

Tudo aqui usa a validação temporal (treino POF 2008-09, teste POF 2017-18,
Platt cruzado por setor), para que nenhum número venha de dados que o modelo
viu. A exceção é a importância média por SHAP, que descreve o modelo final
da aplicação (`final.py`).

Também calcula, em 2017-18, a fatia de guaraná nas ocasiões de pizza com
refrigerante por segmento, que alimenta a regra de conversão do Decision
Engine.

    uv run python -m occasion.modelagem.avaliacao
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from occasion.modelagem import final
from occasion.modelagem import treino as t
from occasion.modelagem.experimentos import FEATURES_FINAIS, _tabela_md, classe
from occasion.paths import PROCESSED_DIR

N_PERMUTACOES = 20
PUBLICO = ["20-24", "25-29", "30-34", "35-39", "40-44"]
# Abaixo disso a fatia de guaraná do segmento é instável e o Decision Engine
# usa a fatia nacional.
MIN_OCASIOES_GUARANA = 40


def importancia_permutacao(
    modelo: t.Modelo,
    te: pd.DataFrame,
    lado: np.ndarray,
    n: int = N_PERMUTACOES,
    semente: int = 3,
) -> pd.DataFrame:
    """Queda da PR-AUC ponderada em 2017-18 ao embaralhar cada feature.

    O Platt é monótono e não muda a ordenação, então a PR-AUC do score bruto
    é a mesma do calibrado dentro de cada metade; usamos o bruto.
    """
    y = te[t.ALVO].astype(int).to_numpy()
    w = te["peso_metrica"].to_numpy()
    ref = average_precision_score(y, modelo.predict(te), sample_weight=w)
    rng = np.random.default_rng(semente)
    linhas = []
    for f in modelo.features:
        quedas = []
        for _ in range(n):
            emb = te.copy()
            emb[f] = emb[f].to_numpy()[rng.permutation(len(emb))]
            if isinstance(te[f].dtype, pd.CategoricalDtype):
                emb[f] = pd.Categorical(emb[f], categories=te[f].cat.categories)
            quedas.append(ref - average_precision_score(y, modelo.predict(emb), sample_weight=w))
        linhas.append(
            {
                "feature": f,
                "queda_pr_auc": float(np.mean(quedas)),
                "queda_pct": 100 * float(np.mean(quedas)) / ref,
                "dp": float(np.std(quedas, ddof=1)),
            }
        )
    return pd.DataFrame(linhas).set_index("feature").sort_values("queda_pr_auc", ascending=False)


def importancia_shap(modelo: final.OccasionScore, te: pd.DataFrame) -> pd.Series:
    """Média ponderada do |SHAP| (logit calibrado) por feature, no modelo final."""
    contrib = modelo.contribuicoes(te).abs()
    w = te["peso_metrica"].to_numpy()
    return contrib.apply(lambda c: np.average(c, weights=w)).sort_values(ascending=False)


def faixas(score: np.ndarray, y: np.ndarray, w: np.ndarray, corte_alta: float, corte_media: float):
    """Peso da população, ocasiões e taxa observada em cada faixa do score."""
    faixa = np.where(score >= corte_alta, "alta", np.where(score >= corte_media, "media", "baixa"))
    linhas = []
    for f in ["alta", "media", "baixa"]:
        s = faixa == f
        if w[s].sum() == 0:  # faixa vazia (cortes iguais por empates)
            linhas.append({"faixa": f, "populacao_pct": 0.0, "ocasioes_pct": 0.0, "positivos": 0})
            continue
        linhas.append(
            {
                "faixa": f,
                "populacao_pct": 100 * w[s].sum() / w.sum(),
                "ocasioes_pct": 100 * (w * y)[s].sum() / (w * y).sum(),
                "taxa_obs_pct": 100 * np.average(y[s], weights=w[s]),
                "score_medio_pct": 100 * np.average(score[s], weights=w[s]),
                "positivos": int(y[s].sum()),
            }
        )
    return pd.DataFrame(linhas).set_index("faixa"), faixa


def matriz_confusao(y: np.ndarray, pred: np.ndarray, w: np.ndarray) -> pd.DataFrame:
    """Matriz de confusão ponderada, em % da população de pessoa-turno."""
    m = pd.DataFrame(
        {
            "previsto 1": [(w * (pred == 1) * (y == 1)).sum(), (w * (pred == 1) * (y == 0)).sum()],
            "previsto 0": [(w * (pred == 0) * (y == 1)).sum(), (w * (pred == 0) * (y == 0)).sum()],
        },
        index=pd.Index(["real 1", "real 0"], name="real"),
    )
    return 100 * m / w.sum()


def _cortes(te: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "turno": te["turno"].astype(str),
        "dia_semana": te["dia_semana"].astype(str),
        "regiao": te["regiao"].astype(str),
        "tipo_area": te["tipo_area"].astype(str),
        "faixa_idade": te["faixa_idade"].astype(str),
        "publico": pd.Series(
            np.where(te["faixa_idade"].astype(str).isin(PUBLICO), "20-44", "outras idades"),
            index=te.index,
        ),
        "sexo": te["sexo"].astype(str).map({"1": "homem", "2": "mulher"}),
        "classe": classe(te["renda_sm"]).astype(str),
    }


def segmentos(te: pd.DataFrame, score: np.ndarray, base: np.ndarray) -> pd.DataFrame:
    """Ranking (PR-AUC) e calibração (previsto/observado) dentro de cada segmento."""
    y = te[t.ALVO].astype(int).to_numpy()
    w = te["peso_metrica"].to_numpy()
    linhas = []
    for nome, grupo in _cortes(te).items():
        for valor in sorted(grupo.dropna().unique()):
            s = (grupo == valor).fillna(False).to_numpy(dtype=bool)
            pos = int(y[s].sum())
            obs = np.average(y[s], weights=w[s])
            prev = np.average(score[s], weights=w[s])
            linha = {
                "segmento": nome,
                "valor": valor,
                "positivos": pos,
                "taxa_obs_pct": 100 * obs,
                "score_medio_pct": 100 * prev,
                "previsto_sobre_obs": prev / obs if obs else np.nan,
            }
            if 0 < pos < s.sum():
                linha["pr_auc_modelo"] = average_precision_score(y[s], score[s], sample_weight=w[s])
                linha["pr_auc_baseline"] = average_precision_score(
                    y[s], base[s], sample_weight=w[s]
                )
            linhas.append(linha)
    return pd.DataFrame(linhas).set_index(["segmento", "valor"])


def perfil_erros(te: pd.DataFrame, faixa: np.ndarray) -> pd.DataFrame:
    """Onde estão os erros: perfil das ocasiões perdidas e dos alarmes falsos.

    - perdidas: ocasiões reais (alvo = 1) fora da faixa alta;
    - falsos alarmes: pessoa-turnos na faixa alta sem ocasião.
    Cada coluna é a distribuição ponderada (%) dentro do grupo, ao lado da
    distribuição de todas as ocasiões reais e de toda a faixa alta.
    """
    y = te[t.ALVO].astype(bool).to_numpy()
    w = te["peso_metrica"].to_numpy()
    alta = faixa == "alta"
    grupos = {
        "ocasioes_reais": y,
        "perdidas": y & ~alta,
        "faixa_alta": alta,
        "falsos_alarmes": alta & ~y,
    }
    cortes = _cortes(te)
    cortes["local_pizza"] = (
        te["fora_de_casa"].map({True: "fora de casa", False: "em casa"}).astype("string")
    )
    linhas = []
    for nome, grupo in cortes.items():
        for valor in sorted(grupo.dropna().unique()):
            s = (grupo == valor).fillna(False).to_numpy(dtype=bool)
            linha = {"segmento": nome, "valor": valor}
            for g, m in grupos.items():
                if nome == "local_pizza" and g in ("faixa_alta", "falsos_alarmes"):
                    continue  # local só existe quando houve pizza
                linha[f"{g}_pct"] = 100 * w[m & s].sum() / w[m].sum()
            linha["n_perdidas"] = int((y & ~alta & s).sum())
            linhas.append(linha)
    return pd.DataFrame(linhas).set_index(["segmento", "valor"])


def fatia_guarana(te: pd.DataFrame) -> pd.DataFrame:
    """Fatia (%) das ocasiões de pizza com refrigerante que tiveram guaraná, por segmento.

    Ponderada por `peso_metrica`. `ocasioes` é o número de ocasiões na
    amostra; abaixo de `MIN_OCASIOES_GUARANA` o valor é marcado como instável.
    O IC95 vem de bootstrap por setor (200 réplicas).
    """
    pos = te[te[t.ALVO].astype(bool)].copy()
    pos["guarana"] = pos["teve_guarana"].astype(float)
    cortes = {"nacional": pd.Series("Brasil", index=pos.index), **_cortes(pos)}
    cortes["local_pizza"] = (
        pos["fora_de_casa"].map({True: "fora de casa", False: "em casa"}).astype("string")
    )
    rng = np.random.default_rng(5)
    setores = pos["id_setor"].to_numpy()
    unicos = np.unique(setores)
    reps = [rng.choice(unicos, len(unicos)) for _ in range(200)]
    linhas = []
    for nome, grupo in cortes.items():
        for valor in sorted(grupo.dropna().unique()):
            g = pos[(grupo == valor).fillna(False).to_numpy(dtype=bool)]
            fatia = np.average(g["guarana"], weights=g["peso_metrica"])
            boot = []
            por_setor = g.groupby("id_setor").apply(
                lambda d: pd.Series(
                    {"gw": (d["guarana"] * d["peso_metrica"]).sum(), "w": d["peso_metrica"].sum()}
                ),
                include_groups=False,
            )
            for r in reps:
                sel = por_setor.reindex(r).dropna()
                if sel["w"].sum() > 0:
                    boot.append(sel["gw"].sum() / sel["w"].sum())
            linhas.append(
                {
                    "segmento": nome,
                    "valor": valor,
                    "ocasioes": len(g),
                    "com_guarana": int(g["guarana"].sum()),
                    "fatia_guarana_pct": 100 * fatia,
                    "ic95": f"[{100 * np.quantile(boot, 0.025):.1f}; "
                    f"{100 * np.quantile(boot, 0.975):.1f}]",
                    "estavel": len(g) >= MIN_OCASIOES_GUARANA,
                }
            )
    return pd.DataFrame(linhas).set_index(["segmento", "valor"])


def rodar(saida: Path = PROCESSED_DIR, arquivo_modelo: Path = final.ARQUIVO_MODELO) -> Path:
    tr = t.preparar(t.carregar("2008_2009"))
    te = t.preparar(t.carregar("2017_2018"))
    y = te[t.ALVO].astype(int).to_numpy()
    w = te["peso_metrica"].to_numpy()
    lado = t.metades_setor(te)

    modelo_t = t.Modelo("logistica", FEATURES_FINAIS).fit(tr)
    score = t.calibrar_cruzado(modelo_t.predict(te), te, lado)
    base = t.calibrar_cruzado(t.Baseline().fit(tr).predict(te), te, lado)

    # Mesmos cortes de faixa do modelo final: 10% e 30% maiores do peso.
    corte_alta = final._corte_topo(score, w, final.TOPO_ALTA)
    corte_media = final._corte_topo(score, w, final.TOPO_MEDIA)
    tab_faixas, faixa = faixas(score, y, w, corte_alta, corte_media)
    corte_base = final._corte_topo(base, w, final.TOPO_ALTA)
    tab_faixas_base, _ = faixas(
        base, y, w, corte_base, final._corte_topo(base, w, final.TOPO_MEDIA)
    )

    perm = importancia_permutacao(modelo_t, te, lado)
    modelo_app = final.carregar(arquivo_modelo)
    shap_medio = importancia_shap(modelo_app, te)
    seg = segmentos(te, score, base)
    erros = perfil_erros(te, faixa)
    guarana = fatia_guarana(te)
    confusao = matriz_confusao(y, (faixa == "alta").astype(int), w)

    partes = [
        "# Occasion Score: interpretação, faixas, segmentos e erros\n",
        (
            "Validação temporal: treino POF 2008-09, teste POF 2017-18 "
            f"({len(te):,} pessoa-turnos, {int(y.sum())} ocasiões). "
            "Pesos: `peso_final × fator_dia`.\n"
        ),
        "## Importância das features\n",
        (
            "Queda da PR-AUC em 2017-18 ao embaralhar a feature "
            f"({N_PERMUTACOES} permutações), ao lado da média do |SHAP| no modelo final.\n"
        ),
        _tabela_md(perm.join(shap_medio.rename("shap_medio_abs_logit"))),
        "\n## Faixas do score (alta = 10% maiores do peso, média = 10% a 30%)\n",
        (
            f"Cortes na validação temporal: alta ≥ {100 * corte_alta:.3f}%, "
            f"média ≥ {100 * corte_media:.3f}%.\n"
        ),
        "### Modelo\n",
        _tabela_md(tab_faixas, 3),
        "\n### Baseline (mesma regra de faixa)\n",
        (
            "O baseline tem poucos valores distintos: o bloco de empates que cruza o corte "
            f"entra inteiro, e a faixa alta dele cobre {tab_faixas_base.loc['alta', 'populacao_pct']:.1f}% "
            "do peso, não 10%.\n"
        ),
        _tabela_md(tab_faixas_base, 3),
        "\n## Matriz de confusão com a faixa alta como positivo (% da população)\n",
        _tabela_md(confusao, 3),
        "\n## Segmentos\n",
        "`previsto_sobre_obs` acima de 1 = o modelo superestima o segmento.\n",
        _tabela_md(seg, 3),
        "\n## Perfil dos erros (% ponderado dentro de cada grupo)\n",
        "Perdidas = ocasiões reais fora da faixa alta; falsos alarmes = faixa alta sem ocasião.\n",
        _tabela_md(erros, 1),
        "\n## Fatia de guaraná nas ocasiões de pizza com refrigerante (POF 2017-18)\n",
        f"Segmentos com menos de {MIN_OCASIOES_GUARANA} ocasiões ficam marcados como instáveis.\n",
        _tabela_md(guarana, 1),
    ]
    saida.mkdir(parents=True, exist_ok=True)
    relatorio = saida / "avaliacao_occasion_score.md"
    relatorio.write_text("\n".join(partes) + "\n")
    guarana.reset_index().to_csv(saida / "fatia_guarana_2017_2018.csv", index=False)
    (saida / "avaliacao_resumo.json").write_text(
        json.dumps(
            {
                "faixas": tab_faixas.round(4).to_dict(orient="index"),
                "permutacao": perm["queda_pct"].round(2).to_dict(),
                "shap_medio": shap_medio.round(4).to_dict(),
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
