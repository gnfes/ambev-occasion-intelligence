"""Auditoria de viés por grupo do Occasion Score e validação da correção.

Tudo em previsões de 2017-18 fora da amostra, no esquema do modelo final
(`final.logit_oof`: treino em 2008-09 + uma metade dos setores de 2017-18,
previsão na outra metade).

1. Antes: Platt só (o score da v1.0.0).
2. Depois: Platt + correção por grupo (`correcao.py`).

Para a comparação não usar os mesmos dados para estimar e medir, Platt,
seleção dos grupos e offsets são ajustados numa metade dos setores e
aplicados na outra, nos dois sentidos (validação cruzada em duas metades).
A correção só vale se melhora a calibração por grupo nas metades que não
viu sem piorar o log loss nem a PR-AUC.

Por grupo (região, tipo de área, classe, faixa de idade, sexo, público
20-44): ocasiões, previsto ÷ observado com IC95 por bootstrap de setores,
recall da faixa alta (10% maiores) e fatia da faixa alta. Para as classes,
também o recall do topo 10% dentro da própria classe, que é a regra
alternativa para a expansão B a D.

    uv run python -m occasion.modelagem.vies_grupos
"""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd

from occasion.modelagem import correcao, final
from occasion.modelagem import treino as t
from occasion.modelagem.experimentos import _tabela_md, classe
from occasion.paths import PROCESSED_DIR, RAIZ

ARQUIVO = PROCESSED_DIR / "vies_por_grupo.md"
ARQUIVO_CSV = PROCESSED_DIR / "vies_por_grupo.csv"
RELATORIO_DOCS = RAIZ / "docs" / "relatorios" / "vies_por_grupo.md"
PUBLICO = ["20-24", "25-29", "30-34", "35-39", "40-44"]
SEGMENTOS = ["regiao", "tipo_area", "classe", "faixa_idade", "sexo", "publico"]
# A escolha dos grupos (`final.treinar`) usa as metades de `t.metades_setor`
# (semente 7). A medição aqui usa outra divisão de setores, para não medir a
# correção nas mesmas metades que decidiram mantê-la.
SEMENTE_VALIDACAO = 2026
SEMENTES_ROBUSTEZ = range(1, 21)


def _sig(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def selecao_por_metade(
    rec: pd.DataFrame, logit_bruto: np.ndarray, lado: np.ndarray, n_boot: int = correcao.N_BOOT
) -> list[dict]:
    """Grupos que a regra de significância escolheria com só uma metade dos setores."""
    y = rec[t.ALVO].astype(int).to_numpy()
    w = rec["peso_metrica"].to_numpy()
    saida = []
    for k in (0, 1):
        m = lado == k
        a, b = final._platt(logit_bruto[m], y[m], w[m])
        sel = correcao.selecionar_grupos(
            rec[m], rec.loc[m, "id_setor"], y[m], _sig(a * logit_bruto[m] + b), w[m], n_boot=n_boot
        )
        saida.append({"metade": k, "grupos": sel})
    return saida


def robustez(
    rec: pd.DataFrame, logit_bruto: np.ndarray, grupos: list[tuple[str, str]]
) -> pd.DataFrame:
    """Previsto ÷ observado de cada grupo corrigido em 20 divisões de setores diferentes."""
    y = rec[t.ALVO].astype(int).to_numpy()
    w = rec["peso_metrica"].to_numpy()
    linhas = []
    for semente in SEMENTES_ROBUSTEZ:
        lado = t.metades_setor(rec, semente=semente)
        antes, depois = correcao.cruzar_metades(logit_bruto, y, w, rec, lado, grupos)
        for col, val in grupos:
            m = (rec[col].astype(str) == val).to_numpy()
            obs = (y * w)[m].sum()
            linhas.append(
                {
                    "grupo": f"{col} = {val}",
                    "antes": (antes * w)[m].sum() / obs,
                    "depois": (depois * w)[m].sum() / obs,
                }
            )
    df = pd.DataFrame(linhas)
    df["melhora"] = np.abs(np.log(df["depois"])) < np.abs(np.log(df["antes"]))
    return df.groupby("grupo").agg(
        pso_antes_mediana=("antes", "median"),
        pso_depois_mediana=("depois", "median"),
        pso_depois_min=("depois", "min"),
        pso_depois_max=("depois", "max"),
        divisoes_com_melhora_pct=("melhora", lambda s: 100 * s.mean()),
    )


def _grupos(rec: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "regiao": rec["regiao"].astype(str),
            "tipo_area": rec["tipo_area"].astype(str),
            "classe": classe(rec["renda_sm"]).astype(str),
            "faixa_idade": rec["faixa_idade"].astype(str),
            "sexo": rec["sexo"].astype(str).map({"1": "homem", "2": "mulher"}),
            "publico": np.where(rec["faixa_idade"].astype(str).isin(PUBLICO), "20-44", "outras"),
        },
        index=rec.index,
    )


def _topo_dentro(score: np.ndarray, w: np.ndarray, grupo: pd.Series) -> np.ndarray:
    """Fração de cada linha no topo 10% do peso dentro do próprio grupo."""
    saida = np.zeros(len(score))
    for g in grupo.unique():
        m = (grupo == g).to_numpy()
        saida[m] = t.fracao_topo(score[m], w[m])
    return saida


def tabela_grupos(
    rec: pd.DataFrame, antes: np.ndarray, depois: np.ndarray, n_boot: int = correcao.N_BOOT
) -> pd.DataFrame:
    y = rec[t.ALVO].astype(int).to_numpy()
    w = rec["peso_metrica"].to_numpy()
    g = _grupos(rec)
    topo_a = t.fracao_topo(antes, w)
    topo_d = t.fracao_topo(depois, w)
    dentro = _topo_dentro(depois, w, g["classe"])
    linhas = []
    for seg in SEGMENTOS:
        ra = correcao.razao_por_grupo(g[seg], rec["id_setor"], y, antes, w, n_boot=n_boot)
        rd = correcao.razao_por_grupo(g[seg], rec["id_setor"], y, depois, w, n_boot=n_boot)
        for (_, la), (_, ld) in zip(ra.iterrows(), rd.iterrows(), strict=True):
            m = (g[seg] == la["valor"]).to_numpy()
            oy = (y * w)[m].sum()
            linha = {
                "segmento": seg,
                "valor": la["valor"],
                "positivos": int(la["positivos"]),
                "ocasioes_pct": 100 * oy / (y * w).sum(),
                "pso_antes": la["previsto_sobre_obs"],
                "ic_antes": f"[{la['ic95_baixo']:.2f}; {la['ic95_alto']:.2f}]",
                "pso_depois": ld["previsto_sobre_obs"],
                "ic_depois": f"[{ld['ic95_baixo']:.2f}; {ld['ic95_alto']:.2f}]",
                "recall_alta_antes_pct": 100 * (y * w * topo_a)[m].sum() / oy,
                "recall_alta_depois_pct": 100 * (y * w * topo_d)[m].sum() / oy,
                "fatia_alta_depois_pct": 100 * (w * topo_d)[m].sum() / (w * topo_d).sum(),
            }
            if seg == "classe":
                linha["recall_topo_na_classe_pct"] = 100 * (y * w * dentro)[m].sum() / oy
            linhas.append(linha)
    return pd.DataFrame(linhas)


def metricas_globais(rec: pd.DataFrame, antes: np.ndarray, depois: np.ndarray) -> pd.DataFrame:
    y = rec[t.ALVO].astype(int).to_numpy()
    w = rec["peso_metrica"].to_numpy()
    linhas = {}
    for nome, p in (("antes (Platt)", antes), ("depois (Platt + correção)", depois)):
        m = t.metricas(y, p, w) | t.metricas_prob(y, p, w)
        linhas[nome] = {k: m[k] for k in ("pr_auc", "roc_auc", "recall_top10", "log_loss", "brier")}
    return pd.DataFrame(linhas).T


def _texto_ajustes(ajustes: list[dict]) -> str:
    return "\n".join(
        f"- Só com a metade {a['metade']}: "
        + (", ".join(f"{c} = {v}" for c, v in a["grupos"]) or "nenhum grupo")
        for a in ajustes
    )


def relatorio(
    rec: pd.DataFrame,
    tabela: pd.DataFrame,
    globais: pd.DataFrame,
    ajustes: list[dict],
    modelo: final.OccasionScore,
    rob: pd.DataFrame,
) -> str:
    cg = modelo.metadata.get("correcao_grupos", {})
    offs = "; ".join(
        f"{c} = {v}: {o:+.3f} no logit (×{np.exp(o):.2f} nas chances)"
        for c, vals in cg.get("offsets", {}).items()
        if c != "global"
        for v, o in vals.items()
    )
    colunas = [
        "valor",
        "positivos",
        "ocasioes_pct",
        "pso_antes",
        "ic_antes",
        "pso_depois",
        "ic_depois",
        "recall_alta_antes_pct",
        "recall_alta_depois_pct",
        "fatia_alta_depois_pct",
    ]
    blocos = []
    for seg in SEGMENTOS:
        sub = tabela[tabela["segmento"] == seg]
        cols = colunas + (["recall_topo_na_classe_pct"] if seg == "classe" else [])
        blocos.append(f"### {seg}\n\n{_tabela_md(sub[cols].set_index('valor'), 2)}")
    return f"""# Viés do Occasion Score por grupo e correção (v{modelo.metadata.get("versao")})

Previsões de 2017-18 fora da amostra ({len(rec):,} pessoa-turnos,
{int(rec[t.ALVO].sum())} ocasiões), no esquema do modelo final. Antes =
Platt (o score da v1.0.0). Depois = Platt + correção dos grupos servidos.
Platt e offsets ajustados numa metade dos setores e medidos na outra, nos
dois sentidos, numa divisão de setores diferente da usada para escolher os
grupos. Os grupos são os do modelo servido, escolhidos nos dados inteiros:
a medição não é de todo independente da escolha, por isso a robustez em 20
divisões vem logo abaixo. `pso` = previsto ÷ observado (acima de 1 = o modelo
superestima o grupo), IC95 por bootstrap de setores.

## Correção servida no modelo v{modelo.metadata.get("versao")}

Regra: {cg.get("regra", "sem correção")}.

Grupos corrigidos: {offs or "nenhum"}. Termo global:
{cg.get("offsets", {}).get("global", 0.0):+.3f}.

Candidatos com IC95 que exclui 1 nos dados inteiros:
{", ".join(f"{c} = {v}" for c, v in cg.get("candidatos_significativos", [])) or "nenhum"}.

## Validação cruzada da correção

{_tabela_md(globais, 4)}

Robustez: o mesmo ajuste em {len(SEMENTES_ROBUSTEZ)} divisões de setores diferentes.

{_tabela_md(rob, 2)}

Estabilidade da seleção (a regra de significância rodada em cada metade,
com metade das ocasiões e IC mais largo):

{_texto_ajustes(ajustes)}

## Por grupo

{chr(10).join(blocos)}

## Como ler

- A correção muda o nível do score dos grupos com viés significativo; a
  ordem dentro de cada grupo não muda, porque o offset é o mesmo para todos
  do grupo.
- Classe de renda não é corrigida. Se a classe D tem `pso` perto de 1 e
  recall baixo na faixa alta, o score está certo e a ocasião é de fato
  menos frequente; a regra alternativa (topo 10% dentro da classe, última
  coluna) mostra quantas ocasiões da classe um alvo por classe alcançaria.
- Grupos com poucas ocasiões (menos de 30) têm IC largo; nenhuma conclusão
  sai deles.
"""


def main() -> None:
    antiga = t.preparar(t.carregar("2008_2009"))
    recente = t.preparar(t.carregar("2017_2018"))
    rec, logit_bruto, lado, _ = final.logit_oof(antiga, recente)
    modelo = final.carregar()
    grupos = [tuple(g) for g in modelo.metadata["correcao_grupos"]["grupos"]]
    y = rec[t.ALVO].astype(int).to_numpy()
    w = rec["peso_metrica"].to_numpy()
    lado_val = t.metades_setor(rec, semente=SEMENTE_VALIDACAO)
    antes, depois = correcao.cruzar_metades(logit_bruto, y, w, rec, lado_val, grupos)
    ajustes = selecao_por_metade(rec, logit_bruto, lado)
    rob = robustez(rec, logit_bruto, grupos)
    tabela = tabela_grupos(rec, antes, depois)
    globais = metricas_globais(rec, antes, depois)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tabela.round(4).to_csv(ARQUIVO_CSV, index=False)
    ARQUIVO.write_text(relatorio(rec, tabela, globais, ajustes, modelo, rob), encoding="utf-8")
    shutil.copy(ARQUIVO, RELATORIO_DOCS)
    print(globais.round(4).to_string())
    print(rob.round(3).to_string())
    print(tabela.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
