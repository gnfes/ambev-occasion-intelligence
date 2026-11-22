"""Viés do Occasion Score por tipo de área e região, medido como no modelo final.

O Mapa de Oportunidade soma o score por município, então um viés por região
ou tipo de área (capital, resto da RM, interior urbano, rural) desloca
municípios inteiros de categoria. Os fatores de `oportunidade.CORRECAO_REGIAO`
e `oportunidade.PREVISTO_SOBRE_OBS` saem daqui. Aqui o viés é medido do mesmo jeito que o modelo
final foi calibrado: previsões de 2017-18 fora da amostra (treino em
2008-09 + a outra metade dos setores de 2017-18), com o Platt do modelo
final. O intervalo vem de bootstrap por setor.

`previsto_sobre_obs` acima de 1 = o modelo superestima o grupo.

    uv run python -m occasion.mapa.vies
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from occasion.modelagem import final
from occasion.modelagem import treino as t
from occasion.modelagem.experimentos import FEATURES_FINAIS
from occasion.paths import PROCESSED_DIR

ARQUIVO = PROCESSED_DIR / "vies_occasion_score.csv"
N_BOOT = 500


def previsoes_fora_da_amostra(
    antiga: pd.DataFrame, recente: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    """2017-18 com o score calibrado fora da amostra (mesmo esquema de `final.treinar`)."""
    tudo = final._juntar(antiga, recente)
    eh_recente = (tudo["edicao"] == recente["edicao"].iloc[0]).to_numpy()
    rec = tudo[eh_recente].reset_index(drop=True)
    lado = t.metades_setor(rec)
    logit = np.empty(len(rec))
    for k in (0, 1):
        m = final._ajustar(final._juntar(tudo[~eh_recente], rec[lado != k]), FEATURES_FINAIS)
        logit[lado == k] = t._logit(m.predict(rec[lado == k]))
    y = rec[t.ALVO].astype(int).to_numpy()
    w = rec["peso_metrica"].to_numpy()
    a, b = final._platt(logit, y, w)
    return rec, 1 / (1 + np.exp(-(a * logit + b)))


def razao_por_grupo(
    rec: pd.DataFrame, score: np.ndarray, grupo: str, n_boot: int = N_BOOT, semente: int = 7
) -> pd.DataFrame:
    """Previsto ÷ observado por grupo, com IC95 por bootstrap de setores."""
    y = rec[t.ALVO].astype(float).to_numpy()
    w = rec["peso_metrica"].to_numpy()
    df = pd.DataFrame(
        {"g": rec[grupo].astype(str), "setor": rec["id_setor"], "py": score * w, "oy": y * w}
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
    pos = rec.groupby(rec[grupo].astype(str))[t.ALVO].sum()
    return pd.DataFrame(
        {
            "grupo": grupo,
            "valor": ponto.index,
            "positivos": pos.reindex(ponto.index).to_numpy(),
            "previsto_sobre_obs": ponto.to_numpy(),
            "ic95_baixo": boots.quantile(0.025).to_numpy(),
            "ic95_alto": boots.quantile(0.975).to_numpy(),
        }
    )


def main() -> None:
    antiga = t.preparar(t.carregar("2008_2009"))
    recente = t.preparar(t.carregar("2017_2018"))
    rec, score = previsoes_fora_da_amostra(antiga, recente)
    rec = rec.assign(regiao_tipo=rec["regiao"].astype(str) + " | " + rec["tipo_area"].astype(str))
    tabela = pd.concat(
        [razao_por_grupo(rec, score, g) for g in ("tipo_area", "regiao", "regiao_tipo")],
        ignore_index=True,
    )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tabela.round(3).to_csv(ARQUIVO, index=False)
    print(tabela.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
