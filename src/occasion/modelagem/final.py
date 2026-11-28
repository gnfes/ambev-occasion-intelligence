"""Modelo final do Occasion Score, usado pela aplicação.

A validação temporal (treino 2008-09, teste 2017-18) escolheu o modelo e
mediu o desempenho; este módulo não muda essa escolha. Para a aplicação, a
mesma logística é reajustada nas duas edições, como decidido em 16/11, e
calibrada em 2017-18 (Platt):

1. Calibração sem vazamento: os setores de 2017-18 são divididos em duas
   metades. Para cada metade, o modelo é treinado em 2008-09 + a outra
   metade e prevê esta. O Platt é ajustado nessas previsões fora da amostra.
2. Modelo final: a logística é reajustada em 2008-09 + 2017-18 inteiras, e
   o Platt do passo 1 é aplicado na saída dela.

Pesos de treino: `peso_final` dividido pela média da edição, para as duas
edições terem a mesma importância por linha. Calibração: `peso_metrica`
(cada dia da semana vale 1/7), a população em que o score é usado.

Explicação (`fatores`): na logística, o SHAP com features independentes é
exato e fechado. A contribuição de cada termo é coef × (x − média do termo
na população de referência), no logit. As colunas de uma mesma feature
(one-hot ou splines) são somadas. A calibração de Platt multiplica o logit
por `a`, então as contribuições também são multiplicadas por `a` e somam,
junto com o valor base, o logit do Occasion Score.

Correção por grupo (v1.1.0, `correcao.py`): depois do Platt, o logit recebe
um offset por grupo (região, tipo de área) cujo previsto ÷ observado fora
da amostra em 2017-18 tem IC95 que exclui 1, mais um termo global. Os
offsets são estimados nas mesmas previsões fora da amostra do Platt. Na
explicação, o offset do grupo entra na contribuição da feature dele.

    uv run python -m occasion.modelagem.final
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from occasion.modelagem import correcao
from occasion.modelagem import treino as t
from occasion.modelagem.experimentos import FEATURES_FINAIS, classe
from occasion.paths import RAIZ

VERSAO = "1.1.0"
MODELOS_DIR = RAIZ / "models"
ARQUIVO_MODELO = MODELOS_DIR / f"occasion_score_v{VERSAO}.joblib"

# Faixas do score pela posição no peso da população de 2017-18 (pessoa-turno,
# cada dia da semana 1/7): alta = 10% maiores, média = os 20% seguintes.
TOPO_ALTA = 0.10
TOPO_MEDIA = 0.30

ROTULOS = {
    "turno": "turno",
    "dia_semana": "dia da semana",
    "regiao": "região",
    "tipo_area": "tipo de área",
    "faixa_idade": "faixa de idade",
    "sexo": "sexo",
    "renda_pct": "renda (percentil)",
}


def _corte_topo(score: np.ndarray, w: np.ndarray, fracao: float) -> float:
    """Menor score que ainda está no topo `fracao` do peso."""
    ordem = np.argsort(-score)
    acum = np.cumsum(w[ordem]) / w.sum()
    return float(score[ordem][np.searchsorted(acum, fracao)])


def _peso_treino(df: pd.DataFrame) -> pd.Series:
    return df["peso_final"] / df.groupby("edicao")["peso_final"].transform("mean")


def _ajustar(df: pd.DataFrame, features: list[str]):
    """Logística das features finais com pesos normalizados por edição."""
    m = t.Modelo("logistica", features)
    m._cats = {c: df[c].cat.categories for c in features if c in t.CATEGORICAS}
    m.est = t._logistica(features).fit(
        df[features], df[t.ALVO].astype(int), logisticregression__sample_weight=_peso_treino(df)
    )
    return m


def _juntar(*partes: pd.DataFrame) -> pd.DataFrame:
    """Concatena edições preparadas, unindo as categorias das colunas categóricas."""
    df = pd.concat([p.astype({c: "string" for c in t.CATEGORICAS}) for p in partes])
    for c in t.CATEGORICAS:
        df[c] = df[c].astype("category")
    return df.reset_index(drop=True)


@dataclass
class OccasionScore:
    """Logística final + Platt + referência para explicar cada previsão."""

    features: list[str]
    modelo: t.Modelo
    platt_a: float
    platt_b: float
    media_termos: np.ndarray
    corte_alta: float
    corte_media: float
    metadata: dict = field(default_factory=dict)
    # Renda em salários mínimos → percentil de renda, na distribuição de
    # 2017-18 (a entrada do app é em SM; o modelo usa o percentil).
    renda_sm_ref: np.ndarray = field(default_factory=lambda: np.zeros(0))
    renda_pct_ref: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # Correção por grupo no logit calibrado: {"global": c, coluna: {valor: offset}}.
    offsets: dict = field(default_factory=dict)
    # Média ponderada do offset de cada coluna em 2017-18 (referência da explicação).
    media_offsets: dict = field(default_factory=dict)
    # Score de 2017-18 por classe de renda, em percentis 0..100 (posição na classe).
    score_classe_ref: dict = field(default_factory=dict)

    # ------------------------------------------------------------ previsão

    def _termos(self, df: pd.DataFrame) -> np.ndarray:
        X = self.modelo._X(df)
        Z = self.modelo.est[0].transform(X)
        return np.asarray(Z.todense() if hasattr(Z, "todense") else Z, dtype=float)

    def _logit_bruto(self, df: pd.DataFrame) -> np.ndarray:
        lr = self.modelo.est[-1]
        return self._termos(df) @ lr.coef_[0] + lr.intercept_[0]

    def offset(self, df: pd.DataFrame) -> np.ndarray:
        """Correção por grupo de cada linha, no logit (zero sem correção)."""
        return correcao.aplicar(self.offsets, df) if self.offsets else np.zeros(len(df))

    def logit(self, df: pd.DataFrame) -> np.ndarray:
        """Logit do Occasion Score: Platt sobre o logit bruto + correção por grupo."""
        return self.platt_a * self._logit_bruto(df) + self.platt_b + self.offset(df)

    def prever(self, df: pd.DataFrame) -> np.ndarray:
        """Occasion Score em probabilidade (0 a 1)."""
        return 1 / (1 + np.exp(-self.logit(df)))

    def vetor_offsets(self) -> np.ndarray:
        """Offsets alinhados às colunas transformadas do modelo (zero nas sem correção).

        `logit = platt_a × (termos @ coef + intercepto) + platt_b + global + termos @ vetor`,
        porque as colunas de região e tipo de área são one-hot.
        """
        nomes = self.modelo.est[0].get_feature_names_out()
        vetor = np.zeros(len(nomes))
        for i, n in enumerate(nomes):
            bruto = n.split("__", 1)[1]
            for col, valores in self.offsets.items():
                if col == "global":
                    continue
                for val, off in valores.items():
                    if bruto == f"{col}_{val}":
                        vetor[i] = off
        return vetor

    def percentil_na_classe(self, score: np.ndarray, classes: list[str]) -> np.ndarray:
        """Posição (0 a 100) do score entre as pessoas-turno da mesma classe em 2017-18."""
        saida = np.full(len(classes), np.nan)
        for i, (s, c) in enumerate(zip(np.asarray(score, dtype=float), classes, strict=True)):
            ref = self.score_classe_ref.get(c)
            if ref is not None and len(ref):
                saida[i] = np.interp(s, ref, np.linspace(0, 100, len(ref)))
        return saida

    def renda_pct(self, renda_sm: float | np.ndarray) -> np.ndarray:
        """Percentil de renda (0 a 100) da renda mensal do domicílio em salários mínimos."""
        return np.interp(renda_sm, self.renda_sm_ref, self.renda_pct_ref)

    def faixa(self, score: np.ndarray) -> np.ndarray:
        return np.where(
            score >= self.corte_alta, "alta", np.where(score >= self.corte_media, "media", "baixa")
        )

    # ----------------------------------------------------------- explicação

    def _grupos(self) -> list[str]:
        """Feature de origem de cada coluna transformada."""
        nomes = self.modelo.est[0].get_feature_names_out()
        grupos = []
        for n in nomes:
            bruto = n.split("__", 1)[1]
            grupos.append(next(f for f in self.features if bruto.startswith(f)))
        return grupos

    def valor_base(self) -> float:
        """Logit calibrado da linha média de referência (onde as contribuições somam zero)."""
        lr = self.modelo.est[-1]
        bruto = float(self.media_termos @ lr.coef_[0] + lr.intercept_[0])
        extra = float(self.offsets.get("global", 0.0)) + sum(self.media_offsets.values())
        return self.platt_a * bruto + self.platt_b + extra

    def contribuicoes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Contribuição de cada feature no logit calibrado (SHAP linear), uma linha por previsão."""
        coef = self.modelo.est[-1].coef_[0]
        termos = (self._termos(df) - self.media_termos) * coef * self.platt_a
        contrib = (
            pd.DataFrame(termos, columns=self._grupos(), index=df.index)
            .T.groupby(level=0)
            .sum()
            .T[self.features]
        )
        for col, valores in self.offsets.items():
            if col == "global":
                continue
            off = df[col].astype(str).map(valores).fillna(0.0).to_numpy(dtype=float)
            contrib[col] += off - self.media_offsets.get(col, 0.0)
        return contrib

    def fatores(self, df: pd.DataFrame, n: int = 3) -> list[list[dict]]:
        """Os `n` fatores de maior efeito de cada previsão, com direção e razão de chances."""
        contrib = self.contribuicoes(df)
        saida = []
        for i, linha in contrib.iterrows():
            top = linha.reindex(linha.abs().sort_values(ascending=False).index)[:n]
            saida.append(
                [
                    {
                        "feature": f,
                        "rotulo": ROTULOS.get(f, f),
                        "valor": _valor_legivel(df.at[i, f]),
                        "contribuicao_logit": round(float(c), 4),
                        "razao_chances": round(float(np.exp(c)), 3),
                        "direcao": "aumenta" if c > 0 else "reduz",
                    }
                    for f, c in top.items()
                ]
            )
        return saida

    # --------------------------------------------------------- persistência

    def salvar(self, caminho: Path = ARQUIVO_MODELO) -> Path:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, caminho)
        caminho.with_suffix(".json").write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return caminho


def _valor_legivel(v: object) -> object:
    if isinstance(v, float | np.floating):
        return round(float(v), 1)
    if isinstance(v, np.integer):
        return int(v)
    return str(v) if not isinstance(v, int | str) else v


def carregar(caminho: Path = ARQUIVO_MODELO) -> OccasionScore:
    return joblib.load(caminho)


def _platt(logit_bruto: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    cal = LogisticRegression(C=1e6).fit(logit_bruto.reshape(-1, 1), y, sample_weight=w)
    return float(cal.coef_[0, 0]), float(cal.intercept_[0])


def logit_oof(
    antiga: pd.DataFrame, recente: pd.DataFrame, features: list[str] = FEATURES_FINAIS
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    """Logit bruto de 2017-18 fora da amostra (metades de setores).

    Para cada metade, a logística é treinada em 2008-09 + a outra metade e
    prevê esta. Devolve (2017-18, logit fora da amostra, metade, as duas edições).
    """
    tudo = _juntar(antiga, recente)
    eh_recente = (tudo["edicao"] == recente["edicao"].iloc[0]).to_numpy()
    rec = tudo[eh_recente].reset_index(drop=True)
    lado = t.metades_setor(rec)
    saida = np.empty(len(rec))
    for k in (0, 1):
        m = _ajustar(_juntar(tudo[~eh_recente], rec[lado != k]), features)
        saida[lado == k] = t._logit(m.predict(rec[lado == k]))
    return rec, saida, lado, tudo


def _ref_por_classe(score: np.ndarray, w: np.ndarray, classes: pd.Series) -> dict:
    """Percentis 0..100 (ponderados) do score dentro de cada classe."""
    ref = {}
    for c in classes.dropna().unique():
        m = (classes == c).fillna(False).to_numpy(dtype=bool)
        ordem = np.argsort(score[m])
        s, ww = score[m][ordem], w[m][ordem]
        acum = (np.cumsum(ww) - ww / 2) / ww.sum()
        ref[str(c)] = np.interp(np.linspace(0, 1, 101), acum, s)
    return ref


def treinar(
    antiga: pd.DataFrame,
    recente: pd.DataFrame,
    features: list[str] = FEATURES_FINAIS,
    metricas_validacao: dict | None = None,
    corrigir: bool = True,
) -> OccasionScore:
    """Treina o modelo da aplicação a partir das duas edições já preparadas (`t.preparar`)."""
    if "renda_sm" not in recente or recente["renda_sm"].isna().all():
        raise ValueError("a edição recente precisa de `renda_sm` para converter a renda do app")
    # 1. Previsões de 2017-18 fora da amostra, para o Platt e a correção por grupo.
    rec, logit_fora, lado, tudo = logit_oof(antiga, recente, features)
    y = rec[t.ALVO].astype(int).to_numpy()
    w = rec["peso_metrica"].to_numpy()
    a, b = _platt(logit_fora, y, w)
    z = a * logit_fora + b
    selecionados = (
        correcao.selecionar_grupos(rec, rec["id_setor"], y, 1 / (1 + np.exp(-z)), w)
        if corrigir
        else []
    )
    candidatos = list(selecionados)
    selecionados = correcao.validar_grupos(logit_fora, y, w, rec, lado, selecionados)
    offsets = correcao.estimar_offsets(z, y, w, rec, selecionados) if selecionados else {}

    # 2. Modelo final nas duas edições inteiras.
    modelo = _ajustar(tudo, features)
    os_ = OccasionScore(features, modelo, a, b, np.zeros(1), 0.0, 0.0, offsets=offsets)

    # Referência da explicação e cortes das faixas: população de 2017-18.
    termos = os_._termos(rec)
    os_.media_termos = np.average(termos, axis=0, weights=w)
    os_.media_offsets = {
        col: float(np.average(rec[col].astype(str).map(vals).fillna(0.0), weights=w))
        for col, vals in offsets.items()
        if col != "global"
    }
    score = os_.prever(rec)
    os_.score_classe_ref = _ref_por_classe(score, w, classe(rec["renda_sm"]).astype("string"))
    os_.corte_alta = _corte_topo(score, w, TOPO_ALTA)
    os_.corte_media = _corte_topo(score, w, TOPO_MEDIA)
    ref = rec[["renda_sm", "renda_pct"]].dropna().drop_duplicates("renda_sm")
    ref = ref.sort_values("renda_sm")
    os_.renda_sm_ref = ref["renda_sm"].to_numpy(dtype=float)
    os_.renda_pct_ref = ref["renda_pct"].to_numpy(dtype=float)

    os_.metadata = {
        "modelo": "occasion_score",
        "versao": VERSAO,
        "treinado_em": date.today().isoformat(),
        "algoritmo": "regressão logística (one-hot + splines na renda) + calibração de Platt"
        + (" + correção por grupo no logit" if offsets else ""),
        "features": features,
        "treino": "POF 2008-09 + POF 2017-18 (pessoa × dia de registro × turno)",
        "calibracao": "Platt em previsões fora da amostra de 2017-18 (metades de setores)",
        "alvo": "pizza com refrigerante (genérico ou marca) com até 1 hora de diferença",
        "unidade": "pessoa × dia de registro × turno",
        "linhas_treino": len(tudo),
        "positivos_treino": int(tudo[t.ALVO].sum()),
        "platt": {"a": a, "b": b},
        "correcao_grupos": {
            "regra": "grupo de região ou tipo de área com previsto ÷ observado fora da amostra "
            f"em 2017-18 cujo IC95 (bootstrap por setor) exclui 1 e com ≥ {correcao.MIN_POSITIVOS} "
            "ocasiões, mantido só se a correção estimada numa metade dos setores melhora a "
            "calibração do grupo na outra; offset no logit calibrado estimado nas mesmas previsões",
            "candidatos_significativos": [list(g) for g in candidatos],
            "grupos": [list(g) for g in selecionados],
            "offsets": offsets,
        },
        "faixas": {
            "alta": f">= {os_.corte_alta:.5f} (10% maiores da população de 2017-18)",
            "media": f">= {os_.corte_media:.5f} (10% a 30%)",
            "baixa": "resto",
            "corte_alta": os_.corte_alta,
            "corte_media": os_.corte_media,
        },
        "taxa_media_2017_18": float(np.average(y, weights=w)),
        "validacao_temporal": (
            {
                **metricas_validacao,
                "nota": "logística + Platt, sem a correção por grupo (treino só em 2008-09); "
                "a correção é medida em docs/relatorios/vies_por_grupo.md",
            }
            if metricas_validacao
            else {}
        ),
    }
    return os_


def metricas_temporais(antiga: pd.DataFrame, recente: pd.DataFrame) -> dict:
    """Métricas da validação temporal (treino 2008-09, teste 2017-18), para o metadata."""
    y = recente[t.ALVO].astype(int).to_numpy()
    w = recente["peso_metrica"].to_numpy()
    lado = t.metades_setor(recente)
    bruto = t.Modelo("logistica", FEATURES_FINAIS).fit(antiga).predict(recente)
    score = t.calibrar_cruzado(bruto, recente, lado)
    base = t.Baseline().fit(antiga).predict(recente)
    return {
        "treino": "POF 2008-09",
        "teste": "POF 2017-18",
        "modelo": {k: round(float(v), 4) for k, v in t.metricas(y, score, w).items()},
        "baseline": {k: round(float(v), 4) for k, v in t.metricas(y, base, w).items()},
    }


def rodar(destino: Path = ARQUIVO_MODELO) -> Path:
    antiga = t.preparar(t.carregar("2008_2009"))
    recente = t.preparar(t.carregar("2017_2018"))
    modelo = treinar(antiga, recente, metricas_validacao=metricas_temporais(antiga, recente))
    return modelo.salvar(destino)


def main() -> None:
    # Importa pelo nome do pacote: rodando com `-m`, este arquivo é `__main__`
    # e o pickle gravaria a classe como `__main__.OccasionScore`.
    modulo = importlib.import_module("occasion.modelagem.final")
    print(modulo.rodar())


if __name__ == "__main__":
    main()
