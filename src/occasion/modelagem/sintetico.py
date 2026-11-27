"""Experimento controlado: dados sintéticos melhoram o Occasion Score?

Pergunta: somar linhas sintéticas ao treino (POF 2008-09) melhora o modelo
fora do tempo, medido só em dado real da POF 2017-18, sem distorcer a
probabilidade? O modelo servido (`models/`) não muda com este experimento.

Desenho (fixado antes de rodar):

- Os geradores aprendem só com 80% dos setores de 2008-09, numa reamostra
  sorteada proporcional a `peso_final` (imita a população, não a amostra).
  Os outros 20% dos setores são o holdout de fidelidade.
- Braços: A0 só real; A1 duplicação de positivos reais; A2 SMOTE-NC;
  A3 marginais independentes por classe (controle negativo); A4 VAE tabular
  condicional ao alvo.
- Mistura ponderada: treino = real + sintético, com peso `lam` nas linhas
  sintéticas (o peso médio de uma linha real é 1). Modo "positivos": só
  positivos sintéticos, em 1, 3 ou 10 vezes os positivos reais. Modo
  "completo": uma base sintética inteira do tamanho da real (A3 e A4).
- O modelo é sempre a logística das 7 features finais (one-hot, splines na
  renda, C=1). Só o conjunto de treino muda.
- A configuração (volume, `lam`) de cada braço é escolhida pela PR-AUC média
  das sementes na metade de seleção dos setores de 2017-18; a metade de
  avaliação só mede.
- Incerteza: bootstrap pareado de setores na metade de avaliação, com a
  métrica de cada braço como média das sementes em cada réplica.

Regra de decisão (fixada antes): um braço só é útil se, na avaliação,
(1) PR-AUC > A0 com IC95 acima de zero, (2) PR-AUC > A1, (3) log loss depois
de Platt não pior que A0, (4) nenhuma região ou classe de renda com recall
no top 10% pior que A0 com IC95 abaixo de zero. Se A3 não sair pior que A0,
as métricas não separam bom de ruim e o resultado é inconclusivo.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

from occasion.modelagem import treino as t
from occasion.modelagem.experimentos import FEATURES_FINAIS, _tabela_md, classe
from occasion.paths import PROCESSED_DIR

CATS = [c for c in FEATURES_FINAIS if c in t.CATEGORICAS]
NUM = "renda_pct"
ALVO = t.ALVO

SEMENTES = (0, 1, 2, 3, 4)
VOLUMES = (1, 3, 10)
LAMBDAS = (0.1, 0.25, 0.5, 1.0)
FRACAO_HOLDOUT = 0.2
N_BOOTSTRAP = 1000
N_FIDELIDADE = 5000


# ------------------------------------------------------------- codificação


@dataclass
class Codificador:
    """One-hot das categóricas + renda padronizada, nas categorias do real."""

    categorias: dict[str, pd.Index]
    media_renda: float
    dp_renda: float

    @classmethod
    def ajustar(cls, df: pd.DataFrame) -> Codificador:
        return cls(
            {c: df[c].cat.categories for c in CATS},
            float(df[NUM].mean()),
            float(df[NUM].std()),
        )

    @property
    def tamanhos(self) -> list[int]:
        return [len(self.categorias[c]) for c in CATS]

    def codigos(self, df: pd.DataFrame) -> np.ndarray:
        """Índice da categoria de cada coluna (n × n_categóricas)."""
        return np.column_stack(
            [pd.Categorical(df[c], categories=self.categorias[c]).codes for c in CATS]
        )

    def renda_z(self, df: pd.DataFrame) -> np.ndarray:
        return ((df[NUM].to_numpy(float) - self.media_renda) / self.dp_renda).astype("float32")

    def one_hot(self, df: pd.DataFrame) -> np.ndarray:
        cod = self.codigos(df)
        blocos = [np.eye(k, dtype="float32")[cod[:, j]] for j, k in enumerate(self.tamanhos)]
        return np.hstack(blocos)

    def matriz(self, df: pd.DataFrame) -> np.ndarray:
        """One-hot + renda padronizada: o espaço da PCA e da distância de Fréchet."""
        return np.hstack([self.one_hot(df), self.renda_z(df)[:, None]])

    def montar(self, codigos: np.ndarray, renda_z: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """DataFrame com as features finais a partir de códigos e renda padronizada."""
        df = pd.DataFrame(
            {
                c: pd.Categorical.from_codes(codigos[:, j], categories=self.categorias[c])
                for j, c in enumerate(CATS)
            }
        )
        df[NUM] = np.clip(renda_z * self.dp_renda + self.media_renda, 0.0, 100.0)
        df[ALVO] = np.asarray(y).astype(bool)
        return df


def reamostra_ponderada(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sorteio com reposição proporcional a `peso_final` (imita a população)."""
    p = df["peso_final"].to_numpy(float)
    idx = rng.choice(len(df), size=n, replace=True, p=p / p.sum())
    return df.iloc[idx].reset_index(drop=True)


def holdout_setores(df: pd.DataFrame, fracao: float = FRACAO_HOLDOUT, semente: int = 3):
    """Máscara booleana: True nas linhas dos setores separados para fidelidade."""
    setores = df[t.GRUPO].unique()
    rng = np.random.default_rng(semente)
    fora = set(rng.choice(setores, size=round(len(setores) * fracao), replace=False))
    return df[t.GRUPO].isin(fora).to_numpy()


# ---------------------------------------------------------------- geradores


class Duplicacao:
    """A1: cópias de linhas reais sorteadas por peso (controle trivial)."""

    nome = "A1 duplicação"

    def ajustar(self, real: pd.DataFrame, cod: Codificador, semente: int) -> Duplicacao:
        self.real, self.cod, self.rng = real, cod, np.random.default_rng(semente)
        return self

    def gerar(self, n: int, y: int) -> pd.DataFrame:
        base = self.real[self.real[ALVO] == bool(y)]
        amostra = reamostra_ponderada(base, n, self.rng)
        return self.cod.montar(self.cod.codigos(amostra), self.cod.renda_z(amostra), [y] * n)


class SmoteNC:
    """A2: SMOTE-NC sobre a reamostra ponderada dos positivos (só gera positivos)."""

    nome = "A2 SMOTE-NC"
    k_vizinhos = 5

    def ajustar(self, real: pd.DataFrame, cod: Codificador, semente: int) -> SmoteNC:
        self.real, self.cod, self.semente = real, cod, semente
        self.rng = np.random.default_rng(semente)
        return self

    def gerar(self, n: int, y: int) -> pd.DataFrame:
        if y != 1:
            raise ValueError("SMOTE-NC só gera a classe minoritária (positivos)")
        from imblearn.over_sampling import SMOTENC

        pos = self.real[self.real[ALVO]]
        base_pos = reamostra_ponderada(pos, len(pos), self.rng)
        # Negativos só cumprem a API (o SMOTE busca vizinhos entre positivos).
        base_neg = reamostra_ponderada(self.real[~self.real[ALVO]], len(pos) + n + 1, self.rng)
        base = pd.concat([base_pos, base_neg], ignore_index=True)
        X = np.column_stack([self.cod.codigos(base), self.cod.renda_z(base)])
        yb = base[ALVO].astype(int).to_numpy()
        sm = SMOTENC(
            categorical_features=list(range(len(CATS))),
            sampling_strategy={1: len(base_pos) + n},
            k_neighbors=self.k_vizinhos,
            random_state=self.semente,
        )
        Xr, yr = sm.fit_resample(X, yb)
        novos = Xr[len(X) :]
        assert (yr[len(X) :] == 1).all() and len(novos) == n
        return self.cod.montar(novos[:, : len(CATS)].astype(int), novos[:, -1], [1] * n)


class Marginais:
    """A3: cada variável sorteada sozinha dentro da classe (sem correlação entre elas).

    Controle negativo: preserva as marginais de cada classe e destrói as
    interações (dia × turno, região × idade...). Deve sair pior que A0.
    """

    nome = "A3 marginais"

    def ajustar(self, real: pd.DataFrame, cod: Codificador, semente: int) -> Marginais:
        self.real, self.cod, self.rng = real, cod, np.random.default_rng(semente)
        return self

    def gerar(self, n: int, y: int) -> pd.DataFrame:
        base = self.real[self.real[ALVO] == bool(y)]
        cod, rz = self.cod.codigos(base), self.cod.renda_z(base)
        p = base["peso_final"].to_numpy(float)
        p = p / p.sum()
        cols = [cod[self.rng.choice(len(base), size=n, p=p), j] for j in range(len(CATS))]
        renda = rz[self.rng.choice(len(base), size=n, p=p)]
        return self.cod.montar(np.column_stack(cols), renda, [y] * n)


class RotuloAleatorio:
    """A3b: "positivos" sorteados de todas as linhas reais, sem olhar o alvo.

    Controle negativo extra, acrescentado depois do teste curto do código: a
    logística do modelo é aditiva (sem interações), então A3 (marginais certas
    por classe, sem correlação) quase não a prejudica e não serve sozinho para
    provar que as métricas detectam dado ruim. Aqui o rótulo não tem relação
    com as features, o que deve piorar o modelo.
    """

    nome = "A3b rótulo aleatório"

    def ajustar(self, real: pd.DataFrame, cod: Codificador, semente: int) -> RotuloAleatorio:
        self.real, self.cod, self.rng = real, cod, np.random.default_rng(semente)
        return self

    def gerar(self, n: int, y: int) -> pd.DataFrame:
        amostra = reamostra_ponderada(self.real, n, self.rng)
        return self.cod.montar(self.cod.codigos(amostra), self.cod.renda_z(amostra), [y] * n)


class VAE:
    """A4: VAE tabular condicional ao alvo (JAX), treinado na reamostra ponderada.

    Entrada: one-hot das 6 categóricas + renda padronizada + alvo. O decodificador
    recebe o latente e o alvo e devolve uma softmax por categórica e uma normal
    para a renda. Positivos entram reamostrados (`mult_pos` vezes o número real)
    para o decodificador ver a classe rara o bastante.
    """

    nome = "A4 VAE"

    def __init__(
        self,
        latente: int = 8,
        oculta: int = 64,
        epocas: int = 30,
        lote: int = 256,
        taxa: float = 1e-3,
        beta: float = 1.0,
        n_negativos: int = 40_000,
        mult_pos: int = 10,
    ):
        self.latente, self.oculta, self.epocas = latente, oculta, epocas
        self.lote, self.taxa, self.beta = lote, taxa, beta
        self.n_negativos, self.mult_pos = n_negativos, mult_pos

    # -- rede
    @staticmethod
    def _camadas(chave, dims):
        import jax

        params = []
        for a, b in pairwise(dims):
            chave, k = jax.random.split(chave)
            params.append((jax.random.normal(k, (a, b)) * np.sqrt(2.0 / a), np.zeros(b)))
        return params

    @staticmethod
    def _mlp(params, x):
        import jax.numpy as jnp

        for W, b in params[:-1]:
            x = jnp.maximum(x @ W + b, 0.0)
        W, b = params[-1]
        return x @ W + b

    def ajustar(self, real: pd.DataFrame, cod: Codificador, semente: int) -> VAE:
        import jax
        import jax.numpy as jnp
        import optax

        self.cod = cod
        rng = np.random.default_rng(semente)
        pos = real[real[ALVO]]
        neg = real[~real[ALVO]]
        base = pd.concat(
            [
                reamostra_ponderada(pos, len(pos) * self.mult_pos, rng),
                reamostra_ponderada(neg, self.n_negativos, rng),
            ],
            ignore_index=True,
        )
        X = jnp.asarray(cod.one_hot(base))
        R = jnp.asarray(cod.renda_z(base))
        C = jnp.asarray(base[ALVO].astype("float32").to_numpy())
        d_cat = X.shape[1]
        limites = np.cumsum([0, *cod.tamanhos])
        L, H = self.latente, self.oculta

        chave = jax.random.PRNGKey(semente)
        k1, k2, chave = jax.random.split(chave, 3)
        params = {
            "enc": self._camadas(k1, [d_cat + 2, H, H, 2 * L]),
            "dec": self._camadas(k2, [L + 1, H, H, d_cat + 2]),
        }
        mlp, beta = self._mlp, self.beta

        def perda(params, x, r, c, k):
            h = mlp(params["enc"], jnp.concatenate([x, r[:, None], c[:, None]], axis=1))
            mu, logvar = h[:, :L], h[:, L:]
            z = mu + jnp.exp(0.5 * logvar) * jax.random.normal(k, mu.shape)
            out = mlp(params["dec"], jnp.concatenate([z, c[:, None]], axis=1))
            ce = 0.0
            for a, b in pairwise(limites):
                ce = ce - jnp.sum(x[:, a:b] * jax.nn.log_softmax(out[:, a:b]), axis=1)
            m, log_s = out[:, d_cat], out[:, d_cat + 1]
            nll = 0.5 * ((r - m) / jnp.exp(log_s)) ** 2 + log_s
            kl = -0.5 * jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar), axis=1)
            return jnp.mean(ce + nll + beta * kl)

        otim = optax.adam(self.taxa)
        estado = otim.init(params)

        @jax.jit
        def passo(params, estado, x, r, c, k):
            valor, grad = jax.value_and_grad(perda)(params, x, r, c, k)
            upd, estado = otim.update(grad, estado, params)
            return optax.apply_updates(params, upd), estado, valor

        n = X.shape[0]
        self.historico = []
        for _ in range(self.epocas):
            ordem = rng.permutation(n)
            soma = 0.0
            for i in range(0, n - self.lote + 1, self.lote):
                j = ordem[i : i + self.lote]
                chave, k = jax.random.split(chave)
                params, estado, v = passo(params, estado, X[j], R[j], C[j], k)
                soma += float(v)
            self.historico.append(soma / (n // self.lote))
        self.params, self.limites, self.d_cat = params, limites, d_cat
        self._chave = jax.random.PRNGKey(semente + 10_000)
        return self

    def gerar(self, n: int, y: int) -> pd.DataFrame:
        import jax
        import jax.numpy as jnp

        self._chave, kz, kc, kr = jax.random.split(self._chave, 4)
        z = jax.random.normal(kz, (n, self.latente))
        c = jnp.full((n, 1), float(y))
        out = self._mlp(self.params["dec"], jnp.concatenate([z, c], axis=1))
        cods = []
        for a, b in pairwise(self.limites):
            kc, k = jax.random.split(kc)
            cods.append(np.asarray(jax.random.categorical(k, out[:, a:b], axis=1)))
        m, log_s = out[:, self.d_cat], out[:, self.d_cat + 1]
        renda = np.asarray(m + jnp.exp(log_s) * jax.random.normal(kr, (n,)))
        return self.cod.montar(np.column_stack(cods), renda, [y] * n)


GERADORES = {g.nome: g for g in (Duplicacao, SmoteNC, Marginais, RotuloAleatorio, VAE)}


def base_completa(gerador, n: int, taxa: float, rng: np.random.Generator) -> pd.DataFrame:
    """Base sintética inteira de `n` linhas com a taxa de positivos `taxa`."""
    n_pos = int(rng.binomial(n, taxa))
    return pd.concat([gerador.gerar(n_pos, 1), gerador.gerar(n - n_pos, 0)], ignore_index=True)


# ------------------------------------------------------- treino com mistura


def treinar_mistura(real: pd.DataFrame, sintetico: pd.DataFrame | None, lam: float):
    """Logística das features finais em real + sintético (peso `lam` no sintético).

    Linhas reais pesam `peso_final / média`, como no modelo servido; cada linha
    sintética pesa `lam`. Com `sintetico=None` é o A0. Com `real=None` é o
    treino só no sintético (TSTR).
    """
    partes, pesos = [], []
    if real is not None:
        partes.append(real[FEATURES_FINAIS + [ALVO]])
        pesos.append((real["peso_final"] / real["peso_final"].mean()).to_numpy())
    if sintetico is not None and len(sintetico):
        partes.append(sintetico[FEATURES_FINAIS + [ALVO]])
        pesos.append(np.full(len(sintetico), lam))
    df = pd.concat([p.astype({c: "string" for c in CATS}) for p in partes], ignore_index=True)
    ref = real if real is not None else sintetico
    for c in CATS:
        df[c] = pd.Categorical(df[c], categories=ref[c].cat.categories)
    m = t.Modelo("logistica", FEATURES_FINAIS)
    m._cats = {c: df[c].cat.categories for c in CATS}
    m.est = t._logistica(FEATURES_FINAIS).fit(
        df[FEATURES_FINAIS],
        df[ALVO].astype(int),
        logisticregression__sample_weight=np.concatenate(pesos),
    )
    return m


# ---------------------------------------------------------------- fidelidade


def frechet(a: np.ndarray, b: np.ndarray) -> float:
    """Distância de Fréchet entre gaussianas ajustadas em `a` e `b` (a conta do FID)."""
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    raiz = linalg.sqrtm(s1 @ s2)
    raiz = np.real(raiz)
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * raiz))


def variacao_total(a: pd.Series, b: pd.Series) -> float:
    pa = a.value_counts(normalize=True)
    pb = b.value_counts(normalize=True)
    return float(0.5 * pa.subtract(pb, fill_value=0).abs().sum())


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import wasserstein_distance

    return float(wasserstein_distance(a, b))


def _chaves(df: pd.DataFrame, cod: Codificador) -> pd.Series:
    """Linha como texto: categóricas + renda arredondada a 1 ponto de percentil."""
    cods = cod.codigos(df)
    renda = np.round(df[NUM].to_numpy(float)).astype(int)
    return pd.Series(["|".join(map(str, r)) for r in np.column_stack([cods, renda])])


def _combos(df: pd.DataFrame, cod: Codificador) -> set:
    return {tuple(r) for r in cod.codigos(df)}


def fidelidade(
    amostra: pd.DataFrame,
    holdout: pd.DataFrame,
    treino: pd.DataFrame,
    cod: Codificador,
    pca: PCA,
    combos_real: set,
) -> dict[str, float]:
    """Compara uma amostra (sintética ou real) com o holdout real da mesma classe."""
    za, zh = pca.transform(cod.matriz(amostra)), pca.transform(cod.matriz(holdout))
    tv = {c: variacao_total(amostra[c], holdout[c]) for c in CATS}
    par_dt = variacao_total(
        amostra["dia_semana"].astype(str) + "×" + amostra["turno"].astype(str),
        holdout["dia_semana"].astype(str) + "×" + holdout["turno"].astype(str),
    )
    par_ri = variacao_total(
        amostra["regiao"].astype(str) + "×" + amostra["faixa_idade"].astype(str),
        holdout["regiao"].astype(str) + "×" + holdout["faixa_idade"].astype(str),
    )
    return {
        "frechet_pca": frechet(za, zh),
        "tv_media": float(np.mean(list(tv.values()))),
        "tv_max": float(np.max(list(tv.values()))),
        "w1_renda": wasserstein_1d(amostra[NUM].to_numpy(float), holdout[NUM].to_numpy(float)),
        "tv_dia_x_turno": par_dt,
        "tv_regiao_x_idade": par_ri,
        "copias_pct": _pct_copias(amostra, treino, cod),
        "fora_do_suporte_pct": _pct_fora(amostra, combos_real, cod),
    }


def _pct_copias(amostra: pd.DataFrame, treino: pd.DataFrame, cod: Codificador) -> float:
    return 100 * float(_chaves(amostra, cod).isin(set(_chaves(treino, cod))).mean())


def _pct_fora(amostra: pd.DataFrame, combos: set, cod: Codificador) -> float:
    return 100 * float(np.mean([tuple(r) not in combos for r in cod.codigos(amostra)]))


# --------------------------------------------------------------- utilidade


def limiar_balanced(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    """Limiar que maximiza a balanced accuracy ponderada com a regra `p >= limiar`."""
    pos = pd.Series(y * w).groupby(p).sum().sort_index(ascending=False)
    neg = pd.Series((1 - y) * w).groupby(p).sum().sort_index(ascending=False)
    tpr = pos.cumsum().to_numpy() / pos.sum()
    tnr = 1 - neg.cumsum().to_numpy() / neg.sum()
    return float(pos.index[np.argmax(0.5 * (tpr + tnr))])


def balanced(y: np.ndarray, p: np.ndarray, w: np.ndarray, limiar: float) -> dict[str, float]:
    pred = p >= limiar
    tp = (w * pred * (y == 1)).sum()
    fp = (w * pred * (y == 0)).sum()
    tpr = tp / (w * (y == 1)).sum()
    tnr = (w * ~pred * (y == 0)).sum() / (w * (y == 0)).sum()
    return {
        "balanced_acc": float(0.5 * (tpr + tnr)),
        "recall_limiar": float(tpr),
        "precisao_limiar": float(tp / (tp + fp)) if tp + fp else 0.0,
        "marcados_pct": float(100 * (w * pred).sum() / w.sum()),
    }


def platt(p_ajuste, y_ajuste, w_ajuste, p_aplicar) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    cal = LogisticRegression(C=1e6).fit(
        t._logit(p_ajuste).reshape(-1, 1), y_ajuste, sample_weight=w_ajuste
    )
    return cal.predict_proba(t._logit(p_aplicar).reshape(-1, 1))[:, 1]


def recall_topo_grupos(y, p, w, grupos: dict[str, np.ndarray]) -> dict[str, float]:
    """Recall no top 10% nacional, dentro de cada grupo (mesmo corte para todos)."""
    topo = w * t.fracao_topo(p, w)
    return {g: float((y * topo)[m].sum() / (y * w)[m].sum()) for g, m in grupos.items()}


def _replica(y, w, linhas, previsoes, previsoes_cal, grupos, semente):
    """Uma réplica do bootstrap de setores: PR-AUC, log loss e recall por grupo."""
    rng = np.random.default_rng(semente)
    idx = np.concatenate([linhas[s] for s in rng.integers(0, len(linhas), len(linhas))])
    yi, wi = y[idx], w[idx]
    gi = {g: m[idx] for g, m in grupos.items()}
    pr = {
        k: np.mean([average_precision_score(yi, p[idx], sample_weight=wi) for p in ps])
        for k, ps in previsoes.items()
    }
    ll = {
        k: np.mean([log_loss(yi, p[idx], sample_weight=wi, labels=[0, 1]) for p in ps])
        for k, ps in previsoes_cal.items()
    }
    rec = {}
    for k, ps in previsoes.items():
        por = [recall_topo_grupos(yi, p[idx], wi, gi) for p in ps]
        for g in grupos:
            rec[f"{k}|{g}"] = np.mean([r[g] for r in por])
    return pr, ll, rec


def bootstrap(
    df: pd.DataFrame,
    previsoes: dict[str, list[np.ndarray]],
    previsoes_cal: dict[str, list[np.ndarray]],
    grupos: dict[str, np.ndarray],
    n: int = N_BOOTSTRAP,
    semente: int = 11,
    n_jobs: int = -1,
) -> dict[str, pd.DataFrame]:
    """Bootstrap pareado de setores. Cada braço = média das sementes em cada réplica.

    Todas as previsões usam a mesma reamostragem em cada réplica (comparação
    pareada). Devolve, por métrica, uma tabela réplica × braço (braço|grupo no
    recall).
    """
    from joblib import Parallel, delayed

    y = df[ALVO].astype(int).to_numpy()
    w = df["peso_metrica"].to_numpy()
    _, setor = np.unique(df[t.GRUPO].to_numpy(), return_inverse=True)
    linhas = pd.Series(np.arange(len(df))).groupby(setor).apply(np.array).to_list()
    sementes = np.random.default_rng(semente).integers(0, 2**32, n)
    res = Parallel(n_jobs=n_jobs)(
        delayed(_replica)(y, w, linhas, previsoes, previsoes_cal, grupos, s) for s in sementes
    )
    return {
        "pr_auc": pd.DataFrame([r[0] for r in res]),
        "log_loss": pd.DataFrame([r[1] for r in res]),
        "recall": pd.DataFrame([r[2] for r in res]),
    }


def _ic(serie: pd.Series) -> tuple[float, float, float]:
    return float(serie.mean()), float(serie.quantile(0.025)), float(serie.quantile(0.975))


def _fmt(trio, casas: int = 4) -> str:
    m, lo, hi = trio
    return f"{m:+.{casas}f} [{lo:+.{casas}f}; {hi:+.{casas}f}]"


# ------------------------------------------------------------------- rodada


def rodar(
    saida: Path = PROCESSED_DIR,
    sementes=SEMENTES,
    volumes=VOLUMES,
    lambdas=LAMBDAS,
    n_bootstrap: int = N_BOOTSTRAP,
    vae_kwargs: dict | None = None,
) -> Path:
    tr = t.preparar(t.carregar("2008_2009")).reset_index(drop=True)
    te = t.preparar(t.carregar("2017_2018")).reset_index(drop=True)
    cod = Codificador.ajustar(tr)
    fora = holdout_setores(tr)
    gen_tr, hold = tr[~fora].reset_index(drop=True), tr[fora].reset_index(drop=True)
    n_pos_real = int(tr[ALVO].sum())
    taxa_real = float(np.average(tr[ALVO], weights=tr["peso_final"]))

    y = te[ALVO].astype(int).to_numpy()
    w = te["peso_metrica"].to_numpy()
    sel = t.metades_setor(te) == 0
    ava = ~sel
    te_ava = te[ava].reset_index(drop=True)
    classes = classe(te["renda_sm"]).astype(str)
    grupos = {f"região {r}": (te["regiao"] == r).to_numpy() for r in te["regiao"].cat.categories}
    grupos |= {f"classe {c}": (classes == c).to_numpy() for c in sorted(classes.unique())}
    grupos_ava = {g: m[ava] for g, m in grupos.items()}

    # PCA ajustada só no real (reamostra ponderada do treino do gerador).
    rng0 = np.random.default_rng(0)
    pca = PCA(n_components=10, random_state=0).fit(
        cod.matriz(reamostra_ponderada(gen_tr, 50_000, rng0))
    )
    combos_real = _combos(tr, cod)
    hold_cls = {
        k: reamostra_ponderada(hold[hold[ALVO] == bool(k)], N_FIDELIDADE, rng0) for k in (0, 1)
    }
    tr_cls = {
        k: reamostra_ponderada(gen_tr[gen_tr[ALVO] == bool(k)], N_FIDELIDADE, rng0) for k in (0, 1)
    }

    # ---------- A0
    p0 = treinar_mistura(tr, None, 0.0).predict(te)
    configs = []  # (braço, modo, volume, lam, semente, previsão)
    fid = []
    fid.append(
        {
            "braço": "referência: real (outros setores)",
            "classe": "positivos",
            **fidelidade(tr_cls[1], hold_cls[1], gen_tr, cod, pca, combos_real),
            "copias_pct": _pct_copias(hold_cls[1], gen_tr, cod),
            "fora_do_suporte_pct": _pct_fora(hold_cls[1], _combos(gen_tr, cod), cod),
        }
    )
    fid.append(
        {
            "braço": "referência: real (outros setores)",
            "classe": "negativos",
            **fidelidade(tr_cls[0], hold_cls[0], gen_tr, cod, pca, combos_real),
            "copias_pct": _pct_copias(hold_cls[0], gen_tr, cod),
            "fora_do_suporte_pct": _pct_fora(hold_cls[0], _combos(gen_tr, cod), cod),
        }
    )
    pca_pontos = {"real holdout": pca.transform(cod.matriz(hold_cls[1]))[:, :2]}
    historicos_vae = []

    for nome, Gerador in GERADORES.items():
        for s in sementes:
            ger = Gerador(**(vae_kwargs or {})) if Gerador is VAE else Gerador()
            ger.ajustar(gen_tr, cod, s)
            if Gerador is VAE:
                historicos_vae.append(ger.historico)
            pos = ger.gerar(n_pos_real * max(volumes), 1)
            if s == sementes[0]:
                amostra_pos = (
                    pos.iloc[:N_FIDELIDADE]
                    if len(pos) >= N_FIDELIDADE
                    else ger.gerar(N_FIDELIDADE, 1)
                )
                fid.append(
                    {
                        "braço": nome,
                        "classe": "positivos",
                        **fidelidade(amostra_pos, hold_cls[1], gen_tr, cod, pca, combos_real),
                    }
                )
                pca_pontos[nome] = pca.transform(cod.matriz(amostra_pos))[:, :2]
                if nome != SmoteNC.nome:
                    fid.append(
                        {
                            "braço": nome,
                            "classe": "negativos",
                            **fidelidade(
                                ger.gerar(N_FIDELIDADE, 0),
                                hold_cls[0],
                                gen_tr,
                                cod,
                                pca,
                                combos_real,
                            ),
                        }
                    )
            for v in volumes:
                sint = pos.iloc[: n_pos_real * v]
                for lam in lambdas:
                    p = treinar_mistura(tr, sint, lam).predict(te)
                    configs.append((nome, "positivos", v, lam, s, p))
            if Gerador in (Marginais, VAE):
                rng = np.random.default_rng(100 + s)
                completa = base_completa(ger, len(tr), taxa_real, rng)
                for lam in lambdas:
                    p = treinar_mistura(tr, completa, lam).predict(te)
                    configs.append((nome, "completo", 1, lam, s, p))
                p = treinar_mistura(None, completa, 1.0).predict(te)
                configs.append((nome, "TSTR (só sintético)", 1, 1.0, s, p))
            print(f"{nome} semente {s}: ok", flush=True)

    # ---------- escolha na metade de seleção
    grade = pd.DataFrame(
        [
            {
                "braço": b,
                "modo": m,
                "volume": v,
                "lam": lam,
                "semente": s,
                "pr_auc_selecao": average_precision_score(y[sel], p[sel], sample_weight=w[sel]),
            }
            for b, m, v, lam, s, p in configs
        ]
    )
    media = grade.groupby(["braço", "modo", "volume", "lam"], as_index=False)["pr_auc_selecao"].agg(
        ["mean", "std"]
    )
    pr0_sel = average_precision_score(y[sel], p0[sel], sample_weight=w[sel])
    escolhas = {}
    for (b, m), g in media.groupby(["braço", "modo"]):
        melhor = g.loc[g["mean"].idxmax()]
        escolhas[f"{b} · {m}"] = (b, m, melhor["volume"], melhor["lam"])

    def prevs(b, m, v, lam):
        return [p for bb, mm, vv, ll, _, p in configs if (bb, mm, vv, ll) == (b, m, v, lam)]

    braços = {"A0 real": [p0]} | {k: prevs(*cfg) for k, cfg in escolhas.items()}

    # ---------- métricas na avaliação
    linhas = []
    cal = {}
    for k, ps in braços.items():
        por_semente = []
        cal[k] = []
        for p in ps:
            lim = limiar_balanced(y[sel], p[sel], w[sel])
            pc = platt(p[sel], y[sel], w[sel], p[ava])
            cal[k].append(pc)
            m = t.metricas(y[ava], p[ava], w[ava])
            por_semente.append(
                {
                    **m,
                    **balanced(y[ava], p[ava], w[ava], lim),
                    "media_prevista_bruta": float(np.average(p[ava], weights=w[ava])),
                    "log_loss_bruto": log_loss(y[ava], p[ava], sample_weight=w[ava]),
                    "brier_platt": brier_score_loss(y[ava], pc, sample_weight=w[ava]),
                    "log_loss_platt": log_loss(y[ava], pc, sample_weight=w[ava]),
                    "media_prevista_platt": float(np.average(pc, weights=w[ava])),
                }
            )
        tab = pd.DataFrame(por_semente)
        linha = {"braço": k, **tab.mean().to_dict(), "pr_auc_dp_sementes": tab["pr_auc"].std()}
        linhas.append(linha)
    util = pd.DataFrame(linhas).set_index("braço")
    util["media_observada"] = float(np.average(y[ava], weights=w[ava]))

    bs = bootstrap(
        te_ava,
        {k: [p[ava] for p in ps] for k, ps in braços.items()},
        cal,
        grupos_ava,
        n=n_bootstrap,
    )
    dif = []
    for k in braços:
        if k == "A0 real":
            continue
        d_pr = _ic(bs["pr_auc"][k] - bs["pr_auc"]["A0 real"])
        a1 = next((kk for kk in braços if kk.startswith("A1") and "positivos" in kk), None)
        d_a1 = _ic(bs["pr_auc"][k] - bs["pr_auc"][a1]) if a1 and k != a1 else (np.nan,) * 3
        d_ll = _ic(bs["log_loss"][k] - bs["log_loss"]["A0 real"])
        piores = [
            g for g in grupos if _ic(bs["recall"][f"{k}|{g}"] - bs["recall"][f"A0 real|{g}"])[2] < 0
        ]
        regra = {
            "1_pr_auc_vs_A0": d_pr[1] > 0,
            "2_pr_auc_vs_A1": (d_a1[1] > 0) if a1 and k != a1 else None,
            "3_log_loss_platt": d_ll[1] <= 0,
            "4_nenhum_grupo_pior": not piores,
        }
        dif.append(
            {
                "braço": k,
                "Δ PR-AUC vs A0 [IC95]": _fmt(d_pr),
                "Δ PR-AUC vs A1 [IC95]": _fmt(d_a1) if not np.isnan(d_a1[0]) else "",
                "Δ log loss (Platt) vs A0 [IC95]": _fmt(d_ll, 5),
                "grupos com recall pior": ", ".join(piores) or "nenhum",
                "passa (1/2/3/4)": "/".join(
                    "n/a" if v is None else "sim" if v else "não" for v in regra.values()
                ),
                "útil": all(v is True for v in regra.values()),
                "_pr_auc_hi": d_pr[2],
            }
        )
    dif = pd.DataFrame(dif).set_index("braço")

    rec_grupos = pd.DataFrame(
        {
            k: {
                g: _fmt(_ic(bs["recall"][f"{k}|{g}"] - bs["recall"][f"A0 real|{g}"]), 3)
                for g in grupos
            }
            for k in braços
            if k != "A0 real"
        }
    )
    rec_a0 = pd.Series({g: bs["recall"][f"A0 real|{g}"].mean() for g in grupos}, name="recall A0")

    # ---------- figura da PCA
    figura = _figura_pca(pca_pontos, saida / "sintetico_pca.png")

    fid = pd.DataFrame(fid).set_index(["braço", "classe"])
    escolhas_tab = pd.DataFrame(
        [
            {"braço": k, "volume (× positivos reais)": v, "lam": lam}
            for k, (_, _, v, lam) in escolhas.items()
        ]
    ).set_index("braço")

    rel = _relatorio(
        tr,
        te,
        gen_tr,
        hold,
        n_pos_real,
        pr0_sel,
        media,
        escolhas_tab,
        util,
        dif,
        fid,
        rec_grupos,
        rec_a0,
        figura,
        historicos_vae,
        sementes,
        n_bootstrap,
        vae_kwargs,
    )
    destino = saida / "dados_sinteticos.md"
    destino.write_text(rel)
    grade.to_csv(saida / "dados_sinteticos_grade.csv", index=False)
    return destino


def _figura_pca(pontos: dict[str, np.ndarray], destino: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    real = pontos["real holdout"]
    nomes = [k for k in pontos if k != "real holdout"]
    fig, eixos = plt.subplots(1, len(nomes), figsize=(4 * len(nomes), 4), sharex=True, sharey=True)
    for ax, nome in zip(np.atleast_1d(eixos), nomes, strict=True):
        s = pontos[nome]
        ax.scatter(s[:, 0], s[:, 1], s=4, alpha=0.15, color="#c2410c", label="sintético")
        ax.scatter(real[:, 0], real[:, 1], s=4, alpha=0.15, color="#1f2937", label="real (holdout)")
        ax.set_title(nome)
        ax.set_xlabel("PC1")
    np.atleast_1d(eixos)[0].set_ylabel("PC2")
    np.atleast_1d(eixos)[0].legend(loc="upper right", markerscale=3)
    fig.suptitle("Positivos: real (holdout 2008-09) × sintético, PCA ajustada no real")
    fig.tight_layout()
    fig.savefig(destino, dpi=120)
    plt.close(fig)
    return destino


def _n(x: int) -> str:
    """Inteiro com ponto de milhar."""
    return f"{x:,}".replace(",", ".")


def _md(df: pd.DataFrame, casas: int = 4) -> str:
    return _tabela_md(df, casas)


def _relatorio(
    tr,
    te,
    gen_tr,
    hold,
    n_pos_real,
    pr0_sel,
    media,
    escolhas,
    util,
    dif,
    fid,
    rec_grupos,
    rec_a0,
    figura,
    historicos_vae,
    sementes,
    n_bootstrap,
    vae_kwargs,
) -> str:
    uteis = dif.index[dif["útil"]].tolist()
    controles = {
        k: bool(dif.loc[k, "_pr_auc_hi"] < 0)
        for k in dif.index
        if k.startswith("A3") and "positivos" in k
    }
    a3_pior = any(controles.values())
    conclusao = (
        "Nenhum braço sintético passou na regra de decisão. "
        if not uteis
        else f"Braços que passaram na regra: {', '.join(uteis)}. "
    )
    conclusao += (
        "Controles negativos (IC95 da PR-AUC inteiro abaixo de A0): "
        + ", ".join(f"{k} {'pior' if v else 'não pior'}" for k, v in controles.items())
        + ". "
        + (
            "Pelo menos um controle ruim foi detectado, então as métricas separam dado bom de ruim."
            if a3_pior
            else "Nenhum controle ruim foi detectado: pela regra, o experimento é inconclusivo."
        )
    )
    vae_final = [h[-1] for h in historicos_vae]
    partes = [
        "# Dados sintéticos: experimento controlado\n",
        (
            "Pergunta: somar linhas sintéticas ao treino melhora o Occasion Score fora do tempo, "
            "medido só em dado real da POF 2017-18, sem distorcer a probabilidade? O modelo servido "
            "não muda com este experimento.\n"
        ),
        f"**Resultado.** {conclusao}\n",
        "## Dados\n",
        f"- Treino real: POF 2008-09, {_n(len(tr))} linhas, {n_pos_real} positivos.",
        (
            f"- Geradores aprendem em {_n(gen_tr[t.GRUPO].nunique())} setores de 2008-09 "
            f"({int(gen_tr[ALVO].sum())} positivos); holdout de fidelidade: "
            f"{_n(hold[t.GRUPO].nunique())} setores ({int(hold[ALVO].sum())} positivos)."
        ),
        (
            f"- Teste: POF 2017-18, {_n(len(te))} linhas, {int(te[ALVO].sum())} positivos, em duas "
            "metades fixas de setores (seleção e avaliação)."
        ),
        f"- Sementes por gerador: {len(sementes)}. Bootstrap pareado de setores: {n_bootstrap} réplicas.",
        f"- VAE: {vae_kwargs or 'parâmetros padrão'}; perda final por semente: "
        + ", ".join(f"{v:.3f}" for v in vae_final)
        + ".\n",
        "## Fidelidade (sintético × holdout real de 2008-09)\n",
        (
            "Cada linha compara 5.000 linhas com o holdout real da mesma classe (reamostras por "
            "`peso_final`). A linha de referência compara o real dos setores de treino com o holdout: "
            "é o quanto o próprio real varia entre setores, o piso esperado. Fréchet = distância de "
            "Fréchet entre gaussianas nos 10 primeiros componentes da PCA ajustada no real. TV = "
            "variação total (0 a 1). Cópias = linhas iguais a uma linha do treino (categóricas + renda "
            "arredondada a 1 ponto). Fora do suporte = combinação das 6 categóricas que não existe no "
            "real de 2008-09.\n"
        ),
        _md(fid, 3),
        "",
    ]
    if figura:
        partes += [f"![PCA dos positivos]({figura.name})\n"]
    partes += [
        "## Escolha na metade de seleção\n",
        (
            f"PR-AUC de A0 na seleção: {pr0_sel:.4f}. Configuração escolhida por braço (maior PR-AUC "
            "média das sementes):\n"
        ),
        _md(escolhas, 2),
        "\nGrade completa (média e desvio entre sementes):\n",
        _md(media.set_index(["braço", "modo", "volume", "lam"]), 4),
        "\n## Utilidade na metade de avaliação (dado real 2017-18)\n",
        (
            "Médias das sementes. Balanced accuracy com limiar escolhido na seleção. Calibração bruta "
            "e depois de Platt ajustado na seleção.\n"
        ),
        _md(
            util[
                [
                    "pr_auc",
                    "pr_auc_dp_sementes",
                    "roc_auc",
                    "recall_top10",
                    "lift_top10",
                    "balanced_acc",
                    "precisao_limiar",
                    "marcados_pct",
                ]
            ],
            4,
        ),
        "",
        _md(
            util[
                [
                    "media_observada",
                    "media_prevista_bruta",
                    "log_loss_bruto",
                    "media_prevista_platt",
                    "log_loss_platt",
                    "brier_platt",
                ]
            ],
            5,
        ),
        "\n## Regra de decisão\n",
        (
            "(1) PR-AUC > A0 com IC95 acima de zero; (2) PR-AUC > A1 (duplicação) com IC95 acima de "
            "zero; (3) log loss depois de Platt não pior que A0; (4) nenhuma região ou classe de renda "
            "com recall no top 10% pior que A0 com IC95 abaixo de zero.\n"
        ),
        _md(dif.drop(columns="_pr_auc_hi")),
        "\n## Recall no top 10% por grupo\n",
        "Recall de A0 e diferença de cada braço contra A0 [IC95], mesmo corte nacional.\n",
        _md(rec_a0.to_frame(), 3),
        "",
        _md(rec_grupos),
        "",
    ]
    return "\n".join(partes) + "\n"


def main() -> None:
    print(rodar())


if __name__ == "__main__":
    main()
