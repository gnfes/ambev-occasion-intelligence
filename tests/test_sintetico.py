"""Testes do experimento de dados sintéticos, sem os microdados da POF."""

import numpy as np
import pandas as pd
import pytest

from occasion.modelagem import sintetico as s

CATEGORIAS = {
    "turno": ["manha", "almoco", "tarde", "noite"],
    "dia_semana": [str(d) for d in range(1, 8)],
    "regiao": ["Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste"],
    "tipo_area": ["capital", "rm", "interior_urbano", "rural"],
    "faixa_idade": ["10-19", "20-29", "30-44", "45-59", "60+"],
    "sexo": ["homem", "mulher"],
}


def _tabela(n: int = 4000, semente: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(semente)
    df = pd.DataFrame(
        {c: pd.Categorical(rng.choice(v, n), categories=v) for c, v in CATEGORIAS.items()}
    )
    df["renda_pct"] = rng.uniform(0, 100, n)
    # Alvo ligado a noite e fim de semana, para o modelo ter o que aprender.
    logit = -5 + 3 * (df["turno"] == "noite") + 1.5 * df["dia_semana"].isin(["6", "7"])
    df[s.ALVO] = rng.random(n) < 1 / (1 + np.exp(-logit.to_numpy(float)))
    df["peso_final"] = rng.uniform(0.5, 2.0, n)
    df["fator_dia"] = 1.0
    df["peso_metrica"] = df["peso_final"]
    df["id_setor"] = (np.arange(n) // 20).astype(str)
    return df


def test_codificador_ida_e_volta():
    df = _tabela(300)
    cod = s.Codificador.ajustar(df)
    volta = cod.montar(cod.codigos(df), cod.renda_z(df), df[s.ALVO])
    assert (cod.codigos(volta) == cod.codigos(df)).all()
    np.testing.assert_allclose(volta["renda_pct"], df["renda_pct"], rtol=1e-4)
    assert cod.one_hot(df).sum(axis=1).tolist() == [len(s.CATS)] * len(df)


@pytest.mark.parametrize("Gerador", [s.Duplicacao, s.Marginais, s.RotuloAleatorio])
def test_geradores_simples_devolvem_classe_e_categorias(Gerador):
    df = _tabela()
    cod = s.Codificador.ajustar(df)
    out = Gerador().ajustar(df, cod, semente=1).gerar(500, 1)
    assert len(out) == 500 and out[s.ALVO].all()
    for c in s.CATS:
        assert list(out[c].cat.categories) == CATEGORIAS[c]
    assert out["renda_pct"].between(0, 100).all()


def test_duplicacao_so_copia_positivos():
    df = _tabela()
    cod = s.Codificador.ajustar(df)
    out = s.Duplicacao().ajustar(df, cod, 0).gerar(300, 1)
    assert s._pct_copias(out, df[df[s.ALVO]], cod) == 100.0


def test_marginais_preservam_a_marginal_da_classe():
    df = _tabela(20000)
    cod = s.Codificador.ajustar(df)
    out = s.Marginais().ajustar(df, cod, 0).gerar(20000, 1)
    pos = df[df[s.ALVO]]
    real = np.average(pos["turno"] == "noite", weights=pos["peso_final"])
    assert abs((out["turno"] == "noite").mean() - real) < 0.03


def test_mistura_com_peso_zero_reproduz_o_real():
    df = _tabela()
    cod = s.Codificador.ajustar(df)
    sint = s.Marginais().ajustar(df, cod, 0).gerar(200, 1)
    p0 = s.treinar_mistura(df, None, 0.0).predict(df)
    p_zero = s.treinar_mistura(df, sint, 0.0).predict(df)
    np.testing.assert_allclose(p0, p_zero, rtol=1e-4, atol=1e-7)
    p_um = s.treinar_mistura(df, sint, 1.0).predict(df)
    assert p_um.mean() > p0.mean()  # positivos sintéticos sobem a taxa prevista


def test_treino_so_no_sintetico():
    df = _tabela()
    cod = s.Codificador.ajustar(df)
    completa = s.base_completa(
        s.Marginais().ajustar(df, cod, 0), 3000, 0.2, np.random.default_rng(0)
    )
    p = s.treinar_mistura(None, completa, 1.0).predict(df)
    assert p.shape == (len(df),) and np.isfinite(p).all()


def test_frechet():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4000, 3))
    assert s.frechet(a, a) == pytest.approx(0.0, abs=1e-6)
    assert s.frechet(a, a + 1.0) == pytest.approx(3.0, rel=1e-6)


def test_variacao_total():
    a = pd.Series(["x", "x", "y", "y"])
    assert s.variacao_total(a, a) == 0.0
    assert s.variacao_total(a, pd.Series(["z"] * 4)) == 1.0


def test_limiar_balanced_separa_classes():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    w = np.ones(5)
    lim = s.limiar_balanced(y, p, w)
    assert lim == 0.8
    assert s.balanced(y, p, w, lim)["balanced_acc"] == 1.0


def test_bootstrap_pareado_da_diferenca_zero_para_previsoes_iguais():
    df = _tabela(2000)
    p = np.random.default_rng(0).random(len(df))
    grupos = {"noite": (df["turno"] == "noite").to_numpy()}
    bs = s.bootstrap(df, {"a": [p], "b": [p.copy()]}, {"a": [p], "b": [p]}, grupos, n=5, n_jobs=1)
    assert (bs["pr_auc"]["a"] == bs["pr_auc"]["b"]).all()
    assert (bs["recall"]["a|noite"] == bs["recall"]["b|noite"]).all()


def test_holdout_separa_setores_inteiros():
    df = _tabela()
    fora = s.holdout_setores(df, 0.25)
    assert not set(df.loc[fora, "id_setor"]) & set(df.loc[~fora, "id_setor"])
    assert 0.15 < fora.mean() < 0.35


def test_smote_nc_gera_positivos():
    pytest.importorskip("imblearn")
    df = _tabela()
    cod = s.Codificador.ajustar(df)
    out = s.SmoteNC().ajustar(df, cod, 0).gerar(100, 1)
    assert len(out) == 100 and out[s.ALVO].all()
    with pytest.raises(ValueError):
        s.SmoteNC().ajustar(df, cod, 0).gerar(10, 0)


def test_vae_treina_e_gera():
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    df = _tabela()
    cod = s.Codificador.ajustar(df)
    vae = s.VAE(epocas=2, n_negativos=1000, mult_pos=2).ajustar(df, cod, 0)
    assert len(vae.historico) == 2 and np.isfinite(vae.historico).all()
    out = vae.gerar(50, 1)
    assert len(out) == 50 and out[s.ALVO].all()
    for c in s.CATS:
        assert list(out[c].cat.categories) == CATEGORIAS[c]
