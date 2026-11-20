"""Treino do Occasion Score: regras com dados sintéticos."""

import numpy as np
import pandas as pd
import pytest

from occasion.modelagem import treino as t


def _tabela(n_setores: int = 60, por_setor: int = 40, edicao: str = "2017-18", semente: int = 0):
    rng = np.random.default_rng(semente)
    n = n_setores * por_setor
    turno = rng.choice(["manha", "almoco", "tarde", "noite"], n)
    renda = rng.uniform(0, 100, n)
    logit = -6 + 3 * (turno == "noite") + 0.02 * renda
    alvo = rng.uniform(size=n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame(
        {
            "edicao": edicao,
            "id_setor": np.repeat([f"s{i}" for i in range(n_setores)], por_setor),
            "alvo_pizza_refri": alvo,
            "turno": turno,
            "dia_semana": rng.integers(1, 8, n),
            "regiao": rng.choice(["Norte", "Sul"], n),
            "tipo_area": rng.choice(["capital", "rural"], n),
            "trends_pizza_rank_uf": pd.array(rng.integers(1, 23, n), dtype="Int8"),
            "faixa_idade": rng.choice(["20-24", "25-29"], n),
            "sexo": rng.integers(1, 3, n),
            "renda_pct": renda,
            "n_moradores": rng.integers(1, 6, n),
            "tem_crianca": rng.integers(0, 2, n).astype(bool),
            "peso_final": rng.uniform(50, 150, n),
            "fator_dia": pd.array(np.ones(n), dtype="Float64"),
        }
    )


def test_preparar_normaliza_trends_e_agrupa_dias():
    df = t.preparar(_tabela())
    assert df["trends_pct_uf"].max() == pytest.approx(1.0)
    assert set(df["dia_grupo"].cat.categories) == {"seg-qui", "sex", "sab", "dom"}
    assert (df["peso_metrica"] == df["peso_final"]).all()


def test_baseline_e_taxa_ponderada_por_dia_e_turno():
    df = t.preparar(_tabela())
    b = t.Baseline().fit(df)
    linha = df.iloc[[0]]
    mesmo = df[
        (df["dia_semana"] == linha["dia_semana"].iloc[0]) & (df["turno"] == linha["turno"].iloc[0])
    ]
    esperado = np.average(mesmo["alvo_pizza_refri"], weights=mesmo["peso_final"])
    assert b.predict(linha)[0] == pytest.approx(esperado)


@pytest.mark.parametrize("tipo", ["logistica", "lgbm"])
def test_modelos_aprendem_o_sinal(tipo):
    treino, teste = t.preparar(_tabela(semente=1)), t.preparar(_tabela(semente=2))
    m = t.Modelo(tipo, t.FEATURES, {"num_leaves": 7, "min_child_samples": 20}, 100).fit(treino)
    p = m.predict(teste)
    y = teste["alvo_pizza_refri"].astype(int).to_numpy()
    assert t.metricas(y, p, teste["peso_metrica"].to_numpy())["roc_auc"] > 0.75


def test_calibracao_cruzada_nao_usa_a_propria_metade():
    df = t.preparar(_tabela())
    lado = t.metades_setor(df)
    # Cada setor fica inteiro de um lado.
    assert pd.Series(lado).groupby(df["id_setor"].to_numpy()).nunique().max() == 1
    p = np.full(len(df), 0.5)
    cal = t.calibrar_cruzado(p, df, lado)
    # Score constante calibrado vira a taxa observada da outra metade.
    y, w = df["alvo_pizza_refri"].to_numpy(), df["peso_final"].to_numpy()
    outra = lado == 1
    assert cal[~outra][0] == pytest.approx(np.average(y[outra], weights=w[outra]), abs=1e-4)


def test_limiar_f1_separa_classes_perfeitas():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert t.limiar_f1(y, p, np.ones(4)) == pytest.approx(0.8)


def test_bootstrap_pareado_tem_uma_coluna_por_variante():
    df = t.preparar(_tabela())
    rng = np.random.default_rng(0)
    bs = t.bootstrap_setor(
        df, {"a": rng.uniform(size=len(df)), "b": rng.uniform(size=len(df))}, n=5
    )
    assert list(bs.columns) == ["a", "b"] and len(bs) == 5


def test_fracao_topo_divide_o_bloco_de_empates():
    p = np.array([0.9, 0.5, 0.5, 0.5, 0.1])
    w = np.array([1.0, 1.0, 1.0, 1.0, 6.0])
    # 10% de 10 = 1 de peso: só a primeira linha.
    assert t.fracao_topo(p, w, 0.1).tolist() == [1, 0, 0, 0, 0]
    # 20% = 2 de peso: a primeira e 1/3 de cada empate.
    np.testing.assert_allclose(t.fracao_topo(p, w, 0.2), [1, 1 / 3, 1 / 3, 1 / 3, 0])


def test_limiar_f1_respeita_empates():
    # O bloco 0.5 tem 1 positivo e 3 negativos: levar o bloco piora o F1.
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.5, 0.5, 0.5, 0.5])
    assert t.limiar_f1(y, p, np.ones(5)) == pytest.approx(0.9)


def test_preparar_recusa_dia_sem_dia_da_semana():
    df = _tabela()
    df["fator_dia"] = df["fator_dia"].astype("Float64")
    df.loc[0, "fator_dia"] = pd.NA
    with pytest.raises(ValueError, match="dia da semana"):
        t.preparar(df)
