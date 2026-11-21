"""Modelo final, explicação e avaliação: regras com dados sintéticos."""

import numpy as np
import pandas as pd
import pytest
import shap
from test_treino import _tabela

from occasion.modelagem import avaliacao as av
from occasion.modelagem import final
from occasion.modelagem import treino as t
from occasion.modelagem.experimentos import FEATURES_FINAIS


def _edicao(edicao: str, semente: int) -> pd.DataFrame:
    df = _tabela(edicao=edicao, semente=semente)
    rng = np.random.default_rng(semente)
    df["renda_sm"] = rng.uniform(0, 30, len(df))
    df["teve_guarana"] = df["alvo_pizza_refri"] & (rng.uniform(size=len(df)) < 0.3)
    df["fora_de_casa"] = pd.array(
        np.where(df["alvo_pizza_refri"], rng.uniform(size=len(df)) < 0.3, pd.NA), dtype="boolean"
    )
    return df


@pytest.fixture(scope="module")
def edicoes():
    return t.preparar(_edicao("2008-09", 1)), t.preparar(_edicao("2017-18", 2))


@pytest.fixture(scope="module")
def modelo(edicoes):
    return final.treinar(*edicoes)


def test_contribuicoes_somam_o_logit_do_score(modelo, edicoes):
    rec = edicoes[1].iloc[:200]
    p = modelo.prever(rec)
    soma = modelo.contribuicoes(rec).sum(axis=1).to_numpy() + modelo.valor_base()
    np.testing.assert_allclose(soma, np.log(p / (1 - p)), atol=1e-9)


def test_contribuicoes_iguais_ao_shap_linear(modelo, edicoes):
    rec = edicoes[1]
    w = rec["peso_metrica"].to_numpy()
    Z = modelo._termos(rec)
    lr = modelo.modelo.est[-1]
    media = np.average(Z, axis=0, weights=w)[None, :]
    ex = shap.LinearExplainer((lr.coef_[0], lr.intercept_[0]), shap.maskers.Independent(media))
    sv = ex.shap_values(Z[:100]) * modelo.platt_a
    esperado = pd.DataFrame(sv, columns=modelo._grupos()).T.groupby(level=0).sum().T
    obtido = modelo.contribuicoes(rec.iloc[:100])
    np.testing.assert_allclose(obtido[esperado.columns].to_numpy(), esperado.to_numpy(), atol=1e-9)


def test_fatores_ordenados_por_efeito_e_com_direcao(modelo, edicoes):
    linha = edicoes[1].iloc[[0]]
    fatores = modelo.fatores(linha, n=3)[0]
    assert len(fatores) == 3
    efeitos = [abs(f["contribuicao_logit"]) for f in fatores]
    assert efeitos == sorted(efeitos, reverse=True)
    for f in fatores:
        assert f["feature"] in FEATURES_FINAIS
        assert f["direcao"] == ("aumenta" if f["contribuicao_logit"] > 0 else "reduz")


def test_turno_noite_e_o_fator_que_mais_pesa(modelo, edicoes):
    # No sintético o sinal forte é a noite (+3 no logit).
    noite = edicoes[1][edicoes[1]["turno"] == "noite"].iloc[[0]]
    assert modelo.fatores(noite, n=1)[0][0]["feature"] == "turno"


def test_faixas_cobrem_10_e_30_por_cento_do_peso(modelo, edicoes):
    rec = edicoes[1]
    w = rec["peso_metrica"].to_numpy()
    faixa = modelo.faixa(modelo.prever(rec))
    assert w[faixa == "alta"].sum() / w.sum() == pytest.approx(0.10, abs=0.02)
    assert w[faixa != "baixa"].sum() / w.sum() == pytest.approx(0.30, abs=0.02)


def test_salvar_e_carregar(modelo, edicoes, tmp_path):
    caminho = modelo.salvar(tmp_path / "m.joblib")
    de_novo = final.carregar(caminho)
    rec = edicoes[1].iloc[:50]
    np.testing.assert_allclose(de_novo.prever(rec), modelo.prever(rec))
    assert caminho.with_suffix(".json").exists()


def test_corte_topo():
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    assert final._corte_topo(s, np.ones(5), 0.2) == pytest.approx(0.9)
    assert final._corte_topo(s, np.ones(5), 0.4) == pytest.approx(0.8)


def test_permutacao_poe_o_turno_no_topo(edicoes):
    tr, te = edicoes
    m = t.Modelo("logistica", FEATURES_FINAIS).fit(tr)
    imp = av.importancia_permutacao(m, te, t.metades_setor(te), n=3)
    assert imp.index[0] == "turno"


def test_fatia_guarana_nacional(edicoes):
    te = edicoes[1]
    tab = av.fatia_guarana(te)
    pos = te[te[t.ALVO]]
    esperado = 100 * np.average(pos["teve_guarana"].astype(float), weights=pos["peso_metrica"])
    assert tab.loc[("nacional", "Brasil"), "fatia_guarana_pct"] == pytest.approx(esperado)
    assert tab.loc[("nacional", "Brasil"), "ocasioes"] == len(pos)


def test_faixas_e_matriz_somam_tudo():
    y = np.array([1, 0, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.1, 0.05, 0.01])
    w = np.ones(5)
    tab, faixa = av.faixas(s, y, w, 0.5, 0.05)
    assert tab["populacao_pct"].sum() == pytest.approx(100)
    assert tab["ocasioes_pct"].sum() == pytest.approx(100)
    m = av.matriz_confusao(y, (faixa == "alta").astype(int), w)
    assert m.to_numpy().sum() == pytest.approx(100)
    assert m.loc["real 1", "previsto 1"] == pytest.approx(20)


def test_treinar_exige_renda_sm(edicoes):
    antiga, recente = edicoes
    with pytest.raises(ValueError, match="renda_sm"):
        final.treinar(antiga, recente.drop(columns="renda_sm"))


def test_faixas_aceita_faixa_vazia():
    y = np.array([1, 0, 0])
    s = np.array([0.9, 0.9, 0.1])
    tab, _ = av.faixas(s, y, np.ones(3), 0.9, 0.9)
    assert tab.loc["media", "populacao_pct"] == 0
