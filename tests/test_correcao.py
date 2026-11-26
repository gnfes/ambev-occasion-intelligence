"""Correção de viés por grupo: estimação, seleção, aplicação e uso no modelo."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_final import edicoes, modelo  # noqa: F401 (fixtures)

from occasion.decisao import engine as e
from occasion.modelagem import correcao


def _sintetico(n: int = 60_000, semente: int = 0, efeito_ne: float = 0.7):
    """Logit calibrado conhecido; no Nordeste a taxa real é exp(efeito) vezes maior."""
    rng = np.random.default_rng(semente)
    grupos = pd.DataFrame(
        {
            "regiao": rng.choice(["Nordeste", "Sudeste", "Sul"], n, p=[0.15, 0.45, 0.4]),
            "tipo_area": rng.choice(["capital", "rural"], n),
        }
    )
    z = rng.normal(-4.5, 1.0, n)
    verdade = z + np.where(grupos["regiao"] == "Nordeste", efeito_ne, 0.0)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-verdade))).astype(int)
    w = rng.uniform(0.5, 1.5, n)
    setor = rng.integers(0, 600, n)
    return z, y, w, grupos, setor


def test_estimar_offsets_recupera_o_efeito_do_grupo():
    z, y, w, g, _ = _sintetico()
    off = correcao.estimar_offsets(z, y, w, g, [("regiao", "Nordeste")])
    assert off["regiao"]["Nordeste"] == pytest.approx(0.7, abs=0.15)
    assert abs(off["global"]) < 0.15


def test_offset_mantem_a_media_calibrada():
    z, y, w, g, _ = _sintetico()
    off = correcao.estimar_offsets(z, y, w, g, [("regiao", "Nordeste")])
    p = 1 / (1 + np.exp(-(z + correcao.aplicar(off, g))))
    assert np.average(p, weights=w) == pytest.approx(np.average(y, weights=w), rel=0.02)
    ne = (g["regiao"] == "Nordeste").to_numpy()
    assert (p * w)[ne].sum() / (y * w)[ne].sum() == pytest.approx(1.0, rel=0.02)


def test_seleciona_so_o_grupo_com_vies():
    z, y, w, g, setor = _sintetico()
    a, b = correcao._platt(z, y, w)  # média nacional calibrada, como depois do Platt
    p = 1 / (1 + np.exp(-(a * z + b)))
    sel = correcao.selecionar_grupos(g, setor, y, p, w, n_boot=200)
    assert ("regiao", "Nordeste") in sel
    # Tipo de área não tem viés (independe da região): não entra.
    assert not [s for s in sel if s[0] == "tipo_area"]


def test_validar_grupos_tira_grupo_sem_vies():
    z, y, w, g, setor = _sintetico()
    lado = setor % 2
    # O Platt dentro da validação recebe o logit "bruto"; aqui ele já é calibrado.
    fica = correcao.validar_grupos(
        z, y, w, g, lado, [("regiao", "Nordeste"), ("tipo_area", "rural")]
    )
    assert ("regiao", "Nordeste") in fica


def test_aplicar_sem_grupo_e_so_global():
    g = pd.DataFrame({"regiao": ["Sul", "Nordeste"], "tipo_area": ["capital", "rural"]})
    off = {"global": 0.1, "regiao": {"Nordeste": 0.5}, "tipo_area": {"rural": -0.2}}
    np.testing.assert_allclose(correcao.aplicar(off, g), [0.1, 0.4])
    assert correcao.selecionados_de(off) == [("regiao", "Nordeste"), ("tipo_area", "rural")]


@pytest.fixture
def modelo_corrigido(modelo, edicoes):  # noqa: F811
    rec = edicoes[1]
    off = {"global": -0.05, "regiao": {"Nordeste": 0.5}, "tipo_area": {"rural": -0.4}}
    w = rec["peso_metrica"].to_numpy()
    medias = {
        c: float(np.average(rec[c].astype(str).map(v).fillna(0.0), weights=w))
        for c, v in off.items()
        if c != "global"
    }
    return replace(modelo, offsets=off, media_offsets=medias)


def test_score_corrigido_soma_o_offset_no_logit(modelo, modelo_corrigido, edicoes):  # noqa: F811
    rec = edicoes[1].iloc[:300]
    dif = modelo_corrigido.logit(rec) - modelo.logit(rec)
    np.testing.assert_allclose(dif, correcao.aplicar(modelo_corrigido.offsets, rec))


def test_contribuicoes_somam_o_logit_com_correcao(modelo_corrigido, edicoes):  # noqa: F811
    rec = edicoes[1].iloc[:300]
    soma = modelo_corrigido.contribuicoes(rec).sum(axis=1) + modelo_corrigido.valor_base()
    np.testing.assert_allclose(soma.to_numpy(), modelo_corrigido.logit(rec), atol=1e-9)


def test_vetor_offsets_alinhado_ao_one_hot(modelo_corrigido, edicoes):  # noqa: F811
    rec = edicoes[1].iloc[:300]
    termos = modelo_corrigido._termos(rec)
    via_vetor = termos @ modelo_corrigido.vetor_offsets() + modelo_corrigido.offsets["global"]
    np.testing.assert_allclose(via_vetor, modelo_corrigido.offset(rec))


def test_percentil_na_classe_cresce_com_o_score(modelo):  # noqa: F811
    if not modelo.score_classe_ref:
        pytest.skip("modelo sem referência por classe")
    classe = next(iter(modelo.score_classe_ref))
    ref = modelo.score_classe_ref[classe]
    pct = modelo.percentil_na_classe(np.array([ref[10], ref[50], ref[95]]), [classe] * 3)
    assert pct[0] < pct[1] < pct[2]
    assert np.isnan(modelo.percentil_na_classe(np.array([0.5]), ["X"])[0])


def test_posicao_na_classe_no_engine(modelo):  # noqa: F811
    pos = e.posicao_na_classe(modelo, 0.05, "D")
    assert pos["classe"] == "D"
    if pos["percentil_na_classe"] is not None:
        assert pos["topo_10_na_classe"] == (pos["percentil_na_classe"] >= e.TOPO_NA_CLASSE)
