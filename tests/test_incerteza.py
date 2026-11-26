"""Incerteza do score: bootstrap por setor, suporte e avisos, com dados sintéticos."""

import numpy as np
import pytest
from test_final import _edicao

from occasion.decisao import engine, fontes
from occasion.modelagem import final
from occasion.modelagem import incerteza as inc
from occasion.modelagem import treino as t


@pytest.fixture(scope="module")
def edicoes():
    return t.preparar(_edicao("2008-09", 1)), t.preparar(_edicao("2017-18", 2))


@pytest.fixture(scope="module")
def modelo(edicoes):
    return final.treinar(*edicoes)


@pytest.fixture(scope="module")
def incerteza(modelo, edicoes):
    return inc.gerar(modelo, *edicoes, n=8, semente=3)


def test_replica_sem_reamostragem_reproduz_o_modelo_servido(modelo, edicoes):
    tudo, eh_rec, lado, Z = inc._dados(modelo, *edicoes)
    beta, c = inc.replica(modelo, Z, tudo, eh_rec, lado, np.ones(len(tudo), dtype=int))
    np.testing.assert_allclose(Z @ beta + c, modelo.logit(tudo), atol=1e-4)


def test_contagens_sorteiam_setores_dentro_da_edicao(edicoes):
    tudo = final._juntar(*edicoes)
    m = inc.contagens_setor(tudo, np.random.default_rng(0))
    for ed in tudo["edicao"].unique():
        parte = tudo["edicao"] == ed
        por_setor = tudo[parte].assign(m=m[parte.to_numpy()]).groupby(t.GRUPO, observed=True)["m"]
        # Mesma contagem em todas as linhas do setor; total = número de setores.
        assert (por_setor.nunique() == 1).all()
        assert por_setor.first().sum() == tudo.loc[parte, t.GRUPO].nunique()


def test_intervalo_e_faixas_das_replicas(modelo, incerteza, edicoes):
    linhas = edicoes[1].iloc[:20]
    saida = incerteza.intervalo(modelo, linhas)
    assert incerteza.beta.shape[0] == 8
    for s in saida:
        lo, hi = s["ic95"]
        assert 0 <= lo <= hi <= 100
        assert sum(s["prob_faixa"].values()) == pytest.approx(1.0)


def test_suporte_conta_ocasioes_reais(incerteza, edicoes):
    tudo = final._juntar(*edicoes)
    noite = int(tudo.loc[tudo["turno"] == "noite", t.ALVO].sum())
    assert incerteza.suporte["por_entrada"]["turno"]["noite"]["ocasioes"] == noite
    total = sum(v["ocasioes"] for v in incerteza.suporte["perfil"].values())
    assert total == int(tudo[t.ALVO].sum())


def test_salvar_e_carregar(incerteza, tmp_path):
    caminho = incerteza.salvar(tmp_path / "inc.npz")
    lido = inc.carregar(caminho)
    np.testing.assert_array_equal(lido.beta, incerteza.beta)
    np.testing.assert_array_equal(lido.c, incerteza.c)
    assert lido.suporte == incerteza.suporte
    assert lido.metadata == incerteza.metadata


CONTEXTO = engine.Contexto(
    turno="noite",
    dia_semana=6,
    regiao="Sudeste",
    tipo_area="capital",
    faixa_idade="25-29",
    sexo="mulher",
    renda_sm=5,
)


def test_engine_traz_intervalo_fontes_e_limites(modelo, incerteza):
    r = engine.decidir(CONTEXTO, modelo, incerteza=incerteza).como_dict()
    u = r["uncertainty"]
    assert u["disponivel"] is True
    lo, hi = u["occasion_score_ic95"]
    assert lo <= hi
    assert r["best_moment"]["occasion_score_ic95"] is not None
    assert u["faixa_estavel"] == (u["prob_faixa"][r["classification"]] >= inc.ESTABILIDADE_MIN)
    assert {f["id"] for f in r["sources"]} >= {"score", "intervalo", "media_populacao"}
    assert r["limitations"] == fontes.LIMITES
    # A regra não muda com a incerteza.
    sem = engine.decidir(CONTEXTO, modelo).como_dict()
    assert sem["rule"] == r["rule"]
    assert sem["occasion_score"] == r["occasion_score"]


def test_engine_sem_replicas_avisa_e_nao_quebra(modelo):
    u = engine.decidir(CONTEXTO, modelo).como_dict()["uncertainty"]
    assert u["disponivel"] is False
    assert u["occasion_score_ic95"] is None
    assert u["avisos"]


def test_perfil_sem_ocasioes_e_informativo_e_entrada_rara_avisa(modelo, incerteza):
    entradas = {c: dict(v) for c, v in incerteza.suporte["por_entrada"].items()}
    vazio = inc.Incerteza(
        incerteza.beta, incerteza.c, {"por_entrada": entradas, "perfil": {}}, incerteza.metadata
    )
    u = engine.decidir(CONTEXTO, modelo, incerteza=vazio).como_dict()["uncertainty"]
    assert u["suporte"]["perfil"]["ocasioes"] == 0
    assert not any("Poucos dados para turno" in a for a in u["avisos"])

    entradas["turno"]["noite"] = {"ocasioes": 5, "linhas": 100}
    u = engine.decidir(CONTEXTO, modelo, incerteza=vazio).como_dict()["uncertainty"]
    assert u["poucos_dados"] is True
    assert any("Poucos dados para turno = noite: 5 ocasiões" in a for a in u["avisos"])


def test_fontes_por_regra():
    ids = {f["id"] for f in fontes.fontes_da_resposta(2, com_mensagens=True)}
    assert {"guarana_21", "zero_worldpanel"} <= ids
    for f in fontes.FONTES.values():
        assert f["ano"] and f["url"].startswith("https://")


def test_ic_guarda_algarismos_significativos_em_score_baixo():
    assert inc._sig(0.000123456) == 0.0001235
    assert inc._sig(5.4321) == 5.432


def test_replicas_de_outro_espaco_de_termos_nao_sao_compativeis(modelo, incerteza):
    assert inc.compativel(incerteza, modelo)
    torta = inc.Incerteza(
        incerteza.beta[:, :-1], incerteza.c, incerteza.suporte, incerteza.metadata
    )
    assert not inc.compativel(torta, modelo)


def test_melhor_momento_traz_estabilidade_da_faixa(modelo, incerteza):
    bm = engine.decidir(CONTEXTO, modelo, incerteza=incerteza).como_dict()["best_moment"]
    assert sum(bm["prob_faixa"].values()) == pytest.approx(1.0)
    assert bm["faixa_estavel"] == (bm["prob_faixa"][bm["classification"]] >= inc.ESTABILIDADE_MIN)
