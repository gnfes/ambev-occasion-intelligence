"""Testes do Mapa de Oportunidade que rodam sem os dados do Censo e do OSM."""

import zipfile

import numpy as np
import pandas as pd
import pytest

from occasion.mapa import censo, cnpj, oportunidade, osm
from occasion.mapa.oportunidade import Categoria, regra


@pytest.mark.parametrize(
    ("potencial", "oferta_alta", "bd", "esperado"),
    [
        ("alto", True, 10.0, (1, Categoria.PIZZARIA)),
        ("alto", False, 90.0, (2, Categoria.DELIVERY)),
        ("medio", True, 60.0, (3, Categoria.AWARENESS)),
        ("medio", False, 40.0, (4, Categoria.MONITORAR)),
        ("baixo", True, 90.0, (4, Categoria.MONITORAR)),
    ],
)
def test_regra_em_ordem(potencial, oferta_alta, bd, esperado):
    assert regra(potencial, oferta_alta, bd) == esperado


def test_nivel_potencial_usa_a_taxa_nacional():
    taxa = pd.Series([200.0, 100.0, 99.9, 50.0, 49.9])
    assert list(oportunidade.nivel_potencial(taxa, 100.0)) == [
        "alto",
        "alto",
        "medio",
        "medio",
        "baixo",
    ]


def _municipios():
    return pd.DataFrame(
        {
            "CD_MUN": [1, 2, 3, 4, 5],
            "municipio": ["A", "B", "C", "D", "Pequeno"],
            "uf": [35, 35, 29, 29, 29],
            "populacao": [500_000, 200_000, 300_000, 150_000, 50_000],
            "publico_20_44": [200_000, 80_000, 120_000, 60_000, 20_000],
            "ocasioes_semana": [40_000.0, 12_000.0, 9_000.0, 2_000.0, 5_000.0],
            "ocasioes_por_mil": [200.0, 150.0, 75.0, 33.0, 250.0],
            "pizzarias_cnpj": [100, 10, 30, 3, 50],
            "pct_classe_bd": [60.0, 70.0, 55.0, 80.0, 50.0],
        }
    )


def test_priorizar_aplica_universo_regras_e_ordem():
    r = oportunidade.priorizar(_municipios(), taxa_nacional=110.0)
    assert "Pequeno" not in set(r["municipio"])  # abaixo de 100 mil habitantes
    # oferta por 100 mil: A 20, B 5, C 10, D 2 → mediana 7,5
    assert r.attrs["mediana_oferta"] == pytest.approx(7.5)
    cat = dict(zip(r["municipio"], r["categoria"], strict=True))
    assert cat == {
        "A": Categoria.PIZZARIA.value,
        "B": Categoria.DELIVERY.value,
        "C": Categoria.AWARENESS.value,
        "D": Categoria.MONITORAR.value,
    }
    assert list(r["prioridade"]) == [1, 2, 3, 4]
    assert r["justificativa"].str.len().gt(0).all()


def test_quantis_ponderados_respeitam_o_peso():
    valor = np.array([1.0, 2.0])
    q = censo._quantis_ponderados(valor, np.array([3.0, 1.0]), n=4)
    # 75% do peso no valor 1: os três primeiros quantis ficam em 1 ou perto dele
    assert q[0] == pytest.approx(1.0)
    assert q[-1] == pytest.approx(2.0)
    assert np.all(np.diff(q) >= 0)


def test_cdf_da_percentil_no_ponto_medio():
    cdf = censo._cdf(pd.DataFrame({"renda": [100.0, 200.0], "peso": [1.0, 1.0]}))
    assert list(cdf["pct"]) == [25.0, 75.0]
    assert censo.percentil_renda(np.array([150.0]), cdf)[0] == pytest.approx(50.0)


def test_tipo_area_segue_os_estratos_da_pof():
    linhas = [(cod * 100000 + 1, nome, f"RM {cod}") for cod, nome in censo.CAPITAIS.items()]
    linhas += [
        (3500002, "Guarulhos", "RM 35"),  # RM da capital, SP tem resto da RM na POF
        (3500003, "Campinas", "RM Campinas"),  # outra RM → interior
        (1100002, "Ariquemes", "RM 11"),  # RO não tem resto da RM na POF → interior
    ]
    rec = pd.DataFrame(linhas, columns=["CD_MUN", "NM_MUN", "COD_CATMETROPOL"])
    tipo = censo.tipo_area_urbana(rec)
    assert tipo[3500001] == "capital"
    assert tipo[3500002] == "resto_rm"
    assert tipo[3500003] == "interior_urbano"
    assert tipo[1100002] == "interior_urbano"


def test_filtro_osm():
    df = pd.DataFrame(
        {
            "osm_id": ["n/1", "n/2", "n/3", "n/4", "n/5"],
            "nome": ["Pizzaria Bella", "Mercado Pizzolato", None, "Bar do Zé", "Pizza Hut"],
            "amenity": ["restaurant", None, "fast_food", "bar", None],
            "cuisine": [None, None, "pizza;burger", "regional", "pizza"],
        }
    )
    assert list(osm.filtrar(df)["osm_id"]) == ["n/1", "n/3", "n/5"]


def test_normalizar_nome_de_municipio():
    assert cnpj.normalizar("Sant'Ana do Livramento") == "SANT ANA DO LIVRAMENTO"
    assert cnpj.normalizar("Igarapé-Açu") == "IGARAPE ACU"


def test_contagem_cnpj_so_ativas_com_cnae_de_comida(tmp_path):
    def linha(nome, situacao, cnae, uf="SP", mun="7107"):
        campos = [""] * 30
        campos[cnpj.COL_NOME_FANTASIA] = nome
        campos[cnpj.COL_SITUACAO] = situacao
        campos[cnpj.COL_CNAE] = cnae
        campos[cnpj.COL_UF] = uf
        campos[cnpj.COL_MUNICIPIO] = mun
        return ";".join(f'"{c}"' for c in campos)

    linhas = [
        linha("PIZZARIA DO ZE", "02", "5611201"),
        linha("Bella Pizza", "02", "5620104"),
        linha("PIZZARIA FECHADA", "08", "5611201"),  # baixada
        linha("PIZZA E CIA", "02", "4721102"),  # padaria: fora dos CNAEs
        linha("CANTINA", "02", "5611201"),
    ]
    caminho = tmp_path / "Estabelecimentos0.zip"
    with zipfile.ZipFile(caminho, "w") as z:
        z.writestr("K3241.ESTABELE", "\n".join(linhas).encode("latin-1"))
    c = cnpj.contar_arquivo(caminho)
    assert c["estabelecimentos"].sum() == 3
    assert c["pizzarias"].sum() == 2


def test_ligacao_receita_ibge_por_nome_e_uf():
    contagem = pd.DataFrame(
        {
            cnpj.COL_UF: ["RS", "SP"],
            cnpj.COL_MUNICIPIO: ["8845", "7107"],
            cnpj.COL_CNAE: ["5611201", "5611201"],
            "estabelecimentos": [5, 9],
            "pizzarias": [1, 2],
        }
    )
    rec = pd.DataFrame({"tom": ["8845", "7107"], "nome": ["SANTANA DO LIVRAMENTO", "SAO PAULO"]})
    ibge = pd.DataFrame(
        {
            "CD_MUN": [4317103, 3550308],
            "municipio": ["Sant'Ana do Livramento", "São Paulo"],
            "uf": ["RS", "SP"],
        }
    )
    c = cnpj.ligar_ibge(contagem, rec, ibge)
    assert list(c["CD_MUN"]) == [4317103, 3550308]
    assert c.attrs["sem_ibge"].empty


def test_pontuar_aplica_a_correcao_por_grupo_do_modelo():
    from dataclasses import replace

    from occasion.mapa import score
    from occasion.modelagem import final

    modelo = final.carregar()
    if not modelo.offsets:
        pytest.skip("modelo sem correção por grupo")
    perfil = pd.DataFrame(
        {
            "CD_MUN": [1, 2],
            "municipio": ["A", "B"],
            "uf": [23, 43],
            "regiao": ["Nordeste", "Sul"],
            "tipo_area": ["interior_urbano", "interior_urbano"],
            "faixa_idade": ["25-29", "25-29"],
            "sexo": ["mulher", "mulher"],
            "renda_pct": [50.0, 50.0],
            "renda_mensal": [3000.0, 3000.0],
            "pessoas": [1000.0, 1000.0],
        }
    )
    com = score.pontuar(perfil, modelo).set_index("CD_MUN")["ocasioes_semana"]
    sem = score.pontuar(perfil, replace(modelo, offsets={}, media_offsets={})).set_index("CD_MUN")[
        "ocasioes_semana"
    ]
    g = modelo.offsets["global"]
    ne = modelo.offsets.get("regiao", {}).get("Nordeste", 0.0)
    # Probabilidades pequenas: a correção no logit vira quase um fator nas ocasiões.
    assert com[1] / sem[1] == pytest.approx(np.exp(g + ne), rel=0.05)
    assert com[2] / sem[2] == pytest.approx(np.exp(g), rel=0.05)
