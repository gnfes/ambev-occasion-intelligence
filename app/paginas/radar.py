"""Trend Radar: alerta de oportunidade por UF, pela API."""

import pandas as pd
import streamlit as st

import cliente

st.title("Trend Radar")
st.caption(
    "Interesse por pizza no Google Trends (categoria Comida e bebida) por UF e semana, "
    "explicado por calendário (feriados nacionais da BrasilAPI, Carnaval, Natal e Ano-Novo, "
    "Dia da Pizza, início do mês) e clima da capital (Open-Meteo). Buscas medem interesse, "
    "não consumo: o Radar não entra no Occasion Score. O nível do alerta é uma regra "
    "documentada, e os fatores são associação, não causa."
)

try:
    radar = cliente.radar()
except cliente.ErroApi as erro:
    st.error(str(erro))
    st.stop()

NIVEIS = {
    "oportunidade": "🟢 Oportunidade",
    "observar": "🔵 Observar",
    "atencao": "🟠 Atenção",
    "estavel": "⚪ Estável",
    "sem_volume": "▫️ Sem volume",
}


def _pct(v: float | None) -> str:
    return "–" if v is None else f"{v:+.1f}%".replace(".", ",")


alertas = radar["alertas"]
st.caption(f"Última semana fechada: {radar['semana_ate']} · gerado em {radar['gerado_em']}")

contagem = pd.Series([a["nivel"] for a in alertas]).value_counts()
cols = st.columns(len(NIVEIS))
for col, (nivel, rotulo) in zip(cols, NIVEIS.items(), strict=True):
    col.metric(rotulo, int(contagem.get(nivel, 0)))


def _linha(a: dict) -> dict:
    t = a["tendencia"] or {}
    j = a["proxima_janela"] or {}
    faixa = t.get("faixa_ruido_pct")
    return {
        "UF": a["sigla"],
        "alerta": NIVEIS[a["nivel"]],
        "variação anual (4 sem.)": _pct(t.get("variacao_anual_pct")),
        "faixa de ruído": "–" if faixa is None else f"±{faixa:.1f}%".replace(".", ","),
        "próximas 4 sem. (previsto)": _pct(j.get("desvio_previsto_pct")),
        "destaque no Mapa": (a["mapa"] or {}).get("municipio_destaque", "–"),
    }


tabela = pd.DataFrame([_linha(a) for a in alertas])
st.dataframe(tabela, hide_index=True, width="stretch")
st.caption(
    '"Guaraná" como sinal de marca só tem volume semanal no Trends (categoria Comida e '
    "bebida) em poucas UFs; o status aparece no detalhe da UF quando existe."
)


def _tendencia_e_fatores(a: dict, t: dict, serie: pd.DataFrame) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Variação anual, últimas 4 semanas",
        _pct(t["variacao_anual_pct"]),
        help=f"Média {t['indice_4s']} contra {t['indice_4s_ano_anterior']} nas mesmas semanas do "
        "ano anterior (índice do Trends da própria UF).",
    )
    c2.metric("Faixa de ruído da UF", f"±{t['faixa_ruido_pct']:.1f}%".replace(".", ","))
    c3.metric(
        "Próximas 4 semanas (previsto)",
        _pct(a["proxima_janela"]["desvio_previsto_pct"]),
        help="Desvio previsto contra a média das últimas 52 semanas, pelos coeficientes de "
        "calendário e clima.",
    )
    st.markdown(f"**{NIVEIS[a['nivel']]}**: {a['motivo']}.")
    marca = a["marca_guarana"]
    if marca["status"] != "sem volume":
        st.caption(
            f'Sinal de marca ("guaraná"): {marca["status"]}, '
            f"{_pct(marca['variacao_anual_pct'])} contra o ano anterior."
        )

    serie = pd.DataFrame(detalhe["serie"])
    if not serie.empty:
        serie["semana"] = pd.to_datetime(serie["semana"])
        st.line_chart(serie.set_index("semana")[["pizza"]], y_label="índice do Trends (0 a 100)")
        st.caption(
            "Índice relativo dentro da UF: 100 = semana de maior interesse nos 5 anos. Não compare "
            "o nível entre UFs."
        )

    e, d = st.columns(2)
    with e:
        st.markdown("**O que explica as últimas 4 semanas** (pontos % de desvio)")
        if a["fatores_4s"]:
            st.dataframe(
                pd.DataFrame(a["fatores_4s"])[["descricao", "pontos_pct"]].rename(
                    columns={"descricao": "fator", "pontos_pct": "pontos %"}
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.write("Nenhum fator passa de 1 ponto: a UF está perto da média do último ano.")
    with d:
        j = a["proxima_janela"]
        st.markdown(f"**Próxima janela** ({j['semanas'][0]} a {j['semanas'][-1]})")
        st.write("Feriados: " + ("; ".join(j["feriados"]) if j["feriados"] else "nenhum nacional"))
        if j["fatores"]:
            st.dataframe(
                pd.DataFrame(j["fatores"])[["descricao", "pontos_pct"]].rename(
                    columns={"descricao": "fator", "pontos_pct": "pontos %"}
                ),
                hide_index=True,
                width="stretch",
            )


st.subheader("Detalhe da UF")
siglas = [a["sigla"] for a in alertas]
sigla = st.selectbox("UF", siglas, index=siglas.index("SP") if "SP" in siglas else 0)
try:
    detalhe = cliente.radar_uf(sigla)
except cliente.ErroApi as erro:
    st.error(str(erro))
    st.stop()
a = detalhe["alerta"]
t = a["tendencia"]

if t is None:
    st.info(f"{NIVEIS[a['nivel']]}: {a['motivo']}.")
else:
    _tendencia_e_fatores(a, t, pd.DataFrame(detalhe["serie"]))

m = a["mapa"]
if m:
    st.markdown(
        f"**No Mapa de Oportunidade:** {m['municipios_no_mapa']} municípios, "
        f"{m['municipios_potencial_alto']} de potencial alto. Destaque: "
        f"**{m['municipio_destaque']}**, {m['recomendacao_destaque']} "
        f"(canal: {m['canal_destaque']})."
    )

with st.expander("Validação fora do tempo e coeficientes"):
    meta = radar["validacao"]["meta"]
    st.caption(
        f"Treino até 2024 ({meta['n_treino']} UF-semanas), teste de {meta['teste_de']} a "
        f"{meta['teste_ate']} ({meta['n_teste']}). Erro absoluto médio do desvio semanal, em "
        "pontos %."
    )
    st.dataframe(pd.DataFrame(radar["validacao"]["modelos"]), hide_index=True)
    st.dataframe(pd.DataFrame(radar["coeficientes"]), hide_index=True)
    st.caption("Fontes: " + " · ".join(radar["fontes"].values()))
