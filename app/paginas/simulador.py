"""Simulador: contexto → Occasion Score, fatores e recomendação, pela API."""

import json

import altair as alt
import pandas as pd
import streamlit as st

import cliente
import rotulos as R

st.title("Simulador de ocasião")
st.caption(
    "Escolha um momento, um lugar e um público. A API calcula o Occasion Score "
    "(probabilidade calibrada de pizza com refrigerante no turno, POF/IBGE) e o Decision "
    "Engine escolhe a categoria de ativação por regras documentadas."
)

try:
    op = cliente.opcoes()
    meta = cliente.modelo()
except cliente.ErroApi as erro:
    st.error(str(erro))
    st.stop()

dias = {d["valor"]: d["dia"].capitalize() for d in op["dia_semana"]}

with st.sidebar:
    st.header("Contexto")
    st.subheader("Momento")
    dia = st.selectbox("Dia da semana", list(dias), index=5, format_func=dias.get)
    turno = st.selectbox("Turno", op["turno"], index=3, format_func=R.TURNOS.get)
    st.subheader("Lugar")
    regiao = st.selectbox("Região", op["regiao"], index=2)
    tipo_area = st.selectbox("Tipo de área", op["tipo_area"], format_func=R.TIPOS_AREA.get)
    st.subheader("Público")
    faixa_idade = st.selectbox("Faixa de idade", op["faixa_idade"], index=2)
    sexo = st.radio("Sexo", op["sexo"], format_func=R.SEXOS.get, horizontal=True)
    renda_sm = st.number_input(
        "Renda do domicílio (salários mínimos)",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=0.5,
    )
    st.caption(f"Classe {R.classe_de_renda(renda_sm)} pelo corte do IBGE.")
    st.subheader("Canal (opcional)")
    canal = st.selectbox(
        "Canal da ativação",
        [None, *op["canal"]],
        format_func=R.CANAIS.get,
        help="Só muda o canal quando a recomendação é de conversão (regra 2).",
    )

entrada = {
    "turno": turno,
    "dia_semana": dia,
    "regiao": regiao,
    "tipo_area": tipo_area,
    "faixa_idade": faixa_idade,
    "sexo": sexo,
    "renda_sm": float(renda_sm),
    "canal": canal,
    "versao_modelo": meta["versao"],
}

try:
    r = cliente.prever(tuple(entrada.items()))
except cliente.ErroApi as erro:
    st.error(str(erro))
    st.stop()

media = 100 * meta["taxa_media_2017_18"]
faixa = r["classification"]

pc = r.get("class_position") or {}
c1, c2, c3, c4 = st.columns(4)
c1.metric("Occasion Score", f"{r['occasion_score']:.2f} / 100")
c2.metric("Faixa", R.FAIXAS[faixa])
c3.metric("Comparado à média da população", f"{r['occasion_score'] / media:.1f}×")
if pc.get("percentil_na_classe") is not None:
    c4.metric(
        f"Posição na classe {pc['classe']}",
        f"top {max(round(100 - pc['percentil_na_classe']), 1)}%",
        "entre os 10% maiores da classe" if pc["topo_10_na_classe"] else None,
        delta_color="off",
    )
st.caption(
    f"O score é a chance, em %, de esta pessoa comer pizza com refrigerante neste turno. "
    f"A média da população em 2017-18 é {media:.2f}. Faixa alta = 10% maiores scores da "
    "população; média = 10% a 30%. A posição na classe compara o score só com pessoas da "
    "mesma classe de renda: serve para mirar as classes B a D sem inflar o score, e não "
    "muda a recomendação. O score já vem corrigido nos grupos em que o modelo errava o "
    "nível fora da amostra (Nordeste, Centro-Oeste e rural)."
)

st.divider()
esq, dir_ = st.columns([3, 2], gap="large")

with esq:
    st.subheader(f":{R.CORES_FAIXA[faixa]}[{r['recommendation_name']}]")
    if r["recommendation_channel"]:
        st.markdown(f"**Canal:** {r['recommendation_channel']}")
    st.markdown(f"**Por quê:** {r['recommendation_reason']}")
    if r["messages"]:
        st.markdown("**Mensagens sugeridas**")
        for m in r["messages"]:
            st.markdown(f"- {m}")
    bm = r["best_moment"]
    st.info(
        f"Melhor momento da semana para este perfil: **{bm['dia']}, "
        f"{R.TURNOS[bm['turno']].split(' (')[0].lower()}**, score {bm['occasion_score']:.2f} "
        f"(faixa {R.FAIXAS[bm['classification']].lower()})."
    )

with dir_:
    st.subheader("Principais fatores")
    st.caption(
        "Contribuição de cada variável em relação a uma pessoa média (SHAP linear da "
        "regressão logística). Razão de chances > 1 aumenta a chance."
    )
    fatores = pd.DataFrame(r["top_factors"])
    fatores["fator"] = fatores["rotulo"] + " = " + fatores["valor"].astype(str)
    fatores["cor"] = fatores["direcao"].map({"aumenta": "#c8102e", "reduz": "#4e79a7"})
    st.altair_chart(
        alt.Chart(fatores)
        .mark_bar()
        .encode(
            x=alt.X("contribuicao_logit:Q", title="contribuição no logit"),
            y=alt.Y("fator:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
            color=alt.Color("cor:N", scale=None),
            tooltip=["fator", "razao_chances", "direcao"],
        )
        .properties(height=150),
        use_container_width=True,
    )
    st.dataframe(
        fatores[["fator", "razao_chances", "direcao"]].rename(
            columns={"razao_chances": "razão de chances", "direcao": "efeito"}
        ),
        hide_index=True,
        use_container_width=True,
    )

st.divider()
st.subheader("A semana deste perfil")
st.caption("Occasion Score de cada dia e turno, mantendo lugar e público (28 chamadas à API).")


@st.cache_data(show_spinner="Calculando a semana...")
def semana(base: tuple) -> pd.DataFrame:
    linhas = []
    for d, nome in dias.items():
        for t in op["turno"]:
            e = {**dict(base), "dia_semana": d, "turno": t}
            linhas.append(
                {
                    "dia": nome,
                    "turno": R.TURNOS[t].split(" (")[0],
                    "score": cliente.prever(tuple(e.items()))["occasion_score"],
                }
            )
    return pd.DataFrame(linhas)


try:
    grade = semana(tuple(entrada.items()))
except cliente.ErroApi as erro:
    st.error(str(erro))
    st.stop()
ordem_turnos = [R.TURNOS[t].split(" (")[0] for t in op["turno"]]
base = alt.Chart(grade).encode(
    x=alt.X("dia:N", sort=list(dias.values()), title=None, axis=alt.Axis(labelAngle=0)),
    y=alt.Y("turno:N", sort=ordem_turnos, title=None),
)
st.altair_chart(
    base.mark_rect().encode(
        color=alt.Color("score:Q", scale=alt.Scale(scheme="reds"), title="Score"),
        tooltip=["dia", "turno", alt.Tooltip("score:Q", format=".2f")],
    )
    + base.mark_text(fontSize=13).encode(
        text=alt.Text("score:Q", format=".2f"),
        color=alt.condition(
            alt.datum.score > grade["score"].max() / 2, alt.value("white"), alt.value("black")
        ),
    ),
    use_container_width=True,
)

with st.expander("Ver a chamada à API (entrada e saída)"):
    st.code(f"POST {cliente.API_URL}/predict", language="text")
    st.code(json.dumps(entrada, ensure_ascii=False, indent=2), language="json")
    st.code(json.dumps(r, ensure_ascii=False, indent=2), language="json")
