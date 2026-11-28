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
        help="Escolhe a categoria na regra 4 e o canal da conversão e do plano de ativação.",
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
    if "409" not in str(erro):
        st.error(str(erro))
        st.stop()
    # A API trocou de modelo: os metadados em cache estão velhos. Relê e tenta de novo.
    cliente.modelo.clear()
    entrada["versao_modelo"] = cliente.modelo()["versao"]
    try:
        r = cliente.prever(tuple(entrada.items()))
    except cliente.ErroApi as erro2:
        st.error(str(erro2))
        st.stop()

media = 100 * meta["taxa_media_2017_18"]
faixa = r["classification"]

pc = r.get("class_position") or {}
inc = r.get("uncertainty") or {}


def pct(x: float) -> str:
    """Percentual com uma casa e vírgula, como na justificativa do engine."""
    return f"{x:.1f}".replace(".", ",")


def num(x: float) -> str:
    """Mesmo formato do engine: 2 algarismos significativos abaixo de 0,1, para não virar 0,00."""
    return f"{x:.2f}" if x >= 0.1 else f"{x:.2g}"


ic = inc.get("occasion_score_ic95")
prob = inc.get("prob_faixa") or {}
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Occasion Score",
    f"{num(r['occasion_score'])} / 100",
    f"IC95 {num(ic[0])} a {num(ic[1])}" if ic else "sem intervalo",
    delta_color="off",
)
c2.metric(
    "Faixa",
    R.FAIXAS[faixa],
    f"em {100 * prob[faixa]:.0f}% das reamostragens" if prob else None,
    delta_color="off",
)
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

if ic:
    cortes = meta["faixas"]
    faixas_df = pd.DataFrame(
        {
            "faixa": ["Baixa", "Média", "Alta"],
            "de": [0.0, 100 * cortes["corte_media"], 100 * cortes["corte_alta"]],
            "ate": [
                100 * cortes["corte_media"],
                100 * cortes["corte_alta"],
                max(ic[1], 100 * cortes["corte_alta"]) * 1.3,
            ],
            "y": "score",
        }
    )
    ponto = pd.DataFrame([{"score": r["occasion_score"], "lo": ic[0], "hi": ic[1], "y": "score"}])
    escala = alt.Scale(type="symlog", constant=0.05, domain=[0, float(faixas_df["ate"].max())])
    fundo = (
        alt.Chart(faixas_df)
        .mark_bar(opacity=0.55, size=36)
        .encode(
            x=alt.X(
                "de:Q",
                scale=escala,
                title=None,
                axis=alt.Axis(values=[0, 0.1, 0.5, 1, 2, 5, 10, 20, 50]),
            ),
            x2="ate:Q",
            y=alt.Y("y:N", axis=None),
            color=alt.Color(
                "faixa:N",
                scale=alt.Scale(
                    domain=["Baixa", "Média", "Alta"], range=["#9aa5b1", "#f2b134", "#c8102e"]
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=[
                "faixa",
                alt.Tooltip("de:Q", format=".2f"),
                alt.Tooltip("ate:Q", format=".2f"),
            ],
        )
    )
    barra = (
        alt.Chart(ponto)
        .mark_rule(strokeWidth=3, color="black")
        .encode(x=alt.X("lo:Q", scale=escala), x2="hi:Q", y=alt.Y("y:N", axis=None))
    )
    marca = (
        alt.Chart(ponto)
        .mark_point(filled=True, size=120, color="black")
        .encode(
            x=alt.X("score:Q", scale=escala),
            y=alt.Y("y:N", axis=None),
            tooltip=[
                alt.Tooltip("score:Q", format=".2f", title="score"),
                alt.Tooltip("lo:Q", format=".2f", title="IC95 de"),
                alt.Tooltip("hi:Q", format=".2f", title="IC95 até"),
            ],
        )
    )
    st.altair_chart((fundo + barra + marca).properties(height=95), use_container_width=True)
    st.caption(
        "Escala logarítmica. Ponto: score. Barra: intervalo de 95% por bootstrap de setores da POF "
        f"({inc.get('metodo', '').split(';')[-1].strip()}). O intervalo mede só a variação da "
        "amostra; não cobre a mudança de hábito desde 2017-18."
    )
for aviso in inc.get("avisos", []):
    st.warning(aviso)

st.divider()
esq, dir_ = st.columns([3, 2], gap="large")

with esq:
    st.subheader(f":{R.CORES_FAIXA[faixa]}[{r['recommendation_name']}]")
    if r["recommendation_channel"] and not r.get("activation_plan"):
        st.markdown(f"**Canal:** {r['recommendation_channel']}")
    st.markdown(f"**Por quê:** {r['recommendation_reason']}")
    assoc = r.get("association") or {}
    for chave, titulo in (("lacuna", "Lacuna de associação"), ("baixa_presenca", "Baixa presença")):
        for seg in assoc.get(chave, []):
            st.caption(
                f"{titulo} em {seg['segmento']}: Guaraná em {pct(seg['guarana_refri_geral_pct'])}% "
                f"do refrigerante em geral e {pct(seg['guarana_na_pizza_pct'])}% na pizza "
                f"({seg['ocasioes_pizza']} ocasiões de pizza, POF 2017-18)."
            )
    if r["messages"]:
        st.markdown("**Mensagens sugeridas**")
        for m in r["messages"]:
            st.markdown(f"- {m}")
    plano = r.get("activation_plan")
    if plano:
        with st.expander("Plano de ativação", expanded=True):
            q, onde, cn, med = plano["quando"], plano["onde"], plano["canal"], plano["medicao"]
            st.markdown(f"**Para quem:** {plano['para_quem']['perfil']}")
            if onde["cidades"]:
                nomes = ", ".join(f"{c['municipio']} ({c['uf']})" for c in onde["cidades"])
                st.markdown(f"**Onde:** {nomes}")
            st.caption(onde["criterio"])
            quando = f"{q['dia']}, {R.TURNOS[q['turno']].split(' (')[0].lower()}"
            if q["horario"]:
                h = q["horario"]
                quando += (
                    f", {h['janela']} ({pct(h['pct_ocasioes_comecam'])}% das ocasiões começam aí)"
                )
            st.markdown(f"**Quando:** {quando}")
            j = q.get("proximas_4_semanas")
            if j:
                feriados = f", com {', '.join(j['feriados'])}" if j["feriados"] else ""
                st.caption(
                    f"Próximas 4 semanas em {j['uf']}: {j['de']} a {j['ate']}{feriados} ({j['natureza']})."
                )
            st.markdown(f"**Canal:** {cn['valor']}")
            st.caption(cn["base"])
            st.markdown(f"**Mecânica (hipótese):** {plano['mecanica']['texto']}")
            st.markdown(f"**Mensagem (hipótese):** {plano['mecanica']['mensagem']}")
            if med["linha_de_base_pct"] is not None:
                st.markdown(
                    f"**Como medir:** {med['indicador']} {med['segmento']}, hoje em "
                    f"{pct(med['linha_de_base_pct'])}% (IC95 {pct(med['ic95'][0])} a "
                    f"{pct(med['ic95'][1])}). Sem meta: não há dado de efeito."
                )
    bm = r["best_moment"]
    bm_ic = bm.get("occasion_score_ic95")
    st.info(
        f"Melhor momento da semana para este perfil: **{bm['dia']}, "
        f"{R.TURNOS[bm['turno']].split(' (')[0].lower()}**, score {bm['occasion_score']:.2f}"
        + (f" (IC95 {num(bm_ic[0])} a {num(bm_ic[1])})" if bm_ic else "")
        + f", faixa {R.FAIXAS[bm['classification']].lower()}."
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
            resp = cliente.prever(tuple(e.items()))
            ic_ = (resp.get("uncertainty") or {}).get("occasion_score_ic95") or [None, None]
            linhas.append(
                {
                    "dia": nome,
                    "turno": R.TURNOS[t].split(" (")[0],
                    "score": resp["occasion_score"],
                    "ic95_de": ic_[0],
                    "ic95_ate": ic_[1],
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
        tooltip=[
            "dia",
            "turno",
            alt.Tooltip("score:Q", format=".2f"),
            alt.Tooltip("ic95_de:Q", format=".2f", title="IC95 de"),
            alt.Tooltip("ic95_ate:Q", format=".2f", title="IC95 até"),
        ],
    )
    + base.mark_text(fontSize=13).encode(
        text=alt.Text("score:Q", format=".2f"),
        color=alt.condition(
            alt.datum.score > grade["score"].max() / 2, alt.value("white"), alt.value("black")
        ),
    ),
    use_container_width=True,
)

st.divider()
fonte_col, limite_col = st.columns(2, gap="large")
with fonte_col:
    st.subheader("Fonte e ano de cada número")
    for f in r.get("sources") or []:
        st.markdown(
            f"- **{f['numero']}.** {f['fonte']} ({f['ano']}). [Dados]({f['url']})"
            + (f" · [cálculo]({f['detalhe']})" if f.get("detalhe") else "")
        )
with limite_col:
    st.subheader("O que o modelo não sabe")
    for limite in r.get("limitations") or []:
        st.markdown(f"- {limite}")
    sup = (inc.get("suporte") or {}).get("perfil")
    if sup:
        st.caption(
            f"Este perfil tem {sup['ocasioes']} ocasiões reais em {sup['linhas']:,} "
            "pessoas-turno nas duas edições da POF.".replace(",", ".")
        )

with st.expander("Ver a chamada à API (entrada e saída)"):
    st.code(f"POST {cliente.API_URL}/predict", language="text")
    st.code(json.dumps(entrada, ensure_ascii=False, indent=2), language="json")
    st.code(json.dumps(r, ensure_ascii=False, indent=2), language="json")
