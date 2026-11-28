"""Modelo e validação: o que está por trás do score, lido da API (/model)."""

import pandas as pd
import streamlit as st

import cliente

st.title("Modelo e validação")

try:
    m = cliente.modelo()
except cliente.ErroApi as erro:
    st.error(str(erro))
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Versão", m["versao"])
c2.metric("Linhas de treino", f"{m['linhas_treino']:,}".replace(",", "."))
c3.metric("Ocasiões positivas", f"{m['positivos_treino']:,}".replace(",", "."))

st.markdown(
    f"""
- **Unidade de análise:** {m["unidade"]}
- **Target:** {m["alvo"]}
- **Algoritmo:** {m["algoritmo"]}
- **Features:** {", ".join(m["features"])}
- **Treino do modelo do app:** {m["treino"]}
- **Calibração:** {m["calibracao"]}
"""
)

st.subheader("Validação temporal")
v = m["validacao_temporal"]
st.caption(
    f"Treino em {v['treino']}, teste em {v['teste']}. Baseline: frequência histórica por dia da "
    'semana e turno, sem Machine Learning (o "sexta à noite é bom para pizza" do brief).'
)
nomes = {
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "lift_top10": "Lift nos 10% maiores",
    "recall_top10": "Recall nos 10% maiores",
    "taxa_base": "Taxa base",
}
tabela = pd.DataFrame({"Modelo": v["modelo"], "Baseline": v["baseline"]}).rename(index=nomes)
tabela["Diferença"] = tabela["Modelo"] - tabela["Baseline"]
st.dataframe(tabela.style.format("{:.4f}"), use_container_width=True)
st.caption(
    "Métrica principal: PR-AUC, porque a ocasião é rara (0,33% dos turnos) e um falso positivo "
    "gera uma ativação pouco relevante. Os 10% maiores scores concentram "
    f"{100 * v['modelo']['recall_top10']:.0f}% das ocasiões (baseline: "
    f"{100 * v['baseline']['recall_top10']:.0f}%)."
)

st.subheader("Faixas do score")
f = m["faixas"]
st.markdown(
    f"""
- **Alta:** score ≥ {100 * f["corte_alta"]:.2f} (10% maiores scores da população de 2017-18,
  cada dia da semana com o mesmo peso)
- **Média:** score ≥ {100 * f["corte_media"]:.2f} (10% a 30%)
- **Baixa:** o resto
"""
)

with st.expander("Metadados completos (GET /model)"):
    st.json(m)
