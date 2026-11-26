"""Trend Radar: o interesse por pizza está subindo ou caindo em cada UF, e por quê.

Unidade: UF x semana (domingo a sábado, a semana do Google Trends). Cidade
não entra porque o Trends só tem série semanal estável no recorte por UF.

Sinal: índice semanal do Trends para "pizza" (categoria Comida e bebida). O
índice vai de 0 a 100 dentro de cada UF, então cada UF só é comparada com ela
mesma. Buscas medem interesse, não consumo: o Radar não alimenta o Occasion
Score e não serve de target.

Três peças:

1. Tendência. Média das últimas 4 semanas fechadas contra as mesmas 4 semanas
   do ano anterior (variação anual, que tira a sazonalidade). A faixa de
   ruído é ±1,28 desvio-padrão das variações anuais de 4 semanas que a
   própria UF teve no histórico (desvio-padrão robusto, pela mediana dos
   desvios absolutos; ~80% das variações caem dentro dela se não houver
   tendência). Acima da faixa = subindo; abaixo = caindo.
2. Fatores. Regressão linear em painel do desvio semanal (índice da semana ÷
   média das 52 semanas anteriores da UF, em %) contra calendário (mês,
   feriados nacionais da BrasilAPI, Carnaval, Natal e Ano-Novo, Dia da Pizza,
   início do mês) e anomalias de clima da capital (Open-Meteo, contra a
   climatologia 2016-2020 da mesma semana do ano). A contribuição de cada
   fator nas últimas 4 semanas é coeficiente x valor; o que sobra é "não
   explicado". Validação fora do tempo: treino até 2024, teste de 2025 em
   diante, contra o baseline "igual à média do último ano".
3. Próxima janela. Feriados das 4 semanas seguintes à semana em curso e previsão de 16 dias do
   Open-Meteo passam pelos mesmos coeficientes, e o alerta cruza isso com o
   Mapa de Oportunidade da UF (municípios por categoria e o primeiro da fila).

Nada aqui é causal: os coeficientes descrevem associação no histórico de
buscas, e a regra que transforma tendência em alerta é documentada, não
aprendida.

    uv run python -m occasion.radar.radar
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from occasion.mapa.oportunidade import ARQUIVO_PONTOS_REFERENCIA
from occasion.paths import PROCESSED_DIR, REFERENCIA_DIR
from occasion.radar import clima, feriados, trends

ARQUIVO_ALERTAS = REFERENCIA_DIR / "radar_alertas.json"
ARQUIVO_RELATORIO = PROCESSED_DIR / "trend_radar.md"

JANELA_SEMANAS = 4
Z_FAIXA = 1.28  # ±1,28 dp: faixa central de 80%
INICIO_TESTE = pd.Timestamp("2025-01-01")
CLIMATOLOGIA = ("2016-01-01", "2020-12-31")  # antes do início do painel (nov/2021)
N_BOOTSTRAP = 300
BLOCO_SEMANAS = 4
SEMENTE = 42
# Acima desta fração de semanas com índice zero (volume insuficiente no Trends),
# a série da UF não é lida: nem tendência, nem ajuste do modelo.
MAX_ZEROS = 0.2

CALENDARIO = ["feriado", "carnaval", "natal_ano_novo", "dia_da_pizza", "inicio_mes"]
CLIMA = ["anom_temp", "anom_chuva"]
MESES = [f"mes_{m:02d}" for m in range(2, 13)]  # janeiro é a referência

NOMES_FATOR = {
    "feriado": "semana com feriado nacional",
    "carnaval": "Carnaval",
    "natal_ano_novo": "Natal e Ano-Novo",
    "dia_da_pizza": "Dia da Pizza (10/07)",
    "inicio_mes": "início do mês (dias 1 a 10)",
    "anom_temp": "temperatura acima do normal",
    "anom_chuva": "chuva acima do normal",
    "sazonalidade": "época do ano",
    "nao_explicado": "não explicado pelo calendário e pelo clima",
}


# --------------------------------------------------------------------------- dados


def calendario_semanal(semanas: pd.Series, tabela_feriados: pd.DataFrame) -> pd.DataFrame:
    """Variáveis de calendário por semana (domingo a sábado)."""
    semanas = pd.to_datetime(pd.Series(semanas).drop_duplicates()).sort_values()
    datas = tabela_feriados["data"]
    nomes = tabela_feriados["nome"].str.lower()
    linhas = []
    for inicio in semanas:
        dias = pd.date_range(inicio, periods=7, freq="D")
        no_periodo = datas.isin(dias)
        nomes_semana = nomes[no_periodo]
        carnaval = nomes_semana.str.contains("carnaval").any()
        natal = any((d.month == 12 and d.day >= 24) or (d.month == 1 and d.day == 1) for d in dias)
        outros = nomes_semana[~nomes_semana.str.contains("carnaval|natal|confraterniza")]
        linhas.append(
            {
                "semana": inicio,
                "feriado": int(len(outros) > 0),
                "carnaval": int(carnaval),
                "natal_ano_novo": int(natal),
                "dia_da_pizza": int(any(d.month == 7 and d.day == 10 for d in dias)),
                "inicio_mes": float(np.mean([d.day <= 10 for d in dias])),
                "mes": dias[3].month,  # mês da quarta-feira
                "feriados_nomes": "; ".join(
                    sorted(set(tabela_feriados.loc[no_periodo, "nome"].tolist()))
                ),
            }
        )
    return pd.DataFrame(linhas)


def clima_semanal(diario: pd.DataFrame) -> pd.DataFrame:
    """Temperatura média e chuva por UF x semana, e anomalias contra 2016-2020."""
    d = diario.copy()
    d["semana"] = d["data"] - pd.to_timedelta((d["data"].dt.day_of_week + 1) % 7, unit="D")
    semanal = (
        d.groupby(["sigla", "semana"])
        .agg(
            temp=("temperature_2m_mean", "mean"),
            chuva=("precipitation_sum", "mean"),
            dias=("data", "size"),
            previsao=("origem", lambda s: bool((s == "previsao").any())),
        )
        .reset_index()
    )
    semanal = semanal[semanal["dias"] >= 4]
    semanal["semana_ano"] = semanal["semana"].dt.isocalendar().week.clip(upper=52).astype(int)
    base = semanal[semanal["semana"].between(*CLIMATOLOGIA)]
    normal = base.groupby(["sigla", "semana_ano"])[["temp", "chuva"]].mean().add_suffix("_normal")
    semanal = semanal.join(normal, on=["sigla", "semana_ano"])
    semanal["anom_temp"] = semanal["temp"] - semanal["temp_normal"]
    semanal["anom_chuva"] = semanal["chuva"] - semanal["chuva_normal"]
    return semanal[["sigla", "semana", "temp", "chuva", "anom_temp", "anom_chuva", "previsao"]]


def painel(serie: pd.DataFrame, cal: pd.DataFrame, clim: pd.DataFrame | None) -> pd.DataFrame:
    """Uma linha por UF x semana fechada, com o desvio e as variáveis explicativas."""
    p = serie[(serie["termo"] == "pizza") & ~serie["parcial"]].copy()
    p = p[~p["sigla"].isin(sem_volume(serie))]
    p = p.sort_values(["sigla", "semana"])
    media_52 = p.groupby("sigla")["indice"].transform(
        lambda s: s.shift(1).rolling(52, min_periods=52).mean()
    )
    p["desvio"] = 100 * (p["indice"] / media_52 - 1)
    p = p.merge(cal, on="semana", how="left")
    if clim is not None:
        p = p.merge(
            clim[["sigla", "semana", "anom_temp", "anom_chuva"]], on=["sigla", "semana"], how="left"
        )
    for m in range(2, 13):
        p[f"mes_{m:02d}"] = (p["mes"] == m).astype(int)
    return p


def sem_volume(serie: pd.DataFrame, termo: str = "pizza") -> list[str]:
    """UFs em que o Trends devolve zero (volume insuficiente) em mais de 20% das semanas."""
    t = serie[(serie["termo"] == termo) & ~serie["parcial"]]
    zeros = (t["indice"] == 0).groupby(t["sigla"]).mean()
    return sorted(zeros[zeros > MAX_ZEROS].index)


# --------------------------------------------------------------------------- modelo


@dataclass
class Ajuste:
    variaveis: list[str]
    coef: np.ndarray  # intercepto primeiro
    ic95: np.ndarray | None = None

    def prever(self, x: pd.DataFrame) -> np.ndarray:
        return self.coef[0] + x[self.variaveis].to_numpy(float) @ self.coef[1:]


def ajustar(df: pd.DataFrame, variaveis: list[str]) -> Ajuste:
    x = np.column_stack([np.ones(len(df)), df[variaveis].to_numpy(float)])
    coef, *_ = np.linalg.lstsq(x, df["desvio"].to_numpy(float), rcond=None)
    return Ajuste(list(variaveis), coef)


def bootstrap(df: pd.DataFrame, variaveis: list[str], n: int = N_BOOTSTRAP) -> np.ndarray:
    """IC95 por bootstrap de blocos de semanas (todas as UFs juntas no bloco).

    Reamostrar semanas inteiras, e não linhas, preserva a correlação entre UFs
    na mesma semana (um feriado nacional mexe em todas ao mesmo tempo).
    """
    rng = np.random.default_rng(SEMENTE)
    semanas = np.sort(df["semana"].unique())
    blocos = [semanas[i : i + BLOCO_SEMANAS] for i in range(0, len(semanas), BLOCO_SEMANAS)]
    por_semana = {s: g for s, g in df.groupby("semana")}
    coefs = []
    for _ in range(n):
        escolhidos = rng.integers(0, len(blocos), len(blocos))
        amostra = pd.concat([por_semana[s] for b in escolhidos for s in blocos[b]])
        coefs.append(ajustar(amostra, variaveis).coef)
    return np.percentile(np.array(coefs), [2.5, 97.5], axis=0)


def validar(p: pd.DataFrame, variaveis_clima: list[str]) -> tuple[pd.DataFrame, dict]:
    """MAE fora do tempo (teste a partir de 2025) dos modelos contra o baseline."""
    base = p.dropna(subset=["desvio", *variaveis_clima])
    treino = base[base["semana"] < INICIO_TESTE]
    teste = base[base["semana"] >= INICIO_TESTE]
    modelos = {
        "baseline (igual à média do último ano)": None,
        "calendário": MESES + CALENDARIO,
    }
    if variaveis_clima:
        modelos["calendário + clima"] = MESES + CALENDARIO + variaveis_clima
    rng = np.random.default_rng(SEMENTE)
    semanas = teste["semana"].unique()
    linhas = []
    erros_base = teste["desvio"].abs().to_numpy()
    evento = teste[["feriado", "carnaval", "natal_ano_novo", "dia_da_pizza"]].sum(axis=1) > 0
    evento = evento.to_numpy()
    semanas_evento = teste.loc[evento, "semana"].unique()
    for nome, vars_ in modelos.items():
        erros = (
            erros_base
            if vars_ is None
            else np.abs(teste["desvio"].to_numpy() - ajustar(treino, vars_).prever(teste))
        )
        ganho = erros_base - erros
        por_semana = pd.Series(ganho).groupby(teste["semana"].to_numpy()).mean()
        boot = [por_semana.loc[rng.choice(semanas, len(semanas))].mean() for _ in range(1000)]
        boot_ev = [
            por_semana.loc[rng.choice(semanas_evento, len(semanas_evento))].mean()
            for _ in range(1000)
        ]
        linhas.append(
            {
                "modelo": nome,
                "mae": round(float(erros.mean()), 2),
                "ganho_vs_baseline": round(float(ganho.mean()), 2),
                "ic95_ganho": [round(float(v), 2) for v in np.percentile(boot, [2.5, 97.5])],
                "mae_semanas_com_evento": round(float(erros[evento].mean()), 2),
                "ganho_semanas_com_evento": round(float(ganho[evento].mean()), 2),
                "ic95_ganho_semanas_com_evento": [
                    round(float(v), 2) for v in np.percentile(boot_ev, [2.5, 97.5])
                ],
            }
        )
    meta = {
        "n_treino": len(treino),
        "n_teste": len(teste),
        "teste_de": str(teste["semana"].min().date()),
        "teste_ate": str(teste["semana"].max().date()),
        "semanas_com_evento_no_teste": len(semanas_evento),
    }
    return pd.DataFrame(linhas), meta


# --------------------------------------------------------------------------- alertas


def tendencia(serie_uf: pd.DataFrame) -> dict:
    """Variação anual das últimas 4 semanas e faixa de ruído da própria UF."""
    s = serie_uf.set_index("semana")["indice"].sort_index()
    janela = s.rolling(JANELA_SEMANAS).mean()
    variacao = 100 * (janela / janela.shift(52) - 1)
    historico = variacao.iloc[:-1].dropna()
    atual = float(variacao.iloc[-1])
    # Desvio-padrão robusto (MAD x 1,4826): um pico isolado, como uma semana de
    # Carnaval, não alarga a faixa da UF inteira.
    mad = float((historico - historico.median()).abs().median())
    faixa = Z_FAIXA * 1.4826 * mad
    status = "subindo" if atual > faixa else "caindo" if atual < -faixa else "estável"
    return {
        "semana_ate": str(s.index[-1].date()),
        "variacao_anual_pct": round(atual, 1),
        "faixa_ruido_pct": round(faixa, 1),
        "status": status,
        "indice_4s": round(float(janela.iloc[-1]), 1),
        "indice_4s_ano_anterior": round(float(janela.shift(52).iloc[-1]), 1),
    }


def contribuicoes(ajuste: Ajuste, linhas: pd.DataFrame) -> dict[str, float]:
    """Pontos de desvio (%) de cada fator, em média nas linhas dadas."""
    pesos = dict(zip(ajuste.variaveis, ajuste.coef[1:], strict=True))
    media = linhas[ajuste.variaveis].astype(float).mean()
    out = {"sazonalidade": float(sum(pesos[m] * media[m] for m in MESES) + ajuste.coef[0])}
    for v in ajuste.variaveis:
        if v not in MESES:
            out[v] = float(pesos[v] * media[v])
    return out


def _texto_fatores(contrib: dict[str, float], limite: float = 1.0) -> list[dict]:
    itens = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))
    return [
        {"fator": k, "descricao": NOMES_FATOR[k], "pontos_pct": round(v, 1)}
        for k, v in itens
        if abs(v) >= limite
    ]


def contexto_mapa(mapa: pd.DataFrame) -> dict[str, dict]:
    out = {}
    for sigla, g in mapa.groupby("uf"):
        g = g.sort_values("prioridade")
        alto = g[g["categoria"].str.startswith("conversao")]
        topo = (alto if len(alto) else g).iloc[0]
        out[sigla] = {
            "municipios_no_mapa": len(g),
            "municipios_potencial_alto": len(alto),
            "ocasioes_semana": int(g["ocasioes_semana"].sum()),
            "municipio_destaque": str(topo["municipio"]),
            "recomendacao_destaque": str(topo["recomendacao"]),
            "canal_destaque": str(topo["canal"]),
        }
    return out


def regra_alerta(
    status: str, janela_pct: float, faixa: float, potencial_alto: int
) -> tuple[str, str]:
    """Nível do alerta (regra documentada, a primeira que vale decide).

    1. Subindo, ou próxima janela prevista acima da faixa de ruído, e há
       município de potencial alto no Mapa → oportunidade.
    2. Subindo ou janela acima da faixa, sem município de potencial alto →
       observar (interesse sem base de ocasiões estimada).
    3. Caindo → atenção.
    4. Resto → estável.
    """
    sinal = status == "subindo" or janela_pct > faixa
    if sinal and potencial_alto > 0:
        return "oportunidade", "interesse em alta numa UF com municípios de potencial alto no Mapa"
    if sinal:
        return "observar", "interesse em alta, mas a UF não tem município de potencial alto no Mapa"
    if status == "caindo":
        return "atencao", "interesse abaixo do mesmo período do ano passado, além da faixa de ruído"
    return "estavel", "interesse dentro da faixa de ruído da UF"


def janela_futura(
    cal: pd.DataFrame, clim: pd.DataFrame | None, sigla: str, apos: pd.Timestamp
) -> pd.DataFrame:
    futuro = cal[cal["semana"] > apos].sort_values("semana").head(JANELA_SEMANAS).copy()
    for m in range(2, 13):
        futuro[f"mes_{m:02d}"] = (futuro["mes"] == m).astype(int)
    if clim is not None:
        c = clim[clim["sigla"] == sigla][["semana", "anom_temp", "anom_chuva"]]
        futuro = futuro.merge(c, on="semana", how="left")
        # Sem previsão (além de 16 dias), o clima fica no normal.
        futuro[CLIMA] = futuro[CLIMA].fillna(0.0)
    return futuro


def montar(hoje: dt.date | None = None) -> tuple[dict, pd.DataFrame]:
    serie = trends.ler()
    tabela_feriados = feriados.ler()
    tem_clima = clima.ARQUIVO.exists()
    clim = clima_semanal(clima.ler()) if tem_clima else None
    variaveis_clima = CLIMA if tem_clima else []

    ultima = serie["semana"].max()
    futuras = pd.Series(pd.date_range(ultima + pd.Timedelta(days=7), periods=8, freq="7D"))
    cal = calendario_semanal(pd.concat([serie["semana"], futuras]), tabela_feriados)
    p = painel(serie, cal, clim)
    variaveis = MESES + CALENDARIO + variaveis_clima
    dados = p.dropna(subset=["desvio", *variaveis_clima])
    ajuste = ajustar(dados, variaveis)
    ajuste.ic95 = bootstrap(dados, variaveis)
    validacao, meta_val = validar(p, variaveis_clima)

    mapa = pd.read_csv(ARQUIVO_PONTOS_REFERENCIA)
    ctx_mapa = contexto_mapa(mapa)
    fechadas = serie[(serie["termo"] == "pizza") & ~serie["parcial"]]

    fora = sem_volume(serie)
    alertas = []
    for sigla in fora:
        alertas.append(
            {
                "sigla": sigla,
                "capital": clima.CAPITAIS[sigla][0],
                "nivel": "sem_volume",
                "motivo": "o Trends devolve volume insuficiente (zero) em mais de 20% das "
                "semanas; a UF não é lida",
                "tendencia": None,
                "desvio_4s_pct": None,
                "fatores_4s": [],
                "proxima_janela": None,
                "marca_guarana": sinal_marca(serie, sigla),
                "mapa": ctx_mapa.get(sigla),
            }
        )
    for sigla, g in fechadas[~fechadas["sigla"].isin(fora)].groupby("sigla"):
        t = tendencia(g)
        ultimas = dados[dados["sigla"] == sigla].sort_values("semana").tail(JANELA_SEMANAS)
        contrib = contribuicoes(ajuste, ultimas)
        observado = float(ultimas["desvio"].mean())
        contrib["nao_explicado"] = observado - sum(contrib.values())
        fut = janela_futura(cal, clim, sigla, ultima)
        contrib_fut = contribuicoes(ajuste, fut)
        janela_pct = float(np.mean(ajuste.prever(fut)))
        ctx = ctx_mapa.get(sigla, {"municipios_potencial_alto": 0})
        nivel, motivo = regra_alerta(
            t["status"], janela_pct, t["faixa_ruido_pct"], ctx["municipios_potencial_alto"]
        )
        alertas.append(
            {
                "sigla": sigla,
                "capital": clima.CAPITAIS[sigla][0],
                "nivel": nivel,
                "motivo": motivo,
                "tendencia": t,
                "desvio_4s_pct": round(observado, 1),
                "fatores_4s": _texto_fatores(contrib),
                "proxima_janela": {
                    "semanas": [str(s.date()) for s in fut["semana"]],
                    "feriados": [n for n in fut["feriados_nomes"] if n],
                    "desvio_previsto_pct": round(janela_pct, 1),
                    "fatores": _texto_fatores(contrib_fut),
                    "clima_previsto": bool(tem_clima),
                },
                "marca_guarana": sinal_marca(serie, sigla),
                "mapa": ctx_mapa.get(sigla),
            }
        )
    ordem = {"oportunidade": 0, "observar": 1, "atencao": 2, "estavel": 3, "sem_volume": 4}
    alertas.sort(
        key=lambda a: (
            ordem[a["nivel"]],
            -(a["tendencia"] or {}).get("variacao_anual_pct", 0),
        )
    )

    coeficientes = pd.DataFrame(
        {
            "variavel": ["intercepto", *variaveis],
            "coef": ajuste.coef,
            "ic95_inf": ajuste.ic95[0],
            "ic95_sup": ajuste.ic95[1],
        }
    )
    saida = {
        "gerado_em": (hoje or dt.date.today()).isoformat(),
        "semana_ate": str(fechadas["semana"].max().date()),
        "fontes": {
            "trends": "Google Trends, 'pizza', categoria Comida e bebida, por UF, semanal, 5 anos",
            "feriados": "BrasilAPI, feriados nacionais",
            "clima": "Open-Meteo (ERA5 e previsão de 16 dias), capital da UF"
            if tem_clima
            else "indisponível nesta execução",
            "mapa": "Mapa de Oportunidade (Occasion Score no Censo 2022 + CNPJ)",
        },
        "validacao": {"meta": meta_val, "modelos": validacao.to_dict(orient="records")},
        "coeficientes": coeficientes.round(3).to_dict(orient="records"),
        "alertas": alertas,
    }
    return saida, coeficientes


def sinal_marca(serie: pd.DataFrame, sigla: str) -> dict:
    """Variação anual de "guaraná" (sinal de marca), só onde há volume."""
    g = serie[(serie["termo"] == "guaraná") & (serie["sigla"] == sigla) & ~serie["parcial"]]
    if g.empty or (g["indice"] == 0).mean() > MAX_ZEROS:
        return {"status": "sem volume", "variacao_anual_pct": None}
    t = tendencia(g)
    return {"status": t["status"], "variacao_anual_pct": t["variacao_anual_pct"]}


# --------------------------------------------------------------------------- relatório


def _pct(v: float) -> str:
    return f"{v:+.1f}%".replace(".", ",")


def relatorio(saida: dict, coeficientes: pd.DataFrame) -> str:
    val = saida["validacao"]
    linhas = [
        "# Trend Radar",
        "",
        f"Gerado em {saida['gerado_em']}, última semana fechada: {saida['semana_ate']}.",
        "",
        "Buscas medem interesse, não consumo. O Radar não entra no Occasion Score.",
        "",
        "## Fontes",
        "",
        *[f"- {k}: {v}" for k, v in saida["fontes"].items()],
        "",
        "## Validação fora do tempo",
        "",
        (
            f"Treino até 2024 ({val['meta']['n_treino']} UF-semanas), teste de "
            f"{val['meta']['teste_de']} a {val['meta']['teste_ate']} ({val['meta']['n_teste']}). "
            "Alvo: desvio % do índice da semana contra a média das 52 semanas anteriores da UF. "
            "IC95 do ganho por bootstrap de semanas."
        ),
        "",
        (
            "| modelo | MAE (pontos %) | ganho vs baseline | IC95 do ganho | "
            "ganho nas semanas com evento | IC95 |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for m in val["modelos"]:
        ic = m["ic95_ganho"]
        ie = m["ic95_ganho_semanas_com_evento"]
        linhas.append(
            f"| {m['modelo']} | {m['mae']} | {m['ganho_vs_baseline']} | [{ic[0]}; {ic[1]}] | "
            f"{m['ganho_semanas_com_evento']} | [{ie[0]}; {ie[1]}] |"
        )
    linhas += [
        "",
        (
            f"Semanas com evento: feriado nacional, Carnaval, Natal e Ano-Novo ou Dia da Pizza "
            f"({val['meta']['semanas_com_evento_no_teste']} semanas no teste)."
        ),
    ]
    linhas += [
        "",
        "## Coeficientes (modelo final, todas as semanas)",
        "",
        (
            "Pontos percentuais de desvio por unidade da variável. IC95 por bootstrap de blocos de "
            f"{BLOCO_SEMANAS} semanas ({N_BOOTSTRAP} reamostras). Anomalia de temperatura em °C e de "
            "chuva em mm/dia contra a média 2016-2020 da mesma semana do ano."
        ),
        "",
        "| variável | coef | IC95 |",
        "|---|---|---|",
    ]
    for r in coeficientes.itertuples():
        linhas.append(f"| {r.variavel} | {r.coef:.2f} | [{r.ic95_inf:.2f}; {r.ic95_sup:.2f}] |")
    linhas += [
        "",
        "## Alertas por UF",
        "",
        "| UF | nível | variação anual (4 sem.) | faixa de ruído | próxima janela | fatores (4 sem.) |",
        "|---|---|---|---|---|---|",
    ]
    for a in saida["alertas"]:
        t = a["tendencia"]
        if t is None:
            linhas.append(f"| {a['sigla']} | {a['nivel']} | – | – | – | {a['motivo']} |")
            continue
        fatores = ", ".join(
            f"{f['descricao']} {_pct(f['pontos_pct'])}" for f in a["fatores_4s"][:3]
        )
        linhas.append(
            f"| {a['sigla']} | {a['nivel']} | {_pct(t['variacao_anual_pct'])} | "
            f"±{t['faixa_ruido_pct']:.1f}% | {_pct(a['proxima_janela']['desvio_previsto_pct'])} | "
            f"{fatores} |"
        )
    linhas += [
        "",
        "## Regras do alerta",
        "",
        (
            "1. Subindo (ou próxima janela prevista acima da faixa de ruído) e município de "
            "potencial alto no Mapa → oportunidade."
        ),
        "2. Subindo sem município de potencial alto → observar.",
        "3. Caindo → atenção.",
        "4. Resto → estável.",
        "",
        "## Limitações",
        "",
        (
            "- Índice do Trends é relativo dentro de cada UF e amostral (dois pedidos iguais diferem "
            "um pouco); a coleta fica versionada."
        ),
        (
            '- "pizza" mistura receita e delivery; a categoria Comida e bebida tira buscas '
            'fora de comida (ex.: "fantasia de pizza" no Carnaval de 2025).'
        ),
        "- Clima da capital representa a UF inteira (hipótese).",
        "- Só feriados nacionais (BrasilAPI); estaduais e municipais ficam de fora.",
        "- Associação, não causa. O alerta não afirma que uma ativação vende mais.",
        "",
    ]
    return "\n".join(linhas)


def main(destino: Path = PROCESSED_DIR) -> None:
    saida, coeficientes = montar()
    ARQUIVO_ALERTAS.parent.mkdir(parents=True, exist_ok=True)
    ARQUIVO_ALERTAS.write_text(json.dumps(saida, ensure_ascii=False, indent=1))
    destino.mkdir(parents=True, exist_ok=True)
    (destino / ARQUIVO_RELATORIO.name).write_text(relatorio(saida, coeficientes))
    print(ARQUIVO_ALERTAS)
    print(destino / ARQUIVO_RELATORIO.name)


if __name__ == "__main__":
    main()
