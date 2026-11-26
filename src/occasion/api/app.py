"""API de inferência: contexto → Occasion Score, fatores e recomendação.

Camada de serviço sobre o modelo final (`occasion.modelagem.final`) e o
Decision Engine (`occasion.decisao.engine`). A API não calcula nada por
conta própria: valida a entrada, chama o engine e devolve a saída no
contrato do case (seção 8 do brief): `occasion_score`, `classification`,
`top_factors`, `recommendation_category`, `recommendation_reason` e
`model_metadata`, mais o detalhe da regra aplicada.

    uv run uvicorn occasion.api.app:app --reload

Documentação interativa em http://localhost:8000/docs. O modelo é lido uma
vez, na subida, de `OCCASION_MODEL_PATH` (padrão: o modelo versionado em
`models/`).
"""

from __future__ import annotations

import json
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from occasion.decisao import engine, fontes
from occasion.mapa.oportunidade import ARQUIVO_PONTOS_REFERENCIA
from occasion.modelagem import final
from occasion.modelagem import incerteza as inc
from occasion.modelagem.tabela import FAIXAS_IDADE, TURNOS
from occasion.radar import radar as trend_radar
from occasion.radar import trends as radar_trends

Turno = Literal[tuple(TURNOS)]
Regiao = Literal[tuple(engine.REGIOES)]
TipoArea = Literal[tuple(engine.TIPOS_AREA)]
FaixaIdade = Literal[tuple(FAIXAS_IDADE)]
Sexo = Literal[tuple(engine.SEXOS)]
Canal = Literal[tuple(engine.CANAIS)]


class ContextoEntrada(BaseModel):
    """Momento, lugar, público e canal do contexto a avaliar."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "turno": "noite",
                    "dia_semana": 6,
                    "regiao": "Sudeste",
                    "tipo_area": "capital",
                    "faixa_idade": "25-29",
                    "sexo": "mulher",
                    "renda_sm": 5,
                    "canal": "delivery",
                }
            ]
        },
    )

    turno: Turno = Field(description="manha (até 10h), almoco (até 14h), tarde (até 17h), noite")
    # strict: 6.0 ou "6" virariam outra categoria no modelo.
    dia_semana: Annotated[int, Field(strict=True, ge=1, le=7)] = Field(
        description="1 = segunda ... 7 = domingo"
    )
    regiao: Regiao
    tipo_area: TipoArea = Field(
        description="estrato da POF: capital, resto_rm (resto da região metropolitana), "
        "interior_urbano ou rural"
    )
    faixa_idade: FaixaIdade
    sexo: Sexo
    # strict: true ou "5" não podem virar número (o engine também recusa bool).
    renda_sm: Annotated[float, Field(strict=True, ge=0)] = Field(
        description="renda mensal do domicílio em salários mínimos"
    )
    canal: Canal | None = Field(
        default=None, description="canal da ativação; só muda o canal da regra 2"
    )
    versao_modelo: str | None = Field(
        default=None,
        description="versão do modelo esperada; se vier e for outra, a API recusa (409)",
    )

    @field_validator("renda_sm")
    @classmethod
    def _finita(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("renda_sm deve ser um número finito")
        return v


class Fator(BaseModel):
    feature: str
    rotulo: str
    valor: str | int | float
    contribuicao_logit: float = Field(description="SHAP linear no logit calibrado")
    razao_chances: float
    direcao: Literal["aumenta", "reduz"]


class MelhorMomento(BaseModel):
    dia_semana: int
    dia: str
    turno: str
    occasion_score: float
    occasion_score_ic95: list[float] | None = Field(
        description="IC95 do score (bootstrap por setor); vazio sem as réplicas"
    )
    classification: Literal["alta", "media", "baixa"]


class SuporteValor(BaseModel):
    feature: str
    valor: str = Field(description="código do modelo (dia 1 a 7, sexo 1 = homem, 2 = mulher)")
    ocasioes: int = Field(description="ocasiões reais (sem peso) nas duas edições da POF")
    linhas: int


class SuportePerfil(BaseModel):
    chave: dict[str, str]
    ocasioes: int
    linhas: int


class Suporte(BaseModel):
    perfil: SuportePerfil = Field(
        description="perfil = região × tipo de área × faixa de idade × sexo × classe de renda"
    )
    por_entrada: list[SuporteValor]


class Incerteza(BaseModel):
    disponivel: bool
    occasion_score_ic95: list[float] | None = Field(
        description="IC95 do score em 0 a 100: só a variação da amostra da POF"
    )
    prob_faixa: dict[Literal["alta", "media", "baixa"], float] | None = Field(
        description="fração das réplicas do bootstrap em cada faixa"
    )
    faixa_estavel: bool | None = Field(
        description="a faixa do score pontual aparece em pelo menos 80% das réplicas"
    )
    poucos_dados: bool | None
    avisos: list[str]
    suporte: Suporte | None
    metodo: str | None


class Fonte(BaseModel):
    id: str
    numero: str
    fonte: str
    ano: str
    url: str
    detalhe: str | None


class PosicaoClasse(BaseModel):
    classe: Literal["A", "B", "C", "D", "E"]
    percentil_na_classe: float | None = Field(
        description="posição do score entre as pessoas-turno da mesma classe em 2017-18 (0 a 100)"
    )
    topo_10_na_classe: bool | None = Field(
        description="score entre os 10% maiores da própria classe (mesmo corte da faixa alta)"
    )


class Recomendacao(BaseModel):
    occasion_score: float = Field(description="probabilidade calibrada da ocasião × 100 (0 a 100)")
    classification: Literal["alta", "media", "baixa"] = Field(
        description="faixa: alta = 10% maiores da população de 2017-18; média = 10% a 30%"
    )
    top_factors: list[Fator]
    recommendation_category: Literal[tuple(c.value for c in engine.Categoria)]
    recommendation_name: str
    recommendation_reason: str
    recommendation_channel: str | None
    messages: list[str]
    rule: int = Field(ge=1, le=5, description="regra do Decision Engine que decidiu")
    best_moment: MelhorMomento = Field(description="dia e turno de maior score para o mesmo perfil")
    class_position: PosicaoClasse = Field(
        description="posição dentro da classe de renda; não muda a regra, serve à expansão B a D"
    )
    uncertainty: Incerteza = Field(
        description="IC95, estabilidade da faixa e aviso de poucos dados; não muda a regra"
    )
    sources: list[Fonte] = Field(description="fonte, ano e link de cada número da resposta")
    limitations: list[str] = Field(description="o que o modelo não sabe")
    model_metadata: dict


def _caminho_modelo() -> Path:
    return Path(os.environ.get("OCCASION_MODEL_PATH", final.ARQUIVO_MODELO))


@asynccontextmanager
async def _ciclo(app: FastAPI):
    app.state.modelo = final.carregar(_caminho_modelo())
    app.state.incerteza = _ler_incerteza(_caminho_incerteza(), app.state.modelo)
    app.state.mapa = _ler_mapa(ARQUIVO_PONTOS_REFERENCIA)
    app.state.radar = _ler_radar(trend_radar.ARQUIVO_ALERTAS)
    app.state.radar_series = _ler_series(radar_trends.ARQUIVO)
    yield


def _caminho_incerteza() -> Path:
    return Path(os.environ.get("OCCASION_INCERTEZA_PATH", inc.ARQUIVO))


def _ler_incerteza(caminho: Path, modelo: final.OccasionScore) -> inc.Incerteza | None:
    """Réplicas do bootstrap (None se o arquivo não existir ou for de outra versão do modelo)."""
    if not caminho.exists():
        return None
    incerteza = inc.carregar(caminho)
    if incerteza.metadata.get("versao_modelo") != modelo.metadata.get("versao"):
        return None
    return incerteza


def _ler_mapa(caminho: Path) -> list[dict]:
    """Pontos do Mapa de Oportunidade (vazio se o arquivo não existir)."""
    if not caminho.exists():
        return []
    df = pd.read_csv(caminho).sort_values("prioridade")
    return df.astype(object).where(df.notna(), None).to_dict("records")


def _ler_radar(caminho: Path) -> dict | None:
    """Alertas do Trend Radar (None se o arquivo não existir)."""
    if not caminho.exists():
        return None
    return json.loads(caminho.read_text(encoding="utf-8"))


def _ler_series(caminho: Path) -> pd.DataFrame | None:
    if not caminho.exists():
        return None
    return pd.read_csv(caminho)


app = FastAPI(
    title="Occasion Intelligence: Pizza + Guaraná",
    version="1.0.0",
    description=(
        "Occasion Score (propensão de pizza com refrigerante no turno, POF/IBGE) e "
        "recomendação de ativação por regras documentadas. O score não decide a campanha "
        "e o sistema não afirma causalidade."
    ),
    lifespan=_ciclo,
)


def _modelo(request: Request) -> final.OccasionScore:
    return request.app.state.modelo


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "versao_modelo": _modelo(request).metadata.get("versao")}


@app.get("/model")
def model(request: Request) -> dict:
    """Metadados do modelo: algoritmo, treino, calibração, faixas, validação e incerteza."""
    incerteza = request.app.state.incerteza
    return {
        **_modelo(request).metadata,
        "incerteza": incerteza.metadata if incerteza else None,
        "limitacoes": fontes.LIMITES,
    }


@app.get("/opcoes")
def opcoes() -> dict:
    """Valores aceitos em cada campo da entrada (para montar formulários)."""
    return {
        "turno": list(TURNOS),
        "dia_semana": [{"valor": d, "dia": nome} for d, nome in engine.DIAS.items()],
        "regiao": engine.REGIOES,
        "tipo_area": engine.TIPOS_AREA,
        "faixa_idade": list(FAIXAS_IDADE),
        "sexo": list(engine.SEXOS),
        "canal": engine.CANAIS,
    }


@app.post("/predict", response_model=Recomendacao)
def predict(entrada: ContextoEntrada, request: Request) -> dict:
    """Occasion Score, os 3 fatores de maior efeito e a categoria de ativação, com justificativa."""
    modelo = _modelo(request)
    versao = modelo.metadata.get("versao")
    if entrada.versao_modelo is not None and entrada.versao_modelo != versao:
        raise HTTPException(
            status_code=409,
            detail=f"versão pedida {entrada.versao_modelo}, versão servida {versao}",
        )
    try:
        contexto = engine.Contexto(**entrada.model_dump(exclude={"versao_modelo"}))
    except ValueError as erro:  # regra do engine que o schema não cobre
        raise HTTPException(status_code=422, detail=str(erro)) from erro
    return engine.decidir(contexto, modelo, incerteza=request.app.state.incerteza).como_dict()


@app.get("/mapa")
def mapa(request: Request) -> list[dict]:
    """Ranking do Mapa de Oportunidade: municípios de 100 mil habitantes ou mais, em prioridade.

    Gerado por `occasion.mapa.oportunidade` (regra documentada aplicada ao território, não
    uma nova previsão).
    """
    pontos = request.app.state.mapa
    if not pontos:
        raise HTTPException(status_code=404, detail="mapa de oportunidade não encontrado")
    return pontos


def _radar(request: Request) -> dict:
    radar = request.app.state.radar
    if radar is None:
        raise HTTPException(status_code=404, detail="alertas do trend radar não encontrados")
    return radar


@app.get("/radar")
def radar(request: Request) -> dict:
    """Trend Radar: alerta por UF com tendência, fatores, próxima janela e contexto do Mapa.

    Gerado por `occasion.radar.radar` a partir do Google Trends (interesse, não consumo),
    feriados da BrasilAPI e clima do Open-Meteo. O nível do alerta é uma regra documentada.
    """
    return _radar(request)


@app.get("/radar/{sigla}")
def radar_uf(sigla: str, request: Request) -> dict:
    """Alerta de uma UF e a série semanal de buscas ("pizza" e "guaraná"), semanas fechadas."""
    sigla = sigla.upper()
    alerta = next((a for a in _radar(request)["alertas"] if a["sigla"] == sigla), None)
    if alerta is None:
        raise HTTPException(status_code=404, detail=f"UF desconhecida: {sigla}")
    series = request.app.state.radar_series
    serie: list[dict] = []
    if series is not None:
        # Só semanas fechadas: a semana em curso ainda é parcial no Trends.
        uf = series[(series["sigla"] == sigla) & ~series["parcial"].astype(bool)]
        serie = (
            uf.pivot_table(index="semana", columns="termo", values="indice")
            .reset_index()
            .to_dict("records")
        )
    return {"alerta": alerta, "serie": serie}
