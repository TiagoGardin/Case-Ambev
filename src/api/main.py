"""API do Occasion Score (FastAPI): contexto → score, classe, fatores e recomendação.

- POST /predict: um contexto (pessoa × dia × período, com canal e associação à marca
  opcionais) → os 6 campos do contrato: occasion_score, classification, top_factors,
  recommendation_category, recommendation_reason e model_metadata.
- GET /health: se os artefatos carregaram.
- GET /model: metadados do modelo congelado.
- GET /rules: categorias e regras do Decision Engine, na ordem em que são avaliadas.
- GET /oportunidade: o Mapa de Oportunidade (públicos-momento e estados), gerado só com o
  treino e os dados externos versionados (docs/oportunidade.md).
- GET /: redireciona para /docs, a documentação interativa.

Rodar: `uvicorn src.api.main:app --reload` (documentação interativa em /docs).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from src.api.servico import UFS, metadados_publicos, oportunidade, pontuar, recursos
from src.decision_engine.regras import CATEGORIAS, REGRAS
from src.features.build_grid import DIAS, PERIODOS

SIGLAS = tuple(sorted(UFS.values()))
RENDA_MAXIMA = 250_000.0  # R$ de jan/2018; a maior renda per capita da POF é R$ 235.192


class Contexto(BaseModel):
    """Contexto a pontuar: quando, onde, quem, e o que o Decision Engine precisa saber."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{
        "periodo": "noite", "dia_semana": "Domingo", "uf": "SP", "urbano": True, "idade": 25,
        "sexo": "feminino", "anos_estudo": 12, "n_moradores": 3, "renda_pc": 4000.0,
        "canal": "delivery", "associacao_marca": None, "model_version": "1.0.0"}]})

    # Quando
    periodo: Literal[tuple(PERIODOS)] = Field(
        description="manha (5h–10h), almoco (11h–14h), tarde (15h–17h) ou noite (18h–4h)")
    dia_semana: Literal[tuple(DIAS)]
    # Onde
    uf: Literal[SIGLAS] = Field(description="Sigla da UF; o modelo usa a região")
    urbano: bool
    # Quem
    idade: int = Field(ge=10, le=110, description="A POF só registra o consumo de quem tem 10+")
    sexo: Literal["feminino", "masculino"]
    anos_estudo: int = Field(ge=0, le=16, description="16 = 16 ou mais")
    n_moradores: int = Field(ge=1, le=20)
    renda_pc: float = Field(
        ge=0, le=RENDA_MAXIMA,
        description="Renda mensal per capita da família, em R$ de janeiro de 2018 (a base de "
                    "preços da POF; um valor de hoje precisa ser descontado da inflação). O "
                    "limite fica logo acima do maior valor da POF (R$ 235 mil): além dele, o "
                    "modelo extrapolaria")
    # Para o Decision Engine (não entram no modelo)
    canal: Literal["delivery", "varejo", "estabelecimento"] | None = Field(
        default=None, description="Canal da ativação, se já definido")
    associacao_marca: Literal["baixa", "normal"] | None = Field(
        default=None, description="Associação atual à marca no segmento, se conhecida")
    # Versão
    model_version: str | None = Field(
        default=None, description="Se informada, precisa ser a versão do modelo carregado")


class Fator(BaseModel):
    factor: str
    value: str
    effect: Literal["aumenta", "reduz", "neutro"]
    odds_multiplier: float = Field(description="Quanto o fator multiplica a chance da ocasião, "
                                               "em relação ao contexto médio do treino")
    log_odds: float


class Resposta(BaseModel):
    occasion_score: float = Field(ge=0, le=1)
    classification: Literal["alta", "media", "baixa"]
    top_factors: list[Fator]
    recommendation_category: Literal[tuple(CATEGORIAS)]
    recommendation_reason: str
    model_metadata: dict


@asynccontextmanager
async def ciclo(app: FastAPI):
    recursos()  # carrega e confere os artefatos na subida, não na primeira requisição
    yield


app = FastAPI(title="Occasion Intelligence: Pizza + Guaraná",
              description="Occasion Score da ocasião pizza com refrigerante (POF 2017-2018) e "
                          "recomendação de ativação por regras documentadas.",
              version="1.0.0", lifespan=ciclo)


@app.get("/", include_in_schema=False)
def raiz() -> RedirectResponse:
    """A raiz leva à documentação interativa; sem isso, o navegador mostraria "Not Found"."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def saude() -> dict:
    meta = recursos().metadados
    return {"status": "ok", "model_version": meta["model_version"],
            "model_file_sha256": meta["model_file_sha256"]}


@app.get("/model")
def modelo() -> dict:
    return metadados_publicos(recursos().metadados)


@app.get("/rules")
def regras() -> dict:
    return {"categories": CATEGORIAS,
            "rules": [{"rule": r, "when": quando, "category": c} for r, quando, c in REGRAS],
            "order": "a primeira regra que se aplica decide"}


@app.get("/oportunidade")
def mapa_de_oportunidade() -> dict:
    """Públicos-momento e estados priorizados pela regra de docs/oportunidade.md."""
    try:
        return oportunidade()
    except FileNotFoundError as erro:
        raise HTTPException(status_code=503, detail="models/oportunidade.json não foi gerado "
                            "(python -m src.oportunidade.ranking)") from erro


@app.post("/predict", response_model=Resposta)
def prever(contexto: Contexto) -> dict:
    rec = recursos()
    versao = rec.metadados["model_version"]
    if contexto.model_version is not None and contexto.model_version != versao:
        raise HTTPException(status_code=409, detail=f"model_version {contexto.model_version!r} "
                                                    f"pedida, mas a carregada é {versao!r}")
    return pontuar(contexto.model_dump(), rec)
