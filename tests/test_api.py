"""Testes da API (FastAPI TestClient) com os artefatos versionados em models/."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from starlette.exceptions import StarletteDeprecationWarning

with warnings.catch_warnings():
    # O Starlette desta versão recomenda o httpx2 no TestClient; o httpx 0.28 segue funcionando
    warnings.simplefilter("ignore", StarletteDeprecationWarning)
    from fastapi.testclient import TestClient

from src.decision_engine.regras import CATEGORIAS, REGRAS
from src.models.preparo import FEATURES, UFS

MODELOS = Path(__file__).resolve().parents[1] / "models"
pytestmark = pytest.mark.skipif(
    not all((MODELOS / f).exists() for f in ("model_metadata.json", "explicacao.json",
                                             "contexto_decisao.json")),
    reason="artefatos do modelo não gerados")
CAMPOS = {"occasion_score", "classification", "top_factors", "recommendation_category",
          "recommendation_reason", "model_metadata"}


def contexto(**mudancas) -> dict:
    base = {"periodo": "noite", "dia_semana": "Domingo", "uf": "SP", "urbano": True, "idade": 25,
            "sexo": "feminino", "anos_estudo": 12, "n_moradores": 3, "renda_pc": 4000.0}
    return {**base, **mudancas}


@pytest.fixture(scope="module")
def cliente():
    from src.api.main import app

    with TestClient(app) as c:
        yield c


def test_raiz_leva_a_documentacao(cliente):
    r = cliente.get("/", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/docs"


def test_health_e_modelo(cliente):
    saude = cliente.get("/health").json()
    assert saude["status"] == "ok" and len(saude["model_file_sha256"]) == 64
    modelo = cliente.get("/model").json()
    assert 0 < modelo["classification_cuts"]["media"] < modelo["classification_cuts"]["alta"] < 1


def test_predict_devolve_os_6_campos_do_contrato(cliente):
    r = cliente.post("/predict", json=contexto(canal="delivery"))
    assert r.status_code == 200
    corpo = r.json()
    assert set(corpo) == CAMPOS
    assert 0 < corpo["occasion_score"] < 1
    assert corpo["classification"] in ("alta", "media", "baixa")
    assert len(corpo["top_factors"]) == 3
    assert corpo["recommendation_category"] in CATEGORIAS
    assert corpo["model_metadata"]["decision_rule"] in {r for r, _, _ in REGRAS}
    assert corpo["model_metadata"]["model_version"] == "1.0.0"


def test_score_da_api_e_o_do_modelo_congelado(cliente):
    """A API aplica o mesmo preparo do treino: o score bate com o do arquivo do modelo."""
    from src.models.artefato import carregar

    modelo, _ = carregar(MODELOS)
    casos = [contexto(), contexto(periodo="manha", dia_semana="Terca-Feira", uf="BA",
                                  urbano=False, idade=65, renda_pc=300.0),
             contexto(periodo="almoco", dia_semana="Sabado", uf="RS", sexo="masculino",
                      anos_estudo=0, n_moradores=7, renda_pc=0.0)]
    sigla_para_codigo = {s: c for c, s in UFS.items()}
    for caso in casos:
        linha = {**{k: caso[k] for k in FEATURES if k != "UF"},
                 "UF": sigla_para_codigo[caso["uf"]], "urbano": int(caso["urbano"])}
        esperado = modelo.predict_proba(pd.DataFrame([linha])[FEATURES])[0, 1]
        obtido = cliente.post("/predict", json=caso).json()["occasion_score"]
        assert obtido == pytest.approx(esperado, rel=1e-12)


def test_todas_as_ufs_e_dias(cliente):
    for uf in UFS.values():
        assert cliente.post("/predict", json=contexto(uf=uf)).status_code == 200
    for dia in ("Segunda-Feira", "Quarta-Feira", "Sexta-Feira", "Sabado"):
        assert cliente.post("/predict", json=contexto(dia_semana=dia)).status_code == 200


@pytest.mark.parametrize("mudanca", [
    {"periodo": "madrugada"}, {"dia_semana": "Sábado"}, {"uf": "XX"}, {"idade": 9},
    {"anos_estudo": 17}, {"n_moradores": 0}, {"renda_pc": -1}, {"renda_pc": 300_000},
    {"canal": "tv"},
    {"associacao_marca": "alta"}, {"campo_extra": 1},
])
def test_entrada_invalida_da_422(cliente, mudanca):
    assert cliente.post("/predict", json=contexto(**mudanca)).status_code == 422


def test_campo_obrigatorio_faltando_da_422(cliente):
    corpo = contexto()
    del corpo["renda_pc"]
    assert cliente.post("/predict", json=corpo).status_code == 422


def test_versao_do_modelo(cliente):
    assert cliente.post("/predict", json=contexto(model_version="1.0.0")).status_code == 200
    r = cliente.post("/predict", json=contexto(model_version="9.9.9"))
    assert r.status_code == 409 and "1.0.0" in r.json()["detail"]


def test_recomendacoes_de_ponta_a_ponta(cliente):
    alta = cliente.post("/predict", json=contexto(canal="delivery")).json()
    assert alta["classification"] == "alta"
    assert alta["recommendation_category"] == "ativacao_delivery"
    baixa = cliente.post("/predict", json=contexto(
        periodo="manha", dia_semana="Terca-Feira", uf="BA", urbano=False, idade=65,
        anos_estudo=2, renda_pc=300.0)).json()
    assert baixa["classification"] == "baixa"
    assert baixa["recommendation_category"] == "sem_ativacao_dedicada"
    awareness = cliente.post("/predict", json=contexto(associacao_marca="baixa")).json()
    assert awareness["recommendation_category"] == "awareness_associacao"


def test_regras_em_ordem(cliente):
    regras = cliente.get("/rules").json()
    assert [r["rule"] for r in regras["rules"]] == ["R1", "R2", "R3", "R4", "R5", "R6"]
    assert set(regras["categories"]) == set(CATEGORIAS)


def test_fatores_somam_o_logit_do_score(cliente):
    from src.models.artefato import carregar
    from src.models.calibracao import logit
    from src.models.explicar import carregar as carregar_explicador

    modelo, meta = carregar(MODELOS)
    explicador = carregar_explicador(modelo, meta, MODELOS)
    corpo = cliente.post("/predict", json=contexto()).json()
    # os 3 fatores da resposta são parte da soma completa dos 8 fatores
    sigla_para_codigo = {s: c for c, s in UFS.items()}
    caso = contexto()
    linha = {**{k: caso[k] for k in FEATURES if k != "UF"}, "UF": sigla_para_codigo["SP"],
             "urbano": 1}
    contrib = explicador.contribuicoes(pd.DataFrame([linha])[FEATURES]).iloc[0]
    assert explicador.log_odds_referencia + contrib.sum() == pytest.approx(
        float(logit(np.array([corpo["occasion_score"]]))[0]), abs=1e-8)
    for f in corpo["top_factors"]:
        assert f["log_odds"] == pytest.approx(contrib[f["factor"]], abs=1e-4)


def test_mapa_de_oportunidade(cliente):
    r = cliente.get("/oportunidade")
    assert r.status_code == 200
    mapa = r.json()
    assert len(mapa["publicos"]) == 480 and len(mapa["estados"]) == 27
    # o mapa é do mesmo modelo que a API carregou
    assert mapa["modelo"]["model_file_sha256"] == cliente.get("/health").json()[
        "model_file_sha256"]
    assert {"chance", "parcela_das_ocasioes", "prioridade", "anuncio"} <= set(mapa["publicos"][0])
    assert {"uf", "categoria", "pizzarias_100mil"} <= set(mapa["estados"][0])
