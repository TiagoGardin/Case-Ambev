"""Testes do Decision Engine: regras, justificativas e contexto observado."""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.decision_engine import contexto as ctx
from src.decision_engine.regras import (
    ASSOCIACOES,
    CANAIS,
    CATEGORIAS,
    REGRAS,
    recomendar,
)
from src.models.artefato import CLASSES

RAIZ = Path(__file__).resolve().parents[1]
MODELOS = RAIZ / "models"

MEDIDA = {"participacao": 0.6, "ic95": [0.5, 0.7]}
CONTEXTO = {"brasil": {m: MEDIDA for m in ctx.MEDIDAS},
            "regioes": {"Sudeste": {m: MEDIDA for m in ctx.MEDIDAS}}}
TOP = [{"factor": "momento", "value": "noite de sexta a domingo", "effect": "aumenta",
        "odds_multiplier": 3.4, "log_odds": 1.22},
       {"factor": "renda", "value": "R$ 4.000 por pessoa ao mês", "effect": "aumenta",
        "odds_multiplier": 1.8, "log_odds": 0.6},
       {"factor": "idade", "value": "60 anos ou mais", "effect": "reduz",
        "odds_multiplier": 0.7, "log_odds": -0.36}]


def rec(classe: str, canal=None, associacao=None, score: float = 0.02):
    return recomendar(score, classe, TOP, "Sudeste", CONTEXTO, 0.003, canal, associacao)


@pytest.mark.parametrize("classe, canal, associacao, regra, categoria", [
    ("baixa", None, None, "R1", "sem_ativacao_dedicada"),
    ("baixa", "delivery", "baixa", "R1", "sem_ativacao_dedicada"),   # R1 vem antes de R2
    ("alta", "delivery", "baixa", "R2", "awareness_associacao"),     # R2 vem antes de R3
    ("media", None, "baixa", "R2", "awareness_associacao"),
    ("alta", "delivery", None, "R3", "ativacao_delivery"),
    ("alta", "delivery", "normal", "R3", "ativacao_delivery"),
    ("alta", "estabelecimento", None, "R4", "ativacao_ponto_de_consumo"),
    ("alta", "varejo", None, "R5", "combo_consumo_em_casa"),
    ("alta", None, None, "R5", "combo_consumo_em_casa"),
    ("media", "delivery", None, "R6", "presenca_de_marca"),
    ("media", None, "normal", "R6", "presenca_de_marca"),
])
def test_cada_regra_e_sua_categoria(classe, canal, associacao, regra, categoria):
    r = rec(classe, canal, associacao)
    assert (r.regra, r.categoria) == (regra, categoria)


def test_toda_combinacao_cai_numa_regra_documentada():
    categoria_da_regra = {regra: categoria for regra, _, categoria in REGRAS}
    assert set(categoria_da_regra.values()) == set(CATEGORIAS)
    usadas = set()
    for classe, canal, associacao in product(CLASSES, (*CANAIS, None), (*ASSOCIACOES, None)):
        r = rec(classe, canal, associacao)
        assert categoria_da_regra[r.regra] == r.categoria
        usadas.add(r.regra)
    assert usadas == set(categoria_da_regra)


def test_justificativa_traz_score_fatores_e_contexto():
    r = rec("alta", "delivery", score=0.0288)
    assert "2,88%" in r.justificativa and "9,6 vezes a chance média" in r.justificativa
    assert "a prioridade é alta" in r.justificativa
    assert "noite de sexta a domingo" in r.justificativa and "× 3,4)" in r.justificativa
    assert "× 0,70)" in r.justificativa  # fator que reduz: duas casas
    assert "60% das ocasiões" in r.justificativa
    assert "faixa provável de 50% a 70%; no Brasil, 60%" in r.justificativa
    assert "inclui delivery" in r.justificativa


@pytest.mark.parametrize("score, trecho", [(0.009, "3,0 vezes a chance média"),
                                           (0.00006, "2% da chance média"),
                                           (0.000001, "menos de 1% da chance média")])
def test_justificativa_compara_com_a_chance_media(score, trecho):
    assert trecho in rec("baixa", score=score).justificativa


def test_justificativa_fala_a_lingua_de_quem_decide():
    """Sem jargão de dados: quem lê é quem decide marketing ("classe" lembra classe social)."""
    jargao = ("score", "classe", "ic 95", "treino", "pof", "log-odds", "shap", "modelo")
    for classe, canal, associacao in product(CLASSES, (*CANAIS, None), (*ASSOCIACOES, None)):
        texto = rec(classe, canal, associacao).justificativa.lower()
        assert not any(p in texto for p in jargao), texto


def test_justificativa_nao_afirma_efeito_causal():
    proibidas = ("vender mais", "aumenta as vendas", "aumentará", "vai vender", "garante",
                 "incremental", "causa")
    for classe, canal, associacao in product(CLASSES, (*CANAIS, None), (*ASSOCIACOES, None)):
        texto = rec(classe, canal, associacao).justificativa.lower()
        assert not any(p in texto for p in proibidas), texto


@pytest.mark.parametrize("mudanca", [{"classe": "altissima"}, {"canal": "tv"},
                                     {"associacao": "alta"}, {"regiao": "Marte"}])
def test_entrada_invalida_da_erro_claro(mudanca):
    args = {"classe": "alta", "canal": None, "associacao": None, "regiao": "Sudeste", **mudanca}
    with pytest.raises(ValueError):
        recomendar(0.02, args["classe"], TOP, args["regiao"], CONTEXTO, 0.003, args["canal"],
                   args["associacao"])


def test_contexto_calcula_participacoes_ponderadas():
    rng = np.random.default_rng(0)
    n = 400
    treino = pd.DataFrame({
        "y": np.r_[np.ones(300, int), np.zeros(100, int)],
        "local_pizza": pd.Categorical(rng.choice(["casa", "estabelecimento", "outro_fora"], n),
                                      ["casa", "estabelecimento", "outro_fora"]),
        "y_guarana": rng.integers(0, 2, n),
        "peso_semana": rng.uniform(1, 3, n),
        "COD_UPA": rng.integers(0, 60, n),
        "regiao": rng.choice(["Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste"], n),
    })
    c = ctx.calcular(treino)
    br = c["brasil"]
    assert br["ocasioes"] == 300
    assert sum(br[m]["participacao"] for m in ("casa", "estabelecimento", "outro_fora")) == (
        pytest.approx(1.0))
    oc = treino[treino["y"] == 1]
    esperado = np.average(oc["local_pizza"] == "casa", weights=oc["peso_semana"])
    assert br["casa"]["participacao"] == pytest.approx(esperado)
    for medida in ctx.MEDIDAS:
        lo, hi = br[medida]["ic95"]
        assert lo <= br[medida]["participacao"] <= hi
    assert sum(r["ocasioes"] for r in c["regioes"].values()) == 300
    assert 0 < br["ocasioes_efetivas"] <= 300


@pytest.mark.skipif(not (MODELOS / ctx.ARQUIVO).exists(), reason="contexto não gerado")
def test_casa_e_o_local_mais_comum_em_todas_as_regioes():
    """A justificativa da regra R5 sem canal diz isso; o teste avisa se os dados mudarem."""
    contexto = ctx.carregar(MODELOS)
    for regiao, c in contexto["regioes"].items():
        casa = c["casa"]["participacao"]
        assert casa > c["estabelecimento"]["participacao"], regiao
        assert casa > c["outro_fora"]["participacao"], regiao
