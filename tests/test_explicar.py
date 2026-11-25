"""Testes do SHAP por fator (src.models.explicar).

O teste principal compara o SHAP calculado pela fórmula fechada com o valor de
Shapley exato, obtido pela definição: as 2^8 coalizões de fatores, com os fatores
de fora tirados de linhas de referência (SHAP intervencional).
"""

from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.build_grid import DIAS, PERIODOS
from src.models.artefato import carregar as carregar_modelo
from src.models.calibracao import ModeloCalibrado, logit
from src.models.explicar import (
    ENTRADAS,
    FATORES,
    Explicador,
    carregar,
    descrever,
    referencia,
    salvar,
    transformar,
)
from src.models.modelos import construir
from src.models.preparo import UFS, preparar

RAIZ = Path(__file__).resolve().parents[1]
MODELOS = RAIZ / "models"
CONFIG = {"modelo": "regressao_logistica", "C": 1.0, "dia": "fim_de_semana",
          "renda": "continua"}


def amostra(n: int, semente: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """n linhas com todas as entradas variando e um target que depende de várias delas."""
    rng = np.random.default_rng(semente)
    X = pd.DataFrame({
        "periodo": rng.choice(PERIODOS, n), "dia_semana": rng.choice(DIAS, n),
        "UF": rng.choice(list(UFS), n), "urbano": rng.integers(0, 2, n),
        "idade": rng.integers(10, 90, n), "sexo": rng.choice(["feminino", "masculino"], n),
        "anos_estudo": rng.integers(0, 17, n), "n_moradores": rng.integers(1, 8, n),
        "renda_pc": rng.gamma(2.0, 800.0, n),
    })
    chance = 0.05 + 0.25 * (X["periodo"] == "noite") + 0.1 * X["urbano"]
    return X, (rng.random(n) < chance).astype(int)


@pytest.fixture(scope="module")
def ajustado():
    X, y = amostra(600)
    modelo = ModeloCalibrado(construir(CONFIG).fit(X, y), a=1.7, b=0.3)
    return modelo, X


def log_odds(modelo, X: pd.DataFrame) -> np.ndarray:
    """Log-odds do score calibrado, sem o corte numérico de calibracao.logit."""
    return modelo.a * modelo.pipeline.decision_function(X) + modelo.b


def shapley_exato(modelo, x: pd.Series, fundo: pd.DataFrame) -> dict[str, float]:
    """Valor de Shapley de cada fator pela definição, com `fundo` como referência."""
    fatores = list(FATORES)
    n = len(fatores)

    def valor(coalizao: tuple[str, ...]) -> float:
        hibrido = fundo.copy()
        for f in coalizao:
            for entrada in ENTRADAS[f]:
                hibrido[entrada] = x[entrada]
        return float(log_odds(modelo, hibrido).mean())

    cache = {S: valor(S) for r in range(n + 1) for S in combinations(fatores, r)}
    phi = {}
    for f in fatores:
        outros = [g for g in fatores if g != f]
        total = 0.0
        for r in range(n):
            peso = factorial(r) * factorial(n - r - 1) / factorial(n)
            for S in combinations(outros, r):
                com_f = tuple(sorted(S + (f,), key=fatores.index))
                total += peso * (cache[com_f] - cache[S])
        phi[f] = total
    return phi


def test_contribuicoes_somam_o_logit_do_score(ajustado):
    modelo, X = ajustado
    explicador = Explicador(modelo, referencia(modelo, X, np.ones(len(X))))
    contrib = explicador.contribuicoes(X)
    assert list(contrib.columns) == list(FATORES)
    np.testing.assert_allclose(explicador.log_odds_referencia + contrib.sum(axis=1),
                               logit(modelo.predict_proba(X)[:, 1]), atol=1e-8)


def test_shap_por_fator_e_o_valor_de_shapley_exato(ajustado):
    modelo, X = ajustado
    fundo = X.iloc[:25].reset_index(drop=True)
    explicador = Explicador(modelo, referencia(modelo, fundo, np.ones(len(fundo))))
    contrib = explicador.contribuicoes(X.iloc[100:103])
    for i in range(3):
        exato = shapley_exato(modelo, X.iloc[100 + i], fundo)
        np.testing.assert_allclose(contrib.iloc[i].to_numpy(),
                                   [exato[f] for f in FATORES], atol=1e-9)


def test_shap_bate_com_a_biblioteca_shap(ajustado):
    shap = pytest.importorskip("shap")
    modelo, X = ajustado
    fundo = transformar(modelo.pipeline, X)
    lr = modelo.pipeline.named_steps["modelo"]
    mascara = shap.maskers.Independent(fundo, max_samples=len(fundo))  # referência inteira
    valores = shap.LinearExplainer((lr.coef_[0], lr.intercept_[0]), mascara).shap_values(fundo[:5])
    explicador = Explicador(modelo, fundo.mean(axis=0))
    por_fator = pd.DataFrame(valores).T.groupby(np.array(
        [next(f for f, cols in FATORES.items() if o in cols) for o in explicador.origem])).sum().T
    np.testing.assert_allclose(explicador.contribuicoes(X.iloc[:5]).to_numpy(),
                               modelo.a * por_fator[list(FATORES)].to_numpy(), atol=1e-9)


def test_top_fatores_ordena_pelo_tamanho_da_contribuicao(ajustado):
    modelo, X = ajustado
    explicador = Explicador(modelo, referencia(modelo, X, np.ones(len(X))))
    tops = explicador.top_fatores(X.iloc[:20], k=3)
    contrib = explicador.contribuicoes(X.iloc[:20])
    for top, (_, c) in zip(tops, contrib.iterrows(), strict=True):
        assert [t["factor"] for t in top] == list(c.abs().sort_values(ascending=False).index[:3])
        for t in top:
            assert set(t) == {"factor", "value", "effect", "odds_multiplier", "log_odds"}
            assert t["effect"] == ("aumenta" if c[t["factor"]] > 0 else "reduz")
            assert t["odds_multiplier"] == pytest.approx(np.exp(c[t["factor"]]), abs=1e-3)


def test_descrever_usa_o_que_o_modelo_enxerga():
    p = preparar(pd.DataFrame([{
        "periodo": "noite", "dia_semana": "Sabado", "UF": "35", "urbano": 1, "idade": 25,
        "sexo": "feminino", "anos_estudo": 16, "n_moradores": 1, "renda_pc": 1500.0}])).iloc[0]
    assert descrever("momento", p) == "noite de sexta a domingo"
    assert descrever("regiao", p) == "Sudeste"
    assert descrever("idade", p) == "18 a 29 anos"
    assert descrever("escolaridade", p) == "16 anos ou mais de estudo"
    assert descrever("moradores", p) == "1 morador"
    assert descrever("renda", p) == "R$ 1.500 por pessoa ao mês, a preços de jan/2018"
    assert descrever("area", p) == "urbana"


def test_carregar_recusa_referencia_de_outro_modelo(ajustado, tmp_path):
    modelo, X = ajustado
    explicador = Explicador(modelo, referencia(modelo, X, np.ones(len(X))))
    salvar(explicador, {"model_file_sha256": "a" * 64}, "teste", tmp_path)
    np.testing.assert_allclose(
        carregar(modelo, {"model_file_sha256": "a" * 64}, tmp_path).media, explicador.media)
    with pytest.raises(ValueError, match="outro arquivo de modelo"):
        carregar(modelo, {"model_file_sha256": "b" * 64}, tmp_path)


@pytest.mark.skipif(not (MODELOS / "explicacao.json").exists(),
                    reason="referência do SHAP não gerada")
def test_explicacao_do_modelo_final():
    modelo, meta = carregar_modelo(MODELOS)
    explicador = carregar(modelo, meta, MODELOS)
    X = pd.DataFrame([{"periodo": "noite", "dia_semana": "Domingo", "UF": "35", "urbano": 1,
                       "idade": 25, "sexo": "feminino", "anos_estudo": 12, "n_moradores": 3,
                       "renda_pc": 1500.0}])
    contrib = explicador.contribuicoes(X)
    assert explicador.log_odds_referencia + contrib.sum(axis=1).iloc[0] == pytest.approx(
        logit(modelo.predict_proba(X)[:, 1])[0], abs=1e-8)
    assert len(explicador.top_fatores(X)[0]) == 3
