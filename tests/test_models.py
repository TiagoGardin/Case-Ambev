"""Testes do preparo de features, dos modelos, das métricas e da validação.

Usam dados pequenos montados à mão. O teste de coerência com a grade é pulado se
ela não tiver sido gerada.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.build_grid import DIAS, PERIODOS
from src.models.metricas import captura_topo
from src.models.modelos import baseline, regressao_logistica
from src.models.preparo import CATEGORIAS, COLUNAS_PREPARADAS, FEATURES, preparar
from src.models.validacao import comparar, particoes, validar_cv

GRID = Path(__file__).resolve().parents[1] / "data" / "processed" / "grid.parquet"


def linha(**mudancas) -> dict:
    """Uma pessoa × dia × período válida; `mudancas` troca campos."""
    base = {"periodo": "noite", "dia_semana": "Sexta-Feira", "UF": "35", "urbano": 1,
            "idade": 25, "sexo": "feminino", "anos_estudo": 12, "n_moradores": 3,
            "renda_pc": 1500.0}
    return {**base, **mudancas}


def amostra(n: int, semente: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """n linhas com período, dia e idade sorteados e um target de 20%."""
    rng = np.random.default_rng(semente)
    X = pd.DataFrame([linha(periodo=str(rng.choice(PERIODOS)), dia_semana=str(rng.choice(DIAS)),
                            idade=int(rng.integers(10, 90))) for _ in range(n)])
    return X, (rng.random(n) < 0.2).astype(int)


# --------------------------------------------------------------------------
# Preparo
# --------------------------------------------------------------------------
def test_preparar_deriva_as_colunas():
    X = preparar(pd.DataFrame([
        linha(),
        linha(UF="43", idade=65, dia_semana="Terca-Feira", periodo="almoco", renda_pc=0.0),
    ]))
    assert list(X.columns) == COLUNAS_PREPARADAS
    assert X["regiao"].astype(str).tolist() == ["Sudeste", "Sul"]
    assert X["faixa_idade"].astype(str).tolist() == ["18_29", "60_mais"]
    assert X["periodo_dia"].astype(str).tolist() == ["Sexta-Feira|noite", "Terca-Feira|almoco"]
    assert X["fim_de_semana"].tolist() == [1, 0]
    assert X["log_renda_pc"].tolist() == pytest.approx([np.log1p(1500.0), 0.0])
    assert X["faixa_renda"].astype(str).tolist() == ["1_2sm", "ate_0.5sm"]


def test_preparar_uma_linha_tem_as_categorias_do_treino():
    """Uma requisição da API sai com as mesmas categorias do treino."""
    X = preparar(pd.DataFrame([linha()]))
    for col, categorias in CATEGORIAS.items():
        assert list(X[col].cat.categories) == list(categorias)


@pytest.mark.parametrize("mudanca, mensagem", [
    ({"UF": "99"}, "UF"),
    ({"periodo": "madrugada"}, "periodo"),
    ({"dia_semana": "sexta"}, "dia_semana"),
    ({"idade": 8}, "idade"),
    ({"renda_pc": -1.0}, "renda_pc"),
    ({"urbano": 2}, "urbano"),
    ({"sexo": None}, "nulos"),
])
def test_preparar_recusa_entrada_invalida(mudanca, mensagem):
    with pytest.raises(ValueError, match=mensagem):
        preparar(pd.DataFrame([linha(**mudanca)]))


def test_preparar_recusa_coluna_faltando():
    with pytest.raises(ValueError, match="renda_pc"):
        preparar(pd.DataFrame([linha()]).drop(columns="renda_pc"))


@pytest.mark.skipif(not GRID.exists(), reason="grade não gerada")
def test_preparar_reproduz_as_colunas_da_grade():
    """As colunas derivadas no preparo são as mesmas que a montagem da grade gravou."""
    derivadas = ["regiao", "faixa_idade", "faixa_renda", "fim_de_semana"]
    g = pd.read_parquet(GRID, columns=FEATURES + derivadas)
    X = preparar(g[FEATURES])
    for col in derivadas:
        assert (X[col].astype(str) == g[col].astype(str)).all(), col


# --------------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------------
def test_baseline_usa_a_taxa_ponderada_da_celula():
    domingo_noite = linha(periodo="noite", dia_semana="Domingo")
    domingo_manha = linha(periodo="manha", dia_semana="Domingo")
    X = pd.DataFrame([domingo_noite] * 4 + [domingo_manha] * 2)
    y = np.array([1, 0, 0, 0, 0, 0])
    peso = np.array([3.0, 1, 1, 1, 1, 1])
    m = baseline().fit(X, y, modelo__sample_weight=peso)

    novo = pd.DataFrame([domingo_noite, domingo_manha, linha(periodo="noite", dia_semana="Sabado")])
    p = m.predict_proba(novo)[:, 1]
    assert p[0] == pytest.approx(3 / 6)  # domingo à noite: 3 de peso positivo em 6
    assert p[1] == 0.0
    assert p[2] == pytest.approx(3 / 8)  # sábado à noite não estava no treino: taxa geral


def test_regressao_logistica_preve_uma_linha_de_uf_nao_vista():
    X, y = amostra(300)  # todas as linhas em SP
    m = regressao_logistica(C=1.0).fit(X, y)
    p = m.predict_proba(pd.DataFrame([linha(UF="11")]))[:, 1]
    assert 0 < p[0] < 1
    nomes = m[:-1].get_feature_names_out()
    assert len(nomes) == m[-1].coef_.shape[1]
    assert "cat__regiao_Norte" in nomes  # as categorias são fixas, não as do treino


# --------------------------------------------------------------------------
# Métricas e validação
# --------------------------------------------------------------------------
def test_captura_topo_divide_o_empate_no_corte():
    """4 linhas de peso 1 e 2 ocasiões; as duas primeiras empatam, com 1 ocasião."""
    y, score, peso = [1, 0, 1, 0], [0.9, 0.9, 0.5, 0.1], [1, 1, 1, 1]
    assert captura_topo(y, score, peso, 0.25) == pytest.approx(0.25)  # metade do empate
    assert captura_topo(y, score, peso, 0.50) == pytest.approx(0.50)
    assert captura_topo(y, score, peso, 0.75) == pytest.approx(1.0)


def test_captura_topo_conta_o_peso():
    """A ocasião tem peso 3 de 4: o top 50% do peso alcança um terço dela."""
    assert captura_topo([0, 1], [0.9, 0.1], [1, 3], 0.5) == pytest.approx(1 / 3)
    assert captura_topo([0, 1], [0.9, 0.1], [1, 1], 0.5) == 0.0


def test_particoes_mantem_cada_domicilio_de_um_lado():
    grupos = np.repeat(np.arange(40), 5)
    y = np.zeros(200, dtype=int)
    y[::9] = 1
    folds = particoes(y, grupos, k=5, repeticoes=2)
    assert len(folds) == 10
    for tr, va in folds:
        assert not set(grupos[tr]) & set(grupos[va])
    for r in range(2):  # em cada repetição, toda linha é validada uma vez
        validadas = np.concatenate([va for _, va in folds[5 * r: 5 * r + 5]])
        assert sorted(validadas) == list(range(200))


def test_validar_cv_preenche_as_previsoes_fora_da_amostra():
    X, y = amostra(400, semente=1)
    folds = particoes(y, np.repeat(np.arange(100), 4), k=5, repeticoes=2)
    r = validar_cv(baseline(), X, y, np.ones(400), folds, peso_ajuste=np.ones(400), n_jobs=1)
    assert r.oof.shape == (400, 2) and not np.isnan(r.oof).any()
    assert len(r.prauc) == len(r.captura10) == 10


def test_comparar_usa_a_variancia_corrigida():
    ref, cand = np.zeros(50), np.tile([0.01, 0.03], 25)
    r = comparar(ref, cand, k=5)
    se = np.sqrt((1 / 50 + 1 / 4) * cand.var(ddof=1))  # 1/n + 1/(k − 1)
    assert r["ganho"] == pytest.approx(0.02)
    assert r["t"] == pytest.approx(0.02 / se)
    assert r["passa"]
