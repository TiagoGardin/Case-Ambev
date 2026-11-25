"""Testes do preparo de features, dos modelos, das métricas, da validação, da
calibração e do arquivo do modelo final.

Usam dados pequenos montados à mão. Os testes com a grade e com o modelo final
gravado são pulados se eles não tiverem sido gerados.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.build_grid import DIAS, PERIODOS
from src.models.artefato import carregar, classificar, salvar
from src.models.calibracao import (
    ModeloCalibrado,
    ajustar_sigmoide,
    aplicar_sigmoide,
    calibrar_fora_do_fold,
    confiabilidade,
)
from src.models.metricas import captura_topo, percentil_ponderado
from src.models.modelos import (
    BINARIAS_LR,
    CATEGORICAS_LR,
    NUMERICAS_LR,
    baseline,
    colunas_lr,
    construir,
    regressao_logistica,
)
from src.models.preparo import CATEGORIAS, COLUNAS_PREPARADAS, FEATURES, preparar
from src.models.validacao import comparar, particoes, validar_cv

RAIZ = Path(__file__).resolve().parents[1]
GRID = RAIZ / "data" / "processed" / "grid.parquet"
MODELOS = RAIZ / "models"


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
    assert X["periodo_fds"].astype(str).tolist() == ["sex-dom|noite", "seg-qui|almoco"]
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


def test_validar_cv_normaliza_o_peso_do_ajuste():
    """Multiplicar o peso do ajuste por uma constante não muda a Regressão Logística."""
    X, y = amostra(400, semente=2)
    folds = particoes(y, np.repeat(np.arange(100), 4), k=5, repeticoes=1)
    peso = np.random.default_rng(0).uniform(0.5, 2, 400)
    a = validar_cv(regressao_logistica(0.1), X, y, np.ones(400), folds, peso_ajuste=peso, n_jobs=1)
    b = validar_cv(regressao_logistica(0.1), X, y, np.ones(400), folds,
                   peso_ajuste=1000 * peso, n_jobs=1)
    np.testing.assert_allclose(a.oof, b.oof, rtol=1e-6)


def test_percentil_ponderado():
    assert percentil_ponderado([4, 1, 3, 2], [1, 1, 1, 1], 0.5) == 2
    assert percentil_ponderado([1, 2, 3, 4], [1, 1, 1, 5], 0.5) == 4  # o 4 tem mais da metade
    assert percentil_ponderado([1, 2, 3, 4], [1, 1, 1, 1], 1.0) == 4


# --------------------------------------------------------------------------
# Alternativas do Data Card e LightGBM
# --------------------------------------------------------------------------
def test_colunas_lr_alternativas_do_data_card():
    assert colunas_lr() == {"categoricas": CATEGORICAS_LR, "numericas": NUMERICAS_LR,
                            "binarias": BINARIAS_LR}
    fds = colunas_lr("fim_de_semana")
    assert "periodo_fds" in fds["categoricas"] and "fim_de_semana" in fds["binarias"]
    assert not {"dia_semana", "periodo_dia"} & set(fds["categoricas"])
    faixas = colunas_lr(renda="faixas")
    assert "faixa_renda" in faixas["categoricas"] and "log_renda_pc" not in faixas["numericas"]
    with pytest.raises(ValueError, match="alternativa"):
        colunas_lr("mes")


@pytest.mark.parametrize("config", [
    {"modelo": "regressao_logistica", "C": 1.0, "dia": "fim_de_semana", "renda": "faixas"},
    {"modelo": "lightgbm", "num_leaves": 4, "min_child_samples": 5},
])
def test_construir_preve_uma_linha(config):
    X, y = amostra(300)
    m = construir(config).fit(X, y)
    p = m.predict_proba(pd.DataFrame([linha(UF="11", dia_semana="Domingo")]))[:, 1]
    assert 0 < p[0] < 1


# --------------------------------------------------------------------------
# Calibração
# --------------------------------------------------------------------------
def test_sigmoide_recupera_a_e_b():
    rng = np.random.default_rng(0)
    score = rng.uniform(0.01, 0.5, 100_000)
    y = (rng.random(100_000) < aplicar_sigmoide(score, 1.3, 0.4)).astype(int)
    a, b = ajustar_sigmoide(score, y, np.ones_like(score))
    assert a == pytest.approx(1.3, abs=0.1)
    assert b == pytest.approx(0.4, abs=0.1)


def test_sigmoide_ponderada_acerta_a_taxa_ponderada():
    """Com o intercepto livre, a média ponderada do score calibrado é a taxa ponderada."""
    rng = np.random.default_rng(1)
    score = rng.uniform(0.01, 0.2, 5000)
    y = (rng.random(5000) < score).astype(int)
    peso = np.where(y == 1, 3.0, 1.0)  # as ocasiões pesam mais, como na POF
    calibrado = aplicar_sigmoide(score, *ajustar_sigmoide(score, y, peso))
    taxa = np.average(y, weights=peso)
    assert np.average(score, weights=peso) < taxa  # sem calibrar, fica abaixo
    assert np.average(calibrado, weights=peso) == pytest.approx(taxa, rel=1e-3)


def test_sigmoide_com_a_positivo_nao_muda_a_ordem():
    assert (np.diff(aplicar_sigmoide([0.001, 0.01, 0.02, 0.2], 0.8, 0.5)) > 0).all()


def test_calibrar_fora_do_fold_preenche_tudo():
    X, y = amostra(400, semente=3)
    folds = particoes(y, np.repeat(np.arange(100), 4), k=5, repeticoes=2)
    r = validar_cv(regressao_logistica(1.0), X, y, np.ones(400), folds, n_jobs=1)
    calibrado = calibrar_fora_do_fold(r.oof, y, np.ones(400), folds)
    assert calibrado.shape == r.oof.shape and not np.isnan(calibrado).any()


def test_confiabilidade_pondera_e_divide_as_copias():
    t = confiabilidade([0.1, 0.1, 0.3, 0.3], [0, 1, 0, 1], [1, 1, 1, 3], ["a", "a", "b", "b"])
    assert t.loc["a", "observada"] == pytest.approx(0.5)
    assert t.loc["b", "observada"] == pytest.approx(0.75)
    assert t.loc["b", "prevista"] == pytest.approx(0.3)
    duas = confiabilidade([0.1] * 4, [1, 0, 1, 0], [1] * 4, ["a"] * 4, copias=2)
    assert duas.loc["a", "ocasioes"] == 1


# --------------------------------------------------------------------------
# Modelo final: classificação e arquivo
# --------------------------------------------------------------------------
def test_classificar_usa_os_cortes():
    cortes = {"alta": 0.5, "media": 0.2}
    assert classificar([0.6, 0.5, 0.3, 0.2, 0.1], cortes).tolist() == [
        "alta", "alta", "media", "media", "baixa"]


def test_modelo_calibrado_sem_sigmoide_e_o_proprio_modelo():
    X, y = amostra(300)
    pipeline = regressao_logistica(1.0).fit(X, y)
    np.testing.assert_allclose(ModeloCalibrado(pipeline).predict_proba(X),
                               pipeline.predict_proba(X), rtol=1e-9)


def test_salvar_e_carregar_o_modelo(tmp_path):
    X, y = amostra(300)
    modelo = ModeloCalibrado(regressao_logistica(1.0).fit(X, y), a=1.1, b=0.2)
    salvar(modelo, {"model_version": "teste"}, tmp_path)
    carregado, meta = carregar(tmp_path)
    novo = pd.DataFrame([linha(), linha(periodo="manha")])
    np.testing.assert_allclose(carregado.predict_proba(novo), modelo.predict_proba(novo))
    assert meta["model_version"] == "teste" and len(meta["model_file_sha256"]) == 64
    (tmp_path / meta["model_file"]).write_bytes(b"outro arquivo")
    with pytest.raises(ValueError, match="hash"):
        carregar(tmp_path)


# Scores do modelo congelado em contextos fixos: o exemplo do README, os dois extremos
# citados na saída do teste e um contexto de score quase nulo. O arquivo do modelo guarda
# referências ao código de src/ (o preparo e as funções de feature), não o código em si.
# Uma mudança nesse código mudaria as previsões sem mudar o hash do arquivo; este teste pega.
SCORES_CONGELADOS = [
    (linha(dia_semana="Domingo", renda_pc=4000.0), 0.02880324108288343),
    (linha(UF="51", idade=27, sexo="masculino", anos_estudo=16, n_moradores=2,
           renda_pc=7569.44), 0.08347372519727868),
    (linha(periodo="almoco", dia_semana="Quinta-Feira", UF="25", urbano=0, idade=13,
           anos_estudo=6, n_moradores=4, renda_pc=296.5275), 8.632783506516647e-05),
    (linha(periodo="manha", dia_semana="Terca-Feira", UF="13", idade=70, sexo="masculino",
           anos_estudo=0, n_moradores=7, renda_pc=0.0), 2.5507926328693376e-06),
]


@pytest.mark.skipif(not (MODELOS / "model_metadata.json").exists(),
                    reason="modelo final não gerado")
@pytest.mark.parametrize("contexto, esperado", SCORES_CONGELADOS)
def test_modelo_congelado_mantem_os_scores_conhecidos(contexto, esperado):
    modelo, _ = carregar(MODELOS)
    obtido = modelo.predict_proba(pd.DataFrame([contexto]))[0, 1]
    assert obtido == pytest.approx(esperado, rel=1e-9)


@pytest.mark.skipif(not (MODELOS / "model_metadata.json").exists(),
                    reason="modelo final não treinado")
def test_modelo_final_preve_e_classifica_uma_linha():
    modelo, meta = carregar(MODELOS)
    assert meta["features"] == FEATURES
    p = modelo.predict_proba(pd.DataFrame([linha()]))[:, 1]
    assert 0 < p[0] < 1
    cortes = meta["classification"]["cuts"]
    assert 0 < cortes["media"] < cortes["alta"] < 1
    assert classificar(p, cortes)[0] in ("baixa", "media", "alta")
