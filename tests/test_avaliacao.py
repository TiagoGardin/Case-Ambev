"""Testes das funções da avaliação no teste (src.models.avaliacao), com dados montados à mão."""

import numpy as np
import pandas as pd
import pytest

from src.models.avaliacao import (
    bootstrap_upa,
    calibracao_por_faixa,
    captura_por_local,
    intervalo,
    matriz_confusao,
    mes_do_domicilio,
    metricas,
    nome_mes,
    perfil,
    por_segmento,
)


def dados(n: int = 4000, semente: int = 0):
    rng = np.random.default_rng(semente)
    score = rng.random(n) * 0.02
    y = (rng.random(n) < score * 3).astype(int)
    peso = rng.uniform(0.5, 2.0, n)
    upa = rng.integers(0, 200, n)
    return y, score, peso, upa


def test_metricas_com_score_perfeito_e_contas_da_matriz():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05])
    m = metricas(y, score, np.ones(10), corte_alta=0.5)
    assert m["PR-AUC"] == pytest.approx(1.0)
    assert m["ROC-AUC"] == pytest.approx(1.0)
    assert m["precisão em alta"] == m["recall em alta"] == m["F1 em alta"] == pytest.approx(1.0)
    assert m["fração do peso em alta"] == pytest.approx(0.2)
    assert m["captura no top 20%"] == pytest.approx(1.0)
    assert m["taxa observada"] == pytest.approx(0.2)
    assert m["Brier"] == pytest.approx(np.mean((score - y) ** 2))
    cm = matriz_confusao(y, score, np.ones(10), 0.5)
    assert cm["linhas"].tolist() == [2, 0, 0, 8]
    assert cm["% do peso"].sum() == pytest.approx(1.0)


def test_metricas_precisao_e_recall_ponderados():
    y = np.array([1, 0, 1, 0])
    score = np.array([0.9, 0.8, 0.1, 0.1])
    m = metricas(y, score, np.array([1.0, 3.0, 1.0, 1.0]), corte_alta=0.5)
    assert m["precisão em alta"] == pytest.approx(1 / 4)
    assert m["recall em alta"] == pytest.approx(1 / 2)
    assert m["F1 em alta"] == pytest.approx(2 * 0.25 * 0.5 / 0.75)


def test_bootstrap_e_pareado_e_reprodutivel():
    y, score, peso, upa = dados()
    scores = {"a": score, "b": score}
    r1 = bootstrap_upa(y, scores, peso, upa, repeticoes=40, semente=3, n_jobs=2)
    r2 = bootstrap_upa(y, scores, peso, upa, repeticoes=40, semente=3, n_jobs=1)
    assert len(r1) == 40
    pd.testing.assert_frame_equal(r1, r2)
    # o mesmo score nas mesmas reamostragens dá diferença zero em todas
    assert np.allclose(r1["PR-AUC a"], r1["PR-AUC b"])
    lo, hi = intervalo(r1["PR-AUC a"])
    assert lo < hi


def test_bootstrap_equivale_a_repetir_as_linhas_da_upa():
    """Peso × número de sorteios da UPA = concatenar as linhas da UPA sorteada."""
    from src.models.metricas import prauc

    y, score, peso, upa = dados(800)
    r = bootstrap_upa(y, {"a": score}, peso, upa, repeticoes=1, semente=7, n_jobs=1)
    unicas, idx = np.unique(upa, return_inverse=True)
    rng = np.random.default_rng(7)
    sorteio = rng.integers(0, len(unicas), len(unicas))
    linhas = np.concatenate([np.where(idx == u)[0] for u in sorteio])
    assert r["PR-AUC a"].iloc[0] == pytest.approx(prauc(y[linhas], score[linhas], peso[linhas]))


def test_por_segmento_so_mede_niveis_com_ocasioes_suficientes():
    y, score, peso, _ = dados()
    seg = pd.Series(np.where(np.arange(len(y)) < 3000, "grande", "pequeno"))
    y = y.copy()
    y[3000:] = 0
    y[3000:3005] = 1  # 5 ocasiões no nível pequeno
    t = por_segmento(seg, y, {"m": score}, {"m": 0.015}, peso, minimo=30)
    grande, pequeno = t.set_index("nível").loc["grande"], t.set_index("nível").loc["pequeno"]
    assert not np.isnan(grande["PR-AUC m"]) and np.isnan(pequeno["PR-AUC m"])
    assert pequeno["ocasiões"] == 5


def test_captura_por_local_conta_so_as_ocasioes():
    local = pd.Series(pd.Categorical(["casa"] * 40 + ["estabelecimento"] * 40 + [None] * 20,
                                     ["casa", "estabelecimento", "outro_fora"]))
    y = np.array([1] * 80 + [0] * 20)
    classe = np.array(["alta"] * 30 + ["baixa"] * 10 + ["media"] * 40 + ["alta"] * 20)
    t = captura_por_local(local, y, {"m": classe}, np.ones(100), minimo=30).set_index("local")
    assert t.loc["casa", "% em alta m"] == pytest.approx(0.75)
    assert t.loc["estabelecimento", "% em alta ou média m"] == pytest.approx(1.0)
    assert "outro_fora" not in t.index  # nível sem nenhuma linha


def test_mes_do_domicilio_segue_a_regra_do_split():
    df = pd.DataFrame({
        "COD_UPA": [1, 1, 1, 2, 2], "NUM_DOM": [1, 1, 1, 1, 1],
        # domicílio 1: a fonte mais confiável (aluguel_estimado) manda, mesmo com mês maior
        "fonte_mes": ["vizinhos_upa", "aluguel_estimado", "aluguel_estimado",
                      "aluguel_pago", "aluguel_pago"],
        "ordem_mes": [3.0, 9.0, 9.0, 8.0, 7.0],
    })
    assert mes_do_domicilio(df).tolist() == [9.0, 9.0, 9.0, 7.0, 7.0]


def test_nome_mes():
    assert [nome_mes(o) for o in (0, 6, 7, 8, 12)] == [
        "jul/2017", "jan/2018", "fev/2018", "mar/2018", "jul/2018"]


def test_calibracao_por_faixa_soma_o_peso_e_as_ocasioes():
    y, score, peso, _ = dados()
    t = calibracao_por_faixa(score, y, peso, faixas=10)
    assert len(t) == 10 and t["% do peso"].sum() == pytest.approx(1.0)
    assert t["ocasiões"].sum() == y.sum()
    assert (t["% do peso"].between(0.09, 0.11)).all()
    assert t["taxa prevista"].is_monotonic_increasing


def test_perfil_soma_um_em_cada_grupo():
    col = pd.Series(pd.Categorical(["a", "b", "b", None], ["a", "b", "c"]))
    t = perfil(col, {"todos": np.ones(4, bool), "dois": np.array([1, 1, 0, 0], bool)},
               np.array([1.0, 1.0, 2.0, 1.0]))
    assert t.index.tolist() == ["a", "b", "sem informação"]
    assert t.sum().tolist() == pytest.approx([1.0, 1.0])
    assert t.loc["b", "todos"] == pytest.approx(3 / 5)
