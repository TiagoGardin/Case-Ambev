"""Validação cruzada repetida e comparação entre opções (README, Metodologia).

- Partições: StratifiedGroupKFold com 5 folds, agrupado por domicílio (a família
  divide a pizza), repetido 10 vezes com sementes 0 a 9 (50 folds). Elas dependem
  só do target e dos domicílios, então são as mesmas em todas as comparações da
  modelagem e da triagem de features.
- Métricas em cada fold: PR-AUC ponderada por `peso_semana` e captura no top 10%.
- Comparação: a opção mais complexa só vence se o ganho de PR-AUC for
  significativo a 5% (unilateral) no teste t corrigido para validação cruzada
  repetida (Nadeau e Bengio, 2003).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold

from src.models.metricas import captura_topo, prauc

K, REPETICOES, ALFA = 5, 10, 0.05


def domicilio(df: pd.DataFrame) -> pd.Series:
    """Rótulo do domicílio (UPA e número do domicílio), o grupo das partições."""
    return df["COD_UPA"].astype(str) + "-" + df["NUM_DOM"].astype(str)


def particoes(y, grupos, k: int = K,
              repeticoes: int = REPETICOES) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pares (índices de treino, índices de validação), repetição por repetição."""
    y = np.asarray(y)
    return [(tr, va)
            for r in range(repeticoes)
            for tr, va in StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=r)
            .split(np.zeros(len(y)), y, np.asarray(grupos))]


@dataclass
class ResultadoCV:
    prauc: np.ndarray      # PR-AUC ponderada de cada fold
    captura10: np.ndarray  # captura no top 10% de cada fold
    oof: np.ndarray        # previsão fora da amostra: linhas × repetições

    def resumo(self) -> str:
        return (f"PR-AUC {self.prauc.mean():.4f} (dp {self.prauc.std(ddof=1):.4f}) | "
                f"captura no top 10%: {self.captura10.mean():.1%}")


def normalizar(peso) -> np.ndarray:
    """Peso do ajuste com média 1.

    A Regressão Logística minimiza C × Σ peso × perda + penalidade, então a força
    da regularização dependeria da escala do peso (o `peso_semana` está na casa dos
    milhares). Com média 1, o C tem o mesmo sentido com e sem peso.
    """
    peso = np.asarray(peso, dtype=float)
    return peso / peso.mean()


def _prever_fold(modelo, X: pd.DataFrame, y: np.ndarray, peso_ajuste, tr, va) -> np.ndarray:
    m = clone(modelo)
    extra = {} if peso_ajuste is None else {"modelo__sample_weight": normalizar(peso_ajuste[tr])}
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)  # sem convergência, a validação para
        m.fit(X.iloc[tr], y[tr], **extra)
    return m.predict_proba(X.iloc[va])[:, 1]


def validar_cv(modelo, X: pd.DataFrame, y, peso, folds, peso_ajuste=None, k: int = K,
               n_jobs: int = -1) -> ResultadoCV:
    """Ajusta uma cópia de `modelo` em cada fold e mede na parte de validação.

    `folds` vem de `particoes` (k folds por repetição, em ordem). `peso` pondera as
    métricas; `peso_ajuste`, se dado, entra no ajuste do modelo, com média 1 em cada
    fold (`normalizar`).
    """
    y = np.asarray(y)
    peso = np.asarray(peso, dtype=float)
    if peso_ajuste is not None:
        peso_ajuste = np.asarray(peso_ajuste, dtype=float)
    previsoes = Parallel(n_jobs=n_jobs)(
        delayed(_prever_fold)(modelo, X, y, peso_ajuste, tr, va) for tr, va in folds)

    oof = np.full((len(y), len(folds) // k), np.nan)
    for i, ((_, va), p) in enumerate(zip(folds, previsoes)):
        oof[va, i // k] = p
    assert not np.isnan(oof).any(), "linha sem previsão fora da amostra"
    pr = [prauc(y[va], p, peso[va]) for (_, va), p in zip(folds, previsoes)]
    cap = [captura_topo(y[va], p, peso[va], 0.10) for (_, va), p in zip(folds, previsoes)]
    return ResultadoCV(np.array(pr), np.array(cap), oof)


def comparar(ref, cand, k: int = K, alfa: float = ALFA) -> dict:
    """Ganho médio de `cand` sobre `ref` e teste t corrigido para validação cruzada repetida.

    Os folds compartilham dados de treino, então a variância ingênua subestima a
    incerteza. A correção de Nadeau e Bengio (2003) soma n_validação / n_treino,
    que é 1 / (k − 1) em k folds.
    """
    d = np.asarray(cand) - np.asarray(ref)
    n = len(d)
    se = np.sqrt((1 / n + 1 / (k - 1)) * d.var(ddof=1))
    t = d.mean() / se if se > 0 else 0.0
    p = float(stats.t.sf(t, df=n - 1))
    return {"ganho": float(d.mean()), "t": float(t), "p": p, "passa": p < alfa}
