"""Calibração do Occasion Score: sigmoide ponderada sobre o logit do score.

O Occasion Score é uma probabilidade da população (README, Metodologia). Um modelo
ajustado sem peso acerta a taxa da amostra, que não é a da população: as ocasiões
têm PESO_FINAL médio maior que o resto das linhas. A sigmoide

    p_calibrado = 1 / (1 + exp(−(a · logit(p) + b)))

é ajustada por máxima verossimilhança ponderada por `peso_semana` sobre as
previsões out-of-fold do treino. Com a > 0 ela é crescente e não muda a ordem do
score, então PR-AUC e captura não mudam: mudam só o nível (b) e a escala (a).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from src.models.validacao import K, normalizar

EPS = 1e-7


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def ajustar_sigmoide(score, y, peso) -> tuple[float, float]:
    """(a, b) da sigmoide por máxima verossimilhança ponderada, sem regularização."""
    m = LogisticRegression(C=np.inf, max_iter=1000)
    m.fit(logit(score).reshape(-1, 1), np.asarray(y), sample_weight=normalizar(peso))
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def aplicar_sigmoide(score, a: float, b: float) -> np.ndarray:
    return expit(a * logit(score) + b)


def calibrar_fora_do_fold(oof: np.ndarray, y, peso, folds, k: int = K) -> np.ndarray:
    """Previsões out-of-fold calibradas sem usar a própria parte para ajustar a sigmoide.

    Em cada fold, a sigmoide é ajustada nas previsões das outras k − 1 partes da
    mesma repetição e aplicada à parte de validação. Serve para medir o efeito da
    calibração; a sigmoide do modelo final usa todas as previsões.
    """
    y, peso = np.asarray(y), np.asarray(peso, dtype=float)
    calibrado = np.full(oof.shape, np.nan)
    for i, (tr, va) in enumerate(folds):
        r = i // k
        a, b = ajustar_sigmoide(oof[tr, r], y[tr], peso[tr])
        calibrado[va, r] = aplicar_sigmoide(oof[va, r], a, b)
    return calibrado


def log_loss_por_fold(oof: np.ndarray, y, peso, folds, k: int = K) -> np.ndarray:
    """Log loss ponderado em cada fold: piora quando o score fica acima ou abaixo da taxa real."""
    y, peso = np.asarray(y), np.asarray(peso, dtype=float)
    return np.array([log_loss(y[va], oof[va, i // k], sample_weight=peso[va], labels=[0, 1])
                     for i, (_, va) in enumerate(folds)])


def confiabilidade(score, y, peso, grupo, copias: int = 1) -> pd.DataFrame:
    """Taxa prevista e observada, ponderadas, em cada grupo de linhas.

    `copias` > 1 quando as linhas vêm empilhadas das várias repetições da validação
    cruzada: as ocasiões aparecem uma vez por repetição e são divididas por isso.
    """
    peso = np.asarray(peso, dtype=float)
    y = np.asarray(y, dtype=float)
    d = pd.DataFrame({"grupo": grupo, "w": peso, "pw": np.asarray(score, dtype=float) * peso,
                      "yw": y * peso, "y": y})
    g = d.groupby("grupo", observed=True)[["w", "pw", "yw", "y"]].sum()
    return pd.DataFrame({"ocasioes": (g["y"] / copias).round().astype(int),
                         "prevista": g["pw"] / g["w"], "observada": g["yw"] / g["w"],
                         "observada / prevista": g["yw"] / g["pw"]})


@dataclass
class ModeloCalibrado:
    """Pipeline ajustado (preparo + modelo) seguido da sigmoide; recebe as 9 features brutas.

    Sem calibração, a = 1 e b = 0, e o score sai como o modelo deu.
    """

    pipeline: object
    a: float = 1.0
    b: float = 0.0

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = aplicar_sigmoide(self.pipeline.predict_proba(X)[:, 1], self.a, self.b)
        return np.column_stack([1 - p, p])
