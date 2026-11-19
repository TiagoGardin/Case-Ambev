"""Modelos do Occasion Score: o baseline dia × período e a Regressão Logística.

Cada modelo é um Pipeline cujo primeiro passo é o preparo único
(src.models.preparo.preparar), então recebe as 9 features brutas, e o artefato
salvo aplica sozinho o mesmo processamento do treino. O último passo se chama
sempre "modelo", para a validação passar o peso do ajuste do mesmo jeito a todos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.models.preparo import CATEGORIAS, COLUNAS_PREPARADAS, preparar

# Regressão Logística (README, Metodologia): `regiao` no lugar de `UF` e `faixa_idade` no
# lugar de `idade`. Com a interação período × dia da semana, ela contém o baseline.
CATEGORICAS_LR = ("periodo", "dia_semana", "periodo_dia", "regiao", "faixa_idade", "sexo")
NUMERICAS_LR = ("anos_estudo", "n_moradores", "log_renda_pc")
BINARIAS_LR = ("urbano",)


def nomes_preparados(transformer, nomes_entrada) -> np.ndarray:
    """Nomes das colunas que saem do preparo, para o get_feature_names_out do Pipeline."""
    return np.asarray(COLUNAS_PREPARADAS, dtype=object)


def passo_preparo() -> FunctionTransformer:
    return FunctionTransformer(preparar, feature_names_out=nomes_preparados)


class BaselineDiaPeriodo(ClassifierMixin, BaseEstimator):
    """Taxa ponderada de y em cada célula dia da semana × período do treino, usada como score.

    É o "sexta à noite é bom para pizza" em forma de número. Uma célula sem peso no
    treino recebe a taxa geral; na POF isso não acontece, porque as 28 células têm
    milhares de linhas.
    """

    def fit(self, X: pd.DataFrame, y, sample_weight=None):
        y = np.asarray(y, dtype=float)
        w = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        celula = X["periodo_dia"].astype(str).to_numpy()
        soma = (pd.DataFrame({"celula": celula, "yw": y * w, "w": w})
                .groupby("celula")[["yw", "w"]].sum())
        self.taxa_ = (soma["yw"] / soma["w"]).to_dict()
        self.taxa_geral_ = float((y * w).sum() / w.sum())
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = X["periodo_dia"].astype(str).map(self.taxa_).fillna(self.taxa_geral_)
        p = p.to_numpy(dtype=float)
        return np.column_stack([1 - p, p])


def baseline() -> Pipeline:
    return Pipeline([("preparo", passo_preparo()), ("modelo", BaselineDiaPeriodo())])


def regressao_logistica(C: float, categoricas=CATEGORICAS_LR, numericas=NUMERICAS_LR,
                        binarias=BINARIAS_LR) -> Pipeline:
    """Categóricas em one-hot, numéricas padronizadas e regularização L2 de força 1/C."""
    colunas = ColumnTransformer([
        ("cat", OneHotEncoder(categories=[CATEGORIAS[c] for c in categoricas]),
         list(categoricas)),
        ("num", StandardScaler(), list(numericas)),
        ("bin", "passthrough", list(binarias)),
    ])
    return Pipeline([("preparo", passo_preparo()), ("colunas", colunas),
                     ("modelo", LogisticRegression(C=C, max_iter=5000))])
