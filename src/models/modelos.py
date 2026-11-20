"""Modelos do Occasion Score: o baseline dia × período, a Regressão Logística e o LightGBM raso.

Cada modelo é um Pipeline cujo primeiro passo é o preparo único
(src.models.preparo.preparar), então recebe as 9 features brutas, e o artefato
salvo aplica sozinho o mesmo processamento do treino. O último passo se chama
sempre "modelo", para a validação passar o peso do ajuste do mesmo jeito a todos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.models.preparo import CATEGORIAS, COLUNAS_PREPARADAS, preparar


def colunas_lr(dia: str = "dia_semana", renda: str = "continua") -> dict[str, tuple[str, ...]]:
    """Colunas da Regressão Logística em cada alternativa do Data Card.

    - `dia`: "dia_semana" usa os 7 dias e as 28 células período × dia;
      "fim_de_semana" usa sexta a domingo contra segunda a quinta e as 8 células
      período × fim de semana.
    - `renda`: "continua" usa o log da renda per capita, padronizado; "faixas" usa as
      5 faixas de salário mínimo em one-hot.

    Em todas, `regiao` fica no lugar de `UF` e `faixa_idade` no lugar de `idade`
    (README, Metodologia), e a interação com o período contém o baseline.
    """
    if dia not in ("dia_semana", "fim_de_semana") or renda not in ("continua", "faixas"):
        raise ValueError(f"alternativa desconhecida: dia={dia!r}, renda={renda!r}")
    if dia == "dia_semana":
        categoricas, binarias = ["periodo", "dia_semana", "periodo_dia"], ["urbano"]
    else:
        categoricas, binarias = ["periodo", "periodo_fds"], ["urbano", "fim_de_semana"]
    categoricas += ["regiao", "faixa_idade", "sexo"]
    numericas = ["anos_estudo", "n_moradores"]
    if renda == "continua":
        numericas.append("log_renda_pc")
    else:
        categoricas.append("faixa_renda")
    return {"categoricas": tuple(categoricas), "numericas": tuple(numericas),
            "binarias": tuple(binarias)}


_PADRAO_LR = colunas_lr()
CATEGORICAS_LR = _PADRAO_LR["categoricas"]
NUMERICAS_LR = _PADRAO_LR["numericas"]
BINARIAS_LR = _PADRAO_LR["binarias"]

# LightGBM raso (README, Metodologia): as 9 features como estão, com as categóricas nativas
# do LightGBM. Sem amostragem de linhas e colunas, o ajuste é determinístico. Um núcleo
# por ajuste, porque a validação cruzada já roda os folds em paralelo.
COLUNAS_LGB = ("periodo", "dia_semana", "UF", "urbano", "idade", "sexo", "anos_estudo",
               "n_moradores", "renda_pc")
PARAMS_LGB = {"n_estimators": 300, "learning_rate": 0.03, "reg_lambda": 5.0,
              "deterministic": True, "force_col_wise": True, "random_state": 0, "n_jobs": 1,
              "verbose": -1}


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


class SelecionarColunas(TransformerMixin, BaseEstimator):
    """Passa adiante só as colunas pedidas, com os tipos do preparo (as categóricas do LightGBM)."""

    def __init__(self, colunas=COLUNAS_LGB):
        self.colunas = colunas

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[list(self.colunas)]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.colunas, dtype=object)


def lightgbm_raso(num_leaves: int, min_child_samples: int, colunas=COLUNAS_LGB) -> Pipeline:
    """Poucas folhas e muitas linhas por folha, porque o treino tem só 523 positivos."""
    modelo = LGBMClassifier(num_leaves=num_leaves, min_child_samples=min_child_samples,
                            **PARAMS_LGB)
    return Pipeline([("preparo", passo_preparo()), ("colunas", SelecionarColunas(colunas)),
                     ("modelo", modelo)])


def construir(config: dict) -> Pipeline:
    """O Pipeline de uma configuração escolhida (ver CONFIG em src.models.treinar)."""
    if config["modelo"] == "regressao_logistica":
        return regressao_logistica(config["C"], **colunas_lr(config["dia"], config["renda"]))
    if config["modelo"] == "lightgbm":
        return lightgbm_raso(config["num_leaves"], config["min_child_samples"])
    raise ValueError(f"modelo desconhecido: {config['modelo']!r}")
