"""Preparo único das features: o mesmo código no treino, na API e no app.

Recebe as 9 features do modelo como estão na grade e devolve as colunas que os
modelos usam, com categorias fixas. Nada aqui é aprendido dos dados: faixas e
regiões vêm das constantes e funções de src.features.build_grid, então uma
requisição de uma linha só sai igual à mesma linha dentro do treino. O que
depende dos dados (a padronização das numéricas) fica dentro de cada modelo e é
ajustado só no treino.

Todo modelo de src.models.modelos tem este preparo como primeiro passo, então o
artefato salvo recebe as features brutas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.build_grid import (
    DIAS,
    FAIXAS_IDADE,
    FAIXAS_RENDA,
    PERIODOS,
    REGIOES,
    faixa_idade,
    faixa_renda,
    fim_de_semana,
    regiao_da_uf,
)

FEATURES = ["periodo", "dia_semana", "UF", "urbano", "idade", "sexo", "anos_estudo",
            "n_moradores", "renda_pc"]

# Código IBGE da UF → sigla, para a API e o app
UFS = {"11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP", "17": "TO",
       "21": "MA", "22": "PI", "23": "CE", "24": "RN", "25": "PB", "26": "PE", "27": "AL",
       "28": "SE", "29": "BA", "31": "MG", "32": "ES", "33": "RJ", "35": "SP", "41": "PR",
       "42": "SC", "43": "RS", "50": "MS", "51": "MT", "52": "GO", "53": "DF"}
SEXOS = ["feminino", "masculino"]
PERIODO_DIA = [f"{d}|{p}" for d in DIAS for p in PERIODOS]
# Alternativa do Data Card: sexta a domingo contra segunda a quinta, em cada período
PERIODO_FDS = [f"{g}|{p}" for g in ("seg-qui", "sex-dom") for p in PERIODOS]

CATEGORIAS = {
    "periodo": PERIODOS,
    "dia_semana": DIAS,
    "periodo_dia": PERIODO_DIA,
    "periodo_fds": PERIODO_FDS,
    "UF": list(UFS),
    "regiao": list(REGIOES.values()),
    "sexo": SEXOS,
    "faixa_idade": FAIXAS_IDADE,
    "faixa_renda": FAIXAS_RENDA,
}
# O recordatório alimentar da POF só cobre quem tem 10 anos ou mais, e a escolaridade
# vai de 0 a 16 anos de estudo (16 = 16 ou mais). Na POF, a idade chega a 105 e a
# família a 16 pessoas.
LIMITES = {"idade": (10, 110), "anos_estudo": (0, 16), "n_moradores": (1, 20),
           "renda_pc": (0, np.inf)}

COLUNAS_PREPARADAS = ["periodo", "dia_semana", "periodo_dia", "fim_de_semana", "periodo_fds",
                      "UF", "regiao", "urbano", "idade", "faixa_idade", "sexo", "anos_estudo",
                      "n_moradores", "renda_pc", "log_renda_pc", "faixa_renda"]


def validar_entrada(df: pd.DataFrame) -> None:
    """Erro com mensagem clara se faltar coluna, houver nulo ou valor que o modelo não conhece."""
    faltam = [c for c in FEATURES if c not in df.columns]
    if faltam:
        raise ValueError(f"faltam as colunas {faltam}")
    nulas = [c for c in FEATURES if df[c].isna().any()]
    if nulas:
        raise ValueError(f"valores nulos em {nulas}")
    for col in ["periodo", "dia_semana", "UF", "sexo"]:
        fora = set(df[col].astype(str).unique()) - set(CATEGORIAS[col])
        if fora:
            raise ValueError(f"{col} com valores desconhecidos: {sorted(fora)}")
    if not df["urbano"].isin([0, 1]).all():
        raise ValueError("urbano deve ser 0 (rural) ou 1 (urbano)")
    for col, (minimo, maximo) in LIMITES.items():
        valor = pd.to_numeric(df[col], errors="coerce")
        if valor.isna().any() or not valor.between(minimo, maximo).all():
            raise ValueError(f"{col} precisa ser um número entre {minimo} e {maximo}")


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """As 9 features brutas (FEATURES) → as colunas dos modelos (COLUNAS_PREPARADAS)."""
    validar_entrada(df)
    out = pd.DataFrame(index=df.index)
    out["periodo"] = pd.Categorical(df["periodo"].astype(str), PERIODOS)
    out["dia_semana"] = pd.Categorical(df["dia_semana"].astype(str), DIAS)
    out["periodo_dia"] = pd.Categorical(
        out["dia_semana"].astype(str) + "|" + out["periodo"].astype(str), PERIODO_DIA)
    out["fim_de_semana"] = fim_de_semana(out["dia_semana"])
    grupo_dias = out["fim_de_semana"].map({0: "seg-qui", 1: "sex-dom"})
    out["periodo_fds"] = pd.Categorical(grupo_dias + "|" + out["periodo"].astype(str), PERIODO_FDS)
    out["UF"] = pd.Categorical(df["UF"].astype(str), CATEGORIAS["UF"])
    out["regiao"] = pd.Categorical(regiao_da_uf(out["UF"]), CATEGORIAS["regiao"])
    out["urbano"] = df["urbano"].astype(int)
    for col in ["idade", "anos_estudo", "n_moradores", "renda_pc"]:
        out[col] = pd.to_numeric(df[col]).astype(float)
    out["faixa_idade"] = faixa_idade(out["idade"])
    out["sexo"] = pd.Categorical(df["sexo"].astype(str), SEXOS)
    out["log_renda_pc"] = np.log1p(out["renda_pc"])
    out["faixa_renda"] = faixa_renda(out["renda_pc"])
    return out[COLUNAS_PREPARADAS]
