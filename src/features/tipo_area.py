"""Tipo de área do domicílio: capital, região metropolitana, interior urbano ou rural.

A POF sorteia os domicílios dentro de estratos geográficos. O arquivo
"Estratos POF 2017-2018.xls" (documentação oficial) diz a que área pertence
cada código de ESTRATO_POF, por UF:

    URBANO: Município da Capital | Resto da RM | Resto da UF      RURAL
    ex. Pará:   1501 a 1503      | 1504 e 1505 | 1506 a 1511 | 1512 a 1519

Por que usar no lugar de "urbano/rural": a diferença dentro do urbano é grande.
Na análise exploratória, a taxa noturna de pizza + refri foi 1,80% na capital,
1,44% na região metropolitana, 1,02% no interior urbano e 0,18% no rural.

Só usamos o nível geográfico do estrato. A POF também estratifica por renda
dentro de cada área, mas essa parte não entra aqui.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

DOC_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "pof" / "documentacao"
ARQ_ESTRATOS = DOC_DIR / "Estratos POF 2017-2018.xls"

# Colunas 1 a 4 da planilha, na ordem em que aparecem
AREAS = ["capital", "regiao_metropolitana", "interior_urbano", "rural"]


def expandir_celula(celula: object) -> set[int]:
    """Converte o texto de uma célula em códigos de estrato.

    "1101 e 1102" -> {1101, 1102}; "1103 a 1107" -> {1103, ..., 1107};
    "1201" -> {1201}; "-" ou vazio -> conjunto vazio.
    """
    texto = str(celula).strip()
    numeros = [int(n) for n in re.findall(r"\d{4}", texto)]
    if not numeros:
        return set()
    if " a " in texto:
        return set(range(numeros[0], numeros[-1] + 1))
    return set(numeros)


def mapa_estratos(arquivo: Path = ARQ_ESTRATOS) -> dict[int, str]:
    """Dicionário estrato -> tipo de área, lido da planilha oficial."""
    tabela = pd.read_excel(arquivo, sheet_name=0, header=None, dtype=str)
    mapa: dict[int, str] = {}
    for _, linha in tabela.iterrows():
        for coluna, area in zip(range(1, 5), AREAS):
            for estrato in expandir_celula(linha[coluna]):
                if estrato in mapa:
                    raise ValueError(f"estrato {estrato} aparece em mais de uma área")
                mapa[estrato] = area
    return mapa


def tipo_area(estrato: pd.Series, mapa: dict[int, str] | None = None) -> pd.Series:
    mapa = mapa if mapa is not None else mapa_estratos()
    area = estrato.astype(int).map(mapa)
    if area.isna().any():
        faltando = sorted(estrato[area.isna()].unique())
        raise ValueError(f"estratos sem tipo de área na planilha: {faltando}")
    return pd.Series(pd.Categorical(area, categories=AREAS), index=estrato.index)
