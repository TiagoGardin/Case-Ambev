"""Leitura dos microdados da POF 2017-2018 (arquivos de largura fixa).

Uso:
    python -m src.ingestion.pof

Gera em data/processed/ (um .parquet por registro, só com as colunas usadas):
    pof_consumo.parquet            – um registro por item consumido (1.175.390 linhas)
    pof_morador.parquet            – um registro por morador
    pof_aluguel_estimado.parquet   – mês de referência das UCs em imóvel próprio/cedido
    pof_pagamentos_moradia.parquet – aluguel, prestação e condomínio pagos (quadro 10)
    pof_domicilio.parquet          – características do domicílio (banheiros, água, rua)
    pof_inventario.parquet         – bens duráveis de cada UC

Este módulo só LÊ e converte tipos. As regras derivadas ficam em src/features/:
mês da entrevista (mes_entrevista.py), classe ABEP (classe_abep.py) e
tipo de área (tipo_area.py).

Por que largura fixa: os .txt do IBGE não têm separador. Cada coluna ocupa
posições fixas na linha, descritas no "Dicionários de variáveis.xls".

As posições abaixo foram copiadas do DICIONÁRIO, não do script em R do IBGE:
o script em R tem as larguras das últimas colunas do Consumo Alimentar erradas
(PESO_FINAL, RENDA_TOTAL e DIA_SEMANA saem deslocados).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DADOS_DIR = ROOT / "data" / "raw" / "pof" / "dados"
OUT_DIR = ROOT / "data" / "processed"

# Chaves que identificam domicílio, unidade de consumo (UC) e pessoa em todos os registros
CHAVE_DOM = ["COD_UPA", "NUM_DOM"]
CHAVE_UC = CHAVE_DOM + ["NUM_UC"]
CHAVE_PESSOA = CHAVE_UC + ["COD_INFORMANTE"]

# Códigos do quadro 10 (despesas com moradia) usados para estimar o mês da entrevista
ALUGUEL_PAGO = 1000101
PRESTACAO_IMOVEL = 1000301
CONDOMINIO = 1000501

# Layouts: nome -> (posição inicial 1-based, tamanho), como no dicionário
LAYOUT_CONSUMO = {
    "UF": (1, 2),
    "TIPO_SITUACAO_REG": (7, 1),  # 1 urbano, 2 rural
    "COD_UPA": (8, 9),
    "NUM_DOM": (17, 2),
    "NUM_UC": (19, 1),
    "COD_INFORMANTE": (20, 2),
    "QUADRO": (22, 2),  # 72 = 1º dia de recordatório, 73 = 2º dia
    "SEQ": (24, 2),
    "V9001": (32, 7),  # código do alimento
    "V9015": (39, 3),  # horário do consumo (ex.: "20H")
    "V9017": (44, 1),  # ocasião: 1 café, 2 almoço, 3 lanche, 4 jantar, 5 ceia, 6 outra, 9 n.d.
    "V9018": (45, 1),  # local: 1 em casa ... 9 n.d.
    "PESO_FINAL": (504, 15),
    "RENDA_TOTAL": (519, 10),
    "DIA_SEMANA": (529, 15),
    "DIA_ATIPICO": (544, 1),  # 1 sim, 2 não
}

LAYOUT_MORADOR = {
    "UF": (1, 2),
    "ESTRATO_POF": (3, 4),  # estrato geográfico: capital, RM, interior urbano ou rural
    "COD_UPA": (8, 9),
    "NUM_DOM": (17, 2),
    "NUM_UC": (19, 1),
    "COD_INFORMANTE": (20, 2),
    "V0306": (22, 2),  # condição na UC: 1 = pessoa de referência
    "V0403": (33, 3),  # idade em anos
    "V0404": (36, 1),  # sexo: 1 homem, 2 mulher
    "V0405": (37, 1),  # cor ou raça (lida, mas NÃO usada como feature)
    "ANOS_ESTUDO": (76, 2),  # base: ensino fundamental de 9 anos
    "NIVEL_INSTRUCAO": (116, 1),  # 1 sem instrução ... 7 superior completo
    "RENDA_DISP_PC": (117, 20),  # renda disponível per capita da UC
}

LAYOUT_ALUGUEL_ESTIMADO = {
    "COD_UPA": (8, 9),
    "NUM_DOM": (17, 2),
    "NUM_UC": (19, 1),
    "V9010": (41, 2),  # = mês anterior à data de abertura da entrevista
    "DEFLATOR": (45, 12),  # fator que leva valores a preços de 15/01/2018
}

LAYOUT_DESPESA_COLETIVA = {
    "COD_UPA": (8, 9),
    "NUM_DOM": (17, 2),
    "NUM_UC": (19, 1),
    "QUADRO": (20, 2),
    "V9001": (24, 7),  # código da despesa
    "V9010": (47, 2),  # último mês em que a despesa foi paga
    "V9011": (49, 2),  # número de meses pagos no período de referência
}

LAYOUT_DOMICILIO = {
    "COD_UPA": (8, 9),
    "NUM_DOM": (17, 2),
    "V0207": (26, 1),  # abastecimento de água: 1 = rede geral
    "V02111": (34, 2),  # banheiros de uso exclusivo (com chuveiro e vaso)
    "V0220": (49, 1),  # rua pavimentada: 1 sim, 2 não
}

LAYOUT_INVENTARIO = {
    "COD_UPA": (8, 9),
    "NUM_DOM": (17, 2),
    "NUM_UC": (19, 1),
    "V9001": (24, 7),  # código do bem durável
    "V9005": (31, 2),  # quantidade do bem no domicílio
}


def _colspecs(layout: dict[str, tuple[int, int]]) -> list[tuple[int, int]]:
    """Converte (início 1-based, tamanho) em intervalos 0-based [início, fim)."""
    return [(ini - 1, ini - 1 + tam) for ini, tam in layout.values()]


def ler_registro(arquivo: str, layout: dict[str, tuple[int, int]]) -> pd.DataFrame:
    """Lê um registro de largura fixa mantendo tudo como texto.

    dtype=str evita perder zeros à esquerda em códigos (ex.: NUM_DOM "02")
    e evita que o pandas adivinhe tipos errados. Conversões são feitas depois,
    de forma explícita.
    """
    return pd.read_fwf(
        DADOS_DIR / arquivo,
        colspecs=_colspecs(layout),
        names=list(layout),
        header=None,
        dtype=str,
        encoding="latin1",
    )


def ler_consumo() -> pd.DataFrame:
    df = ler_registro("CONSUMO_ALIMENTAR.txt", LAYOUT_CONSUMO)
    df["V9001"] = df["V9001"].astype(int)
    df["hora"] = df["V9015"].str.replace("H", "", regex=False).astype(int)
    df["PESO_FINAL"] = df["PESO_FINAL"].astype(float)
    df["RENDA_TOTAL"] = df["RENDA_TOTAL"].astype(float)
    df["DIA_SEMANA"] = df["DIA_SEMANA"].str.strip()
    return df


def ler_morador() -> pd.DataFrame:
    df = ler_registro("MORADOR.txt", LAYOUT_MORADOR)
    df["ESTRATO_POF"] = df["ESTRATO_POF"].astype(int)
    df["V0306"] = df["V0306"].astype(int)
    df["V0403"] = df["V0403"].astype(int)
    df["ANOS_ESTUDO"] = pd.to_numeric(df["ANOS_ESTUDO"], errors="coerce")
    df["NIVEL_INSTRUCAO"] = pd.to_numeric(df["NIVEL_INSTRUCAO"], errors="coerce")
    df["RENDA_DISP_PC"] = df["RENDA_DISP_PC"].astype(float)
    return df


def ler_aluguel_estimado() -> pd.DataFrame:
    """Uma linha por UC em imóvel próprio ou cedido (quem aluga não tem este registro)."""
    df = ler_registro("ALUGUEL_ESTIMADO.txt", LAYOUT_ALUGUEL_ESTIMADO)
    df = df.dropna(subset=["V9010"]).drop_duplicates(subset=CHAVE_UC)
    df["V9010"] = df["V9010"].astype(int)
    df["DEFLATOR"] = df["DEFLATOR"].astype(float)
    return df


def ler_pagamentos_moradia() -> pd.DataFrame:
    """Aluguel, prestação e condomínio pagos (quadro 10 da Despesa Coletiva)."""
    df = ler_registro("DESPESA_COLETIVA.txt", LAYOUT_DESPESA_COLETIVA)
    df = df[df["QUADRO"] == "10"].copy()
    df["V9001"] = df["V9001"].astype(int)
    df = df[df["V9001"].isin([ALUGUEL_PAGO, PRESTACAO_IMOVEL, CONDOMINIO])].copy()
    df["V9010"] = df["V9010"].astype(int)
    df["V9011"] = df["V9011"].astype(int)
    return df.drop(columns="QUADRO")


def ler_domicilio() -> pd.DataFrame:
    df = ler_registro("DOMICILIO.txt", LAYOUT_DOMICILIO)
    df["V02111"] = df["V02111"].astype(int)
    return df


def ler_inventario() -> pd.DataFrame:
    df = ler_registro("INVENTARIO.txt", LAYOUT_INVENTARIO)
    df["V9001"] = df["V9001"].astype(int)
    df["V9005"] = df["V9005"].astype(int)
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for nome, func in [
        ("pof_consumo", ler_consumo),
        ("pof_morador", ler_morador),
        ("pof_aluguel_estimado", ler_aluguel_estimado),
        ("pof_pagamentos_moradia", ler_pagamentos_moradia),
        ("pof_domicilio", ler_domicilio),
        ("pof_inventario", ler_inventario),
    ]:
        df = func()
        df.to_parquet(OUT_DIR / f"{nome}.parquet", index=False)
        print(f"[ok] {nome}: {len(df):,} linhas")


if __name__ == "__main__":
    main()
