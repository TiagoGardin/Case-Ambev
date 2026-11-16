"""Leitura dos microdados da POF 2017-2018 (arquivos de largura fixa).

Uso:
    python -m src.ingestion.pof

Gera em data/processed/:
    pof_consumo.parquet   – um registro por item consumido (1.175.390 linhas)
    pof_morador.parquet   – um registro por morador (idade, sexo, cor, escolaridade)
    pof_mes_entrevista.parquet – mês aproximado da entrevista por unidade de consumo

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

# Chave que identifica uma pessoa em todos os registros da POF
CHAVE_UC = ["COD_UPA", "NUM_DOM", "NUM_UC"]
CHAVE_PESSOA = CHAVE_UC + ["COD_INFORMANTE"]

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
    "COD_UPA": (8, 9),
    "NUM_DOM": (17, 2),
    "NUM_UC": (19, 1),
    "COD_INFORMANTE": (20, 2),
    "V0306": (22, 2),  # condição na unidade de consumo
    "V0403": (33, 3),  # idade em anos
    "V0404": (36, 1),  # sexo: 1 homem, 2 mulher
    "V0405": (37, 1),  # cor ou raça
    "ANOS_ESTUDO": (76, 2),
    "RENDA_DISP_PC": (117, 20),  # renda disponível per capita da UC
}

LAYOUT_ALUGUEL = {
    "COD_UPA": (8, 9),
    "NUM_DOM": (17, 2),
    "NUM_UC": (19, 1),
    "V9010": (41, 2),  # = mês anterior à data de abertura da entrevista
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
    df["V0403"] = df["V0403"].astype(int)
    df["ANOS_ESTUDO"] = pd.to_numeric(df["ANOS_ESTUDO"], errors="coerce")
    df["RENDA_DISP_PC"] = df["RENDA_DISP_PC"].astype(float)
    return df


def ler_mes_entrevista() -> pd.DataFrame:
    """Mês aproximado da entrevista por unidade de consumo (UC).

    Os microdados públicos não trazem a data da entrevista. No registro de
    Aluguel Estimado, V9010 é, por definição do dicionário, "o mês anterior à
    data de abertura da pesquisa". Logo: mês da entrevista = V9010 + 1.

    Limitações (documentar no Data Card):
      - só existe para UCs com aluguel estimado (~86% das pessoas);
      - não informa o ano: julho pode ser 2017 ou 2018.
    """
    df = ler_registro("ALUGUEL_ESTIMADO.txt", LAYOUT_ALUGUEL)
    df = df.dropna(subset=["V9010"]).drop_duplicates(subset=CHAVE_UC)
    df["mes_entrevista"] = df["V9010"].astype(int) % 12 + 1
    return df[CHAVE_UC + ["mes_entrevista"]]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for nome, func in [
        ("pof_consumo", ler_consumo),
        ("pof_morador", ler_morador),
        ("pof_mes_entrevista", ler_mes_entrevista),
    ]:
        df = func()
        df.to_parquet(OUT_DIR / f"{nome}.parquet", index=False)
        print(f"[ok] {nome}: {len(df):,} linhas")


if __name__ == "__main__":
    main()
