"""Códigos de alimentos da POF que definem a ocasião Pizza + Refrigerante.

Fonte principal: classificação oficial do IBGE ("Classificação dos grupos de
Consumo Alimentar.xlsx", aba Gruali), que agrupa os itens em subgrupos como
"Pizzas", "Refrigerantes" e "Refrigerantes diet/light".

Usar a classificação oficial, em vez de uma lista feita à mão, torna o target
rastreável e defensável. Os ajustes feitos sobre ela estão explícitos abaixo.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DOC_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "pof" / "documentacao"
ARQ_GRUPOS = DOC_DIR / "Classifica__o dos grupos de Consumo Alimentar.xlsx"
ARQ_CADASTRO = DOC_DIR / "Cadastro de Produtos do Consumo Alimentar.xls"

SUBGRUPOS_PIZZA = {"Pizzas"}
SUBGRUPOS_REFRI = {"Refrigerantes", "Refrigerantes diet/light"}

# O subgrupo oficial "Refrigerantes" inclui bebidas que não são refrigerante
# no sentido da ocasião (não gaseificadas ou de outra categoria). Removidas:
REFRI_EXCLUIR = {
    8200308,  # Refresco de guaraná natural (não é refrigerante)
    8201015,  # Água mineral gaseificada com sumo de fruta
    8201104,  # Mate tradicional
    8212802,  # Mate diet
    8216501,  # Ice tea tradicional
    8217401,  # Tampico
}


def subgrupos_oficiais() -> pd.DataFrame:
    """Lê a aba Gruali e devolve uma tabela código → grupo → subgrupo.

    Na planilha, a linha de subgrupo tem a descrição preenchida e o código vazio;
    os itens vêm logo abaixo. Por isso, propagamos o subgrupo para baixo (ffill).
    """
    g = pd.read_excel(ARQ_GRUPOS, sheet_name="Gruali", header=None, skiprows=2, dtype=str)
    g.columns = ["grupo", "codigo", "descricao"]
    g["grupo"] = g["grupo"].ffill()
    eh_subgrupo = g["codigo"].isna() & g["descricao"].notna()
    g["subgrupo"] = g["descricao"].where(eh_subgrupo).ffill()
    itens = g[g["codigo"].notna()].copy()
    itens["codigo"] = itens["codigo"].astype(int)
    itens["descricao"] = itens["descricao"].str.strip()
    itens["subgrupo"] = itens["subgrupo"].str.strip()
    return itens[["codigo", "descricao", "grupo", "subgrupo"]]


def cadastro_produtos() -> pd.DataFrame:
    c = pd.read_excel(ARQ_CADASTRO, header=None, skiprows=1, names=["codigo", "descricao"])
    c["codigo"] = c["codigo"].astype(int)
    return c


def codigos_pizza() -> set[int]:
    """Subgrupo oficial 'Pizzas' + qualquer item do cadastro com 'PIZZA' no nome.

    A união é necessária porque alguns sabores (ex.: pizza doce, hawaiana,
    portuguesa light) estão no cadastro mas fora do subgrupo oficial.
    """
    oficiais = subgrupos_oficiais()
    sub = set(oficiais.loc[oficiais["subgrupo"].isin(SUBGRUPOS_PIZZA), "codigo"])
    cad = cadastro_produtos()
    por_nome = set(cad.loc[cad["descricao"].str.contains("PIZZA", na=False), "codigo"])
    return sub | por_nome


def codigos_refrigerante() -> set[int]:
    """Subgrupos oficiais de refrigerantes + itens do cadastro com nome de refrigerante.

    Mesma lógica da pizza: alguns itens (ex.: Fanta light, Tubaína light) estão no
    cadastro mas fora dos subgrupos oficiais.
    """
    oficiais = subgrupos_oficiais()
    sub = set(oficiais.loc[oficiais["subgrupo"].isin(SUBGRUPOS_REFRI), "codigo"])
    cad = cadastro_produtos()
    por_nome = set(
        cad.loc[
            cad["descricao"].str.contains(
                r"REFRIGERANTE|FANTA|COCA COLA|TUBAINA|^GUARANA (?:TRADICIONAL|LIGHT|DIET)",
                na=False,
                regex=True,
            ),
            "codigo",
        ]
    )
    return (sub | por_nome) - REFRI_EXCLUIR


def codigos_guarana() -> set[int]:
    """Refrigerantes de guaraná (tradicional, light e diet). Não identifica marca."""
    cad = cadastro_produtos()
    refri = cad[cad["codigo"].isin(codigos_refrigerante())]
    return set(refri.loc[refri["descricao"].str.contains("GUARANA", na=False), "codigo"])


def tabela_codigos() -> pd.DataFrame:
    """Tabela com todos os códigos usados e seu papel, para documentação."""
    cad = cadastro_produtos().set_index("codigo")["descricao"]
    linhas = []
    for papel, cods in [
        ("pizza", codigos_pizza()),
        ("refrigerante", codigos_refrigerante()),
        ("guarana", codigos_guarana()),
    ]:
        for c in sorted(cods):
            linhas.append({"papel": papel, "codigo": c, "descricao": cad.get(c, "")})
    return pd.DataFrame(linhas)


if __name__ == "__main__":
    t = tabela_codigos()
    print(t.groupby("papel").size())
    print(t.to_string(index=False))
