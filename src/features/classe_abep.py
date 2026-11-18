"""Classe social pelo Critério Brasil (ABEP), aproximada com dados da POF.

Referência: ABEP, "Critério de Classificação Econômica Brasil", alterações
válidas a partir de 01/06/2019 (https://abep.org/criterioBr/01_cceb_2019).
A tabela de pontos e os cortes abaixo foram copiados desse documento.

O Critério Brasil soma pontos por bens do domicílio, banheiros, instrução do
chefe da família e acesso a serviços públicos, e corta a soma em 6 classes.
É a classificação usada no mercado (inclusive no TAPI, que fala em "Classe A").

Aproximações (documentadas no Data Card, seção 8):
- Empregados domésticos e secadora de roupa não existem no inventário da POF
  e pontuam 0. Isso subestima um pouco as classes mais altas.
- Banheiro: a POF conta banheiros "com chuveiro e vaso sanitário"; a ABEP conta
  qualquer cômodo com vaso (inclui lavabos).
- Automóvel: a POF não separa uso pessoal de uso profissional.
- DVD: a ABEP também conta videogames e computadores que leem DVD; a POF só
  tem "aparelho de DVD ou Blu-ray".
- Chefe da família: a ABEP usa quem tem a maior renda; usamos a pessoa de
  referência da UC (V0306 = 1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ingestion.pof import CHAVE_DOM, CHAVE_UC

# Pontos por quantidade possuída: índice 0, 1, 2, 3, 4 ou mais
PONTOS = {
    "banheiros": [0, 3, 7, 10, 14],
    "automoveis": [0, 3, 5, 8, 11],
    "microcomputador": [0, 3, 6, 8, 11],
    "lava_louca": [0, 3, 6, 6, 6],
    "geladeira": [0, 2, 3, 5, 5],
    "freezer": [0, 2, 4, 6, 6],
    "lava_roupa": [0, 2, 4, 6, 6],
    "dvd": [0, 1, 3, 4, 6],
    "micro_ondas": [0, 2, 4, 4, 4],
    "motocicleta": [0, 1, 3, 3, 3],
}
PONTOS_AGUA_ENCANADA = 4
PONTOS_RUA_PAVIMENTADA = 2

# Códigos do inventário da POF (quadro 14) usados em cada item
COD_AUTOMOVEL = 1403001
COD_MICROCOMPUTADOR = 1401901  # "microcomputador ou notebook" (tablet tem código próprio)
COD_LAVA_LOUCA = 1400801
COD_GELADEIRA_1_PORTA = 1400301
COD_GELADEIRA_2_PORTAS = 1400401
COD_FREEZER = 1400201
COD_LAVA_ROUPA = 1401201
COD_DVD = 1401701
COD_MICRO_ONDAS = 1400901
COD_MOTOCICLETA = 1403101

# Classes em ordem crescente e cortes da ABEP (pontos máximos de cada classe):
# DE 0–16, C2 17–22, C1 23–28, B2 29–37, B1 38–44, A 45+
CLASSES = ["DE", "C2", "C1", "B2", "B1", "A"]
CORTES = [-np.inf, 16, 22, 28, 37, 44, np.inf]

# Distribuição oficial dos domicílios no Brasil (ABEP 2019), para comparação
DISTRIBUICAO_OFICIAL_PCT = {"DE": 28.3, "C2": 26.8, "C1": 21.5, "B2": 16.5, "B1": 4.4, "A": 2.5}


def pontos_item(quantidade: pd.Series, tabela: list[int]) -> pd.Series:
    """Pontos de um item: quantidades acima do último índice valem o último valor."""
    q = quantidade.fillna(0).astype(int).clip(0, len(tabela) - 1)
    return pd.Series(np.asarray(tabela)[q.to_numpy()], index=quantidade.index)


def pontos_instrucao(nivel: pd.Series, anos_estudo: pd.Series) -> pd.Series:
    """Instrução do chefe nos 5 níveis da ABEP.

    NIVEL_INSTRUCAO da POF: 1 sem instrução, 2 fundamental incompleto,
    3 fundamental completo, 4 médio incompleto, 5 médio completo,
    6 superior incompleto, 7 superior completo.
    O "Fundamental I completo" (1 ponto) só se separa pelos anos de estudo:
    a POF usa o fundamental de 9 anos, então Fundamental I = 5 anos de estudo.
    """
    condicoes = [
        nivel.isin([1]) | (nivel.eq(2) & (anos_estudo < 5)),  # analfabeto / Fund. I incompleto
        nivel.eq(2),  # Fund. I completo / Fund. II incompleto
        nivel.isin([3, 4]),  # Fund. II completo / Médio incompleto
        nivel.isin([5, 6]),  # Médio completo / Superior incompleto
        nivel.eq(7),  # Superior completo
    ]
    return pd.Series(np.select(condicoes, [0, 1, 2, 4, 7], default=0), index=nivel.index)


def classe_por_pontos(pontos: pd.Series) -> pd.Series:
    return pd.cut(pontos, CORTES, labels=CLASSES, right=True)


def _quantidades_inventario(inventario: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por UC com a quantidade de cada item do Critério Brasil."""
    q = inventario.pivot_table(
        index=CHAVE_UC, columns="V9001", values="V9005", aggfunc="sum", fill_value=0
    )

    def col(codigo: int) -> pd.Series:
        return q[codigo] if codigo in q.columns else pd.Series(0, index=q.index)

    itens = pd.DataFrame(index=q.index)
    itens["automoveis"] = col(COD_AUTOMOVEL)
    itens["microcomputador"] = col(COD_MICROCOMPUTADOR)
    itens["lava_louca"] = col(COD_LAVA_LOUCA)
    itens["geladeira"] = col(COD_GELADEIRA_1_PORTA) + col(COD_GELADEIRA_2_PORTAS)
    # Regra da ABEP: a 2ª porta da geladeira duplex conta como freezer
    itens["freezer"] = col(COD_FREEZER) + col(COD_GELADEIRA_2_PORTAS)
    itens["lava_roupa"] = col(COD_LAVA_ROUPA)
    itens["dvd"] = col(COD_DVD)
    itens["micro_ondas"] = col(COD_MICRO_ONDAS)
    itens["motocicleta"] = col(COD_MOTOCICLETA)
    return itens.reset_index()


def calcular_classe_abep(
    morador: pd.DataFrame, domicilio: pd.DataFrame, inventario: pd.DataFrame
) -> pd.DataFrame:
    """Pontos e classe ABEP de cada UC."""
    ucs = morador[CHAVE_UC].drop_duplicates()

    # UCs sem nenhuma linha no inventário não têm bens duráveis: 0 pontos nos bens
    base = ucs.merge(_quantidades_inventario(inventario), on=CHAVE_UC, how="left")
    base = base.merge(domicilio, on=CHAVE_DOM, how="left", validate="many_to_one")
    if base["V02111"].isna().any():
        raise ValueError("UC sem registro de domicílio: verifique a leitura do DOMICILIO.txt")

    chefe = morador.loc[morador["V0306"] == 1, CHAVE_UC + ["NIVEL_INSTRUCAO", "ANOS_ESTUDO"]]
    base = base.merge(chefe, on=CHAVE_UC, how="left", validate="one_to_one")

    base["banheiros"] = base["V02111"]
    pontos = sum(pontos_item(base[item], tabela) for item, tabela in PONTOS.items())
    pontos += pontos_instrucao(base["NIVEL_INSTRUCAO"], base["ANOS_ESTUDO"])
    pontos += np.where(base["V0207"] == "1", PONTOS_AGUA_ENCANADA, 0)
    pontos += np.where(base["V0220"] == "1", PONTOS_RUA_PAVIMENTADA, 0)

    base["abep_pontos"] = pontos.astype(int)
    base["classe_abep"] = classe_por_pontos(base["abep_pontos"])
    return base[CHAVE_UC + ["abep_pontos", "classe_abep"]]
