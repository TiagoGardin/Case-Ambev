"""Testes do Mapa de Oportunidade (docs/oportunidade.md).

Por enquanto: a regra da coleta e a coesão dos dados externos versionados.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.models.preparo import UFS
from src.oportunidade.coleta import CONSULTA, criterio

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"
POPULACAO_BRASIL_2022 = 203_080_756  # SIDRA, tabela 4709, nível Brasil


# --------------------------------------------------------------------------
# Regra das pizzarias
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tags, esperado", [
    ({"amenity": "restaurant", "cuisine": "pizza"}, "cozinha"),
    ({"amenity": "fast_food", "cuisine": "burger;Pizza"}, "cozinha"),
    ({"amenity": "restaurant", "name": "Pizzaria do Bairro"}, "nome"),
    ({"amenity": "fast_food", "name": "PIZZA HUT"}, "nome"),
    ({"amenity": "restaurant", "cuisine": "pizza", "name": "Pizzaria Roma"}, "ambos"),
    ({"amenity": "cafe", "cuisine": "pizza"}, ""),
    ({"amenity": "restaurant_bar", "cuisine": "pizza"}, ""),
    ({"amenity": "restaurant", "cuisine": "italian", "name": "Cantina"}, ""),
    ({"cuisine": "pizza", "name": "Pizzaria"}, ""),
])
def test_criterio_da_pizzaria(tags, esperado):
    assert criterio(tags) == esperado


def test_consulta_e_a_do_documento():
    doc = (ROOT / "docs" / "oportunidade.md").read_text(encoding="utf-8")
    consulta_sp = CONSULTA.format(sigla="SP")
    # o documento traz a consulta indentada dentro de um item de lista
    linhas_doc = [linha.strip() for linha in doc.splitlines()]
    for linha in consulta_sp.splitlines():
        assert linha.strip() in linhas_doc, linha


# --------------------------------------------------------------------------
# Coesão dos dados externos versionados
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def populacao():
    return pd.read_csv(EXT / "populacao_uf_censo2022.csv", dtype={"codigo_uf": str})


@pytest.fixture(scope="module")
def contagem():
    return pd.read_csv(EXT / "osm_pizzarias_uf.csv", dtype={"codigo_uf": str})


@pytest.fixture(scope="module")
def elementos():
    return pd.read_csv(EXT / "osm_pizzarias_elementos.csv", keep_default_na=False)


def test_populacao_tem_as_27_ufs_e_soma_o_brasil(populacao):
    assert len(populacao) == 27
    assert dict(zip(populacao["codigo_uf"], populacao["uf"])) == UFS
    assert (populacao["populacao"] > 0).all()
    assert populacao["populacao"].sum() == POPULACAO_BRASIL_2022
    assert populacao["fonte"].nunique() == 1 and "4709" in populacao["fonte"].iloc[0]


def test_contagem_tem_as_27_ufs_e_fecha_por_tipo_e_criterio(contagem):
    assert len(contagem) == 27
    assert dict(zip(contagem["codigo_uf"], contagem["uf"])) == UFS
    assert (contagem["pizzarias"] == contagem[["nos", "vias", "relacoes"]].sum(axis=1)).all()
    assert (contagem["pizzarias"]
            == contagem[["so_cozinha", "so_nome", "ambos"]].sum(axis=1)).all()
    assert contagem["data_coleta"].nunique() == 1


def test_cada_elemento_cumpre_a_regra(elementos):
    assert set(elementos["amenity"]) <= {"restaurant", "fast_food"}
    colunas = zip(elementos["amenity"], elementos["cuisine"], elementos["nome"])
    recalculado = [criterio({"amenity": a, "cuisine": c, "name": n}) for a, c, n in colunas]
    assert recalculado == elementos["criterio"].tolist()
    assert not elementos.duplicated(["uf", "tipo", "osm_id"]).any()


def test_elementos_reproduzem_a_contagem(elementos, contagem):
    por_uf = elementos.groupby("uf").size().reindex(contagem["uf"], fill_value=0)
    assert por_uf.tolist() == contagem["pizzarias"].tolist()
    por_criterio = pd.crosstab(elementos["uf"], elementos["criterio"]).reindex(
        contagem["uf"], fill_value=0)
    for coluna, nome in [("so_cozinha", "cozinha"), ("so_nome", "nome"), ("ambos", "ambos")]:
        esperado = por_criterio[nome] if nome in por_criterio else pd.Series(0, por_criterio.index)
        assert esperado.tolist() == contagem[coluna].tolist()
