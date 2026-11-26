"""Testes do Mapa de Oportunidade (docs/oportunidade.md).

A regra da coleta, a coesão dos dados externos versionados e o ranking (as conferências
do documento rodam sobre o arquivo versionado models/oportunidade.json).
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.decision_engine import contexto as contexto_decisao
from src.features.build_grid import periodo_do_dia
from src.models import artefato
from src.models.preparo import UFS
from src.oportunidade import ranking
from src.oportunidade.coleta import CONSULTA, criterio

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"
ARQUIVO_RANKING = ROOT / "models" / "oportunidade.json"
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


# --------------------------------------------------------------------------
# Ranking: regras puras
# --------------------------------------------------------------------------

@pytest.mark.parametrize("idade, faixa", [(18, "18-24"), (24, "18-24"), (25, "25-34"),
                                          (44, "35-44"), (45, "45-54"), (64, "55-64"),
                                          (65, "65+"), (105, "65+")])
def test_faixas_etarias_do_google_ads(idade, faixa):
    assert ranking.faixa_etaria(pd.Series([idade]))[0] == faixa


@pytest.mark.parametrize("chance_acima, densidade_acima, categoria", [
    (True, True, "ativacao_delivery"), (True, False, "combo_consumo_em_casa"),
    (False, True, "presenca_de_marca"), (False, False, "presenca_de_marca")])
def test_regra_do_canal_dos_estados(chance_acima, densidade_acima, categoria):
    assert ranking.categoria_do_estado(chance_acima, densidade_acima) == categoria


def test_categorias_vem_do_decision_engine_sem_canal():
    meta = json.loads((ROOT / "models" / "model_metadata.json").read_text(encoding="utf-8"))
    cat = ranking.categorias_por_classe(contexto_decisao.carregar(), meta["reference_rate"])
    assert cat == {"baixa": ("sem_ativacao_dedicada", "R1"),
                   "media": ("presenca_de_marca", "R6"),
                   "alta": ("combo_consumo_em_casa", "R5")}


@pytest.mark.parametrize("hora", range(24))
def test_horarios_de_anuncio_seguem_os_periodos_da_grade(hora):
    periodo = periodo_do_dia(pd.Series([hora]))[0]
    faixas = [tuple(int(x[:2]) for x in f.split("-")) for f in ranking.HORARIOS[periodo]]
    assert any(ini <= hora < fim for ini, fim in faixas)


def test_anuncio_de_um_publico_momento():
    a = ranking.anuncio("25-34", "masculino", "Centro-Oeste", "noite", "sex_dom")
    assert a == {"faixa_etaria": "25-34", "genero": "masculino",
                 "localizacao_ufs": ["DF", "GO", "MS", "MT"],
                 "dias": ["Sexta-Feira", "Sabado", "Domingo"],
                 "horarios": ["18:00-24:00", "00:00-05:00"]}


# --------------------------------------------------------------------------
# Ranking: o arquivo versionado (as conferências de docs/oportunidade.md)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mapa():
    return json.loads(ARQUIVO_RANKING.read_text(encoding="utf-8"))


def test_conferencias_do_documento(mapa):
    ranking.conferir(mapa)  # públicos completos, parcelas, média, 27 UFs, regra do canal


def test_mapa_usa_o_modelo_congelado(mapa):
    meta = json.loads((ROOT / "models" / "model_metadata.json").read_text(encoding="utf-8"))
    assert mapa["modelo"]["model_file_sha256"] == meta["model_file_sha256"]
    arquivo_modelo = ROOT / "models" / meta["model_file"]
    assert mapa["modelo"]["model_file_sha256"] == artefato.sha256(arquivo_modelo)
    assert mapa["cortes_prioridade"] == pytest.approx(meta["classification"]["cuts"])
    assert mapa["dados"]["idade_minima"] == 18 and "não usado" in mapa["dados"]["treino"]


def test_prioridade_e_categoria_seguem_os_cortes(mapa):
    cortes = mapa["cortes_prioridade"]
    regra = {"alta": "R5", "media": "R6", "baixa": "R1"}
    for p in mapa["publicos"]:
        assert p["prioridade"] == artefato.classificar([p["chance"]], cortes)[0]
        assert p["regra"] == regra[p["prioridade"]]


def test_posicoes_seguem_as_medidas(mapa):
    pub = pd.DataFrame([p for p in mapa["publicos"] if p["amostra_suficiente"]])
    assert (pub["pessoas_amostra"] >= mapa["amostra_minima_pessoas"]).all()
    assert pub.sort_values("chance", ascending=False)["posicao_chance"].tolist() == list(
        range(1, len(pub) + 1))
    assert pub.sort_values("parcela_das_ocasioes", ascending=False)["posicao_parcela"].tolist() \
        == list(range(1, len(pub) + 1))
    est = pd.DataFrame(mapa["estados"])
    assert est["posicao"].tolist() == list(range(1, 28))
    acima = est["chance_acima"].tolist()
    assert acima == sorted(acima, reverse=True)  # primeiro os de chance acima da mediana
    for grupo in (True, False):
        parcelas = est.loc[est["chance_acima"] == grupo, "parcela_das_ocasioes"].tolist()
        assert parcelas == sorted(parcelas, reverse=True)


def test_estados_usam_os_dados_externos_versionados(mapa, populacao, contagem):
    est = pd.DataFrame(mapa["estados"]).set_index("codigo_uf")
    pop = populacao.set_index("codigo_uf")
    cont = contagem.set_index("codigo_uf")
    assert (est["populacao"] == pop["populacao"].reindex(est.index)).all()
    assert (est["pizzarias"] == cont["pizzarias"].reindex(est.index)).all()
    assert est["pizzarias_100mil"].to_numpy() == pytest.approx(
        (est["pizzarias"] / est["populacao"] * 1e5).to_numpy())
    med = mapa["medianas_estados"]
    assert (est["chance_acima"] == (est["chance"] >= med["chance"])).all()
    assert (est["densidade_acima"] == (est["pizzarias_100mil"] >= med["pizzarias_100mil"])).all()
    assert mapa["fontes"]["pizzarias"]["data_coleta_utc"] == contagem["data_coleta"].iloc[0]


@pytest.mark.skipif(not (ranking.PROC / "grid.parquet").exists(),
                    reason="precisa da grade (python -m src.features.build_grid)")
def test_arquivo_versionado_e_reproduzido_pelo_codigo():
    # Na mesma máquina o arquivo sai idêntico (o script confere). Em outra, a última casa de
    # um número pode mudar com a biblioteca de álgebra, então aqui a comparação tem tolerância.
    def igual(a, b):
        if isinstance(a, float) or isinstance(b, float):
            return a == pytest.approx(b, rel=1e-8, abs=1e-12)
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(igual(a[k], b[k]) for k in a)
        if isinstance(a, list):
            return len(a) == len(b) and all(igual(x, y) for x, y in zip(a, b))
        return a == b

    versionado = json.loads(ARQUIVO_RANKING.read_text(encoding="utf-8"))
    assert igual(json.loads(ranking.serializar(ranking.gerar())), versionado)


@pytest.mark.parametrize("estrago", ["categoria_do_estado", "parcela", "publico_faltando",
                                     "chance_de_um_publico"])
def test_conferencias_pegam_um_mapa_errado(mapa, estrago):
    errado = json.loads(json.dumps(mapa))
    if estrago == "categoria_do_estado":
        atual = errado["estados"][0]["categoria"]
        errado["estados"][0]["categoria"] = next(
            c for c in ["ativacao_delivery", "combo_consumo_em_casa", "presenca_de_marca"]
            if c != atual)
    elif estrago == "parcela":
        errado["publicos"][0]["parcela_das_ocasioes"] += 0.01
    elif estrago == "publico_faltando":
        errado["publicos"].pop()
    else:
        errado["publicos"][0]["chance"] *= 2
    with pytest.raises(ValueError):
        ranking.conferir(errado)
