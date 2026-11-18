"""Testes das regras de feature.

Rodar: python -m pytest

Os testes usam dados pequenos montados à mão, então não dependem dos
microdados. O único que lê um arquivo do IBGE é pulado se ele não existir.
"""

import math

import pandas as pd
import pytest

from src.features.build_grid import faixa_renda, periodo_do_dia
from src.features.classe_abep import (
    COD_AUTOMOVEL,
    COD_DVD,
    COD_GELADEIRA_2_PORTAS,
    COD_LAVA_ROUPA,
    COD_MICRO_ONDAS,
    COD_MICROCOMPUTADOR,
    calcular_classe_abep,
    classe_por_pontos,
    pontos_instrucao,
)
from src.features.mes_entrevista import (
    ano_mes_da_ordem,
    candidatos_ordem,
    mes_por_aluguel_estimado,
    ordem_por_pagamento,
)
from src.features.tipo_area import AREAS, ARQ_ESTRATOS, expandir_celula, mapa_estratos


# --------------------------------------------------------------------------
# Tempo e renda
# --------------------------------------------------------------------------
def test_periodo_do_dia():
    horas = pd.Series([0, 4, 5, 10, 11, 14, 15, 17, 18, 23])
    esperado = ["noite", "noite", "manha", "manha", "almoco", "almoco",
                "tarde", "tarde", "noite", "noite"]
    assert periodo_do_dia(horas).tolist() == esperado


def test_faixa_renda_limites():
    sm = 954.0
    renda = pd.Series([0, 0.49 * sm, 0.5 * sm, 1 * sm, 2 * sm, 5 * sm])
    esperado = ["ate_0.5sm", "ate_0.5sm", "0.5_1sm", "1_2sm", "2_5sm", "acima_5sm"]
    assert faixa_renda(renda).astype(str).tolist() == esperado


# --------------------------------------------------------------------------
# Classe ABEP
# --------------------------------------------------------------------------
def test_cortes_da_abep():
    pontos = pd.Series([0, 16, 17, 22, 23, 28, 29, 37, 38, 44, 45, 100])
    esperado = ["DE", "DE", "C2", "C2", "C1", "C1", "B2", "B2", "B1", "B1", "A", "A"]
    assert classe_por_pontos(pontos).astype(str).tolist() == esperado


def test_pontos_instrucao():
    nivel = pd.Series([1, 2, 2, 3, 4, 5, 6, 7])
    anos = pd.Series([0, 4, 5, 9, 10, 12, 13, 16])
    assert pontos_instrucao(nivel, anos).tolist() == [0, 0, 1, 2, 2, 4, 4, 7]


def test_classe_abep_de_dois_domicilios():
    """UC 1: domicílio de classe média (28 pontos = C1). UC 2: sem bens nem serviços (DE)."""
    morador = pd.DataFrame({
        "COD_UPA": ["1", "1"], "NUM_DOM": ["1", "2"], "NUM_UC": ["1", "1"],
        "COD_INFORMANTE": ["1", "1"], "V0306": [1, 1],
        "NIVEL_INSTRUCAO": [5, 1], "ANOS_ESTUDO": [12, 0],
    })
    domicilio = pd.DataFrame({
        "COD_UPA": ["1", "1"], "NUM_DOM": ["1", "2"],
        "V0207": ["1", "2"], "V02111": [1, 0], "V0220": ["1", "2"],
    })
    # Só a UC 1 tem bens. A geladeira duplex conta como geladeira E freezer.
    bens = [COD_AUTOMOVEL, COD_MICROCOMPUTADOR, COD_GELADEIRA_2_PORTAS,
            COD_LAVA_ROUPA, COD_DVD, COD_MICRO_ONDAS]
    inventario = pd.DataFrame({
        "COD_UPA": "1", "NUM_DOM": "1", "NUM_UC": "1", "V9001": bens, "V9005": 1,
    })

    r = calcular_classe_abep(morador, domicilio, inventario).set_index("NUM_DOM")

    # banheiro 3 + carro 3 + computador 3 + geladeira 2 + freezer 2 + lava-roupa 2
    # + DVD 1 + micro-ondas 2 + instrução (médio completo) 4 + água 4 + rua 2 = 28
    assert r.loc["1", "abep_pontos"] == 28
    assert r.loc["1", "classe_abep"] == "C1"
    assert r.loc["2", "abep_pontos"] == 0
    assert r.loc["2", "classe_abep"] == "DE"


# --------------------------------------------------------------------------
# Tipo de área
# --------------------------------------------------------------------------
@pytest.mark.parametrize("celula, esperado", [
    ("1101 e 1102", {1101, 1102}),
    ("1103 a 1107", {1103, 1104, 1105, 1106, 1107}),
    ("1201", {1201}),
    ("-", set()),
    (float("nan"), set()),
])
def test_expandir_celula(celula, esperado):
    assert expandir_celula(celula) == esperado


@pytest.mark.skipif(not ARQ_ESTRATOS.exists(), reason="documentação da POF não baixada")
def test_mapa_estratos_oficial():
    mapa = mapa_estratos()
    assert set(mapa.values()) == set(AREAS)
    # Pará: capital 1501–1503, RM 1504–1505, interior 1506–1511, rural 1512–1519
    assert mapa[1501] == "capital"
    assert mapa[1504] == "regiao_metropolitana"
    assert mapa[1506] == "interior_urbano"
    assert mapa[1512] == "rural"


# --------------------------------------------------------------------------
# Mês da entrevista
# --------------------------------------------------------------------------
def test_candidatos_ordem():
    assert candidatos_ordem(7) == [0, 12]  # julho: início ou fim da coleta
    assert candidatos_ordem(8) == [1]
    assert candidatos_ordem(12) == [5]
    assert candidatos_ordem(1) == [6]
    assert candidatos_ordem(6) == [11]


def test_ano_mes_da_ordem():
    ano, mes = ano_mes_da_ordem(pd.Series([0, 5, 6, 12]))
    assert ano.tolist() == [2017, 2017, 2018, 2018]
    assert mes.tolist() == [7, 12, 1, 7]


def test_julho_separado_pelo_deflator():
    aluguel = pd.DataFrame({
        "COD_UPA": ["1"] * 4, "NUM_DOM": ["1", "2", "3", "4"], "NUM_UC": ["1"] * 4,
        "V9010": [6, 6, 7, 12],  # mês anterior à entrevista
        "DEFLATOR": [1.02, 0.98, 1.01, 0.99],
    })
    r = mes_por_aluguel_estimado(aluguel)
    # jul/2017 (deflator > 1), jul/2018 (deflator < 1), ago/2017, jan/2018
    assert r["ordem_mes"].tolist() == [0, 12, 1, 6]


@pytest.mark.parametrize("v9010, mediana, esperado", [
    (3, 9.0, (9, "aluguel_pago")),  # pagou em março → abril (9), coerente com vizinhos
    (6, 11.0, (12, "aluguel_pago")),  # julho ambíguo: vizinhos indicam 2018
    (6, 1.0, (0, "aluguel_pago")),  # julho ambíguo: vizinhos indicam 2017
    (3, 2.0, (2, "vizinhos_upa")),  # pagamento incoerente (7 meses longe) → vizinhos
    (math.nan, 4.5, (5, "vizinhos_upa")),  # sem pagamento → mediana arredondada
    (3, math.nan, (9, "aluguel_pago")),  # sem vizinhos, mês não ambíguo
])
def test_ordem_por_pagamento(v9010, mediana, esperado):
    assert ordem_por_pagamento(v9010, mediana) == esperado


@pytest.mark.parametrize("v9010", [6, math.nan])
def test_ordem_por_pagamento_sem_informacao(v9010):
    """Julho sem vizinhos, ou nada de informação: não dá para estimar."""
    ordem, fonte = ordem_por_pagamento(v9010, math.nan)
    assert math.isnan(ordem) and fonte is None
