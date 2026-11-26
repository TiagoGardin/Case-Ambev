"""Testes do app (Streamlit): fluxogramas, gráficos, conteúdo, números da avaliação e a página.

O app só fala com a API. Aqui a API roda no mesmo processo (TestClient), no lugar do HTTP.
"""

import re
import sys
import warnings
from pathlib import Path
from xml.dom import minidom

import pandas as pd
import pytest
from starlette.exceptions import StarletteDeprecationWarning

with warnings.catch_warnings():
    warnings.simplefilter("ignore", StarletteDeprecationWarning)
    from fastapi.testclient import TestClient

from src.features import build_grid  # noqa: E402
from src.models import preparo  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
APP = RAIZ / "app"
sys.path.insert(0, str(APP))

import cliente  # noqa: E402
import conteudo as C  # noqa: E402
import estilo as E  # noqa: E402
import fluxogramas as F  # noqa: E402
import graficos as G  # noqa: E402

MODELOS = RAIZ / "models"
TABELAS = RAIZ / "reports" / "teste_tabelas.md"
TEM_ARTEFATOS = all((MODELOS / f).exists() for f in (
    "model_metadata.json", "explicacao.json", "contexto_decisao.json"))
CORTES = {"alta": 0.007765464212035935, "media": 0.0014357961786816277}
# Termos de dados que não aparecem nas abas para quem decide marketing
JARGAO = re.compile(r"\b(score|shap|log-odds|pr-auc|roc-auc|baseline|api|json|ic 95|"
                    r"classe (alta|média|media|baixa)|decision engine)\b", re.IGNORECASE)


def _svg(texto: str):
    raiz = minidom.parseString(texto).documentElement
    assert raiz.tagName == "svg"
    return raiz


def _texto_visivel(html: str) -> str:
    return re.sub(r"<[^>]+>", " ", html)


# --------------------------------------------------------------------------
# Fluxogramas
# --------------------------------------------------------------------------
@pytest.mark.parametrize("svg", [F.treino_e_teste(), F.requisicao(), F.decision_engine(),
                                 F.decision_engine(linguagem="simples")])
def test_fluxogramas_sao_svg_valido_e_acessivel(svg):
    raiz = _svg(svg)
    assert raiz.getAttribute("role") == "img" and len(raiz.getAttribute("aria-label")) > 40
    assert raiz.getAttribute("viewBox").startswith("0 0 ")


@pytest.mark.parametrize("linguagem", ["tecnica", "simples"])
@pytest.mark.parametrize("regra", sorted(F.CAMINHOS))
def test_decision_engine_acende_so_o_caminho_da_regra(regra, linguagem):
    svg = F.decision_engine(regra, ("Chance 1,0%", "prioridade alta"), prefixo=f"t{regra}",
                            linguagem=linguagem)
    _svg(svg)
    # a saída escolhida é a única caixa toda vermelha
    assert svg.count(f'fill="{E.VERMELHO}" stroke="{E.VERMELHO}"') == 1
    # os nós fora do caminho ficam apagados: 11 nós (partida, 4 decisões, 6 saídas)
    nos_no_caminho = [n for n in F.CAMINHOS[regra] if not n.startswith("e_")]
    assert svg.count('<g opacity="0.4">') == 11 - len(nos_no_caminho)
    aceso = (f"Caminho aceso: regra {regra}" if linguagem == "tecnica"
             else f"Caminho aceso: {' '.join(F.TEXTOS_DE['simples']['desfechos'][regra])}")
    assert aceso in svg


def test_arvore_simples_fala_a_lingua_de_quem_decide():
    for regra in [None, *sorted(F.CAMINHOS)]:
        texto = " ".join(t.firstChild.data for t in _svg(
            F.decision_engine(regra, linguagem="simples")).getElementsByTagName("text"))
        assert not re.search(r"\bR[1-6]\b|[Cc]lasse|[Ss]core", texto), texto
        assert "Prioridade" in texto and "Onde vai" in texto


def test_decision_engine_recusa_regra_ou_linguagem_desconhecida():
    with pytest.raises(ValueError, match="regra desconhecida"):
        F.decision_engine("R9")
    with pytest.raises(ValueError, match="linguagem desconhecida"):
        F.decision_engine(linguagem="juridiquês")


def test_gravar_as_figuras_do_readme(tmp_path):
    caminhos = F.gravar(tmp_path)
    assert [c.name for c in caminhos] == ["fluxo_1_treino_e_teste.svg", "fluxo_2_requisicao.svg",
                                          "fluxo_3_decision_engine.svg"]
    for c in caminhos:  # com fundo pintado, para o tema escuro do GitHub
        assert f'fill="{E.CREME}"' in c.read_text(encoding="utf-8")
        _svg(c.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Gráficos e formatos
# --------------------------------------------------------------------------
@pytest.mark.parametrize("score", [1e-7, 0.0001, 0.0029, 0.0288, 0.5])
def test_termometro_vale_para_qualquer_score(score):
    raiz = _svg(G.termometro(score, CORTES))
    assert G.pct(score) in raiz.getAttribute("aria-label")


def test_formatos_para_quem_nao_e_da_area():
    assert G.pct(0.0288) == "2,88%" and G.pct(0.000061) == "0,0061%"
    assert G.razao(9.52) == "9,5" and G.razao(0.021) == "0,02" and G.razao(0.001) == "< 0,01"
    assert [G.pct_simples(v) for v in (0.0238, 0.00382, 0.00045, 0.00006)] == [
        "2,4%", "0,38%", "0,045%", "menos de 0,01%"]
    assert G.pct_simples(0.00006, compacto=True) == "< 0,01%"
    assert G.em_cada(0.0238) == "cerca de 24 em cada 1.000"
    assert G.em_cada(0.0025) == "cerca de 25 em cada 10 mil"
    assert G.em_cada(0.00006) == "cerca de 6 em cada 100 mil"
    assert G.em_cada(1e-7) == "menos de 3 em cada 1 milhão"
    assert G.comparado(7.94) == "7,9 vezes a chance média"
    assert G.comparado(0.02) == "2% da chance média"
    assert G.comparado(0.001) == "menos de 1% da chance média"
    assert G.vezes(13.164) == "× 13,2" and G.vezes(1.565) == "× 1,6" and G.vezes(0.61) == "× 0,61"


def test_fatores_com_as_palavras_do_formulario():
    top = [{"factor": "momento", "value": "noite de sexta a domingo", "effect": "aumenta",
            "odds_multiplier": 13.164, "log_odds": 2.5775},
           {"factor": "renda", "value": "R$ 300 por pessoa ao mês, a preços de jan/2018",
            "effect": "reduz", "odds_multiplier": 0.61, "log_odds": -0.4943}]
    html = G.fatores(top, {"renda": "Até ½ salário mínimo por pessoa"})
    assert "× 13,2" in html and "× 0,61" in html
    assert "barra aumenta" in html and "barra reduz" in html
    assert "Até ½ salário mínimo por pessoa" in html and "jan/2018" not in html
    assert "Renda por pessoa" in html and "média" in html


@pytest.mark.parametrize("classe", ["baixa", "media", "alta"])
def test_prioridade_acende_so_a_do_contexto(classe):
    html = G.prioridade(classe)
    assert html.count("✓") == 1 and html.count('class="passo ativo"') == 1
    assert f"background:{E.CLASSE[classe]}" in html
    assert all(f"{p} dos momentos" in html for p in ("70%", "20%", "10%"))


def test_quadros_de_quando_e_onde():
    celulas = {(b, p): (0.001 * (i + 1), "media")
               for i, (b, p) in enumerate((b, p) for b in C.BLOCOS_DE_DIAS for p in C.PERIODOS)}
    celulas[("Sexta a domingo", "noite")] = (0.024, "alta")
    html = G.quando(celulas, C.PERIODOS, C.HORARIOS, ("Sexta a domingo", "noite"))
    assert html.count("seu cenário") == 1 and "2,4%" in html
    assert html.count('class="cel') == 8
    regioes = {"Norte": (0.02, "alta"), "Sul": (0.03, "alta"), "Sudeste": (0.001, "baixa")}
    html = G.onde(regioes, "Sudeste")
    assert html.count("seu cenário") == 1
    assert html.index("Sul") < html.index("Norte") < html.index("Sudeste")  # da maior à menor


# --------------------------------------------------------------------------
# Conteúdo: o formulário fala a língua de quem decide, a API recebe o que espera
# --------------------------------------------------------------------------
def test_corpo_api_converte_o_formulario():
    corpo = C.corpo_api({**C.CENARIOS[0]["valores"], "canal": "nenhum",
                         "associacao_marca": "nenhuma"})
    assert corpo["canal"] is None and corpo["associacao_marca"] is None
    assert corpo["renda_pc"] == C.RENDAS[C.CENARIOS[0]["valores"]["renda"]][1]
    assert isinstance(corpo["renda_pc"], float) and isinstance(corpo["urbano"], bool)
    assert "renda" not in corpo


def test_faixas_de_renda_sao_as_do_projeto():
    assert list(C.RENDAS) == build_grid.FAIXAS_RENDA
    assert C.SALARIO_MINIMO_2018 == build_grid.SALARIO_MINIMO_2018
    limites = [0, 0.5, 1, 2, 5, float("inf")]
    for (faixa, (_, valor)), lo, hi in zip(C.RENDAS.items(), limites[:-1], limites[1:],
                                           strict=True):
        assert lo <= valor / C.SALARIO_MINIMO_2018 < hi, faixa
        assert build_grid.faixa_renda(pd.Series([valor]))[0] == faixa


def test_regioes_e_dias_sao_os_do_modelo():
    for codigo, sigla in preparo.UFS.items():
        assert C.REGIAO_DA_UF[sigla] == build_grid.REGIOES[codigo[0]], sigla
    assert set(C.REGIAO_DA_UF) == set(C.UFS)
    assert C.FIM_DE_SEMANA == tuple(build_grid.FIM_DE_SEMANA)
    assert C.BLOCOS_DE_DIAS["Sexta a domingo"] in C.FIM_DE_SEMANA
    assert C.BLOCOS_DE_DIAS["Segunda a quinta"] not in C.FIM_DE_SEMANA


def test_cenarios_usam_opcoes_do_formulario():
    assert all(0 <= anos <= 16 for anos in C.ESCOLARIDADES)
    for cenario in C.CENARIOS:
        v = cenario["valores"]
        assert set(v) == set(C.CAMPOS), cenario["nome"]
        assert v["anos_estudo"] in C.ESCOLARIDADES and v["renda"] in C.RENDAS
        assert v["canal"] in C.CANAIS and v["associacao_marca"] in C.ASSOCIACOES
    assert set(C.CAMINHO_SIMPLES) == set(C.QUANDO_SIMPLES) == set(F.CAMINHOS)


# --------------------------------------------------------------------------
# Cenários e números
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def api():
    if not TEM_ARTEFATOS:
        pytest.skip("artefatos do modelo não gerados")
    from src.api.main import app

    with TestClient(app) as c:
        yield c


def test_cenarios_caem_na_regra_anunciada(api):
    assert sorted(c["regra"] for c in C.CENARIOS) == ["R1", "R2", "R3", "R4", "R5", "R6"]
    for cenario in C.CENARIOS:
        r = api.post("/predict", json=C.corpo_api(cenario["valores"]))
        assert r.status_code == 200, cenario["nome"]
        assert r.json()["model_metadata"]["decision_rule"] == cenario["regra"], cenario["nome"]


@pytest.mark.skipif(not TABELAS.exists(), reason="avaliação no teste não gerada")
def test_numeros_da_avaliacao_batem_com_o_relatorio_do_teste():
    texto = TABELAS.read_text(encoding="utf-8")
    secoes = {s.split("\n", 1)[0].strip(): s for s in texto.split("\n## ")[1:]}

    def tabela(prefixo: str) -> dict:
        corpo = next(s for titulo, s in secoes.items() if titulo.startswith(prefixo))
        return {c[0].strip(): [x.strip() for x in c[1:]] for c in
                (linha.strip("|").split("|") for linha in corpo.splitlines()
                 if linha.startswith("|"))}

    linhas = {**tabela("1. Métricas"), **tabela("2. Intervalos"), **tabela("3. Classes")}
    a = C.AVALIACAO
    assert linhas["PR-AUC"] == [a["pr_auc"]["modelo"], a["pr_auc"]["baseline"]]
    assert linhas["captura no top 10%"] == [a["captura_10"]["modelo"],
                                            a["captura_10"]["baseline"]]
    assert linhas["ROC-AUC"] == [a["roc_auc"]["modelo"], a["roc_auc"]["baseline"]]
    assert linhas["recall em alta"] == [a["recall_alta"]["modelo"], a["recall_alta"]["baseline"]]
    assert linhas["precisão em alta"] == [a["precisao_alta"]["modelo"],
                                          a["precisao_alta"]["baseline"]]
    assert linhas["fração do peso em alta"] == [a["fracao_alta"]["modelo"],
                                                a["fracao_alta"]["baseline"]]
    assert linhas["taxa prevista"][0] == a["calibracao"]["prevista"]
    assert linhas["taxa observada"][0] == a["calibracao"]["observada"]
    assert linhas["modelo final"] == [a["pr_auc"]["modelo"], a["pr_auc"]["ic_modelo"],
                                      a["captura_10"]["modelo"], a["captura_10"]["ic_modelo"]]
    assert linhas["baseline"] == [a["pr_auc"]["baseline"], a["pr_auc"]["ic_baseline"],
                                  a["captura_10"]["baseline"], a["captura_10"]["ic_baseline"]]
    assert linhas["diferença final − baseline"] == [
        a["diferenca_pr_auc"]["valor"], a["diferenca_pr_auc"]["ic"],
        a["diferenca_captura"]["valor"], a["diferenca_captura"]["ic"]]
    alta, baixa = linhas["alta"], linhas["baixa"]
    assert (alta[0], alta[2], alta[5], alta[6]) == (a["alta"]["peso"], a["alta"]["ocasioes"],
                                                    a["alta"]["taxa"], a["alta"]["vezes"])
    assert (baixa[0], baixa[2]) == (a["baixa"]["peso"], a["baixa"]["ocasioes"])
    assert tabela("4. Calibração geral")["modelo final"][2] == a["calibracao"]["razao"]
    positivas = re.search(r"diferença positiva: ([\d,]+%)", texto).group(1)
    assert positivas == a["diferenca_pr_auc"]["positivas"]

    # "Dá para confiar?" arredonda os mesmos números
    def redondo(s: str) -> str:
        return f"{round(float(s.rstrip('%×').replace(',', '.'))):d}%"

    c = C.CONFIANCA
    assert c["alta_ocasioes"] == redondo(a["alta"]["ocasioes"])
    assert c["alta_momentos"] == redondo(a["alta"]["peso"])
    assert c["regra_simples"] == redondo(a["recall_alta"]["baseline"])
    assert c["baixa_ocasioes"] == redondo(a["baixa"]["ocasioes"])
    assert c["baixa_momentos"] == redondo(a["baixa"]["peso"])
    razao = float(a["calibracao"]["razao"].replace(",", "."))
    assert c["acima_do_estimado"] == f"{round((razao - 1) * 100)}%"
    assert all(f"{c['acima_do_estimado']} acima" in t for t in C.CUIDADOS[:1])


# --------------------------------------------------------------------------
# Onde e para quem ativar
# --------------------------------------------------------------------------
MAPA = MODELOS / "oportunidade.json"


@pytest.fixture(scope="module")
def mapa():
    import json

    return json.loads(MAPA.read_text(encoding="utf-8"))


def test_textos_do_mapa_para_quem_decide():
    p = {"sexo": "masculino", "faixa_etaria": "65+", "regiao": "Sul", "bloco": "seg_qui",
         "periodo": "almoco"}
    assert G.rotulo_publico(p) == ("Homens de 65 anos ou mais · Sul",
                                   "Segunda a quinta, no almoço")
    assert G.dias_texto(["Sexta-Feira", "Sabado", "Domingo"]) == "sexta, sábado e domingo"
    assert G.horario_texto(["18:00-24:00", "00:00-05:00"]) == \
        "das 18h à meia-noite e da meia-noite às 5h"
    assert G.horario_texto(["11:00-15:00"]) == "das 11h às 15h"


def test_rotulos_cobrem_o_mapa(mapa):
    from src.oportunidade import ranking

    assert set(C.FAIXAS_ETARIAS) == set(ranking.FAIXAS_ETARIAS)
    assert set(C.BLOCOS) == set(ranking.BLOCOS.values())
    assert {p["regra"] for p in mapa["publicos"]} <= set(C.ACAO_SEM_CANAL)
    assert {e["categoria"] for e in mapa["estados"]} <= set(C.CANAIS_DOS_ESTADOS)
    assert {p["renda_mais_comum"] for p in mapa["publicos"]} <= set(C.RENDAS)


def test_volume_fora_da_noite_vem_do_mapa(mapa):
    estimado, observado = G.fora_da_noite(mapa["publicos"])
    # os números do adendo de docs/oportunidade.md
    assert round(estimado, 3) == 0.156 and round(observado, 3) == 0.054


def test_filtros_do_mapa(mapa):
    todos = G.filtrar_publicos(mapa["publicos"], "Todas", "Todos")
    assert len(todos) == sum(p["amostra_suficiente"] for p in mapa["publicos"])
    assert G.filtrar_publicos(mapa["publicos"], None, None) == todos
    norte_mulheres = G.filtrar_publicos(mapa["publicos"], "Norte", "Mulheres")
    assert norte_mulheres and all(p["regiao"] == "Norte" and p["sexo"] == "feminino"
                                  for p in norte_mulheres)
    assert len(norte_mulheres) == 6 * 8  # 6 faixas etárias × 8 momentos


def test_quadros_do_mapa(mapa):
    pub = sorted(mapa["publicos"], key=lambda p: p["posicao_chance"])[:3]
    html = G.ranking_publicos(pub, "chance", mapa["chance_media_adultos"])
    assert html.count('class="item"') == 3 and "4,4%" in html and "prioridade alta" in html
    assert "Nenhum público" in G.ranking_publicos([], "volume", 0.003)
    cartao = G.cartao_anuncio(pub[0], mapa["chance_media_adultos"])
    assert "DF, GO, MS e MT (Centro-Oeste)" in cartao and "Sexta, sábado e domingo" in cartao
    estados = G.estados_por_canal(mapa["estados"])
    assert estados.count('class="uf"') == 27  # cada estado em um único cartão


# --------------------------------------------------------------------------
# A página
# --------------------------------------------------------------------------
def _rodar_app(monkeypatch, requisitar):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(cliente, "_requisitar", requisitar)
    at = AppTest.from_file(str(APP / "app.py"), default_timeout=90)
    at.run()
    assert not at.exception, at.exception
    return at, " ".join(m.value for m in at.markdown)


def test_app_mostra_chance_acao_fatores_e_caminho(monkeypatch, api):
    def pela_api(metodo, caminho, corpo=None):
        r = api.request(metodo, caminho, json=corpo)
        if r.status_code >= 400:
            raise cliente.ApiErro(r.status_code, r.json().get("detail"))
        return r.json()

    at, texto = _rodar_app(monkeypatch, pela_api)
    assert len(at.tabs) == 5
    assert "2,4%" in texto and "prioridade alta" in texto  # o primeiro exemplo (R3)
    assert "Ativação via delivery" in texto and "Caminho aceso: Ação no delivery" in texto
    assert "× 13,2" in texto  # o momento, maior fator do exemplo
    assert texto.count("seu cenário") == 2  # um no quadro de quando, outro no de onde
    # as quatro abas para quem decide marketing ficam sem jargão de dados
    for aba in at.tabs[:4]:
        visivel = _texto_visivel(" ".join(m.value for m in aba.markdown))
        assert len(visivel) > 1000  # a aba foi lida de fato
        assert not JARGAO.search(visivel), JARGAO.search(visivel)
    # o Mapa de Oportunidade: as duas listas, o anúncio, os canais, os cuidados e as fontes
    onde = _texto_visivel(" ".join(m.value for m in at.tabs[1].markdown))
    assert "Maior chance por pessoa" in onde and "Mais ocasiões no total" in onde
    assert "Homens de 25 a 34 anos · Centro-Oeste" in onde  # 1º por chance
    assert "Mulheres de 25 a 34 anos · Sudeste" in onde  # 1º por volume
    assert "Parâmetros de anúncio" in onde and "Das 18h à meia-noite" in onde
    assert "Delivery" in onde and "Combo para casa no varejo" in onde
    assert "16% das ocasiões estimadas, contra 5% das observadas" in onde
    assert "© colaboradores do OpenStreetMap" in onde
    # a aba técnica guarda os termos técnicos e a última simulação
    tecnica = " ".join(m.value for m in at.tabs[4].markdown)
    assert "PR-AUC" in tecnica and "Caminho aceso: regra R3" not in tecnica
    assert "regra R3" in tecnica


def test_app_avisa_quando_a_api_esta_fora(monkeypatch):
    def fora(metodo, caminho, corpo=None):
        raise cliente.ApiIndisponivel("conexão recusada")

    _, texto = _rodar_app(monkeypatch, fora)
    assert "O cálculo está fora do ar" in texto and "Cálculo fora do ar" in texto
    assert "API fora do ar" in texto  # na aba técnica
