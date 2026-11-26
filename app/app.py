"""Occasion Intelligence · Pizza + Guaraná: o app (Streamlit), para quem decide marketing.

Simula um cenário (quando, onde, para quem, canal e associação à marca), chama a API e mostra
a chance da ocasião, a prioridade, os 3 fatores que mais pesam e a ação recomendada, com o
caminho da decisão aceso; mostra também quando e onde a chance é maior para o mesmo público.
"Onde e para quem ativar" traz o Mapa de Oportunidade (docs/oportunidade.md): os públicos e
momentos de maior chance e de maior volume, com os parâmetros de anúncio, e os estados por
canal. "Como funciona" e "Dá para confiar?" explicam o método sem jargão; "Detalhes técnicos"
guarda os fluxogramas técnicos, as métricas do teste e o contrato da API.

O app só fala com o modelo pela API (app/cliente.py). Rodar cada comando em um terminal,
os dois com o ambiente ativado (source .venv/bin/activate):
    uvicorn src.api.main:app
    streamlit run app/app.py
"""

from __future__ import annotations

import json
import re
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cliente  # noqa: E402
import conteudo as C  # noqa: E402
import estilo as E  # noqa: E402
import fluxogramas as F  # noqa: E402
import graficos as G  # noqa: E402
import streamlit as st  # noqa: E402

FIGURAS = Path(__file__).resolve().parents[1] / "reports" / "figures"
VERSAO_DO_FORMULARIO = 2  # muda quando os campos mudam: sessões antigas recomeçam do 1º exemplo

st.set_page_config(page_title="Pizza + Guaraná · Occasion Intelligence", page_icon="🍕",
                   layout="wide", initial_sidebar_state="collapsed")
st.html(E.CSS)  # só estilo: o st.html preserva <style>


def bloco(html: str) -> None:
    """HTML com SVG. O st.html limpa SVG; o st.markdown com HTML liberado o preserva.

    O "$" vira entidade para o Markdown não ler "R$ ... R$" como fórmula.
    """
    st.markdown(html.replace("$", "&#36;"), unsafe_allow_html=True)


def lista(itens: list[str]) -> str:
    """Itens já em HTML simples (negrito permitido)."""
    return '<ul class="oi-lista">' + "".join(f"<li>{t}</li>" for t in itens) + "</ul>"


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def metadados() -> tuple[dict, dict]:
    return cliente.modelo(), cliente.regras()


@st.cache_data(ttl=600, show_spinner=False)
def mapa() -> dict:
    return cliente.oportunidade()


@st.cache_data(ttl=600, show_spinner=False)
def _prever(corpo_json: str) -> dict:
    return cliente.prever(json.loads(corpo_json))


def prever(corpo: dict) -> dict:
    """POST /predict com cache: o quadro de quando e onde repete o mesmo perfil 13 vezes."""
    return _prever(json.dumps(corpo, sort_keys=True))


def estado_da_api() -> tuple[dict | None, dict | None, dict | None]:
    """(saúde, metadados do modelo, regras), ou (None, None, None) se a API não responder."""
    try:
        saude = cliente.saude()
        modelo, regras = metadados()
        return saude, modelo, regras
    except (cliente.ApiIndisponivel, cliente.ApiErro):
        return None, None, None


def aviso_fora_do_ar() -> str:
    return ('<div class="oi-aviso"><b>O cálculo está fora do ar.</b> O app depende de um serviço '
            'de cálculo (a API), que precisa estar ligado. Para ligá-lo, abra outro terminal, '
            'com o ambiente do projeto ativado: <code>source .venv/bin/activate</code> e '
            '<code>uvicorn src.api.main:app</code> (endereço esperado: '
            f'<code>{escape(cliente.API_URL)}</code>). Em seguida, recarregue a página.</div>')


# --------------------------------------------------------------------------
# Topo
# --------------------------------------------------------------------------
def topo(modelo: dict | None) -> None:
    chips = ['<span class="chip">46 mil pessoas · pesquisa do IBGE (POF 2017–2018)</span>',
             '<span class="chip">838 ocasiões de pizza com refrigerante</span>',
             '<span class="chip">testado em 5 meses que o modelo não tinha visto</span>']
    if modelo is None:
        chips.insert(0, '<span class="chip fora"><span class="luz"></span>Cálculo fora do ar'
                        '</span>')
    bloco(
        '<div class="oi-topo"><div class="sobre">Occasion Intelligence · case Inteli × Ambev'
        '</div><div class="oi-titulo">Pizza <span class="mais">+</span> Guaraná</div>'
        '<p>Descubra quando, onde e para quem a ocasião de pizza com refrigerante é mais '
        'provável, e qual ação de marketing faz sentido para o Guaraná.</p>'
        f'<div class="chips">{"".join(chips)}</div>{E.ILUSTRACAO}</div>')


# --------------------------------------------------------------------------
# Simulador
# --------------------------------------------------------------------------
def aplicar(valores: dict) -> None:
    for chave, valor in valores.items():
        st.session_state[chave] = valor


def ao_escolher_cenario() -> None:
    nome = st.session_state.get("cenario")
    for cenario in C.CENARIOS:
        if cenario["nome"] == nome:
            aplicar(cenario["valores"])


def ao_mudar() -> None:
    """Mexer num campo tira a marcação do exemplo pronto (o cenário deixou de ser ele)."""
    st.session_state["cenario"] = None


def formulario() -> dict:
    """Os campos do cenário, nos termos de quem decide marketing (os valores do formulário)."""
    if st.session_state.get("versao_formulario") != VERSAO_DO_FORMULARIO:
        aplicar(C.CENARIOS[0]["valores"])
        st.session_state["cenario"] = C.CENARIOS[0]["nome"]
        st.session_state["versao_formulario"] = VERSAO_DO_FORMULARIO

    bloco('<div class="oi-rotulo">Exemplos prontos · cada um leva a uma ação diferente</div>')
    st.pills("Exemplos prontos", [c["nome"] for c in C.CENARIOS], key="cenario",
             on_change=ao_escolher_cenario, label_visibility="collapsed")
    with st.container(border=True, key="formulario"):
        bloco('<div class="oi-rotulo">Quando e onde</div>')
        c1, c2, c3, c4 = st.columns([1.4, 1.8, 0.74, 0.85], gap="medium")
        c1.segmented_control("Período", list(C.PERIODOS), key="periodo", required=True,
                             format_func=C.PERIODOS.get, on_change=ao_mudar,
                             help="Manhã 5h–10h · almoço 11h–14h · tarde 15h–17h · noite 18h–4h")
        c2.segmented_control("Dia da semana", list(C.DIAS), key="dia_semana", required=True,
                             format_func=C.DIAS.get, on_change=ao_mudar)
        c3.selectbox("Estado", sorted(C.UFS, key=C.UFS.get), key="uf", on_change=ao_mudar,
                     format_func=C.UFS.get)
        c4.segmented_control("Área", [True, False], key="urbano", required=True,
                             format_func=C.AREAS.get, on_change=ao_mudar)
        bloco('<div class="oi-rotulo">Para quem</div>')
        q1, q2, q3, q4, q5 = st.columns([1.0, 0.8, 1.05, 0.75, 1.25], gap="medium")
        q1.slider("Idade", 10, 100, key="idade", on_change=ao_mudar)
        q2.segmented_control("Sexo", list(C.SEXOS), key="sexo", required=True,
                             format_func=C.SEXOS.get, on_change=ao_mudar)
        q3.selectbox("Escolaridade", list(C.ESCOLARIDADES), key="anos_estudo",
                     format_func=C.ESCOLARIDADES.get, on_change=ao_mudar)
        q4.number_input("Pessoas na casa", 1, 20, step=1, key="n_moradores", on_change=ao_mudar)
        q5.selectbox("Renda por pessoa", list(C.RENDAS), key="renda",
                     format_func=lambda k: C.RENDAS[k][0], on_change=ao_mudar,
                     help="Renda mensal da família dividida pelo número de pessoas na casa.")
        bloco('<div class="oi-rotulo">Sobre a ação · não mudam a chance, só a recomendação</div>')
        r1, r2, r3 = st.columns([1.65, 0.9, 1.15], gap="medium")
        r1.segmented_control("Onde você pretende agir?", list(C.CANAIS), key="canal",
                             required=True, format_func=C.CANAIS.get, on_change=ao_mudar,
                             help="Supermercado inclui o varejo em geral; pizzaria ou bar inclui "
                                  "restaurantes e lanchonetes.")
        r2.segmented_control("O público já liga Guaraná a pizza?", list(C.ASSOCIACOES),
                             key="associacao_marca", required=True,
                             format_func=C.ASSOCIACOES.get, on_change=ao_mudar)
        r3.html('<div class="oi-nota" style="margin-top:4px">Essas duas respostas vêm de quem '
                'planeja a ação (plano de mídia, pesquisa de marca). Elas não mudam a chance: só '
                'escolhem a recomendação.</div>')

    return {chave: st.session_state[chave] for chave in C.CAMPOS}


def frases(texto: str) -> list[str]:
    """Quebra a justificativa em frases (o português da API usa vírgula decimal)."""
    return [f for f in re.split(r"(?<=[.])\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ])", texto) if f]


def porque(texto: str) -> list[str]:
    """A justificativa sem as frases da chance e dos fatores, que já têm cartão próprio."""
    return [f for f in frases(texto)
            if not f.startswith(("A chance da ocasião é de", "O que mais pesa"))]


def resumo_cenario(v: dict) -> str:
    """O cenário avaliado em etiquetas curtas, com as palavras do formulário."""
    n = v["n_moradores"]
    partes = [f"{C.DIAS_EXTENSO[v['dia_semana']]} {C.NO_PERIODO[v['periodo']]}",
              f"{C.UFS[v['uf']]} · {C.AREAS[v['urbano']].lower()}",
              f"{C.SEXOS[v['sexo']]}, {v['idade']} anos", C.ESCOLARIDADES[v["anos_estudo"]],
              f"{n} pessoa{'s' if n > 1 else ''} na casa",
              f"renda: {C.RENDAS[v['renda']][0].lower()} por pessoa",
              f"canal: {C.CANAIS[v['canal']].lower()}",
              f"liga Guaraná a pizza: {C.ASSOCIACOES[v['associacao_marca']].lower()}"]
    return '<div class="oi-ctx">' + "".join(f"<span>{escape(p)}</span>" for p in partes) + "</div>"


def textos_dos_fatores(v: dict) -> dict[str, str]:
    """Os fatores com as palavras do formulário, no lugar das da API (ex.: reais de 2018)."""
    n = v["n_moradores"]
    return {"renda": f"{C.RENDAS[v['renda']][0]} por pessoa",
            "escolaridade": C.ESCOLARIDADES[v["anos_estudo"]],
            "sexo": C.SEXOS[v["sexo"]], "area": C.AREAS[v["urbano"]],
            "moradores": f"{n} pessoa{'s' if n > 1 else ''}"}


def resultado(resposta: dict, regras: dict, v: dict) -> None:
    score, classe = resposta["occasion_score"], resposta["classification"]
    referencia = resposta["model_metadata"]["reference_rate"]
    categoria = regras["categories"][resposta["recommendation_category"]]
    titulo, _, descricao = categoria.partition(": ")
    razoes = "".join(f"<p>{escape(f)}</p>" for f in porque(resposta["recommendation_reason"]))

    cartao_chance = (
        '<div class="oi-cartao verde"><div class="oi-rotulo">Chance da ocasião</div>'
        f'<div class="oi-score"><span class="valor">{G.pct_simples(score, compacto=True)}'
        f'</span>{E.classe_html(classe, "prioridade ")}</div>'
        f'<div class="oi-freq">Estimativa: <b>{G.em_cada(score)}</b> pessoas com esse perfil '
        'comem pizza com refrigerante nesse período.</div>'
        f'<div class="oi-media">{G.comparado(score / referencia).capitalize()} de um momento '
        f'qualquer ({G.pct_simples(referencia)}).</div>{G.prioridade(classe)}'
        f'<div class="oi-divisor"></div><div class="oi-rotulo">Seu cenário</div>'
        f'{resumo_cenario(v)}</div>')
    cartao_acao = (
        '<div class="oi-cartao oi-reco"><div class="oi-rotulo">Ação recomendada</div>'
        f'<h3>{escape(titulo)}</h3><p class="descricao">{escape(descricao)}</p>'
        f'<div class="razao"><div class="porque">Por quê</div>{razoes}</div>'
        '<div class="oi-nota">A recomendação mostra onde a ocasião costuma acontecer; não é uma '
        'previsão de vendas.</div></div>')
    cartao_fatores = (
        '<div class="oi-cartao"><div class="oi-rotulo">O que mais pesa nessa chance</div>'
        f'{G.fatores(resposta["top_factors"], textos_dos_fatores(v))}'
        '<div class="oi-legenda">Quantas vezes cada fator multiplica a chance, comparado a um '
        'momento médio. Vermelho aumenta; verde reduz.</div></div>')
    bloco(f'<div class="oi-grade r3">{cartao_chance}{cartao_acao}{cartao_fatores}</div>')


def caminho_da_decisao(resposta: dict) -> None:
    classe = resposta["classification"]
    regra = resposta["model_metadata"]["decision_rule"]
    prioridade = E.ROTULO_CLASSE[classe]
    chance = G.pct_simples(resposta["occasion_score"], compacto=True)
    svg = F.decision_engine(regra, (f"Chance {chance}", f"prioridade {prioridade}"),
                            prefixo="sim", linguagem="simples")
    bloco(E.secao("Como a ação foi escolhida",
                  "Perguntas simples, sempre na mesma ordem; o caminho deste cenário está aceso."))
    bloco(E.figura(svg, f"<b>Neste cenário,</b> "
                        f"{escape(C.CAMINHO_SIMPLES[regra].format(prioridade=prioridade))}. As "
                        "regras partem de hipóteses de marketing e dos dados da pesquisa; elas "
                        "não preveem vendas."))


def panorama(v: dict, modelo: dict) -> None:
    """O mesmo perfil em outros momentos (quando) e em outras regiões (onde)."""
    perfil = {**C.corpo_api({**v, "canal": "nenhum", "associacao_marca": "nenhuma"}),
              "model_version": modelo["model_version"]}
    celulas = {}
    for bloco_de_dias, dia in C.BLOCOS_DE_DIAS.items():
        for periodo in C.PERIODOS:
            r = prever({**perfil, "dia_semana": dia, "periodo": periodo})
            celulas[(bloco_de_dias, periodo)] = (r["occasion_score"], r["classification"])
    regioes = {}
    for regiao, ufs in C.REGIOES.items():
        r = prever({**perfil, "uf": ufs[0]})
        regioes[regiao] = (r["occasion_score"], r["classification"])

    bloco_atual = ("Sexta a domingo" if v["dia_semana"] in C.FIM_DE_SEMANA
                   else "Segunda a quinta")
    regiao_atual = C.REGIAO_DA_UF[v["uf"]]
    momento = f"{C.DIAS_EXTENSO[v['dia_semana']].lower()} {C.NO_PERIODO[v['periodo']]}"
    quando = G.quando(celulas, C.PERIODOS, C.HORARIOS, (bloco_atual, v["periodo"]))
    onde = G.onde(regioes, regiao_atual)
    bloco(E.secao("Quando e onde a chance é maior para este público",
                  "O mesmo perfil em outros momentos e regiões, para escolher quando e onde "
                  "ativar."))
    bloco(
        '<div class="oi-grade r2">'
        f'<div class="oi-cartao"><div class="oi-rotulo">Quando · {escape(regiao_atual)}</div>'
        f'{quando}<div class="oi-legenda">A cor é a prioridade. O modelo separa a semana em dois '
        'blocos, segunda a quinta e sexta a domingo: dentro de cada bloco, todos os dias têm a '
        'mesma chance.</div></div>'
        f'<div class="oi-cartao"><div class="oi-rotulo">Onde · {escape(momento)}</div>{onde}'
        '<div class="oi-legenda">O modelo estima por região: os estados de uma mesma região têm '
        'a mesma chance.</div></div></div>')


def aba_simular(modelo: dict | None, regras: dict | None) -> None:
    bloco(E.secao("Monte o cenário",
                  "Escolha um exemplo pronto ou ajuste quando, onde e para quem. O resultado "
                  "aparece logo abaixo."))
    v = formulario()
    if modelo is None:
        bloco(aviso_fora_do_ar())
        return
    corpo = {**C.corpo_api(v), "model_version": modelo["model_version"]}
    try:
        resposta = prever(corpo)
    except cliente.ApiErro as erro:
        bloco(f'<div class="oi-aviso"><b>O cálculo recusou o cenário ({erro.status}).</b> '
              f'{escape(str(erro.detalhe))}</div>')
        return
    except cliente.ApiIndisponivel:
        bloco('<div class="oi-aviso"><b>O cálculo parou de responder.</b> Confira o terminal '
              'da API (uvicorn) e recarregue a página.</div>')
        return
    st.session_state["ultima_simulacao"] = (corpo, resposta)
    bloco(E.secao("Resultado"))
    resultado(resposta, regras, v)
    caminho_da_decisao(resposta)
    try:
        panorama(v, modelo)
    except (cliente.ApiErro, cliente.ApiIndisponivel):
        bloco('<div class="oi-aviso">Não deu para calcular os outros momentos e regiões agora. '
              'Recarregue a página.</div>')


# --------------------------------------------------------------------------
# Onde e para quem ativar
# --------------------------------------------------------------------------
TOPO_DAS_LISTAS = 8


def aba_onde(modelo: dict | None) -> None:
    bloco(E.secao("Onde e para quem ativar",
                  "A mesma estimativa do simulador, agora para todos os públicos de adultos de "
                  "uma vez: quem priorizar, em que momento, em quais estados e por qual canal."))
    if modelo is None:
        bloco(aviso_fora_do_ar())
        return
    try:
        m = mapa()
    except (cliente.ApiErro, cliente.ApiIndisponivel):
        bloco('<div class="oi-aviso"><b>Não deu para carregar o mapa agora.</b> Recarregue a '
              'página.</div>')
        return
    media = m["chance_media_adultos"]

    bloco(E.secao("Para quem e quando anunciar",
                  "Cada público junta idade, sexo e região, num momento da semana. Duas listas, "
                  "porque são duas decisões: onde cada pessoa alcançada vale mais, e onde está o "
                  f"maior volume. A chance média dos adultos é {G.pct_simples(media)}."))
    f1, f2, _ = st.columns([1.9, 1.0, 0.9])
    regiao = f1.segmented_control("Região", ["Todas", *C.REGIOES], default="Todas",
                                  key="onde_regiao")
    genero = f2.segmented_control("Sexo", ["Todos", *C.GENEROS.values()], default="Todos",
                                  key="onde_sexo")
    itens = G.filtrar_publicos(m["publicos"], regiao, genero)
    por_chance = sorted(itens, key=lambda p: p["posicao_chance"])[:TOPO_DAS_LISTAS]
    por_volume = sorted(itens, key=lambda p: p["posicao_parcela"])[:TOPO_DAS_LISTAS]
    bloco(
        '<div class="oi-grade r2">'
        '<div class="oi-cartao"><div class="oi-rotulo">Maior chance por pessoa</div>'
        f'{G.ranking_publicos(por_chance, "chance", media)}<div class="oi-legenda">Onde a '
        'mensagem tem mais chance de encontrar a ocasião. Bom para ações caras por pessoa '
        'alcançada.</div></div>'
        '<div class="oi-cartao"><div class="oi-rotulo">Mais ocasiões no total</div>'
        f'{G.ranking_publicos(por_volume, "volume", media)}<div class="oi-legenda">A parte de '
        'todas as ocasiões de pizza com refrigerante dos adultos que acontece com esse público, '
        'naquele momento. Bom para alcance.</div></div></div>')

    escolhas = {}
    for p in por_chance + por_volume:
        quem, momento = G.rotulo_publico(p)
        escolhas.setdefault(f"{quem} · {momento.lower()}", p)
    if escolhas:
        bloco(E.secao("Como anunciar para esse público",
                      "Escolha um dos públicos das listas para ver os parâmetros de anúncio."))
        nome = st.selectbox("Público", list(escolhas), key="onde_publico",
                            label_visibility="collapsed")
        bloco(G.cartao_anuncio(escolhas.get(nome, next(iter(escolhas.values()))), media))

    bloco(E.secao("Em quais estados e por qual canal",
                  "Cada estado é comparado com os outros em duas coisas: a chance média dos "
                  "adultos dele e quantas pizzarias tem por 100 mil habitantes. Dentro de cada "
                  "canal, os estados vêm em ordem de volume de ocasiões."))
    bloco(G.estados_por_canal(m["estados"]))
    bloco('<div class="oi-legenda">Ocasiões: a parte das ocasiões dos adultos do país que '
          'acontece no estado. Pizzarias: quantas há por 100 mil habitantes.</div>')
    with st.expander("Ver os 27 estados em ordem de prioridade", icon=":material/list:"):
        st.dataframe([{
            "Ordem": e["posicao"], "Estado": e["nome"], "Região": e["regiao"],
            "Chance média dos adultos": G.pct_simples(e["chance"]),
            "Parte das ocasiões do país": f"{e['parcela_das_ocasioes'] * 100:.1f}%".replace(
                ".", ","),
            "Pizzarias por 100 mil hab.": f"{e['pizzarias_100mil']:.1f}".replace(".", ","),
            "Canal": C.CANAIS_DOS_ESTADOS[e["categoria"]][0],
        } for e in m["estados"]], hide_index=True, width="stretch")

    estimado, observado = G.fora_da_noite(m["publicos"])
    bloco(E.secao("Antes de usar estes números") + lista(C.cuidados_do_mapa(
        f"{estimado * 100:.0f}%", f"{observado * 100:.0f}%")))
    bloco(f'<div class="oi-legenda">{escape(C.FONTES_DO_MAPA)}</div>')


# --------------------------------------------------------------------------
# Como funciona
# --------------------------------------------------------------------------
def cartoes_de_fatos(fatos: list[tuple[str, str, str]]) -> str:
    return ('<div class="oi-grade r4">' + "".join(
        f'<div class="oi-cartao oi-fato"><div class="titulo">{escape(t)}</div>'
        f'<div class="grande">{escape(g)}</div><p>{escape(p)}</p></div>' for t, g, p in fatos)
        + "</div>")


def aba_como_funciona(regras: dict | None) -> None:
    bloco(E.secao("Como funciona",
                  "Em um minuto: de onde vêm os dados, o que a chance quer dizer e como a ação "
                  "é escolhida."))
    bloco(cartoes_de_fatos(C.COMO_FUNCIONA))
    bloco(E.secao("O caminho da decisão",
                  "Perguntas simples, sempre na mesma ordem: a primeira resposta que leva a uma "
                  "ação decide."))
    bloco(E.figura(F.decision_engine(prefixo="geral", linguagem="simples"),
                   "<b>A ordem importa.</b> Com prioridade baixa, não há ação dedicada. Se o "
                   "público liga pouco Guaraná a pizza, a ação de awareness vem antes de qualquer "
                   "canal. Só com prioridade alta o canal escolhe a ação."))
    if regras:
        categoria_da_regra = {r["rule"]: r["category"] for r in regras["rules"]}
        cartoes = []
        for regra, categoria in categoria_da_regra.items():
            titulo, _, descricao = regras["categories"][categoria].partition(": ")
            cartoes.append(
                f'<div class="oi-cartao"><b style="font-size:15.5px">{escape(titulo)}</b>'
                f'<div class="oi-nota">{escape(descricao or titulo)}</div>'
                f'<div class="oi-nota"><b>Quando:</b> {escape(C.QUANDO_SIMPLES[regra])}</div>'
                '</div>')
        bloco(E.secao("As seis ações possíveis") + '<div class="oi-grade r3" '
              'style="grid-template-columns:repeat(3,1fr)">' + "".join(cartoes) + "</div>")
    bloco(E.secao("O que esta ferramenta não faz") + lista([escape(t) for t in C.NAO_FAZ]))


# --------------------------------------------------------------------------
# Dá para confiar?
# --------------------------------------------------------------------------
def aba_confianca() -> None:
    c = C.CONFIANCA
    bloco(E.secao("Dá para confiar?",
                  "O modelo foi construído com dados de julho de 2017 a fevereiro de 2018 e "
                  "testado uma única vez em março a julho de 2018, meses que ele não tinha visto. "
                  "A comparação é com uma regra simples, que só olha o dia e o horário."))
    numeros = [
        ("", c["alta_ocasioes"], f"das ocasiões reais do teste caíram nos momentos de prioridade "
                                 f"alta, que eram só {c['alta_momentos']} do total."),
        ("cinza", c["regra_simples"], "foi o que a regra simples reuniu nos momentos de "
                                      "prioridade alta dela, também cerca de 10% do total."),
        ("", c["baixa_ocasioes"], f"das ocasiões caíram em prioridade baixa, que somava "
                                  f"{c['baixa_momentos']} dos momentos."),
    ]
    bloco('<div class="oi-grade r3" style="grid-template-columns:repeat(3,1fr)">' + "".join(
        f'<div class="oi-cartao"><div class="oi-grande {cor}">{numero}</div>'
        f'<div class="oi-nota" style="font-size:14.5px;color:var(--tinta)">{escape(texto)}</div>'
        '</div>' for cor, numero, texto in numeros) + "</div>")
    bloco('<div class="oi-aviso" style="margin-top:16px"><b>A vantagem sobre a regra simples '
          'ainda pode ser acaso.</b> Pelo critério combinado antes do teste, o modelo não pode ser '
          'declarado melhor que ela. O app usa o modelo porque ele separa pessoas no mesmo dia e '
          'horário (idade, renda, escolaridade, região) e explica a chance de cada perfil; a '
          'regra simples dá a mesma chance para todo mundo no mesmo momento.</div>')
    bloco(E.secao("Cuidados ao usar") + lista(C.CUIDADOS))


# --------------------------------------------------------------------------
# Detalhes técnicos
# --------------------------------------------------------------------------
def numero(rotulo: str, modelo: str, baseline: str, extra: str) -> str:
    return (f'<div class="oi-cartao oi-numero"><div class="rotulo">{escape(rotulo)}</div>'
            f'<div class="linha"><span class="modelo">{modelo}</span>'
            f'<span class="base">baseline {baseline}</span></div>'
            f'<div class="ic">{extra}</div></div>')


def avaliacao_tecnica() -> None:
    a = C.AVALIACAO
    bloco(E.secao("4. Avaliação única no teste (mar–jul/2018)",
                  "O modelo congelado contra o baseline dia × período, com intervalos de 95% "
                  "por bootstrap pareado de UPAs (1.000 reamostragens)."))
    bloco(
        '<div class="oi-aviso"><b>Pela regra fixada antes do teste, o modelo não pode ser '
        'declarado melhor que o baseline.</b> A diferença de PR-AUC foi de '
        f'{a["diferenca_pr_auc"]["valor"]}, com intervalo de {a["diferenca_pr_auc"]["ic"]}, '
        f'que inclui zero ({a["diferenca_pr_auc"]["positivas"]} das reamostragens positivas). '
        'O modelo manteve o nível da validação e ficou à frente em todas as métricas de '
        'ordenação, sem significância; o baseline melhorou no teste. O app usa o modelo '
        'congelado, como o protocolo manda, porque ele diferencia pessoas no mesmo dia e '
        'período, e é isso que alimenta os fatores e o Decision Engine.</div>')
    bloco('<div class="oi-grade r4" style="margin-top:16px">' + "".join([
        numero("PR-AUC (métrica principal)", a["pr_auc"]["modelo"], a["pr_auc"]["baseline"],
               f'IC {a["pr_auc"]["ic_modelo"]} · validação cruzada {a["pr_auc"]["cv"]}'),
        numero("Captura no top 10%", a["captura_10"]["modelo"], a["captura_10"]["baseline"],
               f'diferença {a["diferenca_captura"]["valor"]} (IC {a["diferenca_captura"]["ic"]})'),
        numero("ROC-AUC", a["roc_auc"]["modelo"], a["roc_auc"]["baseline"],
               "complementar: com 99,6% de negativos, ela fica alta mesmo com topo ruim"),
        numero("Recall na classe alta", a["recall_alta"]["modelo"], a["recall_alta"]["baseline"],
               f'precisão {a["precisao_alta"]["modelo"]} contra {a["precisao_alta"]["baseline"]}'),
    ]) + "</div>")
    bloco(E.secao("Ordenação e calibração",
                  f'A classe alta reuniu {a["alta"]["ocasioes"]} das ocasiões em '
                  f'{a["alta"]["peso"]} do peso. Sem recalibrar, a taxa prevista ficou em '
                  f'{a["calibracao"]["prevista"]} e a observada em '
                  f'{a["calibracao"]["observada"]}.'))
    with st.container(border=True):
        st.image(str(FIGURAS / "teste_1_precisao_calibracao.png"), width="stretch")
    e, d = st.columns([1.05, 1], gap="large")
    with e, st.container(border=True):
        st.image(str(FIGURAS / "teste_2_shap_global.png"), width="stretch")
    d.html(E.secao("Onde o modelo erra") + lista(C.ERROS))
    bloco(E.secao("Limitações") + lista([escape(t) for t in C.LIMITACOES]))


def aba_tecnica(modelo: dict | None, regras: dict | None) -> None:
    bloco(E.secao("Detalhes técnicos",
                  "Para a equipe técnica e a avaliação do case: método, arquitetura, regras e "
                  "métricas, com os termos técnicos."))
    if modelo:
        bloco('<div class="oi-status">'
              f'<span class="chip"><span class="luz"></span>API no ar · '
              f'<code>{escape(cliente.API_URL)}</code></span>'
              f'<span class="chip">modelo {escape(modelo["model_version"])} · SHA-256 '
              f'<code>{escape(modelo["model_file_sha256"][:12])}…</code></span>'
              f'<span class="chip">Decision Engine {escape(modelo["decision_engine_version"])}'
              f'</span><span class="chip">treino: {escape(modelo["training_period"])}</span>'
              '</div>')
    else:
        bloco('<div class="oi-status"><span class="chip fora"><span class="luz"></span>API fora '
              f'do ar em <code>{escape(cliente.API_URL)}</code></span></div>')
    bloco(cartoes_de_fatos(C.FATOS))

    bloco(E.secao("1. Como o modelo foi construído e testado"))
    bloco(E.figura(F.treino_e_teste(),
                   "<b>O teste ficou fora de todas as escolhas.</b> Seleção do modelo, "
                   "calibração e cortes das classes usam só o treino. O modelo foi congelado num "
                   "commit, com o hash do arquivo, antes de ser avaliado uma única vez no teste, "
                   "sempre ao lado do baseline dia × período."))
    bloco(E.secao("2. O caminho de uma requisição"))
    bloco(E.figura(F.requisicao(),
                   "<b>Cada requisição usa o mesmo preparo do treino e só artefatos congelados."
                   "</b> A API confere o hash do modelo ao subir. Canal e associação à marca não "
                   "entram no modelo: vão direto para o Decision Engine. A resposta segue o "
                   "contrato de 6 campos."))
    bloco(E.secao("3. As regras do Decision Engine"))
    bloco(E.figura(F.decision_engine(prefixo="tecnica"),
                   "<b>A primeira regra que se aplica decide.</b> A classe diz se vale ativar; a "
                   "associação à marca, se a mensagem é de conversão ou de construção de "
                   "associação; o canal, por onde ativar. No app, a mesma árvore aparece em "
                   "linguagem simples."))
    if regras:
        quando = {r["category"]: (r["rule"], r["when"]) for r in regras["rules"]}
        cartoes = []
        for categoria, texto in regras["categories"].items():
            regra, condicao = quando[categoria]
            titulo, _, descricao = texto.partition(": ")
            cartao = (
                f'<div class="oi-cartao"><div class="oi-reco-topo"><span class="oi-regra">{regra}'
                f'</span><b style="font-size:15px">{escape(titulo)}</b></div>'
                f'<div class="oi-nota">{escape(descricao or titulo)}</div>'
                f'<div class="oi-nota"><b>Quando:</b> {escape(condicao)}.</div></div>')
            cartoes.append((regra, cartao))
        bloco('<div class="oi-grade r3" style="grid-template-columns:repeat(3,1fr);'
              'margin-top:16px">' + "".join(html for _, html in sorted(cartoes)) + "</div>")
    avaliacao_tecnica()

    ultima = st.session_state.get("ultima_simulacao")
    if ultima and modelo:
        corpo, resposta = ultima
        meta = resposta["model_metadata"]
        regra = meta["decision_rule"]
        caminho = C.CAMINHO_DA_REGRA[regra].format(
            classe=E.ROTULO_CLASSE[resposta["classification"]])
        bloco(E.secao("5. A última simulação, como a API vê",
                      f"Score {G.pct(resposta['occasion_score'])} na escala das classes "
                      f"(cortes P70 e P90 do treino) e regra {regra}: {caminho}."))
        bloco('<div class="oi-cartao" style="max-width:420px">'
              f'{G.termometro(resposta["occasion_score"], meta["classification_cuts"])}</div>')
        with st.expander("Contrato da API: o que o app enviou e o que voltou",
                         icon=":material/data_object:"):
            e, d = st.columns(2, gap="large")
            e.caption("Requisição: POST /predict")
            e.json(corpo)
            d.caption("Resposta: os 6 campos do contrato")
            d.json(resposta)


# --------------------------------------------------------------------------
# Página
# --------------------------------------------------------------------------
saude, modelo, regras = estado_da_api()
topo(modelo)
ABAS = [":material/insights: Simular uma ocasião", ":material/map: Onde e para quem ativar",
        ":material/menu_book: Como funciona", ":material/verified: Dá para confiar?",
        ":material/code: Detalhes técnicos"]
simular, onde, como_funciona, confianca, tecnica = st.tabs(ABAS)
with simular:
    aba_simular(modelo, regras)
with onde:
    aba_onde(modelo)
with como_funciona:
    aba_como_funciona(regras)
with confianca:
    aba_confianca()
with tecnica:
    aba_tecnica(modelo, regras)
rodape = ("Projeto acadêmico do case Inteli × Ambev (TAPI). Dados: POF 2017–2018 e Censo 2022, "
          "IBGE; pizzarias: © colaboradores do OpenStreetMap. "
          "Não é um produto oficial da Ambev: as cores lembram o Guaraná, sem uso de marcas.")
if modelo:
    rodape += (f" Modelo {escape(modelo['model_name'])} {escape(modelo['model_version'])}, "
               f"treinado em {escape(modelo['trained_at'][:10])}.")
bloco(f'<div class="oi-rodape">{rodape}</div>')
