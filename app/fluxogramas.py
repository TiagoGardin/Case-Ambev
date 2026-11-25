"""Fluxogramas do Occasion Intelligence, em SVG desenhado à mão (sem bibliotecas).

Cada figura defende uma ideia só:
- `treino_e_teste`: todas as escolhas usam só o treino, e o teste entra uma única vez, no fim;
- `requisicao`: uma requisição passa pelo mesmo preparo do treino e só usa artefatos
  congelados em models/; canal e associação à marca vão direto ao Decision Engine;
- `decision_engine`: a primeira regra que se aplica decide; com `regra`, o caminho dela acende.
  Com `linguagem="simples"`, a mesma árvore fala com quem decide marketing (prioridade no lugar
  de classe, sem os códigos R1 a R6).

O app usa a árvore simples na simulação e as três figuras técnicas na aba técnica.
`python app/fluxogramas.py` grava as três técnicas em reports/figures/, com fundo creme, para
o README.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

import estilo as E

# Aparência de cada estado de nó: preenchimento, borda, espessura, tracejado, cores do texto
ESTADOS = {
    "normal": (E.BRANCO, E.BORDA_FORTE, 1.4, None, E.TINTA, E.SUAVE, 1.0),
    "destaque": (E.VERDE_ESCURO, E.VERDE_ESCURO, 1.4, None, E.BRANCO, "#CFE6D8", 1.0),
    "teste": (E.BRANCO, E.DOURADO, 1.8, "5 4", E.TINTA, E.SUAVE, 1.0),
    "artefato": (E.BRANCO, E.VERDE, 1.3, None, E.VERDE_ESCURO, E.SUAVE, 1.0),
    "ativo": (E.VERMELHO_CLARO, E.VERMELHO, 2.4, None, E.TINTA, E.SUAVE, 1.0),
    "escolhido": (E.VERMELHO, E.VERMELHO, 2.4, None, E.BRANCO, "#FFE1DE", 1.0),
    "apagado": (E.BRANCO, E.BORDA, 1.2, None, E.SUAVE, E.SUAVE, 0.4),
}
COR_SETA = {"normal": E.CINZA_SETA, "ativo": E.VERMELHO, "apagado": E.BORDA_FORTE,
            "verde": E.VERDE, "dourado": "#C9920F"}


@dataclass
class Figura:
    """Uma figura SVG: elementos acumulados em ordem, ids prefixados para não colidir na página."""

    largura: int
    altura: int
    prefixo: str
    descricao: str
    fundo: str | None = None  # None = transparente (no app); cor = fundo pintado (README)
    partes: list[str] = field(default_factory=list)

    @property
    def halo(self) -> str:
        return self.fundo or E.BRANCO

    def add(self, elemento: str) -> None:
        self.partes.append(elemento)

    def svg(self) -> str:
        p = self.prefixo
        marcadores = "".join(
            f'<marker id="{p}-seta-{nome}" viewBox="0 0 10 10" refX="8.6" refY="5" '
            f'markerWidth="7.5" markerHeight="7.5" orient="auto-start-reverse">'
            f'<path d="M0.6,0.8 L9.4,5 L0.6,9.2 Z" fill="{cor}"/></marker>'
            for nome, cor in COR_SETA.items())
        sombra = (f'<filter id="{p}-sombra" x="-8%" y="-12%" width="116%" height="140%">'
                  f'<feDropShadow dx="0" dy="1.6" stdDeviation="2.4" flood-color="#0B4A30" '
                  f'flood-opacity="0.11"/></filter>')
        fundo = (f'<rect x="0" y="0" width="{self.largura}" height="{self.altura}" rx="18" '
                 f'fill="{self.fundo}"/>' if self.fundo else "")
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.largura} '
                f'{self.altura}" role="img" aria-label="{escape(self.descricao)}" '
                f'font-family="{escape(E.FONTE_TEXTO)}">'
                f'<title>{escape(self.descricao)}</title><defs>{marcadores}{sombra}</defs>'
                f'{fundo}{"".join(self.partes)}</svg>')


# --------------------------------------------------------------------------
# Primitivas
# --------------------------------------------------------------------------
def _texto(x: float, y: float, conteudo: str, tamanho: float, cor: str, peso: int = 400,
           ancora: str = "middle", fonte: str | None = None, halo: str | None = None) -> str:
    familia = f' font-family="{escape(fonte)}"' if fonte else ""
    contorno = (f' paint-order="stroke" stroke="{halo}" stroke-width="5" stroke-linejoin="round"'
                if halo else "")
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{tamanho}" font-weight="{peso}" '
            f'fill="{cor}" text-anchor="{ancora}"{familia}{contorno}>{escape(conteudo)}</text>')


def no(f: Figura, x: float, y: float, w: float, h: float, titulo: str,
       linhas: tuple[str, ...] = (), estado: str = "normal", mono: bool = False,
       raio: float = 13) -> None:
    """Caixa de processo: título em destaque e até três linhas curtas, centralizados."""
    fundo, borda, espessura, traco, cor_titulo, cor_linha, opacidade = ESTADOS[estado]
    tracejado = f' stroke-dasharray="{traco}"' if traco else ""
    sombra = f' filter="url(#{f.prefixo}-sombra)"' if estado not in ("apagado",) else ""
    f.add(f'<g opacity="{opacidade}">')
    f.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{raio}" fill="{fundo}" '
          f'stroke="{borda}" stroke-width="{espessura}"{tracejado}{sombra}/>')
    total = 17 + 15.5 * len(linhas)
    base = y + (h - total) / 2 + 12.5
    if mono:
        f.add(_texto(x + w / 2, base, titulo, 12, cor_titulo, 600, fonte=E.FONTE_MONO))
    else:
        f.add(_texto(x + w / 2, base, titulo, 13.5, cor_titulo, 700, fonte=E.FONTE_TITULO))
    for i, linha in enumerate(linhas):
        f.add(_texto(x + w / 2, base + 17.5 + 15.5 * i, linha, 11.5, cor_linha))
    f.add("</g>")


def losango(f: Figura, cx: float, cy: float, w: float, h: float, linhas: tuple[str, ...],
            estado: str = "normal") -> None:
    """Decisão: losango com a pergunta em duas linhas."""
    fundo, borda, espessura, _, cor_titulo, _, opacidade = ESTADOS[estado]
    if estado == "normal":
        fundo, borda = E.VERDE_NEVOA, E.VERDE
    sombra = f' filter="url(#{f.prefixo}-sombra)"' if estado != "apagado" else ""
    pontos = f"{cx},{cy - h / 2} {cx + w / 2},{cy} {cx},{cy + h / 2} {cx - w / 2},{cy}"
    f.add(f'<g opacity="{opacidade}"><polygon points="{pontos}" fill="{fundo}" stroke="{borda}" '
          f'stroke-width="{espessura}" stroke-linejoin="round"{sombra}/>')
    topo = cy - (len(linhas) - 1) * 8 + 4.5
    for i, linha in enumerate(linhas):
        f.add(_texto(cx, topo + 16 * i, linha, 12.5, cor_titulo, 650))
    f.add("</g>")


def desfecho(f: Figura, x: float, cy: float, w: float, h: float, regra: str,
             linhas: tuple[str, str], estado: str = "normal", etiqueta: bool = True) -> None:
    """Saída do Decision Engine: etiqueta da regra (se houver) e a categoria em duas linhas."""
    fundo, borda, espessura, _, cor_titulo, _, opacidade = ESTADOS[estado]
    sombra = f' filter="url(#{f.prefixo}-sombra)"' if estado != "apagado" else ""
    y = cy - h / 2
    cor_etiqueta, cor_texto_etiqueta = ((E.BRANCO, E.VERMELHO) if estado == "escolhido"
                                        else (E.VERDE_ESCURO, E.BRANCO))
    f.add(f'<g opacity="{opacidade}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" '
          f'fill="{fundo}" stroke="{borda}" stroke-width="{espessura}"{sombra}/>')
    if etiqueta:
        f.add(f'<rect x="{x + 12}" y="{cy - 12}" width="38" height="24" rx="7" '
              f'fill="{cor_etiqueta}"/>')
        f.add(_texto(x + 31, cy + 4.5, regra, 12.5, cor_texto_etiqueta, 700))
        texto_x, ancora = x + 62, "start"
    else:
        texto_x, ancora = x + w / 2, "middle"
    f.add(_texto(texto_x, cy - 3.5, linhas[0], 12.5, cor_titulo, 650, ancora=ancora))
    f.add(_texto(texto_x, cy + 12.5, linhas[1], 12.5, cor_titulo, 650, ancora=ancora))
    f.add("</g>")


def _caminho(pontos: list[tuple[float, float]], raio: float = 10) -> str:
    """Linha poligonal com cantos arredondados (curva quadrática em cada canto)."""
    d = f"M{pontos[0][0]:.1f},{pontos[0][1]:.1f}"
    for anterior, canto, seguinte in zip(pontos, pontos[1:], pontos[2:], strict=False):
        (x0, y0), (x1, y1), (x2, y2) = anterior, canto, seguinte
        d1 = max(abs(x1 - x0), abs(y1 - y0))
        d2 = max(abs(x2 - x1), abs(y2 - y1))
        r = min(raio, d1 / 2, d2 / 2)
        ux0, uy0 = (x1 - x0) / d1, (y1 - y0) / d1
        ux2, uy2 = (x2 - x1) / d2, (y2 - y1) / d2
        d += (f" L{x1 - ux0 * r:.1f},{y1 - uy0 * r:.1f}"
              f" Q{x1:.1f},{y1:.1f} {x1 + ux2 * r:.1f},{y1 + uy2 * r:.1f}")
    return d + f" L{pontos[-1][0]:.1f},{pontos[-1][1]:.1f}"


def seta(f: Figura, pontos: list[tuple[float, float]], estado: str = "normal",
         rotulo: str | None = None, onde: tuple[float, float] | None = None,
         ancora: str = "middle", tracejada: bool = False, cor_rotulo: str | None = None) -> None:
    """Conexão com ponta de seta e, se houver, um rótulo curto com halo sobre a linha."""
    cor = COR_SETA[estado]
    espessura = 2.6 if estado == "ativo" else 1.6
    traco = ' stroke-dasharray="6 5"' if tracejada else ""
    f.add(f'<path d="{_caminho(pontos)}" fill="none" stroke="{cor}" stroke-width="{espessura}" '
          f'stroke-linecap="round" stroke-linejoin="round"{traco} '
          f'marker-end="url(#{f.prefixo}-seta-{estado})"/>')
    if rotulo and onde:
        cor_texto = cor_rotulo or (E.VERMELHO if estado == "ativo" else
                                   E.BORDA_FORTE if estado == "apagado" else E.SUAVE)
        peso = 700 if estado == "ativo" else 500
        f.add(_texto(onde[0], onde[1], rotulo, 11.5, cor_texto, peso, ancora, halo=f.halo))


def faixa(f: Figura, x: float, y: float, w: float, h: float, cor: str, borda: str,
          rotulo: str, cor_rotulo: str, tracejada: bool = False) -> None:
    """Fundo de raia com uma legenda curta no canto superior esquerdo."""
    traco = ' stroke-dasharray="5 4"' if tracejada else ""
    f.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{cor}" '
          f'stroke="{borda}" stroke-width="1"{traco}/>')
    f.add(f'<text x="{x + 16}" y="{y + 19}" font-size="10.5" font-weight="700" '
          f'letter-spacing="1.1" fill="{cor_rotulo}">{escape(rotulo)}</text>')


# --------------------------------------------------------------------------
# 1. Treino e teste
# --------------------------------------------------------------------------
def treino_e_teste(fundo: str | None = None) -> str:
    f = Figura(1316, 272, "tt", "Fluxo do treino e do teste: a POF vira uma grade pessoa por "
               "dia por período; o corte pelo mês da entrevista separa treino e teste; seleção, "
               "calibração e cortes usam só o treino; o modelo é congelado e avaliado uma única "
               "vez no teste.", fundo)
    faixa(f, 470, 8, 838, 128, E.VERDE_NEVOA, E.VERDE_CLARO,
          "TODAS AS ESCOLHAS USAM SÓ O TREINO", E.VERDE)
    faixa(f, 470, 150, 838, 114, E.DOURADO_CLARO, "#EDCB6B",
          "USADO UMA ÚNICA VEZ, NO FIM", "#9A6C00", tracejada=True)
    y1, y2, h = 44, 180, 76
    c1, c2 = y1 + h / 2, y2 + h / 2
    no(f, 16, y1, 152, h, "POF 2017–2018", ("microdados do IBGE", "recordatório de 24h"))
    no(f, 222, y1, 170, h, "Grade", ("pessoa × dia × período", "alvo: pizza + refri"))
    no(f, 496, y1, 150, h, "Treino", ("jul/2017 a fev/2018", "523 ocasiões"))
    no(f, 702, y1, 198, h, "Seleção do modelo",
       ("validação cruzada 5 × 10", "vence a Reg. Logística"))
    no(f, 956, y1, 146, h, "Calibração", ("sigmoide ponderada", "cortes P90 e P70"))
    no(f, 1152, y1, 148, h, "Modelo congelado", ("models/ + SHA-256", "commit antes do teste"),
       estado="destaque")
    no(f, 496, y2, 150, h, "Teste", ("mar a jul/2018", "315 ocasiões"), estado="teste")
    no(f, 1152, y2, 148, h, "Avaliação única", ("contra o baseline", "IC por UPA"),
       estado="teste")

    seta(f, [(168, c1), (222, c1)])
    seta(f, [(392, c1), (496, c1)], rotulo="jul/17–fev/18", onde=(444, c1 - 8))
    seta(f, [(307, y1 + h), (307, c2), (496, c2)], rotulo="mar–jul/18", onde=(402, c2 - 8))
    seta(f, [(646, c1), (702, c1)])
    seta(f, [(900, c1), (956, c1)])
    seta(f, [(1102, c1), (1152, c1)])
    seta(f, [(1226, y1 + h), (1226, y2)], estado="dourado", rotulo="avalia 1 vez",
         onde=(1214, 156), ancora="end", cor_rotulo="#9A6C00")
    seta(f, [(646, c2), (1152, c2)], estado="dourado", rotulo="não entra em nenhuma escolha",
         onde=(899, c2 - 8), cor_rotulo="#9A6C00")
    return f.svg()


# --------------------------------------------------------------------------
# 2. Uma requisição
# --------------------------------------------------------------------------
CAMPOS_RESPOSTA = ("occasion_score", "classification", "top_factors",
                   "recommendation_category", "recommendation_reason", "model_metadata")


def requisicao(fundo: str | None = None) -> str:
    f = Figura(1316, 352, "rq", "Caminho de uma requisição: o app envia o contexto à API, que "
               "valida, aplica o mesmo preparo do treino, calcula o score com o modelo congelado, "
               "explica com SHAP e aplica as regras do Decision Engine; a resposta tem 6 campos.",
               fundo)
    # artefatos congelados
    f.add(f'<rect x="586" y="12" width="520" height="104" rx="16" fill="{E.VERDE_NEVOA}" '
          f'stroke="{E.VERDE_CLARO}" stroke-width="1"/>')
    f.add(_texto(600, 31, "models/", 12, E.VERDE_ESCURO, 700, ancora="start", fonte=E.FONTE_MONO))
    f.add(_texto(657, 31, "artefatos congelados, gerados só com o treino", 11.5, E.SUAVE,
                 ancora="start"))
    no(f, 600, 44, 162, 58, "occasion_model.joblib", ("+ metadados, hash e cortes",),
       estado="artefato", mono=True, raio=10)
    no(f, 776, 44, 142, 58, "explicacao.json", ("referência do SHAP",), estado="artefato",
       mono=True, raio=10)
    no(f, 932, 44, 160, 58, "contexto_decisao.json", ("contexto observado",), estado="artefato",
       mono=True, raio=10)

    y, h = 158, 80
    c = y + h / 2
    no(f, 16, y, 124, h, "App", ("Streamlit", "formulário"))
    no(f, 236, y, 158, h, "API FastAPI", ("valida com Pydantic", "422 fora do domínio"))
    no(f, 422, y, 146, h, "Preparo único", ("o mesmo do treino", "9 features"))
    no(f, 600, y, 162, h, "Modelo congelado", ("score calibrado", "classe pelos cortes"),
       estado="destaque")
    no(f, 790, y, 128, h, "SHAP exato", ("3 fatores", "do score"))
    no(f, 946, y, 146, h, "Decision Engine", ("regras R1 a R6", "+ justificativa"))
    # resposta com os 6 campos do contrato
    f.add(f'<g><rect x="1122" y="128" width="182" height="140" rx="14" fill="{E.BRANCO}" '
          f'stroke="{E.VERDE}" stroke-width="1.6" filter="url(#rq-sombra)"/>')
    f.add(_texto(1213, 151, "Resposta", 13.5, E.TINTA, 700, fonte=E.FONTE_TITULO))
    for i, campo in enumerate(CAMPOS_RESPOSTA):
        f.add(_texto(1138, 175 + 16.5 * i, campo, 11, E.VERDE_ESCURO, 500, ancora="start",
                     fonte=E.FONTE_MONO))
    f.add("</g>")

    seta(f, [(140, c), (236, c)], rotulo="POST /predict", onde=(188, c - 9))
    seta(f, [(394, c), (422, c)])
    seta(f, [(568, c), (600, c)])
    seta(f, [(762, c), (790, c)])
    seta(f, [(918, c), (946, c)])
    seta(f, [(1092, c), (1122, c)])
    for x in (681, 847, 1012):
        seta(f, [(x, 102), (x, y)], estado="verde")
    f.add(_texto(689, 136, "confere o hash", 11, E.VERDE, 600, ancora="start", halo=f.halo))
    # canal e associação não passam pelo modelo
    seta(f, [(315, y + h), (315, 284), (1019, 284), (1019, y + h)], tracejada=True,
         rotulo="canal e associação à marca: não entram no modelo", onde=(667, 279))
    # a resposta volta ao app
    seta(f, [(1213, 268), (1213, 326), (78, 326), (78, y + h)],
         rotulo="o app mostra score, classe, fatores e recomendação", onde=(667, 321))
    return f.svg()


# --------------------------------------------------------------------------
# 3. Decision Engine
# --------------------------------------------------------------------------
DESFECHOS = {
    "R1": ("Sem ativação", "dedicada"),
    "R2": ("Awareness e", "associação"),
    "R3": ("Ativação via", "delivery"),
    "R4": ("Ativação no ponto", "de consumo"),
    "R5": ("Combo para", "consumo em casa"),
    "R6": ("Presença", "de marca"),
}
# Os textos da árvore: "tecnica" (README e aba técnica, com as regras R1 a R6) e "simples"
# (para quem decide marketing: prioridade no lugar de classe, sem os códigos das regras)
TEXTOS_DE = {
    "tecnica": {
        "inicio": ("Score e classe", "do modelo congelado"),
        "perguntas": (("Classe", "baixa?"), ("Associação à", "marca baixa?"), ("Classe", "alta?"),
                      ("Qual o", "canal?")),
        "media": "não: classe média",
        "canais": ("delivery", "estabelecimento", "varejo ou nenhum"),
        "desfechos": DESFECHOS,
        "descricao": ("Árvore de regras do Decision Engine: classe baixa leva a R1; associação à "
                      "marca baixa leva a R2; classe média leva a R6; classe alta segue o canal: "
                      "delivery R3, estabelecimento R4, varejo ou nenhum R5."),
    },
    "simples": {
        "inicio": ("Chance da", "ocasião"),
        "perguntas": (("Prioridade", "baixa?"), ("Marca pouco", "ligada a pizza?"),
                      ("Prioridade", "alta?"), ("Onde vai", "agir?")),
        "media": "não: prioridade média",
        "canais": ("delivery", "pizzaria ou bar", "varejo ou não sei"),
        "desfechos": {
            "R1": ("Sem ação", "dedicada"),
            "R2": ("Ligar Guaraná a pizza", "(awareness)"),
            "R3": ("Ação no", "delivery"),
            "R4": ("Ação em pizzarias,", "bares e restaurantes"),
            "R5": ("Combo para", "comer em casa"),
            "R6": ("Presença de marca", "de baixo custo"),
        },
        "descricao": ("Como a ação é escolhida: prioridade baixa leva a nenhuma ação dedicada; "
                      "marca pouco ligada a pizza leva a awareness; prioridade média leva a "
                      "presença de marca de baixo custo; prioridade alta segue o canal: delivery, "
                      "pizzarias e bares, ou combo para comer em casa."),
    },
}
_BASE = {"inicio", "e_inicio", "d1"}
CAMINHOS = {
    "R1": _BASE | {"e_d1_r1", "R1"},
    "R2": _BASE | {"e_d1_d2", "d2", "e_d2_r2", "R2"},
    "R6": _BASE | {"e_d1_d2", "d2", "e_d2_d3", "d3", "e_d3_r6", "R6"},
    "R3": _BASE | {"e_d1_d2", "d2", "e_d2_d3", "d3", "e_d3_d4", "d4", "e_d4_r3", "R3"},
    "R4": _BASE | {"e_d1_d2", "d2", "e_d2_d3", "d3", "e_d3_d4", "d4", "e_d4_r4", "R4"},
    "R5": _BASE | {"e_d1_d2", "d2", "e_d2_d3", "d3", "e_d3_d4", "d4", "e_d4_r5", "R5"},
}


def decision_engine(regra: str | None = None, inicio: tuple[str, str] | None = None,
                    fundo: str | None = None, prefixo: str = "de",
                    linguagem: str = "tecnica") -> str:
    """A árvore de regras. Com `regra`, o caminho dela acende e o resto apaga.

    `inicio` troca o texto do nó de partida (ex.: ("Score 2,88%", "classe alta")).
    `linguagem` escolhe os textos: "tecnica" (com R1 a R6) ou "simples" (TEXTOS_DE).
    """
    if regra is not None and regra not in CAMINHOS:
        raise ValueError(f"regra desconhecida: {regra!r}")
    if linguagem not in TEXTOS_DE:
        raise ValueError(f"linguagem desconhecida: {linguagem!r}")
    t = TEXTOS_DE[linguagem]
    ativos = CAMINHOS.get(regra)

    def estado_no(nome: str) -> str:
        if ativos is None:
            return "normal"
        if nome in ativos:
            return "escolhido" if nome == regra else "ativo"
        return "apagado"

    def estado_seta(nome: str) -> str:
        if ativos is None:
            return "normal"
        return "ativo" if nome in ativos else "apagado"

    descricao = t["descricao"]
    if regra:
        nome = " ".join(t["desfechos"][regra])
        descricao += (f" Caminho aceso: regra {regra}." if linguagem == "tecnica"
                      else f" Caminho aceso: {nome}.")
    f = Figura(1322, 300, prefixo, descricao, fundo)
    cy = 110
    titulo, linha = inicio or t["inicio"]
    estado_inicio = "destaque" if ativos is None else ("ativo" if "inicio" in ativos else "apagado")
    no(f, 16, cy - 33, 150, 66, titulo, (linha,), estado=estado_inicio, raio=33)

    p1, p2, p3, p4 = t["perguntas"]
    losango(f, 270, cy, 150, 96, p1, estado_no("d1"))
    losango(f, 496, cy, 176, 96, p2, estado_no("d2"))
    losango(f, 724, cy, 150, 96, p3, estado_no("d3"))
    losango(f, 930, cy, 140, 96, p4, estado_no("d4"))

    delivery, ponto, casa = t["canais"]
    seta(f, [(166, cy), (195, cy)], estado_seta("e_inicio"))
    seta(f, [(345, cy), (408, cy)], estado_seta("e_d1_d2"), "não", (376, cy - 9))
    seta(f, [(584, cy), (649, cy)], estado_seta("e_d2_d3"), "não", (616, cy - 9))
    seta(f, [(799, cy), (860, cy)], estado_seta("e_d3_d4"), "sim", (829, cy - 9))
    seta(f, [(270, cy + 48), (270, 229)], estado_seta("e_d1_r1"), "sim", (279, 199), "start")
    seta(f, [(496, cy + 48), (496, 229)], estado_seta("e_d2_r2"), "sim", (505, 199), "start")
    seta(f, [(724, cy + 48), (724, 229)], estado_seta("e_d3_r6"), t["media"], (733, 199), "start")
    seta(f, [(930, cy - 48), (930, 40), (1110, 40)], estado_seta("e_d4_r3"), delivery,
         (1020, 32))
    seta(f, [(1000, cy), (1110, cy)], estado_seta("e_d4_r4"), ponto, (1055, cy - 9))
    seta(f, [(930, cy + 48), (930, 180), (1110, 180)], estado_seta("e_d4_r5"), casa,
         (1020, 196))

    for nome, (x, cy_saida) in {"R3": (1110, 40), "R4": (1110, cy), "R5": (1110, 180),
                                "R1": (172, 258), "R2": (398, 258), "R6": (626, 258)}.items():
        desfecho(f, x, cy_saida, 196, 58, nome, t["desfechos"][nome], estado_no(nome),
                 etiqueta=linguagem == "tecnica")
    return f.svg()


def gravar(pasta: Path) -> list[Path]:
    """As três figuras com fundo creme, para o README."""
    pasta.mkdir(parents=True, exist_ok=True)
    arquivos = {"fluxo_1_treino_e_teste.svg": treino_e_teste(E.CREME),
                "fluxo_2_requisicao.svg": requisicao(E.CREME),
                "fluxo_3_decision_engine.svg": decision_engine(fundo=E.CREME)}
    caminhos = []
    for nome, svg in arquivos.items():
        caminho = pasta / nome
        caminho.write_text(svg + "\n", encoding="utf-8")
        caminhos.append(caminho)
    return caminhos


if __name__ == "__main__":
    raiz = Path(__file__).resolve().parents[1]
    for caminho in gravar(raiz / "reports" / "figures"):
        print(f"[ok] {caminho.relative_to(raiz)}")
    sys.exit(0)
