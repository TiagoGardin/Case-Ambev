"""Gráficos do app: a prioridade, os fatores, os quadros de quando e onde, o Mapa de
Oportunidade e o termômetro.

Todos seguem as mesmas regras: marcas finas, rótulo direto em cada marca (a cor nunca é a
única pista) e cores da paleta validada em estilo.py, com a mesma cor para a mesma
prioridade em todo o app. O termômetro em escala log fica na aba técnica; as abas
principais falam em chance e prioridade.
"""

from __future__ import annotations

import math
from html import escape

import conteudo as C
import estilo as E

# Escala do termômetro: de 0,001% a 30%, em log (o score vai de milionésimos a dezenas de %)
LOG_MIN, LOG_MAX = -5.0, math.log10(0.30)
NOME_FATOR = {"momento": "Momento", "regiao": "Região", "idade": "Idade", "sexo": "Sexo",
              "escolaridade": "Escolaridade", "moradores": "Pessoas na casa",
              "renda": "Renda por pessoa", "area": "Área"}
LOG_ODDS_MAXIMO = 3.2  # |log-odds| que ocupa a meia barra inteira (× 24,5)
# Parcela dos momentos (pessoa × dia × período) do treino em cada prioridade: cortes P70 e P90
PARCELA = {"baixa": "70%", "media": "20%", "alta": "10%"}
ORDEM_CLASSES = ("baixa", "media", "alta")


def pct(v: float, casas: int = 2) -> str:
    """0,0288 → "2,88%"; abaixo de 0,01% usa mais casas para não virar zero."""
    if 0 < v < 0.0001:
        casas = max(casas, 4)
    return f"{v * 100:.{casas}f}%".replace(".", ",")


def pct_simples(v: float, compacto: bool = False) -> str:
    """A chance para quem não é da área: "2,4%", "0,38%", "0,045%" ou "menos de 0,01%".

    `compacto` troca "menos de 0,01%" por "< 0,01%" (no número grande e nos quadros).
    """
    if v < 0.0001:
        return "< 0,01%" if compacto else "menos de 0,01%"
    casas = 1 if v >= 0.01 else (2 if v >= 0.001 else 3)
    return f"{v * 100:.{casas}f}%".replace(".", ",")


def em_cada(v: float) -> str:
    """A chance como frequência: "cerca de 24 em cada 1.000"; chances pequenas, em 10 mil etc."""
    for base, nome in ((1_000, "1.000"), (10_000, "10 mil"), (100_000, "100 mil"),
                       (1_000_000, "1 milhão")):
        if v * base >= 3:  # com menos de 3, o arredondamento distorce: passa à base seguinte
            return f"cerca de {int(v * base + 0.5)} em cada {nome}"
    return "menos de 3 em cada 1 milhão"


def comparado(x: float) -> str:
    """A chance dividida pela chance média: "7,9 vezes a chance média", "2% da chance média"."""
    if x >= 1:
        return f"{x:.1f} vezes a chance média".replace(".", ",")
    if x >= 0.01:
        return f"{x * 100:.0f}% da chance média"
    return "menos de 1% da chance média"


def vezes(m: float) -> str:
    """Multiplicador da chance: 13,164 → "× 13,2"; 1,565 → "× 1,6"; 0,394 → "× 0,39"."""
    casas = 1 if m >= 1 else 2
    return f"× {m:.{casas}f}".replace(".", ",")


def razao(x: float) -> str:
    """Quantas vezes a taxa média: 9,52 → "9,5"; 0,021 → "0,02"; abaixo de 0,01 → "< 0,01"."""
    if x < 0.01:
        return "< 0,01"
    return f"{x:.{1 if x >= 1 else 2}f}".replace(".", ",")


# --------------------------------------------------------------------------
# Para quem decide marketing
# --------------------------------------------------------------------------
def prioridade(classe: str) -> str:
    """As três prioridades lado a lado, com a do contexto acesa e a parcela de cada uma."""
    partes = []
    for c in ORDEM_CLASSES:
        nome = E.ROTULO_CLASSE[c].capitalize()
        estilo = ""
        if c == classe:
            estilo = (f"background:{E.CLASSE[c]};color:{E.CLASSE_TEXTO[c]};"
                      f"border-color:{E.CLASSE[c]}")
            nome = f"✓ {nome}"
        partes.append(f'<div class="passo{" ativo" if c == classe else ""}">'
                      f'<div class="faixa" style="{estilo}">{nome}</div>'
                      f'<div class="parcela">{PARCELA[c]} dos momentos</div></div>')
    return (f'<div class="oi-prioridade" role="img" aria-label="Prioridade '
            f'{E.ROTULO_CLASSE[classe]}: baixa reúne 70% dos momentos, média 20% e alta os 10% '
            f'de maior chance">{"".join(partes)}</div>')


def fatores(top: list[dict], valores: dict[str, str] | None = None) -> str:
    """Os 3 fatores do SHAP como barras em torno da média (× 1, o contexto médio do treino).

    O comprimento é proporcional ao log-odds, então × 2 e × 0,5 ficam do mesmo tamanho, um
    para cada lado. O multiplicador vai escrito na ponta da barra (dentro dela, se for longa).
    `valores` troca o texto de um fator pelo do formulário (ex.: a renda em salários mínimos).
    """
    valores = valores or {}
    linhas = []
    for f in top:
        lo = float(f["log_odds"])
        meia = min(abs(lo) / LOG_ODDS_MAXIMO, 1.0) * 46  # % da largura total
        efeito = "aumenta" if lo > 0 else ("reduz" if lo < 0 else "neutro")
        nome = NOME_FATOR.get(f["factor"], f["factor"])
        valor = valores.get(f["factor"], f["value"])
        dica = escape(f"{nome}: {valor} ({'aumenta' if lo > 0 else 'reduz'} a chance, "
                      f"{vezes(f['odds_multiplier'])})")
        dentro = meia > 34
        lado, oposto = ("left", "right") if lo >= 0 else ("right", "left")
        arredonda = "aumenta" if lo >= 0 else "reduz"
        barra = (f'<div class="barra {arredonda}" style="{lado}:50%;width:{meia:.1f}%;'
                 f'background:{E.FATOR[efeito]}"></div>')
        posicao = (f"{oposto}:{50 - meia:.1f}%;padding-{oposto}:8px;color:#fff" if dentro
                   else f"{lado}:calc(50% + {meia:.1f}% + 7px)")
        linhas.append(
            f'<div class="oi-fator" title="{dica}"><div><span class="nome">{escape(nome)}</span>'
            f'<span class="valor"> · {escape(valor)}</span></div>'
            f'<div class="oi-barra">{barra}<span class="mult" style="{posicao}">'
            f'{vezes(f["odds_multiplier"])}</span></div></div>')
    escala = ('<div class="oi-escala"><span>← reduz a chance</span><span>média</span>'
              '<span>aumenta a chance →</span></div>')
    return "".join(linhas) + escala


def quando(celulas: dict[tuple[str, str], tuple[float, str]], periodos: dict[str, str],
           horarios: dict[str, str], atual: tuple[str, str]) -> str:
    """Quadro bloco de dias × período, na cor da prioridade; a célula do cenário fica marcada.

    `celulas[(bloco, periodo)] = (chance, classe)`.
    """
    blocos = list(dict.fromkeys(b for b, _ in celulas))
    topo = "".join(f'<div class="cab">{escape(periodos[p])}<small>{escape(horarios[p])}</small>'
                   f'</div>' for p in periodos)
    linhas = []
    for bloco in blocos:
        linhas.append(f'<div class="lin">{escape(bloco)}</div>')
        for p in periodos:
            chance, classe = celulas[(bloco, p)]
            marcada = (bloco, p) == atual
            dica = escape(f"{bloco}, {periodos[p].lower()}: chance {pct_simples(chance)}, "
                          f"prioridade {E.ROTULO_CLASSE[classe]}")
            linhas.append(
                f'<div class="cel{" atual" if marcada else ""}" title="{dica}" '
                f'style="background:{E.CLASSE[classe]};color:{E.CLASSE_TEXTO[classe]}">'
                f'<b>{pct_simples(chance, compacto=True)}</b>'
                f'<span>{E.ROTULO_CLASSE[classe]}</span>'
                + ('<i>seu cenário</i>' if marcada else '') + '</div>')
    return f'<div class="oi-quando"><div></div>{topo}{"".join(linhas)}</div>'


def onde(regioes: dict[str, tuple[float, str]], atual: str) -> str:
    """Uma barra por região (escala linear até a maior chance), da maior para a menor."""
    maximo = max(chance for chance, _ in regioes.values()) or 1.0
    linhas = []
    for regiao, (chance, classe) in sorted(regioes.items(), key=lambda kv: -kv[1][0]):
        largura = max(chance / maximo * 100, 1.5)
        marcada = regiao == atual
        dica = escape(f"{regiao}: chance {pct_simples(chance)}, prioridade "
                      f"{E.ROTULO_CLASSE[classe]}")
        linhas.append(
            f'<div class="reg{" atual" if marcada else ""}" title="{dica}">'
            f'<div class="nome">{escape(regiao)}'
            + ('<i>seu cenário</i>' if marcada else '') + '</div>'
            f'<div class="trilho"><div class="barra" style="width:{largura:.1f}%;'
            f'background:{E.CLASSE[classe]}"></div></div>'
            f'<div class="valor"><b>{pct_simples(chance, compacto=True)}</b> · '
            f'{E.ROTULO_CLASSE[classe]}</div>'
            '</div>')
    return f'<div class="oi-onde">{"".join(linhas)}</div>'


# --------------------------------------------------------------------------
# Onde e para quem ativar (Mapa de Oportunidade)
# --------------------------------------------------------------------------
def rotulo_publico(p: dict) -> tuple[str, str]:
    """("Homens de 25 a 34 anos · Centro-Oeste", "Sexta a domingo, à noite")."""
    quem = f"{C.GENEROS[p['sexo']]} de {C.FAIXAS_ETARIAS[p['faixa_etaria']]} · {p['regiao']}"
    return quem, f"{C.BLOCOS[p['bloco']]}, {C.NO_PERIODO[p['periodo']]}"


def _decimal(v: float, casas: int = 1) -> str:
    return f"{v:.{casas}f}".replace(".", ",")


def _juntar(partes: list[str]) -> str:
    return partes[0] if len(partes) == 1 else ", ".join(partes[:-1]) + " e " + partes[-1]


def dias_texto(dias: list[str]) -> str:
    """["Sexta-Feira", "Sabado", "Domingo"] → "sexta, sábado e domingo"."""
    return _juntar([C.DIAS_EXTENSO[d].lower() for d in dias])


def horario_texto(horarios: list[str]) -> str:
    """["18:00-24:00", "00:00-05:00"] → "das 18h à meia-noite e da meia-noite às 5h"."""
    def hora(h: str, inicio: bool) -> str:
        n = int(h[:2])
        if n in (0, 24):
            return "da meia-noite" if inicio else "à meia-noite"
        return f"das {n}h" if inicio else f"às {n}h"
    return _juntar([f"{hora(a, True)} {hora(b, False)}"
                    for a, b in (h.split("-") for h in horarios)])


def fora_da_noite(publicos: list[dict]) -> tuple[float, float]:
    """Parcela das ocasiões de manhã, no almoço e à tarde: (estimada, observada na pesquisa).

    A observada sai do próprio mapa: taxa observada × parcela dos momentos de cada público.
    """
    estimada = sum(p["parcela_das_ocasioes"] for p in publicos if p["periodo"] != "noite")
    obs = [(p["periodo"], p["taxa_observada"] * p["parcela_dos_momentos"]) for p in publicos]
    total = sum(v for _, v in obs)
    return estimada, sum(v for periodo, v in obs if periodo != "noite") / total


def filtrar_publicos(publicos: list[dict], regiao: str | None, genero: str | None) -> list:
    """Públicos com amostra suficiente, da região e do gênero escolhidos ("Todas"/"Todos" ou
    None deixam passar todos)."""
    sexo = {v: k for k, v in C.GENEROS.items()}.get(genero or "")
    return [p for p in publicos if p["amostra_suficiente"]
            and (regiao in (None, "Todas") or p["regiao"] == regiao)
            and (sexo is None or p["sexo"] == sexo)]


def ranking_publicos(itens: list[dict], medida: str, media: float) -> str:
    """Lista numerada com barra. `medida`: "chance" (por pessoa) ou "volume" (do total)."""
    chave = "chance" if medida == "chance" else "parcela_das_ocasioes"
    topo = max((p[chave] for p in itens), default=0) or 1
    linhas = []
    for i, p in enumerate(itens, 1):
        quem, momento = rotulo_publico(p)
        if medida == "chance":
            valor = pct_simples(p["chance"])
            apoio = [comparado(p["chance"] / media).replace("chance média", "média")]
        else:
            valor = f"{_decimal(p['parcela_das_ocasioes'] * 100)}%"
            apoio = ["das ocasiões", f"chance {pct_simples(p['chance'])}"]
        largura = 100 * p[chave] / topo
        linhas.append(
            f'<div class="item"><div class="pos">{i}</div>'
            f'<div class="quem"><b>{escape(quem)}</b><span>{escape(momento)} · '
            f'prioridade {E.ROTULO_CLASSE[p["prioridade"]]}</span>'
            f'<div class="trilho"><div class="barra" style="width:{largura:.1f}%;'
            f'background:{E.CLASSE[p["prioridade"]]}"></div></div></div>'
            f'<div class="valor"><b>{valor}</b>'
            + "".join(f"<span>{escape(a)}</span>" for a in apoio) + "</div></div>")
    if not linhas:
        return '<div class="oi-nota">Nenhum público com esses filtros.</div>'
    return f'<div class="oi-rank">{"".join(linhas)}</div>'


def cartao_anuncio(p: dict, media: float) -> str:
    """Os parâmetros de anúncio de um público-momento, em palavras simples."""
    a = p["anuncio"]
    quem, momento = rotulo_publico(p)
    ufs = _juntar(a["localizacao_ufs"])
    linhas = [
        ("Idade", C.FAIXAS_ETARIAS[a["faixa_etaria"]]),
        ("Gênero", C.GENEROS[a["genero"]]),
        ("Onde", f"{ufs} ({p['regiao']})"),
        ("Dias", dias_texto(a["dias"]).capitalize()),
        ("Horário", horario_texto(a["horarios"]).capitalize()),
        ("Renda mais comum", f"{C.RENDAS[p['renda_mais_comum']][0]} por pessoa na família"),
    ]
    tabela = "".join(f'<div class="par"><span>{escape(r)}</span><b>{escape(v)}</b></div>'
                     for r, v in linhas)
    acao = C.ACAO_SEM_CANAL[p["regra"]]
    return (
        f'<div class="oi-cartao oi-anuncio"><div class="oi-rotulo">Parâmetros de anúncio</div>'
        f'<div class="titulo">{escape(quem)}</div><div class="oi-nota">{escape(momento)} · '
        f'chance {pct_simples(p["chance"])}, '
        f'{escape(comparado(p["chance"] / media).replace("chance média", "média dos adultos"))}'
        f' · {E.classe_html(p["prioridade"], "prioridade ")}</div>'
        f'<div class="pares">{tabela}</div>'
        f'<div class="oi-legenda">Sem canal escolhido, o caminho da decisão sugere '
        f'<b>{escape(acao)}</b>. Estes são parâmetros que as plataformas de anúncio aceitam; '
        'não é um plano de mídia: orçamento, lances e resultado esperado ficam de fora.</div>'
        '</div>')


def estados_por_canal(estados: list[dict]) -> str:
    """Os 27 estados em três cartões, um por canal, na ordem do mapa."""
    cores = {"ativacao_delivery": E.VERDE, "combo_consumo_em_casa": E.DOURADO,
             "presenca_de_marca": E.SUAVE}
    cartoes = []
    for categoria, (nome, regra) in C.CANAIS_DOS_ESTADOS.items():
        grupo = [e for e in estados if e["categoria"] == categoria]
        linhas = "".join(
            f'<div class="uf"><b>{escape(e["uf"])}</b><span class="nome">{escape(e["nome"])}'
            f'</span><span class="num">{_decimal(e["parcela_das_ocasioes"] * 100)}%</span>'
            f'<span class="num">{_decimal(e["pizzarias_100mil"])}</span></div>' for e in grupo)
        cartoes.append(
            f'<div class="oi-cartao oi-canal" style="border-top:4px solid {cores[categoria]}">'
            f'<div class="cab"><b>{escape(nome)}</b><span>{len(grupo)} estados</span></div>'
            f'<div class="oi-nota">{escape(regra)}</div>'
            '<div class="uf cabecalho"><b></b><span class="nome">Estado</span>'
            '<span class="num">Ocasiões</span><span class="num">Pizzarias</span></div>'
            f'{linhas}</div>')
    return ('<div class="oi-grade r3" style="grid-template-columns:repeat(3,1fr)">'
            + "".join(cartoes) + "</div>")


# --------------------------------------------------------------------------
# Para a equipe técnica
# --------------------------------------------------------------------------
def _x(v: float, x0: float, x1: float) -> float:
    lv = math.log10(min(max(v, 10 ** LOG_MIN), 10 ** LOG_MAX))
    return x0 + (lv - LOG_MIN) / (LOG_MAX - LOG_MIN) * (x1 - x0)


def termometro(score: float, cortes: dict, prefixo: str = "tm") -> str:
    """Barra em escala log com as três classes e a posição do score (feita para ~300 px)."""
    x0, x1, y, h = 2.0, 298.0, 22.0, 18.0
    xm, xa, xs = _x(cortes["media"], x0, x1), _x(cortes["alta"], x0, x1), _x(score, x0, x1)
    zonas = [("baixa", x0, xm), ("media", xm, xa), ("alta", xa, x1)]
    partes = [f'<defs><clipPath id="{prefixo}-barra"><rect x="{x0}" y="{y}" width="{x1 - x0}" '
              f'height="{h}" rx="{h / 2}"/></clipPath></defs>',
              f'<g clip-path="url(#{prefixo}-barra)">']
    for classe, a, b in zonas:
        partes.append(f'<rect x="{a:.1f}" y="{y}" width="{b - a:.1f}" height="{h}" '
                      f'fill="{E.CLASSE[classe]}"><title>classe {E.ROTULO_CLASSE[classe]}</title>'
                      f'</rect>')
    for corte in (xm, xa):  # 2 px de respiro entre as faixas
        partes.append(f'<rect x="{corte - 1:.1f}" y="{y}" width="2" height="{h}" fill="#fff"/>')
    partes.append("</g>")
    for classe, a, b in zonas:
        centro, meia_largura = (a + b) / 2, 15  # o rótulo ocupa uns 30 px
        if abs(xs - centro) < meia_largura + 8:  # o marcador cairia em cima do rótulo
            esquerda, direita = (a + xs) / 2, (xs + b) / 2
            centro = esquerda if xs - a > b - xs else direita
            if min(abs(centro - a), abs(b - centro)) < meia_largura or \
                    abs(xs - centro) < meia_largura + 6:
                continue  # sem espaço: a faixa fica sem nome, o rótulo de baixo diz o corte
        partes.append(f'<text x="{centro:.1f}" y="{y + 12.8:.1f}" font-size="10.5" '
                      f'font-weight="700" fill="{E.CLASSE_TEXTO[classe]}" text-anchor="middle">'
                      f'{E.ROTULO_CLASSE[classe]}</text>')
    marcas = ((x0, "start", "0,001%"), (xm, "middle", pct(cortes["media"])),
              (xa, "middle", pct(cortes["alta"])), (x1, "end", "30%"))
    for x, ancora, texto in marcas:
        partes.append(f'<text x="{x:.1f}" y="{y + h + 14:.1f}" font-size="10.5" '
                      f'fill="{E.SUAVE}" text-anchor="{ancora}">{texto}</text>')
    ancora = "start" if xs < x0 + 24 else ("end" if xs > x1 - 24 else "middle")
    partes += [
        f'<line x1="{xs:.1f}" x2="{xs:.1f}" y1="{y - 5}" y2="{y + h + 3}" stroke="{E.TINTA}" '
        f'stroke-width="2"/>',
        f'<circle cx="{xs:.1f}" cy="{y + h / 2}" r="4.5" fill="#fff" stroke="{E.TINTA}" '
        f'stroke-width="2.2"/>',
        f'<text x="{xs:.1f}" y="{y - 8}" font-size="12" font-weight="700" fill="{E.TINTA}" '
        f'text-anchor="{ancora}">{pct(score)}</text>',
    ]
    return (f'<svg viewBox="0 0 300 {y + h + 19:.0f}" role="img" '
            f'aria-label="Score {pct(score)} na escala das classes: baixa até '
            f'{pct(cortes["media"])}, média até {pct(cortes["alta"])}, alta acima" '
            f'font-family="{escape(E.FONTE_TEXTO)}" style="width:100%;height:auto;display:block">'
            + "".join(partes) + "</svg>")
