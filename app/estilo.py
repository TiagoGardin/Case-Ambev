"""Identidade visual do app: paleta inspirada no Guaraná, CSS e pequenos blocos de HTML.

A paleta de dados foi conferida com o validador de paletas (contraste e daltonismo):
- classes (`baixa`, `media`, `alta`): uma rampa de vermelho do fruto, do claro ao escuro;
- fatores: vermelho para o que aumenta a chance e verde para o que reduz, sempre com o
  rótulo ao lado da barra (a cor nunca é a única pista).
O verde-escuro, o dourado e o creme ficam para a interface, não para os dados.
"""

from __future__ import annotations

from html import escape

# Interface
VERDE_ESCURO = "#0B4A30"
VERDE = "#0E7A41"
VERDE_MEDIO = "#13874B"
VERDE_CLARO = "#E4F1E8"
VERDE_NEVOA = "#F3F8F4"
VERMELHO = "#C8262C"
VERMELHO_CLARO = "#FBE6E4"
DOURADO = "#F0B323"
DOURADO_CLARO = "#FDF4DC"
CREME = "#FBF7EF"
TINTA = "#1C2A21"
SUAVE = "#5A675F"
BORDA = "#E3E6DE"
BORDA_FORTE = "#C9D3CA"
CINZA_SETA = "#7C8A81"
BRANCO = "#FFFFFF"

# Dados
CLASSE = {"baixa": "#EFA39C", "media": "#D8473F", "alta": "#9E1A20"}
CLASSE_TEXTO = {"baixa": "#5A1714", "media": "#FFFFFF", "alta": "#FFFFFF"}
ROTULO_CLASSE = {"baixa": "baixa", "media": "média", "alta": "alta"}
FATOR = {"aumenta": "#B3242A", "reduz": "#3E9E6E", "neutro": "#9AA39D"}

FONTE_TEXTO = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"
FONTE_TITULO = "Poppins, Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"
FONTE_MONO = "'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"

CSS = f"""
<style>
:root {{
  --verde-escuro: {VERDE_ESCURO}; --verde: {VERDE}; --verde-claro: {VERDE_CLARO};
  --verde-nevoa: {VERDE_NEVOA}; --vermelho: {VERMELHO}; --vermelho-claro: {VERMELHO_CLARO};
  --dourado: {DOURADO}; --dourado-claro: {DOURADO_CLARO}; --creme: {CREME};
  --tinta: {TINTA}; --suave: {SUAVE}; --borda: {BORDA};
  --sombra: 0 1px 2px rgba(16, 40, 28, .05), 0 10px 28px rgba(16, 40, 28, .07);
}}
.block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1380px; }}
header[data-testid="stHeader"] {{ background: transparent; }}

/* Abas */
[role="tablist"] {{ gap: .3rem; }}
[data-testid="stTab"] {{ padding: .55rem 1.1rem .6rem; border-radius: .85rem .85rem 0 0; }}
[data-testid="stTab"] p {{ font-size: 15.5px; font-weight: 600; }}
[data-testid="stTab"][aria-selected="true"] {{ background: rgba(14, 122, 65, .08); }}

/* Topo */
.oi-topo {{
  position: relative; overflow: hidden; border-radius: 26px; padding: 30px 38px 28px;
  margin-bottom: 6px; color: #fff; box-shadow: 0 14px 34px rgba(11, 74, 48, .22);
  background: radial-gradient(130% 160% at 88% 0%, {VERDE_MEDIO} 0%, {VERDE} 34%,
                                                    {VERDE_ESCURO} 100%);
}}
.oi-topo .sobre {{ font: 600 12px/1.2 {FONTE_TEXTO}; letter-spacing: .16em;
  text-transform: uppercase; color: #F7D77E; }}
.oi-topo .oi-titulo {{ font-family: {FONTE_TITULO}; font-weight: 800; font-size: 46px;
  line-height: 1.05; color: #fff; margin: 10px 0 8px; letter-spacing: -.01em; }}
.oi-topo .oi-titulo .mais {{ color: #FFD25C; }}
.oi-topo p {{ font-size: 16.5px; line-height: 1.5; color: rgba(255, 255, 255, .88);
  max-width: 660px; margin: 0; }}
.oi-topo .chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }}
.oi-topo .ilustracao {{ position: absolute; right: 22px; bottom: -6px; width: 290px; }}
@media (max-width: 1180px) {{ .oi-topo .ilustracao {{ display: none; }} }}
.chip {{ display: inline-flex; align-items: center; gap: 7px; padding: 6px 13px;
  border-radius: 999px; font: 500 12.5px/1 {FONTE_TEXTO}; color: #fff;
  background: rgba(255, 255, 255, .12); border: 1px solid rgba(255, 255, 255, .22); }}
.chip code {{ font-family: {FONTE_MONO}; font-size: 11.5px; color: #fff; background: none; }}
.chip .luz {{ width: 8px; height: 8px; border-radius: 50%; background: #62E39B;
  box-shadow: 0 0 0 3px rgba(98, 227, 155, .25); }}
.chip.fora .luz {{ background: #FF8F87; box-shadow: 0 0 0 3px rgba(255, 143, 135, .25); }}

/* Formulário em cartão branco */
.st-key-formulario {{ background: #fff; border-radius: 20px !important;
  box-shadow: var(--sombra); padding: 10px 14px 6px; }}

/* Títulos de seção */
.oi-secao {{ margin: 26px 0 10px; }}
.oi-secao h2 {{ font-family: {FONTE_TITULO}; font-weight: 700; font-size: 23px; color: var(--tinta);
  margin: 0; padding: 0; }}
.oi-secao p {{ color: var(--suave); margin: 4px 0 0; font-size: 15px; max-width: 900px; }}
.oi-rotulo {{ font: 600 11.5px/1.2 {FONTE_TEXTO}; letter-spacing: .12em; text-transform: uppercase;
  color: var(--suave); margin: 0 0 10px; }}

/* Cartões */
.oi-grade {{ display: grid; gap: 16px; }}
.oi-grade.r3 {{ grid-template-columns: 1fr 1.25fr 1.15fr; }}
.oi-grade.r4 {{ grid-template-columns: repeat(4, 1fr); }}
.oi-grade.r2 {{ grid-template-columns: 1fr 1fr; }}
@media (max-width: 1100px) {{ .oi-grade.r3, .oi-grade.r4 {{ grid-template-columns: 1fr 1fr; }} }}
@media (max-width: 700px) {{ .oi-grade.r3, .oi-grade.r4, .oi-grade.r2 {{
  grid-template-columns: 1fr; }} }}
.oi-cartao {{ background: #fff; border: 1px solid var(--borda); border-radius: 20px;
  padding: 20px 22px; box-shadow: var(--sombra); }}
.oi-cartao.verde {{ background: linear-gradient(180deg, #FFFFFF 0%, {VERDE_NEVOA} 100%); }}

/* Score */
.oi-score {{ display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }}
.oi-score .valor {{ font-family: {FONTE_TITULO}; font-weight: 800; font-size: 54px; line-height: 1;
  color: var(--tinta); letter-spacing: -.02em; }}
.oi-classe {{ display: inline-block; padding: 5px 13px; border-radius: 999px;
  font: 700 12px/1 {FONTE_TEXTO}; letter-spacing: .1em; text-transform: uppercase; }}
.oi-vezes {{ margin: 10px 0 2px; color: var(--tinta); font-size: 15px; }}
.oi-vezes b {{ font-weight: 700; }}
.oi-nota {{ color: var(--suave); font-size: 13px; line-height: 1.45; margin-top: 6px; }}
.oi-ctx {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }}
.oi-ctx span {{ background: {VERDE_CLARO}; color: {VERDE_ESCURO}; border-radius: 8px;
  padding: 3px 9px; font-size: 12.3px; font-weight: 500; }}
.oi-divisor {{ border-top: 1px solid var(--borda); margin: 14px 0 12px; }}

/* Recomendação */
.oi-reco-topo {{ display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }}
.oi-regra {{ display: inline-flex; align-items: center; justify-content: center; min-width: 38px;
  height: 26px; padding: 0 9px; border-radius: 8px; background: var(--vermelho); color: #fff;
  font: 700 13px/1 {FONTE_TEXTO}; }}
.oi-reco h3 {{ font-family: {FONTE_TITULO}; font-weight: 700; font-size: 22px; line-height: 1.2;
  color: var(--tinta); margin: 4px 0 6px; padding: 0; }}
.oi-reco .descricao {{ color: var(--suave); font-size: 14px; margin: 0 0 12px; }}
.oi-reco .razao {{ border-left: 3px solid var(--dourado); background: var(--dourado-claro);
  border-radius: 0 12px 12px 0; padding: 10px 14px; font-size: 12.8px; line-height: 1.5;
  color: var(--tinta); }}
.oi-reco .razao p {{ margin: 0 0 6px; }}
.oi-reco .razao p:last-child {{ margin: 0; }}

/* Fatores */
.oi-fator {{ padding: 11px 0 9px; border-bottom: 1px dashed var(--borda); }}
.oi-fator:last-of-type {{ border-bottom: 0; }}
.oi-fator .nome {{ font-weight: 650; font-size: 14px; color: var(--tinta); }}
.oi-fator .valor {{ color: var(--suave); font-size: 12.5px; line-height: 1.35; }}
.oi-barra {{ position: relative; height: 30px; margin-top: 4px; }}
.oi-barra::before {{ content: ""; position: absolute; left: calc(50% - 1px); top: 1px; bottom: 1px;
  width: 2px; border-radius: 1px; background: {BORDA_FORTE}; }}
.oi-barra .barra {{ position: absolute; top: 8px; height: 14px; }}
.oi-barra .barra.aumenta {{ border-radius: 0 4px 4px 0; }}
.oi-barra .barra.reduz {{ border-radius: 4px 0 0 4px; }}
.oi-barra .mult {{ position: absolute; top: 5px; font: 700 13px/20px {FONTE_TEXTO};
  color: var(--tinta); white-space: nowrap; }}
.oi-escala {{ display: flex; justify-content: space-between; color: var(--suave);
  font-size: 11.5px; margin-top: 2px; }}
.oi-legenda {{ color: var(--suave); font-size: 12px; line-height: 1.45; margin-top: 10px; }}

/* Figuras */
.oi-figura {{ margin: 0; background: #fff; border: 1px solid var(--borda); border-radius: 20px;
  padding: 18px 20px 14px; box-shadow: var(--sombra); }}
.oi-figura svg {{ width: 100%; height: auto; display: block; }}
.oi-figura figcaption {{ color: var(--suave); font-size: 13.5px; line-height: 1.5;
  margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--borda); }}
.oi-figura figcaption b {{ color: var(--tinta); }}

/* Fatos e números */
.oi-fato .titulo {{ font: 600 11.5px/1.2 {FONTE_TEXTO}; letter-spacing: .12em;
  text-transform: uppercase; color: var(--verde); margin-bottom: 8px; }}
.oi-fato .grande {{ font-family: {FONTE_TITULO}; font-weight: 700; font-size: 22px;
  line-height: 1.15; color: var(--tinta); }}
.oi-fato p {{ color: var(--suave); font-size: 13.5px; line-height: 1.5; margin: 6px 0 0; }}
.oi-numero .rotulo {{ color: var(--suave); font-size: 13px; font-weight: 600; }}
.oi-numero .linha {{ display: flex; align-items: baseline; gap: 10px; margin-top: 6px; }}
.oi-numero .modelo {{ font-family: {FONTE_TITULO}; font-weight: 800; font-size: 30px;
  color: var(--tinta); }}
.oi-numero .base {{ color: var(--suave); font-size: 14px; }}
.oi-numero .ic {{ color: var(--suave); font-size: 12.5px; margin-top: 4px; }}
.oi-aviso {{ border-radius: 18px; padding: 18px 22px; background: var(--dourado-claro);
  border: 1px solid #F3D88C; color: var(--tinta); font-size: 15px; line-height: 1.55; }}
.oi-aviso b {{ font-weight: 700; }}
.oi-lista {{ margin: 0; padding-left: 18px; color: var(--tinta); font-size: 14px;
  line-height: 1.6; }}
.oi-lista li {{ margin-bottom: 6px; }}

/* Prioridade e frequência */
.oi-prioridade {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px;
  margin: 14px 0 2px; }}
.oi-prioridade .faixa {{ border: 1px solid var(--borda); background: #F4F6F2; color: var(--suave);
  border-radius: 10px; padding: 8px 4px; text-align: center; font: 700 12.5px/1.1 {FONTE_TEXTO}; }}
.oi-prioridade .parcela {{ color: var(--suave); font-size: 11.5px; line-height: 1.3;
  text-align: center; margin-top: 5px; }}
.oi-prioridade .ativo .parcela {{ color: var(--tinta); font-weight: 600; }}
.oi-freq {{ color: var(--tinta); font-size: 15px; line-height: 1.45; margin: 12px 0 0; }}
.oi-freq b {{ font-weight: 700; }}
.oi-media {{ color: var(--suave); font-size: 13.5px; margin-top: 4px; }}
.oi-reco .porque {{ font: 600 11.5px/1.2 {FONTE_TEXTO}; letter-spacing: .12em;
  text-transform: uppercase; color: #9A6C00; margin: 0 0 6px; }}
.oi-reco .razao p::before {{ content: "• "; color: #C9920F; font-weight: 700; }}

/* Quando: bloco de dias × período */
.oi-quando {{ display: grid; grid-template-columns: 118px repeat(4, 1fr); gap: 14px 7px;
  align-items: stretch; margin-top: 6px; }}
.oi-quando .cab {{ font: 600 13px/1.2 {FONTE_TEXTO}; color: var(--tinta); text-align: center;
  padding-bottom: 4px; }}
.oi-quando .cab small {{ display: block; color: var(--suave); font-weight: 500; font-size: 11px;
  margin-top: 2px; }}
.oi-quando .lin {{ display: flex; align-items: center; font: 600 13px/1.25 {FONTE_TEXTO};
  color: var(--tinta); }}
.oi-quando .cel {{ position: relative; border-radius: 12px; padding: 13px 4px 11px;
  text-align: center; min-height: 64px; }}
.oi-quando .cel b {{ display: block; font: 800 18px/1.1 {FONTE_TITULO}; }}
.oi-quando .cel span {{ display: block; font-size: 11.5px; font-weight: 600; margin-top: 3px; }}
.oi-quando .cel.atual {{ box-shadow: 0 0 0 3px #fff, 0 0 0 5px var(--tinta); }}
.oi-quando .cel i {{ position: absolute; top: -9px; left: 50%; transform: translateX(-50%);
  background: var(--tinta); color: #fff; font: 600 10px/1 {FONTE_TEXTO}; font-style: normal;
  padding: 3px 7px; border-radius: 999px; white-space: nowrap; }}

/* Onde: uma barra por região */
.oi-onde .reg {{ display: grid; grid-template-columns: 118px 1fr 118px; gap: 12px;
  align-items: center; padding: 9px 0; border-bottom: 1px dashed var(--borda); }}
.oi-onde .reg:last-child {{ border-bottom: 0; }}
.oi-onde .nome {{ font-size: 14px; color: var(--tinta); font-weight: 500; }}
.oi-onde .reg.atual .nome {{ font-weight: 750; }}
.oi-onde .nome i {{ display: block; font-style: normal; font-size: 11px; color: var(--vermelho);
  font-weight: 600; margin-top: 2px; }}
.oi-onde .trilho {{ height: 14px; background: #F1F3EF; border-radius: 0 4px 4px 0; }}
.oi-onde .barra {{ height: 14px; border-radius: 0 4px 4px 0; }}
.oi-onde .valor {{ font-size: 13px; color: var(--suave); white-space: nowrap; }}
.oi-onde .valor b {{ color: var(--tinta); font-size: 14px; }}

/* Dá para confiar */
.oi-grande {{ font-family: {FONTE_TITULO}; font-weight: 800; font-size: 44px; line-height: 1;
  color: var(--verde-escuro); letter-spacing: -.02em; margin: 2px 0 8px; }}
.oi-grande.cinza {{ color: var(--suave); }}

/* Status técnico, em fundo claro */
.oi-status {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0 6px; }}
.oi-status .chip {{ color: var(--tinta); background: var(--verde-claro);
  border-color: #CFE6D8; }}
.oi-status .chip code {{ color: var(--verde-escuro); }}

/* Rodapé */
.oi-rodape {{ margin-top: 34px; padding-top: 14px; border-top: 1px solid var(--borda);
  color: var(--suave); font-size: 12.5px; line-height: 1.6; }}
</style>
"""


def _fruto(cx: float, cy: float, r: float) -> str:
    """Um fruto de guaraná aberto: casca vermelha, polpa branca e semente preta (o "olho")."""
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#D9262E" stroke="#A3171D" '
            f'stroke-width="1.5"/>'
            f'<ellipse cx="{cx - r * .38}" cy="{cy - r * .42}" rx="{r * .28}" ry="{r * .16}" '
            f'fill="#fff" opacity=".28" transform="rotate(-35 {cx} {cy})"/>'
            f'<ellipse cx="{cx + r * .12}" cy="{cy + r * .06}" rx="{r * .6}" ry="{r * .7}" '
            f'fill="#FFF7EA"/>'
            f'<circle cx="{cx + r * .16}" cy="{cy + r * .1}" r="{r * .39}" fill="#1B1512"/>'
            f'<circle cx="{cx + r * .04}" cy="{cy - r * .04}" r="{r * .1}" fill="#fff" '
            f'opacity=".9"/>')


def _ilustracao() -> str:
    bolhas = "".join(
        f'<circle cx="{x}" cy="{y}" r="{r}" fill="none" stroke="#fff" stroke-opacity=".35" '
        f'stroke-width="1.4"/>'
        for x, y, r in ((96, 34, 6), (128, 92, 3), (150, 20, 3.5), (174, 58, 2.5),
                        (318, 22, 5), (286, 10, 3)))
    folhas = (
        '<path d="M206 168 C176 120 196 64 262 42 C268 104 246 148 206 168 Z" fill="#1FA25C"/>'
        '<path d="M206 168 C218 128 236 92 262 42" fill="none" stroke="#0B4A30" '
        'stroke-opacity=".45" stroke-width="2"/>'
        '<path d="M232 170 C252 128 296 104 330 112 C314 150 280 172 232 170 Z" fill="#168A4C"/>'
        '<path d="M232 170 C262 150 292 128 330 112" fill="none" stroke="#0B4A30" '
        'stroke-opacity=".45" stroke-width="2"/>')
    frutos = "".join(_fruto(*f) for f in ((214, 112, 21), (252, 92, 24), (286, 122, 22),
                                          (246, 140, 19), (196, 148, 16)))
    return (f'<svg class="ilustracao" viewBox="0 0 330 172" aria-hidden="true">{bolhas}{folhas}'
            f'{frutos}</svg>')


ILUSTRACAO = _ilustracao()


def classe_html(classe: str, prefixo: str = "") -> str:
    """A classe como etiqueta colorida (com o nome escrito, a cor não é a única pista).

    `prefixo` vem antes do nome (ex.: "prioridade "), para "alta" não ler como classe social.
    """
    return (f'<span class="oi-classe" style="background:{CLASSE[classe]};'
            f'color:{CLASSE_TEXTO[classe]}">{prefixo}{ROTULO_CLASSE[classe]}</span>')


def secao(titulo: str, texto: str = "") -> str:
    corpo = f"<p>{escape(texto)}</p>" if texto else ""
    return f'<div class="oi-secao"><h2>{escape(titulo)}</h2>{corpo}</div>'


def figura(svg: str, legenda: str) -> str:
    """Uma figura com legenda; `legenda` já vem em HTML simples (negrito permitido)."""
    return f'<figure class="oi-figura">{svg}<figcaption>{legenda}</figcaption></figure>'
