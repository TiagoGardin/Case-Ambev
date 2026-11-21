"""Decision Engine: do Occasion Score à categoria de ativação, por regras documentadas.

Não existe histórico de contexto → ação → resultado, então não há como treinar um
segundo modelo para escolher a ação. As regras combinam:
- a classe do score (`alta`, `media` ou `baixa`): se vale ativar e com que prioridade;
- o canal, informado por quem usa (delivery, varejo ou estabelecimento): por onde;
- a associação atual à marca no segmento, informada por quem usa (baixa ou normal):
  se a mensagem é de conversão ou de construção de associação;
- o contexto observado na POF (src.decision_engine.contexto) e os fatores do SHAP,
  que entram só na justificativa.

A primeira regra de REGRAS que se aplica decide. As justificativas descrevem onde a
ocasião costuma acontecer; nenhuma afirma que uma ação causa vendas (docs/decision_engine.md).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.models.artefato import CLASSES
from src.models.explicar import NOME_FATOR

VERSAO = "1.0.0"
CANAIS = ("delivery", "varejo", "estabelecimento")
ASSOCIACOES = ("baixa", "normal")
CATEGORIAS = {
    "ativacao_delivery":
        "Ativação via delivery: combo de pizza com Guaraná em aplicativos e pizzarias com "
        "entrega",
    "combo_consumo_em_casa":
        "Promoção ou combo para consumo em casa: pizza com Guaraná no varejo e no delivery",
    "ativacao_ponto_de_consumo":
        "Ativação no ponto de consumo: pizzarias, restaurantes e bares",
    "awareness_associacao":
        "Awareness e construção de associação entre Guaraná e pizza",
    "presenca_de_marca":
        "Presença de marca de baixo custo (mídia contínua, material de ponto de venda), sem "
        "promoção dedicada",
    "sem_ativacao_dedicada":
        "Sem ativação dedicada: o contexto só entra em ações amplas",
}
# (regra, quando se aplica, categoria), na ordem de avaliação
REGRAS = [
    ("R1", "classe baixa", "sem_ativacao_dedicada"),
    ("R2", "classe alta ou média e associação à marca informada como baixa",
     "awareness_associacao"),
    ("R3", "classe alta e canal delivery", "ativacao_delivery"),
    ("R4", "classe alta e canal estabelecimento", "ativacao_ponto_de_consumo"),
    ("R5", "classe alta e canal varejo ou não informado", "combo_consumo_em_casa"),
    ("R6", "classe média", "presenca_de_marca"),
]


@dataclass(frozen=True)
class Recomendacao:
    categoria: str
    justificativa: str
    regra: str


def _pct(v: float, casas: int = 0) -> str:
    return f"{v * 100:.{casas}f}%".replace(".", ",")


def _num(v: float, casas: int = 1) -> str:
    return f"{v:.{casas}f}".replace(".", ",")


def _parcela(contexto: dict, regiao: str, medida: str) -> tuple[str, str]:
    """("68%", "IC 95% de 47% a 87%; Brasil: 65%") para uma medida do contexto observado."""
    r, br = contexto["regioes"][regiao][medida], contexto["brasil"][medida]
    lo, hi = r["ic95"]
    return (_pct(r["participacao"]),
            f"IC 95% de {_pct(lo)} a {_pct(hi)}; Brasil: {_pct(br['participacao'])}")


EFEITO = {"aumenta": "aumenta a chance", "reduz": "reduz a chance", "neutro": "não muda a chance"}


def _fatores(top_fatores: list[dict]) -> str:
    partes = [f"{NOME_FATOR[f['factor']]}: {f['value']} ({EFEITO[f['effect']]}, "
              f"× {_num(f['odds_multiplier'], 2)})" for f in top_fatores]
    return "Principais fatores do score: " + "; ".join(partes) + "."


def validar(classe: str, regiao: str, contexto: dict, canal: str | None,
            associacao_marca: str | None) -> None:
    if classe not in CLASSES:
        raise ValueError(f"classe desconhecida: {classe!r}")
    if regiao not in contexto["regioes"]:
        raise ValueError(f"região sem contexto observado: {regiao!r}")
    if canal is not None and canal not in CANAIS:
        raise ValueError(f"canal deve ser um de {CANAIS} ou vazio")
    if associacao_marca is not None and associacao_marca not in ASSOCIACOES:
        raise ValueError(f"associacao_marca deve ser uma de {ASSOCIACOES} ou vazia")


def recomendar(score: float, classe: str, top_fatores: list[dict], regiao: str, contexto: dict,
               taxa_referencia: float, canal: str | None = None,
               associacao_marca: str | None = None) -> Recomendacao:
    """Categoria de ativação e justificativa para um contexto já pontuado pelo modelo."""
    validar(classe, regiao, contexto, canal, associacao_marca)
    score_txt = (f"O score é {_pct(score, 2)}, {_num(score / taxa_referencia)} vezes a taxa "
                 f"média do treino ({_pct(taxa_referencia, 2)}), na classe {classe}.")
    fatores = _fatores(top_fatores)
    valor, detalhe = _parcela(contexto, regiao, "casa")
    em_casa = (f"Na POF, {valor} das ocasiões de pizza com refrigerante da região {regiao} "
               f"foram em casa, o que inclui delivery ({detalhe}).")

    if classe == "baixa":
        motivo = ("A chance da ocasião fica abaixo do corte da classe média, então o contexto "
                  "não justifica investimento dedicado.")
        return Recomendacao("sem_ativacao_dedicada", f"{score_txt} {motivo} {fatores}", "R1")

    if associacao_marca == "baixa":
        via = f", usando o canal {canal}" if canal else ""
        valor, detalhe = _parcela(contexto, regiao, "guarana")
        guarana = (f"Na POF, o guaraná (qualquer marca) esteve em {valor} das ocasiões de pizza "
                   f"com refrigerante da região {regiao} ({detalhe}).")
        motivo = (f"A associação atual à marca no segmento foi informada como baixa. Com chance "
                  f"{'alta' if classe == 'alta' else 'moderada'} da ocasião, a prioridade é "
                  f"construir a associação entre Guaraná e pizza{via}.")
        return Recomendacao("awareness_associacao", f"{score_txt} {motivo} {guarana} {fatores}",
                            "R2")

    if classe == "alta" and canal == "delivery":
        motivo = "Com chance alta e o canal delivery informado, a ativação é no delivery."
        return Recomendacao("ativacao_delivery", f"{score_txt} {motivo} {em_casa} {fatores}",
                            "R3")

    if classe == "alta" and canal == "estabelecimento":
        valor, detalhe = _parcela(contexto, regiao, "estabelecimento")
        fora = (f"Na POF, {valor} das ocasiões de pizza com refrigerante da região {regiao} "
                f"foram em estabelecimentos ({detalhe}).")
        motivo = ("Com chance alta e o canal estabelecimento informado, a ativação é no ponto "
                  "de consumo.")
        return Recomendacao("ativacao_ponto_de_consumo", f"{score_txt} {motivo} {fora} {fatores}",
                            "R4")

    if classe == "alta":
        if canal == "varejo":
            motivo = ("Com chance alta e o canal varejo informado, a ativação é um combo para "
                      "consumo em casa.")
        else:
            motivo = ("Com chance alta e sem canal informado, a recomendação segue o local mais "
                      "comum da ocasião, que é a casa em todas as regiões.")
        return Recomendacao("combo_consumo_em_casa", f"{score_txt} {motivo} {em_casa} {fatores}",
                            "R5")

    via = f" no canal {canal}" if canal else ""
    motivo = ("A chance é moderada: o contexto fica atrás dos de classe alta na prioridade e pede "
              f"presença de marca de baixo custo{via}, sem promoção dedicada.")
    return Recomendacao("presenca_de_marca", f"{score_txt} {motivo} {fatores}", "R6")
