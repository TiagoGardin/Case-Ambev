"""Monta a base de modelagem: uma linha por pessoa × dia × período.

Uso:
    python -m src.features.build_grid

Entrada: data/processed/pof_*.parquet (gerados por src.ingestion.pof)
Saída:   data/processed/grid.parquet

Unidade de análise
------------------
Cada pessoa da POF relatou tudo o que comeu em 2 dias (QUADRO 72 e 73).
Cada dia é dividido em 4 períodos. Para cada (pessoa, dia, período):
    y = 1 se comeu pizza E tomou refrigerante naquele período.

Instante de previsão: início do dia. As features do modelo são conhecidas
antes de a pessoa comer: quando (período, dia da semana), onde (UF, urbano ou
rural) e quem (idade, sexo, escolaridade, tamanho da família, renda per capita).
Tipo de área e classe ABEP ficam na base para segmentos e para o Decision
Engine: na triagem com validação cruzada no treino
(notebooks/02_triagem_features.py), elas não trouxeram ganho ao modelo.
Variáveis observadas junto com o consumo ou depois dele (local e tipo da
refeição, outros alimentos, dia atípico) ficam de fora para não vazar o target.
O local da pizza fica na base só como coluna de análise (`local_pizza`), para
medir o desempenho por canal e alimentar o Decision Engine.
A lista completa, com o papel de cada coluna, está no Data Card (seções 3 e 6).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.classe_abep import CLASSES, DISTRIBUICAO_OFICIAL_PCT, calcular_classe_abep
from src.features.mes_entrevista import ULTIMO_MES_TREINO, estimar_mes_entrevista
from src.features.tipo_area import tipo_area
from src.ingestion.codigos_alimentos import codigos_guarana, codigos_pizza, codigos_refrigerante
from src.ingestion.pof import CHAVE_DOM, CHAVE_PESSOA, CHAVE_UC

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"

SALARIO_MINIMO_2018 = 954.0

PERIODOS = ["manha", "almoco", "tarde", "noite"]
DIAS = ["Segunda-Feira", "Terca-Feira", "Quarta-Feira", "Quinta-Feira",
        "Sexta-Feira", "Sabado", "Domingo"]
FAIXAS_RENDA = ["ate_0.5sm", "0.5_1sm", "1_2sm", "2_5sm", "acima_5sm"]
FAIXAS_IDADE = ["10_17", "18_29", "30_44", "45_59", "60_mais"]
REGIOES = {"1": "Norte", "2": "Nordeste", "3": "Sudeste", "4": "Sul", "5": "Centro-Oeste"}

# Origem do mês da entrevista, da mais para a menos confiável (Data Card, seção 5)
FONTES_MES = ["aluguel_estimado", "aluguel_pago", "vizinhos_upa"]

# Local da refeição (V9018) agrupado; 9 (não declarado) fica nulo
LOCAIS = ["casa", "estabelecimento", "outro_fora"]
LOCAL_V9018 = {
    "1": "casa",             # em casa (inclui delivery: a POF não separa)
    "3": "estabelecimento",  # restaurante a quilo
    "4": "estabelecimento",  # lanchonete / fast food
    "5": "estabelecimento",  # restaurante
    "2": "outro_fora",       # merenda
    "6": "outro_fora",       # ambulante
    "7": "outro_fora",       # outros lugares fora de casa
}


# --------------------------------------------------------------------------
# Funções de feature reutilizáveis (a API vai importar estas mesmas funções,
# garantindo o mesmo processamento em treino e inferência)
# --------------------------------------------------------------------------
def periodo_do_dia(hora: pd.Series) -> pd.Series:
    """manhã 5–10h, almoço 11–14h, tarde 15–17h, noite 18–4h.

    A madrugada (0–4h) vai para 'noite' porque, na prática, é a continuação
    da noite anterior (pizza à meia-noite é ocasião noturna).
    """
    cond = [hora.between(5, 10), hora.between(11, 14), hora.between(15, 17)]
    return pd.Series(np.select(cond, PERIODOS[:3], default="noite"), index=hora.index)


def faixa_renda(renda_pc: pd.Series) -> pd.Series:
    """Renda per capita em faixas de salário mínimo de 2018."""
    em_sm = renda_pc / SALARIO_MINIMO_2018
    return pd.cut(em_sm, [-np.inf, 0.5, 1, 2, 5, np.inf], labels=FAIXAS_RENDA, right=False)


def faixa_idade(idade: pd.Series) -> pd.Series:
    return pd.cut(idade, [10, 18, 30, 45, 60, np.inf], labels=FAIXAS_IDADE, right=False)


def estacao(mes: pd.Series) -> pd.Series:
    """Estação do ano (hemisfério sul), pelo mês da entrevista. Só para análise."""
    mapa = {12: "verao", 1: "verao", 2: "verao", 3: "outono", 4: "outono", 5: "outono",
            6: "inverno", 7: "inverno", 8: "inverno", 9: "primavera", 10: "primavera",
            11: "primavera"}
    return mes.map(mapa)


# --------------------------------------------------------------------------
# Etapas da montagem
# --------------------------------------------------------------------------
def base_pessoa_dia(consumo: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por pessoa × dia, com o contexto daquele dia."""
    chave = CHAVE_PESSOA + ["QUADRO"]
    base = consumo.groupby(chave, as_index=False).agg(
        UF=("UF", "first"),
        situacao=("TIPO_SITUACAO_REG", "first"),
        dia_semana=("DIA_SEMANA", "first"),
        dia_atipico=("DIA_ATIPICO", "first"),
        peso=("PESO_FINAL", "first"),
        renda_total_uc=("RENDA_TOTAL", "first"),
    )
    # Checagem: um dia de recordatório deve ter um único dia da semana
    n_dias = consumo.groupby(chave)["DIA_SEMANA"].nunique()
    assert (n_dias == 1).all(), "pessoa-dia com mais de um DIA_SEMANA"
    return base


def ocasioes_por_periodo(consumo: pd.DataFrame) -> pd.DataFrame:
    """Marca, para cada pessoa × dia × período, se houve pizza, refri e guaraná,
    e onde a pizza foi comida (só análise)."""
    chave = CHAVE_PESSOA + ["QUADRO", "periodo"]
    itens = consumo.assign(
        periodo=periodo_do_dia(consumo["hora"]),
        pizza=consumo["V9001"].isin(codigos_pizza()),
        refri=consumo["V9001"].isin(codigos_refrigerante()),
        guarana=consumo["V9001"].isin(codigos_guarana()),
    )
    oc = itens.groupby(chave, as_index=False)[["pizza", "refri", "guarana"]].max()
    return oc.merge(local_da_pizza(itens), on=chave, how="left")


def local_refeicao(v9018: pd.Series) -> pd.Series:
    """Agrupa o local da refeição (V9018) em casa, estabelecimento e outro lugar fora."""
    return v9018.map(LOCAL_V9018)


def local_da_pizza(itens: pd.DataFrame) -> pd.DataFrame:
    """Onde a pizza foi comida, para cada pessoa × dia × período com pizza.

    Recebe os itens consumidos já marcados com `periodo` e `pizza`. Com mais de
    uma pizza no período, vale a primeira em ordem de horário que tenha local
    declarado. Na POF, só 1 dos 1.716 períodos com pizza tem locais diferentes.

    Só para análise: o local é observado junto com o consumo e vazaria o target
    se fosse feature.
    """
    chave = CHAVE_PESSOA + ["QUADRO", "periodo"]
    pz = itens.loc[itens["pizza"], chave + ["hora", "SEQ", "V9018"]].copy()
    pz["local_pizza"] = local_refeicao(pz["V9018"])
    # A madrugada (0–4h) vem depois da noite; SEQ é texto ("10" < "2"), então vira número
    pz["instante"] = pz["hora"].where(pz["hora"] >= 5, pz["hora"] + 24)
    pz["ordem"] = pd.to_numeric(pz["SEQ"])
    pz = pz.sort_values(["instante", "ordem"])
    return pz.groupby(chave, as_index=False)["local_pizza"].first()


def perfil_pessoa(morador: pd.DataFrame) -> pd.DataFrame:
    """Idade, sexo, escolaridade e tamanho da unidade de consumo."""
    n_mor = morador.groupby(CHAVE_UC).size().rename("n_moradores").reset_index()
    p = morador[CHAVE_PESSOA + ["V0403", "V0404", "ANOS_ESTUDO"]].rename(
        columns={"V0403": "idade", "V0404": "sexo", "ANOS_ESTUDO": "anos_estudo"}
    )
    p["sexo"] = p["sexo"].map({"1": "masculino", "2": "feminino"})
    return p.merge(n_mor, on=CHAVE_UC, how="left")


def perfil_uc(morador: pd.DataFrame, domicilio: pd.DataFrame, inventario: pd.DataFrame,
              aluguel_estimado: pd.DataFrame, pagamentos: pd.DataFrame) -> pd.DataFrame:
    """Tudo o que é da unidade de consumo: tipo de área, classe ABEP e mês da entrevista."""
    ucs = morador.drop_duplicates(subset=CHAVE_UC)[CHAVE_UC + ["ESTRATO_POF"]].copy()
    ucs["tipo_area"] = tipo_area(ucs["ESTRATO_POF"])
    ucs = ucs.merge(calcular_classe_abep(morador, domicilio, inventario), on=CHAVE_UC, how="left")
    meses = estimar_mes_entrevista(ucs, aluguel_estimado, pagamentos)
    return ucs.drop(columns="ESTRATO_POF").merge(meses, on=CHAVE_UC, how="left")


def split_temporal(ucs: pd.DataFrame) -> pd.Series:
    """Treino, teste ou sem_mes para cada UC, decidido por domicílio.

    As UCs de um domicílio são entrevistadas na mesma visita e a família divide
    a pizza, então ficam sempre do mesmo lado do corte. Vale o mês da UC com a
    fonte mais confiável; em empate, o mês mais cedo (início da visita).
    """
    rank = ucs["fonte_mes"].map({f: i for i, f in enumerate(FONTES_MES)})
    ref = (ucs.assign(_rank=rank)
              .dropna(subset=["ordem_mes"])
              .sort_values(["_rank", "ordem_mes"])
              .drop_duplicates(subset=CHAVE_DOM)[CHAVE_DOM + ["ordem_mes"]]
              .rename(columns={"ordem_mes": "ordem_split"}))
    ordem = ucs[CHAVE_DOM].merge(ref, on=CHAVE_DOM, how="left")["ordem_split"].to_numpy()
    split = np.select([np.isnan(ordem), ordem <= ULTIMO_MES_TREINO], ["sem_mes", "treino"],
                      default="teste")
    return pd.Series(split, index=ucs.index)


def peso_semana(grid: pd.DataFrame) -> pd.Series:
    """PESO_FINAL ajustado para que cada dia da semana tenha 1/7 do peso, dentro de cada split.

    O PESO_FINAL expande pessoas, não dias. Como o recordatório é do dia anterior à
    entrevista e há poucas entrevistas no fim de semana, sextas e sábados ficam bem
    abaixo de 1/7 dos dias da amostra (cerca de 9% e 5%). Taxas e métricas que somam
    a semana usam este peso; a taxa de um dia só não muda, porque o fator é o mesmo
    dentro do dia. Usa só a distribuição dos dias de recordatório, não o target.
    """
    pdias = grid.drop_duplicates(subset=CHAVE_PESSOA + ["QUADRO"])
    soma = pdias.groupby(["split", "dia_semana"], observed=True)["peso"].sum()
    fator = (1 / 7) / (soma / soma.groupby(level="split").transform("sum"))
    chave = pd.MultiIndex.from_frame(grid[["split", "dia_semana"]].astype(str))
    return pd.Series(grid["peso"].to_numpy() * fator.reindex(chave).to_numpy(), index=grid.index)


def montar_grid() -> pd.DataFrame:
    consumo = pd.read_parquet(PROC / "pof_consumo.parquet")
    morador = pd.read_parquet(PROC / "pof_morador.parquet")
    uc = perfil_uc(
        morador,
        domicilio=pd.read_parquet(PROC / "pof_domicilio.parquet"),
        inventario=pd.read_parquet(PROC / "pof_inventario.parquet"),
        aluguel_estimado=pd.read_parquet(PROC / "pof_aluguel_estimado.parquet"),
        pagamentos=pd.read_parquet(PROC / "pof_pagamentos_moradia.parquet"),
    )

    # 1) Grade completa: todo pessoa-dia × 4 períodos.
    #    Sem isso, só existiriam linhas de períodos em que a pessoa comeu algo,
    #    e os períodos "sem pizza" (negativos) ficariam sub-representados.
    base = base_pessoa_dia(consumo)
    grid = base.merge(pd.DataFrame({"periodo": PERIODOS}), how="cross")

    # 2) Target: junta o que foi consumido em cada período
    oc = ocasioes_por_periodo(consumo)
    grid = grid.merge(oc, on=CHAVE_PESSOA + ["QUADRO", "periodo"], how="left")
    for col in ["pizza", "refri", "guarana"]:
        grid[col] = grid[col].fillna(False).astype(bool)
    grid["y"] = (grid["pizza"] & grid["refri"]).astype(int)
    grid["y_guarana"] = (grid["pizza"] & grid["guarana"]).astype(int)  # só análise

    # 3) Perfil da pessoa e da unidade de consumo
    grid = grid.merge(perfil_pessoa(morador), on=CHAVE_PESSOA, how="left")
    grid = grid.merge(uc, on=CHAVE_UC, how="left")
    grid["renda_pc"] = grid["renda_total_uc"] / grid["n_moradores"]
    grid["faixa_renda"] = faixa_renda(grid["renda_pc"])
    grid["faixa_idade"] = faixa_idade(grid["idade"])

    # 4) Território
    grid["regiao"] = grid["UF"].str[0].map(REGIOES)
    grid["urbano"] = (grid["situacao"] == "1").astype(int)

    # 5) Tempo
    fds = ["Sexta-Feira", "Sabado", "Domingo"]
    grid["fim_de_semana"] = grid["dia_semana"].isin(fds).astype(int)
    grid["dia_atipico"] = (grid["dia_atipico"] == "1").astype(int)
    grid["estacao"] = estacao(grid["mes_entrevista"])

    # 6) Split temporal pelo mês estimado da entrevista, decidido por domicílio
    ucs = grid.drop_duplicates(subset=CHAVE_UC)[CHAVE_UC + ["ordem_mes", "fonte_mes"]]
    ucs = ucs.assign(split=split_temporal(ucs))
    grid = grid.merge(ucs[CHAVE_UC + ["split"]], on=CHAVE_UC, how="left")

    # 7) Peso por pessoa-dia: cada dia da semana com 1/7 do peso, em cada split
    grid["peso_semana"] = peso_semana(grid)

    # Categorias ordenadas (ajuda em gráficos e no one-hot)
    grid["periodo"] = pd.Categorical(grid["periodo"], PERIODOS, ordered=True)
    grid["dia_semana"] = pd.Categorical(grid["dia_semana"], DIAS, ordered=True)
    grid["classe_abep"] = pd.Categorical(grid["classe_abep"], CLASSES, ordered=True)
    grid["local_pizza"] = pd.Categorical(grid["local_pizza"], LOCAIS)

    colunas = (
        CHAVE_PESSOA + ["QUADRO", "peso", "peso_semana", "split"]
        # tempo da entrevista (split e análise; NÃO é feature)
        + ["ano_entrevista", "mes_entrevista", "ordem_mes", "fonte_mes", "estacao"]
        # features do modelo (triagem: notebooks/02_triagem_features.py)
        + ["periodo", "dia_semana", "UF", "regiao", "urbano",
           "idade", "faixa_idade", "sexo", "anos_estudo", "n_moradores", "renda_pc"]
        # alternativas, segmentos e análise (ver Data Card, seção 3)
        + ["faixa_renda", "fim_de_semana", "tipo_area", "abep_pontos", "classe_abep",
           "dia_atipico"]
        # targets e marcadores (NÃO usar como feature)
        + ["y", "y_guarana", "pizza", "refri", "guarana", "local_pizza"]
    )
    return grid[colunas]


def validar(grid: pd.DataFrame, n_pessoa_dias: int) -> None:
    assert len(grid) == n_pessoa_dias * len(PERIODOS), "grade incompleta"
    assert not grid.duplicated(CHAVE_PESSOA + ["QUADRO", "periodo"]).any(), "linhas duplicadas"
    assert grid["y"].notna().all(), "target com nulos"
    for col in ["idade", "tipo_area", "classe_abep"]:
        assert grid[col].notna().all(), f"{col} com nulos"
    assert grid.loc[~grid["pizza"], "local_pizza"].isna().all(), "local_pizza sem pizza no período"
    lados = grid.groupby(CHAVE_DOM)["split"].nunique()
    assert (lados == 1).all(), "domicílio nos dois lados do split"
    pdias = grid.drop_duplicates(subset=CHAVE_PESSOA + ["QUADRO"])
    for s in ["treino", "teste"]:
        d = pdias[pdias["split"] == s]
        soma = d.groupby("dia_semana", observed=True)["peso_semana"].sum()
        fatias = soma / soma.sum()
        assert len(fatias) == 7 and np.allclose(fatias, 1 / 7), f"peso_semana desigual ({s})"


def _resumo(grid: pd.DataFrame) -> None:
    """Imprime os números citados no Data Card (seções 1, 2, 5 e 7)."""
    ucs = grid.drop_duplicates(subset=CHAVE_UC)
    dias = grid.drop_duplicates(subset=CHAVE_PESSOA + ["QUADRO"]).groupby(CHAVE_PESSOA).size()
    pos = grid[grid["y"] == 1]
    def taxa(d: pd.DataFrame, peso: str) -> float:
        return (d["y"] * d[peso]).sum() / d[peso].sum() * 100

    n_dom = grid[CHAVE_DOM].drop_duplicates().shape[0]
    print(f"\nCobertura: {dias.size:,} pessoas em {n_dom:,} domicílios; "
          f"{(dias == 2).sum():,} com 2 dias de recordatório e {(dias == 1).sum():,} com 1")
    print(f"Target: {len(pos):,} positivos, em {pos[CHAVE_PESSOA].drop_duplicates().shape[0]:,} "
          f"pessoas de {pos[CHAVE_DOM].drop_duplicates().shape[0]:,} domicílios")
    for s, d in [("base inteira", grid)] + [(s, grid[grid["split"] == s])
                                            for s in ["treino", "teste"]]:
        print(f"  taxa ({s}): {taxa(d, 'peso_semana'):.3f}% com peso_semana "
              f"({taxa(d, 'peso'):.3f}% só com PESO_FINAL)")

    print("\nPor split:")
    print(grid.groupby("split").agg(linhas=("y", "size"), positivos=("y", "sum"),
                                    pizza_guarana=("y_guarana", "sum")).to_string())

    print("\nOrigem do mês da entrevista (UCs):")
    print(ucs["fonte_mes"].value_counts(dropna=False).to_string())

    print("\nClasse ABEP (% ponderado das UCs) × distribuição oficial ABEP 2019:")
    pct = (ucs.groupby("classe_abep", observed=True)["peso"].sum() / ucs["peso"].sum() * 100)
    oficial = pd.Series(DISTRIBUICAO_OFICIAL_PCT)
    comp = pd.DataFrame({"POF (%)": pct.round(1), "ABEP 2019 (%)": oficial})
    print(comp.loc[CLASSES].to_string())

    print("\nTipo de área × situação do domicílio (UCs):")
    print(pd.crosstab(ucs["tipo_area"], ucs["urbano"].map({1: "urbano", 0: "rural"})).to_string())

    print("\nLocal da pizza (só análise; nulo = local não declarado):")
    com_pizza = grid[grid["pizza"]]
    print(pd.DataFrame({
        "períodos com pizza": com_pizza["local_pizza"].value_counts(dropna=False),
        "pizza + refri (y=1)": com_pizza.loc[com_pizza["y"] == 1, "local_pizza"]
        .value_counts(dropna=False),
    }).to_string())

    # local_pizza é nulo por desenho quando não há pizza no período
    nulos = grid.drop(columns="local_pizza").isna().sum()
    print("\nNulos por coluna (só as que têm):")
    print(nulos[nulos > 0].to_string() if (nulos > 0).any() else "nenhum")


def main() -> None:
    grid = montar_grid()
    n_pd = grid.groupby(CHAVE_PESSOA + ["QUADRO"]).ngroups
    validar(grid, n_pd)
    grid.to_parquet(PROC / "grid.parquet", index=False)

    print(f"[ok] grid: {len(grid):,} linhas ({n_pd:,} pessoa-dias × {len(PERIODOS)} períodos)")
    print(f"     positivos (pizza+refri): {grid['y'].sum():,}"
          f"  |  pizza+guaraná: {grid['y_guarana'].sum():,}")
    _resumo(grid)


if __name__ == "__main__":
    main()
