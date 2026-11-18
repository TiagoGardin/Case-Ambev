"""Ano e mês da entrevista de cada unidade de consumo (UC).

Os microdados públicos da POF não trazem a data da entrevista. Ela é
reconstruída em três níveis, do mais preciso para o menos preciso
(a coluna `fonte_mes` diz qual foi usado):

1. "aluguel_estimado" (imóvel próprio ou cedido, ~84% das UCs)
   No registro Aluguel Estimado, V9010 é, pelo dicionário, "o mês anterior à
   data de abertura da pesquisa". Logo, mês = V9010 + 1.
   A coleta foi de jul/2017 a jul/2018, então só julho é ambíguo. O DEFLATOR
   leva valores a preços de 15/01/2018: > 1 para valores de 2017, < 1 para
   2018. Em julho a distribuição é bimodal, sem nenhum valor entre 0,985 e 1,0.

2. "aluguel_pago" (inquilinos, que não têm Aluguel Estimado)
   O quadro 10 da Despesa Coletiva traz o último mês em que o aluguel foi pago
   (V9010). Usamos a mesma regra: mês = V9010 + 1. O ano (e a consistência)
   vêm dos vizinhos: UCs da mesma UPA (setor sorteado) são entrevistadas na
   mesma época, normalmente dentro de 2 a 3 meses. Se o mês do pagamento ficar
   a mais de TOLERANCIA_MESES da mediana dos vizinhos, usamos a mediana.

3. "vizinhos_upa" (sem aluguel pago, ou pagamento incoerente)
   Mediana do mês das UCs da mesma UPA que têm Aluguel Estimado.

A regra 2 foi validada em UCs com mês conhecido que também pagam prestação ou
condomínio (mesmo tipo de despesa mensal). Ver `validar_regra_pagamento` e a
seção 6 de notebooks/01_verificacao_pof.py.

Uso: só para o split temporal. O mês NÃO é feature do modelo (ver Data Card).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.ingestion.pof import ALUGUEL_PAGO, CHAVE_UC

# Ordem do mês na coleta: 0 = jul/2017, 1 = ago/2017, ..., 12 = jul/2018
ULTIMO_MES_TREINO = 7  # treino: jul/2017–fev/2018 | teste: mar/2018–jul/2018
TOLERANCIA_MESES = 2


def candidatos_ordem(mes: int) -> list[int]:
    """Ordens possíveis de um mês do calendário. Só julho aparece duas vezes."""
    if mes == 7:
        return [0, 12]
    return [mes - 7] if mes >= 8 else [mes + 5]


def ano_mes_da_ordem(ordem: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Converte a ordem (0–12) de volta em ano e mês do calendário."""
    ano = 2017 + (ordem + 6) // 12
    mes = (ordem + 6) % 12 + 1
    return ano, mes


def mes_por_aluguel_estimado(aluguel_estimado: pd.DataFrame) -> pd.DataFrame:
    """Nível 1: regra exata para UCs com Aluguel Estimado."""
    df = aluguel_estimado[CHAVE_UC].copy()
    mes = aluguel_estimado["V9010"] % 12 + 1
    ordem = np.where(mes >= 8, mes - 7, mes + 5)  # julho cai em 12 (jul/2018) por padrão
    julho_2017 = (mes == 7) & (aluguel_estimado["DEFLATOR"] > 1)
    df["ordem_mes"] = np.where(julho_2017, 0, ordem).astype(float)
    df["fonte_mes"] = "aluguel_estimado"
    return df


def ordem_por_pagamento(
    v9010: float, mediana_upa: float, tolerancia: int = TOLERANCIA_MESES
) -> tuple[float, str | None]:
    """Níveis 2 e 3: estima a ordem a partir do último pagamento e dos vizinhos.

    Devolve (ordem, fonte). Sem pagamento e sem vizinhos: (NaN, None).
    """
    tem_vizinhos = not pd.isna(mediana_upa)
    referencia = math.floor(mediana_upa + 0.5) if tem_vizinhos else np.nan

    if pd.isna(v9010):
        return (referencia, "vizinhos_upa") if tem_vizinhos else (np.nan, None)

    candidatos = candidatos_ordem(int(v9010) % 12 + 1)
    if not tem_vizinhos:
        # Sem vizinhos, só dá para decidir se o mês não for julho
        return (candidatos[0], "aluguel_pago") if len(candidatos) == 1 else (np.nan, None)

    melhor = min(candidatos, key=lambda c: abs(c - mediana_upa))
    if abs(melhor - mediana_upa) > tolerancia:
        return referencia, "vizinhos_upa"
    return melhor, "aluguel_pago"


def _ultimo_pagamento(pagamentos: pd.DataFrame, codigos: list[int]) -> pd.DataFrame:
    """Um pagamento por UC; se houver mais de um, fica o pago por mais meses."""
    p = pagamentos[pagamentos["V9001"].isin(codigos)]
    return p.sort_values("V9011", ascending=False).drop_duplicates(subset=CHAVE_UC)


def estimar_mes_entrevista(
    ucs: pd.DataFrame, aluguel_estimado: pd.DataFrame, pagamentos: pd.DataFrame
) -> pd.DataFrame:
    """Ano, mês, ordem e fonte da estimativa para cada UC (UCs sem estimativa ficam de fora)."""
    conhecidos = mes_por_aluguel_estimado(aluguel_estimado)
    mediana_upa = conhecidos.groupby("COD_UPA")["ordem_mes"].median()

    restantes = (
        ucs[CHAVE_UC]
        .drop_duplicates()
        .merge(conhecidos[CHAVE_UC], on=CHAVE_UC, how="left", indicator=True)
        .query("_merge == 'left_only'")
        .drop(columns="_merge")
    )
    aluguel = _ultimo_pagamento(pagamentos, [ALUGUEL_PAGO])
    restantes = restantes.merge(aluguel[CHAVE_UC + ["V9010"]], on=CHAVE_UC, how="left")
    restantes["mediana_upa"] = restantes["COD_UPA"].map(mediana_upa)

    estimativas = [
        ordem_por_pagamento(v, m) for v, m in zip(restantes["V9010"], restantes["mediana_upa"])
    ]
    restantes["ordem_mes"] = [e[0] for e in estimativas]
    restantes["fonte_mes"] = [e[1] for e in estimativas]

    todos = pd.concat(
        [conhecidos, restantes[CHAVE_UC + ["ordem_mes", "fonte_mes"]]], ignore_index=True
    ).dropna(subset=["ordem_mes"])
    todos["ano_entrevista"], todos["mes_entrevista"] = ano_mes_da_ordem(todos["ordem_mes"])
    return todos[CHAVE_UC + ["ano_entrevista", "mes_entrevista", "ordem_mes", "fonte_mes"]]


def validar_regra_pagamento(
    aluguel_estimado: pd.DataFrame,
    pagamentos: pd.DataFrame,
    codigos: list[int],
    tolerancia: int = TOLERANCIA_MESES,
) -> pd.DataFrame:
    """Mede o erro da regra em UCs cujo mês é conhecido (têm Aluguel Estimado).

    Para cada UC com mês conhecido e um pagamento dos `codigos`, estima o mês
    como se ela fosse inquilina: usa o pagamento e a mediana dos OUTROS vizinhos
    da UPA (leave-one-out, para a própria UC não "colar" a resposta).
    Compara também com a alternativa mais simples: só a mediana dos vizinhos.
    """
    conhecidos = mes_por_aluguel_estimado(aluguel_estimado)
    alvo = _ultimo_pagamento(pagamentos, codigos)[CHAVE_UC + ["V9010"]].merge(
        conhecidos[CHAVE_UC + ["ordem_mes"]], on=CHAVE_UC
    )

    vizinhos = conhecidos[CHAVE_UC + ["ordem_mes"]].rename(
        columns={"NUM_DOM": "dom_viz", "NUM_UC": "uc_viz", "ordem_mes": "ordem_viz"}
    )
    pares = alvo[CHAVE_UC].merge(vizinhos, on="COD_UPA")
    pares = pares[(pares["NUM_DOM"] != pares["dom_viz"]) | (pares["NUM_UC"] != pares["uc_viz"])]
    mediana = pares.groupby(CHAVE_UC)["ordem_viz"].median().rename("mediana_upa")
    alvo = alvo.merge(mediana, on=CHAVE_UC, how="left")

    real = alvo["ordem_mes"].to_numpy()
    regras = {
        "pagamento + vizinhos (regra usada)": [
            ordem_por_pagamento(v, m, tolerancia)[0]
            for v, m in zip(alvo["V9010"], alvo["mediana_upa"])
        ],
        "só vizinhos (alternativa)": [
            ordem_por_pagamento(np.nan, m)[0] for m in alvo["mediana_upa"]
        ],
    }
    linhas = []
    for nome, estimado in regras.items():
        est = np.asarray(estimado, dtype=float)
        ok = ~np.isnan(est)
        erro = est[ok] - real[ok]
        linhas.append({
            "regra": nome,
            "n": int(ok.sum()),
            "exato": np.mean(erro == 0),
            "erro ≤ 1 mês": np.mean(np.abs(erro) <= 1),
            "erro > 3 meses": np.mean(np.abs(erro) > 3),
            "mesmo lado do corte treino/teste": np.mean(
                (est[ok] <= ULTIMO_MES_TREINO) == (real[ok] <= ULTIMO_MES_TREINO)
            ),
        })
    return pd.DataFrame(linhas)
