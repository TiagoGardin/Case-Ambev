# %% [markdown]
# # 01 – Verificação da leitura da POF e primeira contagem da ocasião
#
# Objetivo: confirmar que a leitura está correta e medir quantos casos positivos
# existem para cada definição de target. Esse número decide se o target é viável.
#
# Rodar: `python notebooks/01_verificacao_pof.py` (ou célula a célula no VS Code).
# Pré-requisito: `python -m src.ingestion.download_pof` e `python -m src.ingestion.pof`.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

from src.ingestion.codigos_alimentos import (  # noqa: E402
    codigos_guarana,
    codigos_pizza,
    codigos_refrigerante,
)
from src.ingestion.pof import CHAVE_PESSOA  # noqa: E402

consumo = pd.read_parquet(ROOT / "data/processed/pof_consumo.parquet")

# %% [markdown]
# ## 1. Checagens de sanidade da leitura

# %%
assert len(consumo) == 1_175_390, "número de linhas diferente do arquivo original"
assert consumo["hora"].between(0, 23).all(), "hora fora de 0–23: colunas deslocadas?"
assert consumo["QUADRO"].isin(["72", "73"]).all(), "QUADRO deveria ser 72 ou 73"
dias_validos = {
    "Domingo", "Segunda-Feira", "Terca-Feira", "Quarta-Feira",
    "Quinta-Feira", "Sexta-Feira", "Sabado",
}
assert set(consumo["DIA_SEMANA"]) <= dias_validos, "DIA_SEMANA com texto cortado"
print("Leitura OK")

# %% [markdown]
# ## 2. Tamanho da amostra

# %%
n_pessoas = consumo.groupby(CHAVE_PESSOA).ngroups
n_pessoa_dias = consumo.groupby(CHAVE_PESSOA + ["QUADRO"]).ngroups
print(f"Pessoas:      {n_pessoas:,}")
print(f"Pessoa-dias:  {n_pessoa_dias:,}")

# %% [markdown]
# ## 3. Marcar itens de interesse

# %%
consumo["eh_pizza"] = consumo["V9001"].isin(codigos_pizza())
consumo["eh_refri"] = consumo["V9001"].isin(codigos_refrigerante())
consumo["eh_guarana"] = consumo["V9001"].isin(codigos_guarana())

# %% [markdown]
# ## 4. Contagem por refeição (pessoa × dia × ocasião V9017)
#
# "Mesma refeição" = mesmo valor de V9017 (café, almoço, lanche, jantar...) no mesmo dia.

# %%
refeicoes = (
    consumo.groupby(CHAVE_PESSOA + ["QUADRO", "V9017"])
    .agg(
        pizza=("eh_pizza", "max"),
        refri=("eh_refri", "max"),
        guarana=("eh_guarana", "max"),
        peso=("PESO_FINAL", "first"),
    )
    .reset_index()
)


def resumo(df: pd.DataFrame, nome_unidade: str) -> pd.DataFrame:
    """Contagem bruta e taxa ponderada pelo peso amostral (PESO_FINAL)."""
    definicoes = {
        "pizza": df["pizza"],
        "pizza + refri": df["pizza"] & df["refri"],
        "pizza + guaraná": df["pizza"] & df["guarana"],
    }
    linhas = []
    for nome, mask in definicoes.items():
        linhas.append({
            "definição": nome,
            "casos": int(mask.sum()),
            "taxa ponderada (%)": round(100 * df.loc[mask, "peso"].sum() / df["peso"].sum(), 3),
        })
    print(f"\nUnidade: {nome_unidade} ({len(df):,} linhas)")
    return pd.DataFrame(linhas)


print(resumo(refeicoes, "pessoa × dia × refeição").to_string(index=False))

# %% [markdown]
# ## 5. Participação do guaraná dentro das ocasiões pizza + refri

# %%
ocasioes = refeicoes[refeicoes["pizza"] & refeicoes["refri"]]
share = ocasioes["guarana"].mean()
print(f"Guaraná em {share:.1%} das ocasiões pizza + refri (n = {len(ocasioes)})")

# %% [markdown]
# ## 6. Validação da estimativa do mês da entrevista para inquilinos
#
# Inquilinos não têm Aluguel Estimado, então o mês deles vem do último aluguel
# pago + vizinhos da mesma UPA (ver `src/features/mes_entrevista.py`).
#
# Como validar sem saber o mês real dos inquilinos: aplicamos a mesma regra a
# UCs cujo mês É conhecido e que também têm uma despesa mensal de moradia
# (prestação ou condomínio, mesmo tipo de pagamento que o aluguel), fingindo
# não saber o mês delas. A mediana dos vizinhos exclui a própria UC.
#
# Também testamos nas poucas UCs que têm Aluguel Estimado E aluguel pago
# (casos atípicos, ex.: aluguel de outro imóvel), que são um teste mais duro.

# %%
from src.features.mes_entrevista import validar_regra_pagamento  # noqa: E402
from src.ingestion.pof import ALUGUEL_PAGO, CONDOMINIO, PRESTACAO_IMOVEL  # noqa: E402

aluguel_estimado = pd.read_parquet(ROOT / "data/processed/pof_aluguel_estimado.parquet")
pagamentos = pd.read_parquet(ROOT / "data/processed/pof_pagamentos_moradia.parquet")

for nome, codigos in [
    ("prestação ou condomínio (UCs com mês conhecido)", [PRESTACAO_IMOVEL, CONDOMINIO]),
    ("aluguel pago (UCs com mês conhecido, casos atípicos)", [ALUGUEL_PAGO]),
]:
    print(f"\nValidação com {nome}:")
    tabela = validar_regra_pagamento(aluguel_estimado, pagamentos, codigos)
    print(tabela.round(3).to_string(index=False))

# %% [markdown]
# ## 7. Evidência da regra do mês (decisão 1 das notas)
#
# No Aluguel Estimado, mês da entrevista = V9010 + 1. Só julho aparece nos dois anos da
# coleta; o DEFLATOR, que leva valores a preços de 15/01/2018, separa os dois: > 1 em 2017
# e < 1 em 2018.

# %%
mes = aluguel_estimado["V9010"] % 12 + 1
julho = aluguel_estimado[mes == 7]
print(f"UCs em julho: {len(julho):,} → 2017 (deflator > 1): {(julho['DEFLATOR'] > 1).sum():,}"
      f" | 2018 (deflator < 1): {(julho['DEFLATOR'] < 1).sum():,}")
print("Deflatores de julho entre 0,985 e 1,0 (esperado: 0):",
      int(julho["DEFLATOR"].between(0.985, 1.0, inclusive="neither").sum()))
anos = julho.assign(ano=np.where(julho["DEFLATOR"] > 1, 2017, 2018)).groupby("COD_UPA")["ano"]
print(f"UPAs com entrevista em julho: {anos.ngroups:,}; com os dois anos misturados: "
      f"{int((anos.nunique() > 1).sum())}")
print(f"Deflator médio: junho {aluguel_estimado.loc[mes == 6, 'DEFLATOR'].mean():.3f} (2018), "
      f"agosto {aluguel_estimado.loc[mes == 8, 'DEFLATOR'].mean():.3f} (2017)")
