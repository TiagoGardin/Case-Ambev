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
