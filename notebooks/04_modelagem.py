# %% [markdown]
# # 04 – Modelagem com validação cruzada (só treino)
#
# Protocolo do README (Metodologia), fixado antes de abrir o teste:
# - **Dados:** só o treino (jul/2017–fev/2018). O teste não entra.
# - **Partições:** as mesmas 50 da triagem de features (5 folds agrupados por
#   domicílio × 10 repetições), de `src/models/validacao.py`.
# - **Métricas em cada fold:** PR-AUC ponderada por `peso_semana` (principal) e
#   captura no top 10% (leitura de negócio).
# - **Regras:** entre duas opções, a mais complexa só vence com ganho de PR-AUC
#   significativo a 5% (unilateral) no teste t corrigido para validação cruzada
#   repetida; hiperparâmetros seguem só a maior PR-AUC média.
#
# Esta versão cobre o baseline dia × período e a Regressão Logística, com o C da
# regularização escolhido pela maior PR-AUC média.
#
# Rodar: `python notebooks/04_modelagem.py` (cerca de 2 minutos).
# Pré-requisito: `python -m src.features.build_grid`.

# %%
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

from src.models.modelos import baseline, regressao_logistica  # noqa: E402
from src.models.preparo import FEATURES  # noqa: E402
from src.models.validacao import comparar, domicilio, particoes, validar_cv  # noqa: E402

pd.set_option("display.width", 200)
PROC = ROOT / "data" / "processed"
GRADE_C = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]

# %% [markdown]
# ## 1. Treino e partições

# %%
grid = pd.read_parquet(PROC / "grid.parquet")
treino = grid[grid["split"] == "treino"].reset_index(drop=True)
del grid
X, y, peso = treino[FEATURES], treino["y"].to_numpy(), treino["peso_semana"].to_numpy()
FOLDS = particoes(y, domicilio(treino))

pos_fold = [int(y[va].sum()) for _, va in FOLDS]
prevalencia = sum((y[va] * peso[va]).sum() / peso[va].sum() for _, va in FOLDS) / len(FOLDS)
print(f"Treino: {len(treino):,} linhas, {y.sum()} positivos, "
      f"{domicilio(treino).nunique():,} domicílios")
print(f"{len(FOLDS)} folds; positivos por fold: {min(pos_fold)} a {max(pos_fold)}")
print(f"Prevalência ponderada média nos folds: {prevalencia:.4f} "
      "(a PR-AUC de um score sem sinal)")

# %% [markdown]
# ## 2. Baseline: taxa ponderada por dia da semana × período
# A taxa de cada uma das 28 células no treino do fold vira o score.

# %%
base = validar_cv(baseline(), X, y, peso, FOLDS, peso_ajuste=peso)
print(f"Baseline: {base.resumo()} | {base.prauc.mean() / prevalencia:.1f} vezes a prevalência")

# %% [markdown]
# ## 3. Regressão Logística: força da regularização
# One-hot de período, dia da semana, período × dia, região, faixa etária e sexo;
# anos de estudo, moradores e log da renda per capita padronizados; urbano como 0/1.
# Ajuste sem peso. Fica o C com a maior PR-AUC média.

# %%
lr = {}
for C in GRADE_C:
    inicio = time.time()
    lr[C] = validar_cv(regressao_logistica(C), X, y, peso, FOLDS)
    print(f"  C = {C:<6} {lr[C].resumo()} ({time.time() - inicio:.0f} s)", flush=True)

C_LR = max(GRADE_C, key=lambda c: lr[c].prauc.mean())
if C_LR in (GRADE_C[0], GRADE_C[-1]):
    print("ATENÇÃO: o melhor C está na borda da grade; amplie a grade.")
grade = pd.DataFrame({
    "C": GRADE_C,
    "PR-AUC média": [lr[c].prauc.mean() for c in GRADE_C],
    "dp entre folds": [lr[c].prauc.std(ddof=1) for c in GRADE_C],
    "captura top 10%": [lr[c].captura10.mean() for c in GRADE_C],
})
print()
print(grade.round(4).to_string(index=False))
print(f"\nC escolhido: {C_LR}")

# %% [markdown]
# ## 4. Regressão Logística contra o baseline
# A Regressão Logística é a opção mais complexa: só vence com ganho significativo.

# %%
melhor = lr[C_LR]
c = comparar(base.prauc, melhor.prauc)
veredito = "a Regressão Logística vence" if c["passa"] else "empate, fica o baseline"
print(f"PR-AUC: baseline {base.prauc.mean():.4f}, Regressão Logística {melhor.prauc.mean():.4f}; "
      f"ganho {c['ganho']:+.4f} (t = {c['t']:.2f}, p = {c['p']:.2g}) → {veredito}")
print(f"Captura no top 10%: baseline {base.captura10.mean():.1%}, "
      f"Regressão Logística {melhor.captura10.mean():.1%}")
print(f"Regressão Logística à frente em {(melhor.prauc > base.prauc).sum()} "
      f"de {len(FOLDS)} folds")
