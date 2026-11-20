# %% [markdown]
# # 04 – Seleção do modelo com validação cruzada (só treino)
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
# Roteiro: baseline e Regressão Logística (seções 2 a 4); alternativas do Data Card,
# peso no ajuste e LightGBM (5 a 9); calibração e cortes de `alta` e `media` (10 e 11).
# O modelo final é treinado e gravado por `src/models/treinar.py`, e a seção 12 confere
# que a configuração dele é a escolhida aqui.
#
# Rodar: `python notebooks/04_modelagem.py` (cerca de 9 minutos).
# Pré-requisito: `python -m src.features.build_grid`.

# %%
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

from src.features.build_grid import DIAS  # noqa: E402
from src.models.artefato import CLASSES, classificar  # noqa: E402
from src.models.calibracao import (  # noqa: E402
    ajustar_sigmoide,
    aplicar_sigmoide,
    calibrar_fora_do_fold,
    confiabilidade,
    log_loss_por_fold,
)
from src.models.metricas import percentil_ponderado  # noqa: E402
from src.models.modelos import (  # noqa: E402
    baseline,
    colunas_lr,
    lightgbm_raso,
    regressao_logistica,
)
from src.models.preparo import FEATURES  # noqa: E402
from src.models.treinar import CONFIG  # noqa: E402
from src.models.validacao import comparar, domicilio, particoes, validar_cv  # noqa: E402

pd.set_option("display.width", 200)
PROC = ROOT / "data" / "processed"
GRADE_C = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
GRADE_LGB = [(folhas, minimo) for folhas in (4, 8, 16) for minimo in (300, 3000)]

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
def grade_lr(colunas: dict, rotulo: str, peso_ajuste=None):
    """Valida a Regressão Logística em cada C da grade; devolve (melhor C, resultado, tabela)."""
    inicio = time.time()
    res = {C: validar_cv(regressao_logistica(C, **colunas), X, y, peso, FOLDS, peso_ajuste)
           for C in GRADE_C}
    C = max(GRADE_C, key=lambda c: res[c].prauc.mean())
    tabela = pd.DataFrame({
        "C": GRADE_C,
        "PR-AUC média": [res[c].prauc.mean() for c in GRADE_C],
        "dp entre folds": [res[c].prauc.std(ddof=1) for c in GRADE_C],
        "captura top 10%": [res[c].captura10.mean() for c in GRADE_C],
    })
    borda = " (ATENÇÃO: na borda da grade)" if C in (GRADE_C[0], GRADE_C[-1]) else ""
    print(f"{rotulo}: melhor C = {C}{borda} | {res[C].resumo()} "
          f"({time.time() - inicio:.0f} s)", flush=True)
    return C, res[C], tabela


C_LR, lr_ref, tabela = grade_lr(colunas_lr(), "Regressão Logística")
print()
print(tabela.round(4).to_string(index=False))

# %% [markdown]
# ## 4. Regressão Logística contra o baseline
# A Regressão Logística é a opção mais complexa: só vence com ganho significativo.

# %%
c = comparar(base.prauc, lr_ref.prauc)
veredito = "a Regressão Logística vence" if c["passa"] else "empate, fica o baseline"
print(f"PR-AUC: baseline {base.prauc.mean():.4f}, Regressão Logística {lr_ref.prauc.mean():.4f}; "
      f"ganho {c['ganho']:+.4f} (t = {c['t']:.2f}, p = {c['p']:.2g}) → {veredito}")
print(f"Captura no top 10%: baseline {base.captura10.mean():.1%}, "
      f"Regressão Logística {lr_ref.captura10.mean():.1%}")
print(f"Regressão Logística à frente em {(lr_ref.prauc > base.prauc).sum()} "
      f"de {len(FOLDS)} folds")

# %% [markdown]
# ## 5. Regras das comparações seguintes (fixadas antes de rodar)
# Cada passo parte do vencedor do anterior. Em cada um, a opção mais simples é a
# referência, e a mais complexa só vence com ganho de PR-AUC significativo:
#
# 6. **Dia:** a mais simples é fim de semana (sexta a domingo × segunda a quinta, com 8
#    células período × fim de semana); a mais complexa, dia da semana (7 dias, 28 células).
# 7. **Renda:** a mais simples é o log da renda per capita (1 coluna); a mais complexa, as
#    5 faixas de salário mínimo.
# 8. **Peso no ajuste:** a mais simples é sem peso; a mais complexa, com `peso_semana`
#    (média 1 em cada fold).
# 9. **Algoritmo:** a mais simples é a Regressão Logística; a mais complexa, o LightGBM raso.
#
# - Nas alternativas da Regressão Logística, o C é reescolhido na mesma grade.
# - O LightGBM usa as 9 features como estão (UF e idade, e as árvores acham as
#   interações), 300 árvores, taxa 0,03, e a decisão de peso do passo 8. Grade de 6
#   configurações: 4, 8 ou 16 folhas × 300 ou 3.000 linhas no mínimo por folha.
# - **Calibração (seção 10):** sigmoide ponderada sobre o logit das previsões
#   out-of-fold. Ela entra se reduzir o log loss ponderado com significância, no mesmo
#   teste t corrigido. Para medir isso, a sigmoide de cada fold é ajustada nas outras 4
#   partes da mesma repetição. Ela não muda a ordem do score, então PR-AUC e captura
#   ficam iguais.


# %%
def decidir(simples, complexa, nome_simples: str, nome_complexa: str) -> bool:
    """True se a opção mais complexa vence (ganho de PR-AUC significativo)."""
    c = comparar(simples.prauc, complexa.prauc)
    fica = nome_complexa if c["passa"] else nome_simples
    print(f"{nome_complexa} − {nome_simples}: PR-AUC {c['ganho']:+.4f} "
          f"(t = {c['t']:.2f}, p = {c['p']:.2g}) → fica {fica}")
    return c["passa"]


# %% [markdown]
# ## 6. Dia da semana ou fim de semana

# %%
C_FDS, lr_fds, _ = grade_lr(colunas_lr("fim_de_semana"), "Com fim de semana")
if decidir(lr_fds, lr_ref, "fim de semana", "dia da semana"):
    DIA, C_ATUAL, atual = "dia_semana", C_LR, lr_ref
else:
    DIA, C_ATUAL, atual = "fim_de_semana", C_FDS, lr_fds

# %% [markdown]
# ## 7. Renda contínua ou em faixas

# %%
C_FX, lr_faixas, _ = grade_lr(colunas_lr(DIA, "faixas"), "Com renda em faixas")
if decidir(atual, lr_faixas, "renda contínua", "renda em faixas"):
    RENDA, C_ATUAL, atual = "faixas", C_FX, lr_faixas
else:
    RENDA = "continua"

# %% [markdown]
# ## 8. Peso no ajuste

# %%
C_PESO, lr_peso, _ = grade_lr(colunas_lr(DIA, RENDA), "Com peso_semana no ajuste",
                              peso_ajuste=peso)
PESO_NO_AJUSTE = decidir(atual, lr_peso, "sem peso", "com peso_semana")
if PESO_NO_AJUSTE:
    C_ATUAL, atual = C_PESO, lr_peso

# %% [markdown]
# ## 9. LightGBM raso contra a Regressão Logística

# %%
lgb = {}
for folhas, minimo in GRADE_LGB:
    inicio = time.time()
    lgb[folhas, minimo] = validar_cv(lightgbm_raso(folhas, minimo), X, y, peso, FOLDS,
                                     peso if PESO_NO_AJUSTE else None)
    print(f"  {folhas:>2} folhas, mínimo de {minimo:>5} linhas: {lgb[folhas, minimo].resumo()} "
          f"({time.time() - inicio:.0f} s)", flush=True)
CFG_LGB = max(GRADE_LGB, key=lambda cfg: lgb[cfg].prauc.mean())
print(f"Melhor LightGBM: {CFG_LGB[0]} folhas, mínimo de {CFG_LGB[1]} linhas por folha")

if decidir(atual, lgb[CFG_LGB], "Regressão Logística", "LightGBM"):
    config = {"modelo": "lightgbm", "num_leaves": CFG_LGB[0], "min_child_samples": CFG_LGB[1],
              "peso_no_ajuste": PESO_NO_AJUSTE}
    final = lgb[CFG_LGB]
else:
    config = {"modelo": "regressao_logistica", "C": C_ATUAL, "dia": DIA, "renda": RENDA,
              "peso_no_ajuste": PESO_NO_AJUSTE}
    final = atual
print(f"\nModelo escolhido: {config}")

# %% [markdown]
# ## 10. Calibração
# A taxa prevista deve bater com a observada, as duas ponderadas por `peso_semana`: no
# geral, ao longo do ranking do score e por dia da semana. As previsões out-of-fold das 10
# repetições entram empilhadas. "Depois" usa a sigmoide de fora do fold.

# %%
REP = final.oof.shape[1]
score = final.oof.ravel(order="F")
y_rep, peso_rep = np.tile(y, REP), np.tile(peso, REP)
calibrado_cv = calibrar_fora_do_fold(final.oof, y, peso, FOLDS)
score_cv = calibrado_cv.ravel(order="F")

# Faixas do ranking pelo peso: o 1% de maior score, do 1% ao 5%, ...
QUANTIS = [0.50, 0.70, 0.90, 0.95, 0.99]
ROTULOS = ["metade de baixo", "top 30–50%", "top 10–30%", "top 5–10%", "top 1–5%", "top 1%"]
limites = [percentil_ponderado(score, peso_rep, q) for q in QUANTIS]
faixa_rank = pd.Categorical(np.array(ROTULOS)[np.searchsorted(limites, score, side="right")],
                            ROTULOS[::-1])
dia_rep = pd.Categorical(np.tile(treino["dia_semana"].astype(str).to_numpy(), REP), DIAS)


def tabela_calibracao(grupo) -> pd.DataFrame:
    antes = confiabilidade(score, y_rep, peso_rep, grupo, copias=REP)
    depois = confiabilidade(score_cv, y_rep, peso_rep, grupo, copias=REP)
    t = pd.DataFrame({"ocasiões": antes["ocasioes"], "observada": antes["observada"],
                      "prevista antes": antes["prevista"], "prevista depois": depois["prevista"]})
    for col in ["observada", "prevista antes", "prevista depois"]:
        t[col] = t[col].map(lambda v: f"{v:.3%}")
    return t


print(tabela_calibracao(np.repeat("todas as linhas", len(score))).to_string())
print("\nAo longo do ranking do score:")
print(tabela_calibracao(faixa_rank).to_string())
print("\nPor dia da semana:")
print(tabela_calibracao(dia_rep).to_string())

ll_antes = log_loss_por_fold(final.oof, y, peso, FOLDS)
ll_depois = log_loss_por_fold(calibrado_cv, y, peso, FOLDS)
c = comparar(-ll_antes, -ll_depois)
CALIBRAR = c["passa"]
print(f"\nLog loss ponderado: {ll_antes.mean():.5f} sem a sigmoide, {ll_depois.mean():.5f} com; "
      f"redução {c['ganho']:+.5f} (t = {c['t']:.2f}, p = {c['p']:.2g}) → "
      f"{'a sigmoide entra' if CALIBRAR else 'a sigmoide não entra'}")
A, B = ajustar_sigmoide(score, y_rep, peso_rep) if CALIBRAR else (1.0, 0.0)
print(f"Sigmoide final, com todas as previsões out-of-fold: a = {A:.3f}, b = {B:+.3f}")

# %% [markdown]
# ## 11. Cortes de `alta` e `media`
# Percentis 90 e 70 do score calibrado, ponderados por `peso_semana`, nas previsões
# out-of-fold (README, Metodologia). Eles ficam fixos daí em diante.

# %%
score_cal = aplicar_sigmoide(score, A, B)
CORTES = {"alta": percentil_ponderado(score_cal, peso_rep, 0.90),
          "media": percentil_ponderado(score_cal, peso_rep, 0.70)}
classe = pd.Categorical(classificar(score_cal, CORTES), list(CLASSES)[::-1])
soma = (pd.DataFrame({"classe": classe, "w": peso_rep, "yw": y_rep * peso_rep})
        .groupby("classe", observed=True)[["w", "yw"]].sum())
taxa_media = (y * peso).sum() / peso.sum()
por_classe = pd.DataFrame({
    "fatia do peso": (soma["w"] / soma["w"].sum()).map(lambda v: f"{v:.1%}"),
    "captura das ocasiões": (soma["yw"] / soma["yw"].sum()).map(lambda v: f"{v:.1%}"),
    "taxa observada": (soma["yw"] / soma["w"]).map(lambda v: f"{v:.2%}"),
    "vezes a média": (soma["yw"] / soma["w"] / taxa_media).round(1),
})
print(f"Cortes: alta com score ≥ {CORTES['alta']:.3%}, media com score ≥ {CORTES['media']:.3%}")
print(por_classe.to_string())

# %% [markdown]
# ## 12. Resumo e conferência com o modelo final
# `src/models/treinar.py` treina e grava o modelo com a configuração abaixo.

# %%
config["calibracao"] = bool(CALIBRAR)
c = comparar(base.prauc, final.prauc)
print(f"Modelo final na validação cruzada: {final.resumo()} | "
      f"{final.prauc.mean() / prevalencia:.1f} vezes a prevalência")
print(f"Contra o baseline: ganho de PR-AUC {c['ganho']:+.4f} (t = {c['t']:.2f}, "
      f"p = {c['p']:.2g}); à frente em {(final.prauc > base.prauc).sum()} de {len(FOLDS)} folds")
assert config == CONFIG, f"atualize CONFIG em src/models/treinar.py para {config}"
print(f"CONFIG de src/models/treinar.py confere: {CONFIG}")
