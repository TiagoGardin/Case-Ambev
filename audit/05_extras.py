"""Checagens extras: PR-AUC média por fold × PR-AUC agregada das previsões out-of-fold.

A validação cruzada reporta a média da PR-AUC de cada fold (cerca de 105 positivos cada). No
teste, a PR-AUC será calculada uma vez sobre todas as linhas (315 positivos). Average precision
não é aditiva: a média dos folds e o valor agregado podem diferir. Só treino.

Rodar da raiz: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python audit/05_extras.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from _comum import Relatorio, carregar_grid

from src.models.metricas import captura_topo, prauc
from src.models.modelos import baseline, construir, lightgbm_raso, regressao_logistica
from src.models.preparo import FEATURES
from src.models.treinar import CONFIG
from src.models.validacao import domicilio, particoes, validar_cv

r = Relatorio("05_extras")
r("# Checagens extras (só treino)")
g = carregar_grid()
treino = g[g["split"] == "treino"].reset_index(drop=True)
X, y, w = treino[FEATURES], treino["y"].to_numpy(), treino["peso_semana"].to_numpy()
FOLDS = particoes(y, domicilio(treino))

r.secao("5.1 PR-AUC média dos folds × PR-AUC agregada das previsões out-of-fold")
linhas = []
CANDIDATOS = [("baseline", baseline(), w),
              ("Regressão Logística final", construir(CONFIG), None),
              ("Regressão Logística dia da semana (C 0,03)", regressao_logistica(0.03), None),
              ("LightGBM (16 folhas, mín. 3000)", lightgbm_raso(16, 3000), None)]
for nome, modelo, peso_aj in CANDIDATOS:
    res = validar_cv(modelo, X, y, w, FOLDS, peso_ajuste=peso_aj)
    agregada = [prauc(y, res.oof[:, k], w) for k in range(res.oof.shape[1])]
    captura = [captura_topo(y, res.oof[:, k], w) for k in range(res.oof.shape[1])]
    linhas.append({"modelo": nome, "PR-AUC média dos folds": res.prauc.mean(),
                   "PR-AUC agregada (média das 10 repetições)": np.mean(agregada),
                   "mín–máx agregada": f"{min(agregada):.4f}–{max(agregada):.4f}",
                   "captura agregada": np.mean(captura)})
t = pd.DataFrame(linhas)
r.tabela(t.round(4), index=False)
base_ag = t.loc[0, "PR-AUC agregada (média das 10 repetições)"]
media = t["PR-AUC média dos folds"]
agregada_col = t["PR-AUC agregada (média das 10 repetições)"]
r(f"Razão final ÷ baseline: {media[1] / media[0]:.2f} na média dos folds; "
  f"{agregada_col[1] / base_ag:.2f} agregada.")

r.gravar()

# ---------------------------------------------------------------------------
r.secao("5.2 Por que a média dos folds é maior: amostra pequena ou mistura de escalas?")
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

res = validar_cv(construir(CONFIG), X, y, w, FOLDS)
dom = domicilio(treino).to_numpy()
linhas = []
for rep in range(3):
    s = res.oof[:, rep]
    folds_rep = FOLDS[5 * rep: 5 * rep + 5]
    media_folds = np.mean([prauc(y[va], s[va], w[va]) for _, va in folds_rep])
    sub = []
    for semente in range(20):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=500 + semente)
        sub += [prauc(y[va], s[va], w[va]) for _, va in cv.split(np.zeros(len(y)), y, dom)]
    linhas.append({"repetição": rep, "agregada (5 modelos misturados)": prauc(y, s, w),
                   "média dos folds originais (sem mistura)": media_folds,
                   "média em quintos aleatórios das previsões agregadas": np.mean(sub)})
r.tabela(pd.DataFrame(linhas).round(4), index=False)
r("Se a média em quintos aleatórios (que também mistura os 5 modelos) fica perto da média dos "
  "folds, a diferença vem do tamanho da amostra, não da mistura de escalas.")
r.gravar()
