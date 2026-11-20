"""Métricas ponderadas do protocolo (README, Metodologia)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def prauc(y, score, peso) -> float:
    """PR-AUC (average precision) ponderada. Um score sem sinal tem PR-AUC igual à prevalência."""
    return float(average_precision_score(y, score, sample_weight=peso))


def captura_topo(y, score, peso, fracao: float = 0.10) -> float:
    """Fração ponderada das ocasiões que cai na `fracao` do peso com maior score.

    Um empate no corte entra pela parte que cabe, o que equivale a desempatar ao
    acaso. Isso importa para o baseline, que só tem 28 valores distintos.
    """
    peso = np.asarray(peso, dtype=float)
    d = pd.DataFrame({"score": np.asarray(score, dtype=float), "w": peso,
                      "yw": np.asarray(y, dtype=float) * peso})
    g = d.groupby("score")[["w", "yw"]].sum().sort_index(ascending=False)
    antes = g["w"].cumsum() - g["w"]
    parte = ((fracao * g["w"].sum() - antes) / g["w"]).clip(0, 1)
    return float((parte * g["yw"]).sum() / g["yw"].sum())


def percentil_ponderado(valores, peso, q: float) -> float:
    """Menor valor v tal que as linhas com valor ≤ v somam pelo menos a fração q do peso."""
    v = np.asarray(valores, dtype=float)
    ordem = np.argsort(v, kind="stable")
    acumulado = np.cumsum(np.asarray(peso, dtype=float)[ordem])
    i = int(np.searchsorted(acumulado, q * acumulado[-1], side="left"))
    return float(v[ordem][min(i, len(v) - 1)])
