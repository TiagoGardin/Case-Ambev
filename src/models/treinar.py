"""Treina o modelo final em todo o treino e o grava em models/, congelado antes do teste.

A configuração (CONFIG) foi escolhida em notebooks/04_modelagem.py com as regras do
README (Metodologia), e o notebook confere que ela bate com as decisões dele. Aqui
ela só é reproduzida:
1. previsões out-of-fold nas mesmas 50 partições da validação cruzada;
2. sigmoide ponderada sobre essas previsões (se a calibração foi escolhida) e cortes
   de `alta` e `media`: percentis 90 e 70 do score calibrado, ponderados por
   `peso_semana`;
3. ajuste em todo o treino e gravação do modelo e dos metadados (src.models.artefato).

Rodar: `python -m src.models.treinar` (cerca de 1 minuto). Pré-requisito: a grade.
"""

from __future__ import annotations

import platform
import warnings
from datetime import datetime
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning

from src.models.artefato import CLASSES, classificar, salvar
from src.models.calibracao import ModeloCalibrado, aplicar_sigmoide, ajustar_sigmoide
from src.models.metricas import percentil_ponderado
from src.models.modelos import baseline, construir
from src.models.preparo import FEATURES
from src.models.validacao import K, domicilio, normalizar, particoes, validar_cv

PROC = Path(__file__).resolve().parents[2] / "data" / "processed"
VERSAO = "1.0.0"

# Escolhida na validação cruzada do treino (notebooks/04_modelagem.py, seções 6 a 10):
# fim de semana no lugar de dia da semana e renda contínua (as opções mais simples, porque
# as mais complexas não ganharam com significância), ajuste sem peso, Regressão Logística
# (o LightGBM não ganhou) e sigmoide ponderada (reduziu o log loss com significância).
CONFIG = {
    "modelo": "regressao_logistica",
    "C": 0.003,
    "dia": "fim_de_semana",
    "renda": "continua",
    "peso_no_ajuste": False,
    "calibracao": True,
}
ALGORITMOS = {"regressao_logistica": "Regressão Logística com regularização L2",
              "lightgbm": "LightGBM raso"}
TOLERANCIA_CALIBRACAO = 0.05  # média do score calibrado no treino × taxa real


def config_final(config: dict, k: int = K) -> dict:
    """Configuração do ajuste em todo o treino.

    No scikit-learn, a Regressão Logística minimiza C × Σ perda + penalidade, então
    com mais linhas o mesmo C regulariza relativamente menos. Cada modelo da validação
    cruzada treina com (k − 1)/k das linhas, e a sigmoide foi ajustada nas previsões
    deles. Com C × (k − 1)/k, o modelo final tem a mesma regularização por linha e a
    sigmoide vale para ele. Com o C da validação, o score calibrado saía 16% acima da
    taxa do treino.
    """
    if config["modelo"] == "regressao_logistica":
        return {**config, "C": round(config["C"] * (k - 1) / k, 12)}
    return config


def carregar_treino() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    grid = pd.read_parquet(PROC / "grid.parquet")
    treino = grid[grid["split"] == "treino"].reset_index(drop=True)
    return treino, treino[FEATURES], treino["y"].to_numpy(), treino["peso_semana"].to_numpy()


def main() -> None:
    treino, X, y, peso = carregar_treino()
    folds = particoes(y, domicilio(treino))
    modelo = construir(CONFIG)
    peso_ajuste = peso if CONFIG["peso_no_ajuste"] else None

    # 1. Previsões out-of-fold (as mesmas da seleção) e o baseline, para os metadados
    cv = validar_cv(modelo, X, y, peso, folds, peso_ajuste=peso_ajuste)
    base = validar_cv(baseline(), X, y, peso, folds, peso_ajuste=peso)

    # 2. Sigmoide e cortes sobre as previsões das 10 repetições empilhadas
    repeticoes = cv.oof.shape[1]
    score = cv.oof.ravel(order="F")
    y_rep, peso_rep = np.tile(y, repeticoes), np.tile(peso, repeticoes)
    a, b = ajustar_sigmoide(score, y_rep, peso_rep) if CONFIG["calibracao"] else (1.0, 0.0)
    calibrado = aplicar_sigmoide(score, a, b)
    cortes = {"alta": percentil_ponderado(calibrado, peso_rep, 0.90),
              "media": percentil_ponderado(calibrado, peso_rep, 0.70)}
    classe = classificar(calibrado, cortes)
    fatia = {c: float(peso_rep[classe == c].sum() / peso_rep.sum()) for c in CLASSES}
    captura = {c: float((y_rep * peso_rep)[classe == c].sum() / (y_rep * peso_rep).sum())
               for c in CLASSES}

    # 3. Ajuste em todo o treino, com a regularização por linha dos modelos da validação
    ajuste = config_final(CONFIG)
    extra = {} if peso_ajuste is None else {"modelo__sample_weight": normalizar(peso)}
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline = construir(ajuste).fit(X, y, **extra)
    final = ModeloCalibrado(pipeline, a, b)

    # Conferência: no treino, a média ponderada do score calibrado bate com a taxa real
    taxa = float((y * peso).sum() / peso.sum())
    score_treino = final.predict_proba(X)[:, 1]
    media = float(np.average(score_treino, weights=peso))
    if abs(media / taxa - 1) > TOLERANCIA_CALIBRACAO:
        raise ValueError(f"score calibrado com média {media:.4%} no treino, taxa {taxa:.4%}: "
                         "o modelo final não está calibrado; nada foi gravado")
    classe_treino = classificar(score_treino, cortes)
    fatia_treino = {c: float(peso[classe_treino == c].sum() / peso.sum()) for c in CLASSES}

    metadados = {
        "model_name": "occasion-score-pizza-refrigerante",
        "model_version": VERSAO,
        "trained_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "algorithm": ALGORITMOS[CONFIG["modelo"]],
        "config": CONFIG,
        "final_fit": {
            "config": ajuste,
            "note": "Regressão Logística: C × (k − 1)/k no treino todo, a mesma regularização "
                    "por linha dos modelos da validação cruzada em que a sigmoide foi ajustada",
        },
        "features": FEATURES,
        "target": "pizza e refrigerante no mesmo período (pessoa × dia × período)",
        "training_data": {
            "source": "POF 2017-2018 (IBGE), recordatório alimentar de 24h",
            "period": "jul/2017 a fev/2018 (teste: mar a jul/2018, não usado)",
            "rows": int(len(y)), "occasions": int(y.sum()),
            "households": int(domicilio(treino).nunique()),
        },
        "reference_rate": taxa,
        "calibration": ({"method": "sigmoide ponderada sobre o logit (Platt)", "a": a, "b": b,
                         "weighted_mean_score_train": media}
                        if CONFIG["calibracao"] else None),
        "classification": {
            "cuts": cortes,
            "rule": "alta se score >= cuts.alta (P90 ponderado das previsões out-of-fold "
                    "calibradas); media se score >= cuts.media (P70); senão baixa",
            "weight_share_oof": fatia,
            "occasion_capture_oof": captura,
            "weight_share_train_final_model": fatia_treino,
        },
        "cross_validation": {
            "scheme": "StratifiedGroupKFold por domicílio, 5 folds × 10 repetições, só treino",
            "metric_weight": "peso_semana",
            "prauc_mean": float(cv.prauc.mean()), "prauc_sd": float(cv.prauc.std(ddof=1)),
            "capture_top10_mean": float(cv.captura10.mean()),
            "baseline_prauc_mean": float(base.prauc.mean()),
            "baseline_capture_top10_mean": float(base.captura10.mean()),
        },
        "libraries": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                      "lightgbm": lightgbm.__version__, "pandas": pd.__version__,
                      "numpy": np.__version__},
    }
    caminho = salvar(final, metadados)

    print(f"Modelo: {metadados['algorithm']} | {CONFIG} | C no treino todo: {ajuste.get('C')}")
    print(f"Validação cruzada: {cv.resumo()} | baseline: {base.resumo()}")
    print(f"Sigmoide: a = {a:.3f}, b = {b:+.3f}")
    print(f"Treino: taxa ponderada {taxa:.4%}, "
          f"média do score calibrado do modelo final {media:.4%}")
    print(f"Cortes: alta ≥ {cortes['alta']:.4%}, media ≥ {cortes['media']:.4%}")
    for c in reversed(CLASSES):
        print(f"  {c:<5}: {fatia[c]:.1%} do peso e {captura[c]:.1%} das ocasiões fora da amostra; "
              f"{fatia_treino[c]:.1%} do peso com o modelo final")
    print(f"Gravado: {caminho} e {caminho.with_name('model_metadata.json').name}")


if __name__ == "__main__":
    main()
