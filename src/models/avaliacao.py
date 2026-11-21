"""Avaliação no teste (README, Metodologia: "Avaliação no teste" e adendo de 20/11).

Funções sobre arrays (y, score, peso, UPA) usadas por notebooks/06_teste.py:
- métricas do protocolo, sempre calculadas igual para o modelo e o baseline;
- bootstrap pareado de UPAs;
- segmentos, canal (local da pizza), meses, calibração e perfil dos erros.

Também traz a receita de src.models.treinar (previsões out-of-fold, sigmoide,
cortes e ajuste final com C × (k − 1)/k), para treinar a comparação exploratória
com o dia da semana do mesmo jeito que o modelo congelado.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.features.build_grid import FONTES_MES
from src.ingestion.pof import CHAVE_DOM
from src.models.calibracao import ModeloCalibrado, ajustar_sigmoide, aplicar_sigmoide
from src.models.metricas import captura_topo, percentil_ponderado, prauc
from src.models.modelos import baseline, construir
from src.models.preparo import FEATURES
from src.models.treinar import config_final
from src.models.validacao import domicilio, normalizar, particoes, validar_cv

MINIMO_POSITIVOS = 30      # segmentos com menos positivos no teste são só descritos
REPETICOES_BOOTSTRAP = 1000
FRACOES = (0.05, 0.10, 0.20)


# --------------------------------------------------------------------------
# Modelos de comparação
# --------------------------------------------------------------------------
def receita(config: dict, treino: pd.DataFrame) -> tuple[ModeloCalibrado, dict]:
    """O procedimento de src.models.treinar.main para uma configuração, sem gravar nada.

    Previsões out-of-fold nas mesmas 50 partições, sigmoide ponderada sobre elas,
    cortes de `alta` e `media` (P90 e P70 ponderados) e ajuste final em todo o
    `treino` com C × (k − 1)/k.
    """
    X, y, peso = treino[FEATURES], treino["y"].to_numpy(), treino["peso_semana"].to_numpy()
    folds = particoes(y, domicilio(treino))
    peso_ajuste = peso if config["peso_no_ajuste"] else None
    cv = validar_cv(construir(config), X, y, peso, folds, peso_ajuste=peso_ajuste)
    repeticoes = cv.oof.shape[1]
    score = cv.oof.ravel(order="F")
    y_rep, peso_rep = np.tile(y, repeticoes), np.tile(peso, repeticoes)
    a, b = ajustar_sigmoide(score, y_rep, peso_rep) if config["calibracao"] else (1.0, 0.0)
    calibrado = aplicar_sigmoide(score, a, b)
    cortes = {"alta": percentil_ponderado(calibrado, peso_rep, 0.90),
              "media": percentil_ponderado(calibrado, peso_rep, 0.70)}
    extra = {} if peso_ajuste is None else {"modelo__sample_weight": normalizar(peso)}
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline = construir(config_final(config)).fit(X, y, **extra)
    return ModeloCalibrado(pipeline, a, b), cortes


def ajustar_baseline(treino: pd.DataFrame) -> tuple[object, float]:
    """Baseline dia × período (taxas ponderadas) e o corte de `alta` dele.

    O corte é o P90 ponderado do score do baseline no treino, o mesmo percentil do
    modelo. Como o baseline só tem 28 valores, a fração do peso acima do corte não
    sai exatamente 10%.
    """
    X, y, peso = treino[FEATURES], treino["y"].to_numpy(), treino["peso_semana"].to_numpy()
    modelo = baseline().fit(X, y, modelo__sample_weight=normalizar(peso))
    return modelo, percentil_ponderado(modelo.predict_proba(X)[:, 1], peso, 0.90)


# --------------------------------------------------------------------------
# Métricas
# --------------------------------------------------------------------------
def metricas(y, score, peso, corte_alta: float) -> dict[str, float]:
    """Métricas do protocolo, ponderadas por `peso`, para um score."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(score, dtype=float)
    w = np.asarray(peso, dtype=float)
    alta = s >= corte_alta
    vp, fp = w[(y == 1) & alta].sum(), w[(y == 0) & alta].sum()
    fn = w[(y == 1) & ~alta].sum()
    precisao = vp / (vp + fp) if vp + fp > 0 else np.nan
    recall = vp / (vp + fn)
    f1 = 2 * precisao * recall / (precisao + recall) if precisao + recall > 0 else 0.0
    saida = {"PR-AUC": prauc(y, s, w)}
    for fracao in FRACOES:
        saida[f"captura no top {fracao:.0%}"] = captura_topo(y, s, w, fracao)
    saida.update({
        "ROC-AUC": float(roc_auc_score(y, s, sample_weight=w)),
        "fração do peso em alta": float(w[alta].sum() / w.sum()),
        "precisão em alta": float(precisao),
        "recall em alta": float(recall),
        "F1 em alta": float(f1),
        "Brier": float(brier_score_loss(y, s, sample_weight=w)),
        "taxa prevista": float(np.average(s, weights=w)),
        "taxa observada": float(np.average(y, weights=w)),
    })
    return saida


def matriz_confusao(y, score, peso, corte_alta: float) -> pd.DataFrame:
    """Previsto (alta ou não) × observado (com ou sem ocasião): linhas e % do peso."""
    y = np.asarray(y, dtype=int)
    alta = np.asarray(score, dtype=float) >= corte_alta
    w = np.asarray(peso, dtype=float)
    linhas = []
    for previsto, m_prev in (("alta", alta), ("média ou baixa", ~alta)):
        for observado, m_obs in (("com ocasião", y == 1), ("sem ocasião", y == 0)):
            m = m_prev & m_obs
            linhas.append({"previsto": previsto, "observado": observado,
                           "linhas": int(m.sum()), "% do peso": float(w[m].sum() / w.sum())})
    return pd.DataFrame(linhas)


def _bloco_bootstrap(y, scores, peso, idx_upa, contagens) -> list[dict]:
    saida = []
    for contagem in contagens:
        w = peso * contagem[idx_upa]
        m = w > 0
        linha = {}
        for nome, s in scores.items():
            linha[f"PR-AUC {nome}"] = prauc(y[m], s[m], w[m])
            linha[f"captura {nome}"] = captura_topo(y[m], s[m], w[m], 0.10)
        saida.append(linha)
    return saida


def bootstrap_upa(y, scores: dict, peso, upa, repeticoes: int = REPETICOES_BOOTSTRAP,
                  semente: int = 0, n_jobs: int = -1) -> pd.DataFrame:
    """PR-AUC e captura no top 10% de cada score em reamostragens de UPAs com reposição.

    Sortear uma UPA k vezes equivale a multiplicar o peso das linhas dela por k. As
    mesmas reamostragens valem para todos os scores (bootstrap pareado), e as
    contagens são sorteadas antes da paralelização, então o resultado só depende da
    semente.
    """
    y = np.asarray(y, dtype=int)
    peso = np.asarray(peso, dtype=float)
    scores = {nome: np.asarray(s, dtype=float) for nome, s in scores.items()}
    unicas, idx = np.unique(np.asarray(upa), return_inverse=True)
    rng = np.random.default_rng(semente)
    contagens = np.stack([np.bincount(rng.integers(0, len(unicas), len(unicas)),
                                      minlength=len(unicas)) for _ in range(repeticoes)])
    blocos = np.array_split(contagens, min(repeticoes, cpu_count()))
    partes = Parallel(n_jobs=n_jobs)(
        delayed(_bloco_bootstrap)(y, scores, peso, idx, bloco) for bloco in blocos)
    return pd.DataFrame([linha for parte in partes for linha in parte])


def intervalo(valores) -> tuple[float, float]:
    """Intervalo de 95% pelos percentis 2,5 e 97,5."""
    lo, hi = np.percentile(np.asarray(valores, dtype=float), [2.5, 97.5])
    return float(lo), float(hi)


# --------------------------------------------------------------------------
# Recortes
# --------------------------------------------------------------------------
def niveis(coluna: pd.Series) -> list:
    """Níveis presentes, na ordem das categorias quando a coluna é categórica."""
    if isinstance(coluna.dtype, pd.CategoricalDtype):
        presentes = set(coluna.dropna().unique())
        return [c for c in coluna.cat.categories if c in presentes]
    return sorted(coluna.dropna().unique())


def por_segmento(segmento: pd.Series, y, scores: dict, cortes: dict, peso,
                 minimo: int = MINIMO_POSITIVOS) -> pd.DataFrame:
    """Métricas de cada nível com pelo menos `minimo` ocasiões; os outros só descritos.

    - PR-AUC dentro do nível (ordena os contextos do próprio nível);
    - % das ocasiões do nível em `alta`, com o corte de cada score;
    - taxa prevista pelo primeiro score de `scores` × taxa observada.
    """
    seg = segmento.to_numpy()
    y = np.asarray(y, dtype=int)
    w = np.asarray(peso, dtype=float)
    primeiro = next(iter(scores))
    linhas = []
    for nivel in niveis(segmento):
        m = seg == nivel
        yy, ww = y[m], w[m]
        linha = {"nível": nivel, "linhas": int(m.sum()), "ocasiões": int(yy.sum()),
                 "taxa observada": float((yy * ww).sum() / ww.sum())}
        if yy.sum() >= minimo:
            for nome, s in scores.items():
                linha[f"PR-AUC {nome}"] = prauc(yy, s[m], ww)
            for nome, s in scores.items():
                linha[f"% em alta {nome}"] = float(
                    (ww * yy * (s[m] >= cortes[nome])).sum() / (ww * yy).sum())
            prevista = float(np.average(scores[primeiro][m], weights=ww))
            linha["taxa prevista"] = prevista
            linha["observada ÷ prevista"] = linha["taxa observada"] / prevista
        linhas.append(linha)
    return pd.DataFrame(linhas)


def captura_por_local(local: pd.Series, y, classes: dict, peso,
                      minimo: int = MINIMO_POSITIVOS) -> pd.DataFrame:
    """Das ocasiões de cada local da pizza, a % ponderada em cada classe de cada modelo.

    `classes`: nome → array com "alta", "media" ou "baixa" por linha. O local só existe
    quando há pizza, então o resultado é a captura das ocasiões de cada local.
    """
    loc = local.to_numpy()
    y = np.asarray(y, dtype=int)
    w = np.asarray(peso, dtype=float)
    linhas = []
    for nivel in niveis(local):
        m = (loc == nivel) & (y == 1)
        linha = {"local": nivel, "ocasiões": int(m.sum())}
        if m.sum() >= minimo:
            for nome, classe in classes.items():
                linha[f"% em alta {nome}"] = float(w[m & (classe == "alta")].sum() / w[m].sum())
                if (classe == "media").any():
                    linha[f"% em alta ou média {nome}"] = float(
                        w[m & (classe != "baixa")].sum() / w[m].sum())
        linhas.append(linha)
    return pd.DataFrame(linhas)


def mes_do_domicilio(df: pd.DataFrame) -> pd.Series:
    """Mês (`ordem_mes`) que decidiu o split de cada domicílio, linha a linha.

    A mesma regra de src.features.build_grid.split_temporal: vale a UC com a fonte
    do mês mais confiável e, em empate, o mês mais cedo.
    """
    rank = df["fonte_mes"].astype(str).map({f: i for i, f in enumerate(FONTES_MES)})
    ref = (df.assign(_rank=rank)
             .dropna(subset=["ordem_mes"])
             .sort_values(["_rank", "ordem_mes"], kind="stable")
             .drop_duplicates(subset=CHAVE_DOM)[CHAVE_DOM + ["ordem_mes"]])
    meses = df[CHAVE_DOM].merge(ref, on=CHAVE_DOM, how="left")["ordem_mes"]
    return pd.Series(meses.to_numpy(), index=df.index, name="mes_split")


def nome_mes(ordem: float) -> str:
    """ordem_mes → "mar/2018" (ordem 0 = julho de 2017)."""
    ordem = int(ordem)
    nomes = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
    return f"{nomes[(ordem + 6) % 12]}/{2017 + (ordem + 6) // 12}"


def calibracao_por_faixa(score, y, peso, faixas: int = 10) -> pd.DataFrame:
    """Taxa prevista × observada em faixas de score com o mesmo peso (decis ponderados)."""
    s = np.asarray(score, dtype=float)
    y = np.asarray(y, dtype=int)
    w = np.asarray(peso, dtype=float)
    ordem = np.argsort(s, kind="stable")
    acumulado = np.cumsum(w[ordem]) / w.sum()
    faixa = np.empty(len(s), dtype=int)
    faixa[ordem] = np.minimum(np.ceil(acumulado * faixas).astype(int), faixas) - 1
    d = pd.DataFrame({"faixa": faixa + 1, "w": w, "sw": s * w, "yw": y * w, "y": y})
    g = d.groupby("faixa")[["w", "sw", "yw", "y"]].sum()
    return pd.DataFrame({"% do peso": g["w"] / w.sum(), "ocasiões": g["y"].astype(int),
                         "taxa prevista": g["sw"] / g["w"], "taxa observada": g["yw"] / g["w"],
                         "observada ÷ prevista": g["yw"] / g["sw"]})


def perfil(coluna: pd.Series, grupos: dict, peso) -> pd.DataFrame:
    """Distribuição ponderada (%) de `coluna` em cada grupo de linhas (máscaras)."""
    w = pd.Series(np.asarray(peso, dtype=float), index=coluna.index)
    valores = coluna.astype(object).where(coluna.notna(), "sem informação")
    ordem = [str(v) for v in niveis(coluna)] + (["sem informação"] if coluna.isna().any() else [])
    valores = valores.astype(str)
    saida = {}
    for nome, mascara in grupos.items():
        m = np.asarray(mascara, dtype=bool)
        saida[nome] = w[m].groupby(valores[m]).sum() / w[m].sum()
    return pd.DataFrame(saida).reindex(ordem).fillna(0.0)
