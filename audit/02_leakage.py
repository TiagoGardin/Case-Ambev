"""Fase 2 – Data leakage.

Rodar da raiz: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python audit/02_leakage.py
Tudo dentro do treino (jul/2017–fev/2018): o teste não é usado para medir nenhum modelo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from _comum import PROC, Relatorio, carregar_grid, taxa
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from src.features.build_grid import periodo_do_dia
from src.ingestion.codigos_alimentos import cadastro_produtos, codigos_pizza, codigos_refrigerante
from src.ingestion.pof import CHAVE_DOM, CHAVE_PESSOA, CHAVE_UC
from src.models.calibracao import aplicar_sigmoide, ajustar_sigmoide
from src.models.metricas import captura_topo, prauc
from src.models.modelos import baseline, construir, lightgbm_raso, regressao_logistica
from src.models.preparo import FEATURES
from src.models.treinar import CONFIG, config_final
from src.models.validacao import domicilio, particoes, validar_cv

r = Relatorio("02_leakage")
r("# Fase 2 – Leakage (só treino)")
g = carregar_grid()
treino = g[g["split"] == "treino"].reset_index(drop=True)
X, y, w = treino[FEATURES], treino["y"].to_numpy(), treino["peso_semana"].to_numpy()
dom = domicilio(treino)
FOLDS = particoes(y, dom)
FINAL = construir(CONFIG)
MODELOS = {"baseline": (baseline(), w), "Regressão Logística final": (FINAL, None),
           "LightGBM (melhor da grade)": (lightgbm_raso(16, 3000), None)}


class TaxaPorFaixa(ClassifierMixin, BaseEstimator):
    """Score = taxa ponderada de y na categoria (ou no decil, se numérica), no treino do fold."""

    def __init__(self, coluna: str = "periodo", bins: int = 10):
        self.coluna, self.bins = coluna, bins

    def _codigo(self, v: pd.Series) -> np.ndarray:
        if self.cortes_ is None:
            return v.astype(str).to_numpy()
        return np.searchsorted(self.cortes_, v.to_numpy(dtype=float), side="right").astype(str)

    def fit(self, X, y, sample_weight=None):
        v = X[self.coluna]
        numerica = pd.api.types.is_numeric_dtype(v) and v.nunique() > 30
        self.cortes_ = (np.unique(np.quantile(v, np.linspace(0, 1, self.bins + 1)[1:-1]))
                        if numerica else None)
        sw = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        d = pd.DataFrame({"c": self._codigo(v), "yw": np.asarray(y) * sw, "w": sw})
        s = d.groupby("c")[["yw", "w"]].sum()
        self.taxa_ = (s["yw"] / s["w"]).to_dict()
        self.geral_ = float(d["yw"].sum() / d["w"].sum())
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = pd.Series(self._codigo(X[self.coluna])).map(self.taxa_).fillna(self.geral_).to_numpy()
        return np.column_stack([1 - p, p])


# ---------------------------------------------------------------------------
r.secao("2.1 Target leakage: força de cada coluna sozinha (taxa por categoria/decil, 50 folds)")
linhas = []
for col in FEATURES + ["dia_atipico", "classe_abep", "tipo_area"]:
    Xc = treino[[col]]
    res = validar_cv(Pipeline([("modelo", TaxaPorFaixa(col))]), Xc, y, w, FOLDS, peso_ajuste=w)
    linhas.append({"coluna": col, "papel": "feature" if col in FEATURES else "fora do modelo",
                   "PR-AUC sozinha": res.prauc.mean(), "captura top 10%": res.captura10.mean()})
uni = pd.DataFrame(linhas).sort_values("PR-AUC sozinha", ascending=False)
r.tabela(uni.round(4), index=False)
r("Referência: prevalência 0,0030; baseline dia × período 0,0160; modelo final 0,0371 (README).")
atip = treino.groupby("dia_atipico").apply(
    lambda d: pd.Series({"linhas": len(d), "positivos": int(d["y"].sum()),
                         "taxa ponderada": taxa(d["y"], d["peso_semana"])}), include_groups=False)
r("dia_atipico (excluída por leakage no Data Card): taxa da ocasião por valor, treino:")
r.tabela(atip)
r("Colunas marcadoras do alvo na grade (pizza, refri, guarana, y_guarana, local_pizza) e "
  "dia_atipico não entram em FEATURES (src/models/preparo.py:31); preparar() devolve só "
  "COLUNAS_PREPARADAS, derivadas das 9 features.")

# ---------------------------------------------------------------------------
r.secao("2.2 Grupos: partições por UPA (vizinhança) em vez de domicílio")
FOLDS_UPA = particoes(y, treino["COD_UPA"])
linhas = []
for nome, (modelo, peso_aj) in MODELOS.items():
    a = validar_cv(modelo, X, y, w, FOLDS, peso_ajuste=peso_aj)
    b = validar_cv(modelo, X, y, w, FOLDS_UPA, peso_ajuste=peso_aj)
    linhas.append({"modelo": nome, "PR-AUC (domicílio)": a.prauc.mean(),
                   "PR-AUC (UPA)": b.prauc.mean(), "diferença": b.prauc.mean() - a.prauc.mean(),
                   "captura (domicílio)": a.captura10.mean(), "captura (UPA)": b.captura10.mean()})
r.tabela(pd.DataFrame(linhas).round(4), index=False)

# ---------------------------------------------------------------------------
r.secao("2.3 Temporal: origem móvel dentro do treino (treina no passado, valida no mês seguinte)")
mes_dom = treino.groupby(CHAVE_DOM)["ordem_mes"].transform("min").to_numpy()
ORIGENS = [("jul–set → out–nov", 2, [3, 4]), ("jul–nov → dez–jan", 4, [5, 6]),
           ("jul–jan → fev", 6, [7])]
CANDIDATOS = {"baseline": (baseline(), True), "Regressão Logística final": (FINAL, False),
              "Regressão Logística dia da semana (C 0,03)": (regressao_logistica(0.03), False),
              "LightGBM (melhor da grade)": (lightgbm_raso(16, 3000), False)}
rng = np.random.default_rng(0)
upa = treino["COD_UPA"].to_numpy()
linhas = []
for rotulo, ate, val in ORIGENS:
    tr, va = np.where(mes_dom <= ate)[0], np.where(np.isin(mes_dom, val))[0]
    prev = {}
    for nome, (modelo, com_peso) in CANDIDATOS.items():
        extra = {"modelo__sample_weight": w[tr] / w[tr].mean()} if com_peso else {}
        m = clone(modelo).fit(X.iloc[tr], y[tr], **extra)
        prev[nome] = m.predict_proba(X.iloc[va])[:, 1]
        linhas.append({"origem": rotulo, "positivos na validação": int(y[va].sum()),
                       "modelo": nome, "PR-AUC": prauc(y[va], prev[nome], w[va]),
                       "captura top 10%": captura_topo(y[va], prev[nome], w[va], 0.10)})
    # bootstrap pareado por UPA da diferença final − baseline
    upas_va = np.unique(upa[va])
    idx_por_upa = {u: np.where(upa[va] == u)[0] for u in upas_va}
    difs = []
    for _ in range(300):
        amostra = rng.choice(upas_va, size=len(upas_va), replace=True)
        ii = np.concatenate([idx_por_upa[u] for u in amostra])
        if y[va][ii].sum() == 0:
            continue
        difs.append(prauc(y[va][ii], prev["Regressão Logística final"][ii], w[va][ii])
                    - prauc(y[va][ii], prev["baseline"][ii], w[va][ii]))
    lo, hi = np.percentile(difs, [2.5, 97.5])
    linhas.append({"origem": rotulo, "positivos na validação": int(y[va].sum()),
                   "modelo": "final − baseline (IC 95%, bootstrap de UPAs)",
                   "PR-AUC": f"[{lo:+.4f}; {hi:+.4f}]", "captura top 10%": ""})
tempo = pd.DataFrame(linhas)
r.tabela(tempo.assign(**{"PR-AUC": tempo["PR-AUC"].map(
    lambda v: f"{v:.4f}" if isinstance(v, float) else v)}), index=False)
r("Comparar com a validação cruzada aleatória agrupada (README): baseline 0,0160, final 0,0371, "
  "Regressão Logística com dia da semana 0,0386, LightGBM 0,0284.")

desvio = treino.drop_duplicates(CHAVE_UC)[CHAVE_UC + ["ordem_mes", "fonte_mes"]]
desvio["mediana_upa"] = desvio.groupby("COD_UPA")["ordem_mes"].transform("median")
desvio["dist"] = (desvio["ordem_mes"] - desvio["mediana_upa"]).abs()
r("UCs do treino com mês a mais de 3 meses da mediana da UPA, por origem do mês "
  "(candidatas a mês estimado errado):")
r.tabela(desvio.assign(longe=desvio["dist"] > 3).groupby("fonte_mes")["longe"]
         .agg(["sum", "size", "mean"]).rename(columns={"sum": "longe", "size": "UCs",
                                                       "mean": "fração"}).round(4))
teste_ucs = g[g["split"] == "teste"].drop_duplicates(CHAVE_UC)[CHAVE_UC + ["ordem_mes"]]
todas = pd.concat([desvio[CHAVE_UC + ["ordem_mes"]], teste_ucs])
todas["mediana_upa"] = todas.groupby("COD_UPA")["ordem_mes"].transform("median")
cruza = ((todas["ordem_mes"] <= 7) != (todas["mediana_upa"] <= 7)) & \
        ((todas["ordem_mes"] - todas["mediana_upa"]).abs() > 3)
r(f"UCs em lado do corte diferente do da mediana da sua UPA e a mais de 3 meses dela: "
  f"{int(cruza.sum())} de {len(todas):,} (limite superior grosseiro de UCs no split errado)")

# ---------------------------------------------------------------------------
r.secao("2.4 Calibração com validação cruzada aninhada (valida também o C × 4/5 fora da amostra)")
dom_arr = dom.to_numpy()


def aninhado(i: int, tr: np.ndarray, va: np.ndarray) -> tuple[np.ndarray, dict]:
    interno = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1000 + i)
    oof = np.empty(len(tr))
    for itr, iva in interno.split(np.zeros(len(tr)), y[tr], dom_arr[tr]):
        m = clone(FINAL).fit(X.iloc[tr[itr]], y[tr[itr]])
        oof[iva] = m.predict_proba(X.iloc[tr[iva]])[:, 1]
    a, b = ajustar_sigmoide(oof, y[tr], w[tr])
    bruto = construir(CONFIG).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
    ajustado = (construir(config_final(CONFIG)).fit(X.iloc[tr], y[tr])
                .predict_proba(X.iloc[va])[:, 1])
    return va, {"sem sigmoide": bruto, "sigmoide + C da validação": aplicar_sigmoide(bruto, a, b),
                "sigmoide + C × 4/5 (como no modelo final)": aplicar_sigmoide(ajustado, a, b)}


saidas = Parallel(n_jobs=-1)(delayed(aninhado)(i, tr, va) for i, (tr, va) in enumerate(FOLDS))
linhas = []
for nome in saidas[0][1]:
    ll = [log_loss(y[va], s[nome], sample_weight=w[va], labels=[0, 1]) for va, s in saidas]
    prev = np.concatenate([s[nome] for _, s in saidas])
    idx = np.concatenate([va for va, _ in saidas])
    linhas.append({"versão": nome, "log loss ponderado (média dos folds)": np.mean(ll),
                   "média prevista": np.average(prev, weights=w[idx]),
                   "taxa observada": np.average(y[idx], weights=w[idx])})
r.tabela(pd.DataFrame(linhas).round(5), index=False)
r("Comparar com o notebook 04 (sigmoide de fora do fold, sem aninhar): log loss 0,01796 sem "
  "sigmoide e 0,01667 com; média prevista 0,304% contra 0,302% observada.")

# ---------------------------------------------------------------------------
r.secao("2.5 Sensibilidade da definição do alvo: bebidas no limite (água tônica e H2O)")
cad = cadastro_produtos().set_index("codigo")["descricao"]
limite = {c for c in codigos_refrigerante()
          if any(t in str(cad.get(c, "")) for t in ("AGUA TONICA", "H2O", "QUININO"))}
consumo = pd.read_parquet(PROC / "pof_consumo.parquet", columns=CHAVE_PESSOA + ["QUADRO", "V9001",
                                                                                 "hora"])
consumo["periodo"] = periodo_do_dia(consumo["hora"])
k = CHAVE_PESSOA + ["QUADRO", "periodo"]
cods_pz, cods_rf = codigos_pizza(), codigos_refrigerante()
oc = consumo.assign(pz=consumo["V9001"].isin(cods_pz), rf=consumo["V9001"].isin(cods_rf),
                    rf_sem=consumo["V9001"].isin(cods_rf - limite)).groupby(k)[
    ["pz", "rf", "rf_sem"]].max()
oc = oc.reset_index().merge(g[k + ["split"]].astype({"periodo": str}), on=k, how="inner")
oc_tr = oc[oc["split"] == "treino"]
r(f"Códigos no limite: {sorted(limite)} ({', '.join(cad.get(c, '') for c in sorted(limite))})")
r(f"Positivos do treino: {int((oc_tr['pz'] & oc_tr['rf']).sum())} com a definição atual; "
  f"{int((oc_tr['pz'] & oc_tr['rf_sem']).sum())} sem as bebidas no limite.")

r.gravar()
