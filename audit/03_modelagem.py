"""Fase 3 – Escolha e avaliação dos modelos (só treino).

Rodar da raiz: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python audit/03_modelagem.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from _comum import Relatorio, carregar_grid, taxa
from sklearn.metrics import average_precision_score

from src.models.artefato import CLASSES, carregar, classificar
from src.models.calibracao import aplicar_sigmoide, calibrar_fora_do_fold, confiabilidade
from src.models.metricas import captura_topo, prauc
from src.models.modelos import baseline, colunas_lr, construir, lightgbm_raso, regressao_logistica
from src.models.preparo import FEATURES
from src.models.treinar import CONFIG
from src.models.validacao import comparar, domicilio, particoes, validar_cv

r = Relatorio("03_modelagem")
r("# Fase 3 – Modelagem (só treino)")
g = carregar_grid()
treino = g[g["split"] == "treino"].reset_index(drop=True)
X, y, w = treino[FEATURES], treino["y"].to_numpy(), treino["peso_semana"].to_numpy()
FOLDS = particoes(y, domicilio(treino))
modelo, meta = carregar()
A, B = meta["calibration"]["a"], meta["calibration"]["b"]
CORTES = meta["classification"]["cuts"]

base = validar_cv(baseline(), X, y, w, FOLDS, peso_ajuste=w)
final = validar_cv(construir(CONFIG), X, y, w, FOLDS)
lgb_cv = validar_cv(lightgbm_raso(16, 3000), X, y, w, FOLDS)

# ---------------------------------------------------------------------------
r.secao("3.1 Reprodução das métricas de validação cruzada citadas no README")
r.tabela(pd.DataFrame({
    "modelo": ["baseline", "Regressão Logística final", "LightGBM (16 folhas, mín. 3000)"],
    "PR-AUC reproduzida": [base.prauc.mean(), final.prauc.mean(), lgb_cv.prauc.mean()],
    "PR-AUC no README": [0.0160, 0.0371, 0.0284],
    "captura reproduzida": [base.captura10.mean(), final.captura10.mean(), lgb_cv.captura10.mean()],
    "captura no README": [0.542, 0.669, None]}).round(4), index=False)

# ---------------------------------------------------------------------------
r.secao("3.2 Overfitting: dentro da amostra × validação cruzada")
p_final = modelo.predict_proba(X)[:, 1]
lgb_full = lightgbm_raso(16, 3000).fit(X, y)
p_lgb = lgb_full.predict_proba(X)[:, 1]
r.tabela(pd.DataFrame({
    "modelo": ["Regressão Logística final (artefato)", "LightGBM (16 folhas)"],
    "PR-AUC no próprio treino": [prauc(y, p_final, w), prauc(y, p_lgb, w)],
    "PR-AUC na validação cruzada": [final.prauc.mean(), lgb_cv.prauc.mean()],
    "captura no treino": [captura_topo(y, p_final, w), captura_topo(y, p_lgb, w)],
    "captura na validação": [final.captura10.mean(), lgb_cv.captura10.mean()]}).round(4),
    index=False)

# ---------------------------------------------------------------------------
r.secao("3.3 Instabilidade entre folds e tamanho efetivo da amostra")
desc = pd.DataFrame({"baseline": base.prauc, "final": final.prauc,
                     "final − baseline": final.prauc - base.prauc}).describe(
    percentiles=[0.1, 0.5, 0.9]).T
r.tabela(desc.round(4))
c = comparar(base.prauc, final.prauc)
r(f"Final à frente em {(final.prauc > base.prauc).sum()} de {len(FOLDS)} folds; teste t corrigido: "
  f"ganho {c['ganho']:+.4f}, t = {c['t']:.2f}, p = {c['p']:.3f}")
pos_w = w[y == 1]
kish_pos = pos_w.sum() ** 2 / (pos_w ** 2).sum()
kish_all = w.sum() ** 2 / (w ** 2).sum()
r(f"Tamanho efetivo (Kish) dos 523 positivos com peso_semana: {kish_pos:.0f} "
  f"({kish_pos / len(pos_w):.0%}); de todas as linhas: {kish_all:,.0f} ({kish_all / len(w):.0%})")
sem_peso = pd.DataFrame({
    "modelo": ["baseline", "final"],
    "PR-AUC sem peso (média dos folds)": [
        np.mean([average_precision_score(y[va], res.oof[va, i // 5]) for i, (_, va) in
                 enumerate(FOLDS)]) for res in (base, final)],
    "PR-AUC ponderada": [base.prauc.mean(), final.prauc.mean()]})
r.tabela(sem_peso.round(4), index=False)

# ---------------------------------------------------------------------------
r.secao("3.4 Custo da regra conservadora: o caminho 'maior média' teria dado quanto?")
GRADE_C = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
res = {C: validar_cv(regressao_logistica(C, **colunas_lr("dia_semana", "continua")), X, y, w,
                     FOLDS, peso_ajuste=w) for C in GRADE_C}
C_max = max(GRADE_C, key=lambda c: res[c].prauc.mean())
alt = res[C_max]
c = comparar(final.prauc, alt.prauc)
r(f"Regressão Logística com dia da semana e peso_semana no ajuste (melhor C = {C_max}): "
  f"PR-AUC {alt.prauc.mean():.4f}, captura {alt.captura10.mean():.1%}; contra o final: "
  f"{c['ganho']:+.4f} (t = {c['t']:.2f}, p = {c['p']:.2f})")

# ---------------------------------------------------------------------------
r.secao("3.5 Viés de calibração por segmento (previsões out-of-fold, sigmoide de fora do fold)")
cal = calibrar_fora_do_fold(final.oof, y, w, FOLDS)
REP = cal.shape[1]
score = cal.ravel(order="F")
y_rep, w_rep = np.tile(y, REP), np.tile(w, REP)
linhas = []
for col in ["periodo", "dia_semana", "regiao", "urbano", "tipo_area", "faixa_idade", "sexo",
            "faixa_renda", "classe_abep"]:
    grupo = np.tile(treino[col].astype(str).to_numpy(), REP)
    t = confiabilidade(score, y_rep, w_rep, grupo, copias=REP)
    for nivel, lin in t.iterrows():
        linhas.append({"segmento": col, "nível": nivel, "ocasiões": lin["ocasioes"],
                       "prevista": lin["prevista"], "observada": lin["observada"],
                       "observada / prevista": lin["observada / prevista"]})
seg = pd.DataFrame(linhas)
seg["alerta"] = np.where((seg["ocasiões"] >= 30)
                         & ((seg["observada / prevista"] > 1.3)
                            | (seg["observada / prevista"] < 1 / 1.3)), "viés > 30%", "")
fmt = seg.assign(prevista=seg["prevista"].map("{:.3%}".format),
                 observada=seg["observada"].map("{:.3%}".format))
r.tabela(fmt.round(2), index=False)

# ---------------------------------------------------------------------------
r.secao("3.6 Análise de erro com os cortes de produção (sigmoide e cortes do artefato)")
score_prod = aplicar_sigmoide(final.oof.ravel(order="F"), A, B)
classe = classificar(score_prod, CORTES)
linhas = []
for col in ["periodo", "fim_de_semana", "regiao", "urbano", "tipo_area", "faixa_idade", "sexo",
            "faixa_renda", "classe_abep"]:
    grupo = np.tile(treino[col].astype(str).to_numpy(), REP)
    d = pd.DataFrame({"g": grupo, "c": classe, "yw": y_rep * w_rep, "w": w_rep})
    for nivel, dd in d.groupby("g"):
        oc = dd["yw"].sum()
        linhas.append({
            "segmento": col, "nível": nivel, "ocasiões": int(round((dd["yw"] > 0).sum() / REP)),
            "ocasiões em alta": dd.loc[dd["c"] == "alta", "yw"].sum() / oc,
            "ocasiões perdidas (baixa)": dd.loc[dd["c"] == "baixa", "yw"].sum() / oc,
            "fatia do segmento em alta": dd.loc[dd["c"] == "alta", "w"].sum() / dd["w"].sum(),
            "taxa dentro de alta": (dd.loc[dd["c"] == "alta", "yw"].sum()
                                    / max(dd.loc[dd["c"] == "alta", "w"].sum(), 1e-12))})
erro = pd.DataFrame(linhas)
for col in ["ocasiões em alta", "ocasiões perdidas (baixa)", "fatia do segmento em alta"]:
    erro[col] = erro[col].map("{:.1%}".format)
erro["taxa dentro de alta"] = erro["taxa dentro de alta"].map("{:.2%}".format)
r.tabela(erro, index=False)
geral = {c: float((y_rep * w_rep)[classe == c].sum() / (y_rep * w_rep).sum()) for c in CLASSES}
r(f"Geral: ocasiões em alta {geral['alta']:.1%}, em media {geral['media']:.1%}, "
  f"em baixa {geral['baixa']:.1%}")

# ---------------------------------------------------------------------------
r.secao("3.7 Sazonalidade e eventos dentro do treino (o mês não é feature)")
rng = np.random.default_rng(0)
NOMES = ["jul/17", "ago/17", "set/17", "out/17", "nov/17", "dez/17", "jan/18", "fev/18"]
linhas = []
for m in range(8):
    d = treino[treino["ordem_mes"] == m]
    upas = d["COD_UPA"].unique()
    grupos = {u: i for u, i in d.groupby("COD_UPA").indices.items()}
    boot = []
    for _ in range(200):
        ii = np.concatenate([grupos[u] for u in rng.choice(upas, len(upas))])
        boot.append(taxa(d["y"].to_numpy()[ii], d["peso_semana"].to_numpy()[ii]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    linhas.append({"mês": NOMES[m], "positivos": int(d["y"].sum()),
                   "taxa ponderada": f"{taxa(d['y'], d['peso_semana']):.3%}",
                   "IC 95% (UPAs)": f"[{lo:.3%}; {hi:.3%}]",
                   "fatia Sudeste": f"{taxa(d['regiao'].eq('Sudeste'), d['peso_semana']):.1%}"})
r.tabela(pd.DataFrame(linhas), index=False)
r("Eventos: o fim de ano (dez/17) e o Carnaval (9 a 14/02/2018) estão no treino. A Copa do Mundo "
  "(14/06 a 15/07/2018) e as festas juninas caem no período de teste, sem equivalente no treino. "
  "A POF não traz a data exata do recordatório, só o dia da semana e o mês estimado, então "
  "feriados não são identificáveis.")

# ---------------------------------------------------------------------------
r.secao("3.8 Calibração por dia: modelo final (fim de semana) × "
        "Regressão Logística com dia da semana")
ref = validar_cv(regressao_logistica(0.03), X, y, w, FOLDS)
cal_ref = calibrar_fora_do_fold(ref.oof, y, w, FOLDS).ravel(order="F")
dia = pd.Categorical(np.tile(treino["dia_semana"].astype(str).to_numpy(), REP),
                     list(treino["dia_semana"].cat.categories))
t_fin = confiabilidade(score, y_rep, w_rep, dia, copias=REP)
t_ref = confiabilidade(cal_ref, y_rep, w_rep, dia, copias=REP)
comp = pd.DataFrame({"ocasiões": t_fin["ocasioes"], "observada": t_fin["observada"],
                     "prevista final": t_fin["prevista"],
                     "prevista dia da semana": t_ref["prevista"]})
comp["erro final"] = comp["prevista final"] / comp["observada"] - 1
comp["erro dia da semana"] = comp["prevista dia da semana"] / comp["observada"] - 1
r.tabela(comp.assign(**{c: comp[c].map("{:.3%}".format) for c in
                        ["observada", "prevista final", "prevista dia da semana"]}).round(2))
r(f"Erro absoluto médio por dia: final {comp['erro final'].abs().mean():.0%}, "
  f"dia da semana {comp['erro dia da semana'].abs().mean():.0%}. PR-AUC: final "
  f"{final.prauc.mean():.4f}, dia da semana {ref.prauc.mean():.4f}.")

r.gravar()
