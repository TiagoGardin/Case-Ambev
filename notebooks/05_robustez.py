# %% [markdown]
# # 05 – Robustez do modelo escolhido (só treino)
#
# Checagens complementares, feitas depois da seleção (notebook 04) e da auditoria de
# 20/11/2026, antes de abrir o teste. Elas descrevem como o modelo escolhido se comporta e não
# escolhem nada: o modelo congelado em `models/` não muda.
#
# 1. PR-AUC média dos folds × PR-AUC agregada das previsões out-of-fold.
# 2. Validação com origem móvel dentro do treino (treina no passado, valida nos meses seguintes).
# 3. Generalização por região: treina com 4 regiões e avalia na 5ª (README, Metodologia).
# 4. Partições por UPA em vez de domicílio.
# 5. Calibração por segmento e por dia da semana.
# 6. Tamanho efetivo da amostra e métricas sem peso.
# 7. Taxa da ocasião por mês da coleta.
# 8. Sensibilidade da definição do alvo às bebidas no limite (água tônica, H2O, quinino).
# 9. Calibração com validação cruzada aninhada, que testa fora da amostra o procedimento do
#    modelo final (sigmoide nas previsões out-of-fold e ajuste final com C × 4/5).
#
# Rodar: `python notebooks/05_robustez.py` (cerca de 4 minutos).
# Pré-requisito: `python -m src.features.build_grid`.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

from joblib import Parallel, delayed  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import average_precision_score, log_loss  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from src.features.build_grid import DIAS, REGIOES, periodo_do_dia  # noqa: E402
from src.ingestion.codigos_alimentos import (  # noqa: E402
    cadastro_produtos,
    codigos_pizza,
    codigos_refrigerante,
)
from src.ingestion.pof import CHAVE_DOM, CHAVE_PESSOA  # noqa: E402
from src.models.calibracao import (  # noqa: E402
    ajustar_sigmoide,
    aplicar_sigmoide,
    calibrar_fora_do_fold,
    confiabilidade,
)
from src.models.metricas import captura_topo, prauc  # noqa: E402
from src.models.modelos import baseline, construir, regressao_logistica  # noqa: E402
from src.models.preparo import FEATURES  # noqa: E402
from src.models.treinar import CONFIG, config_final  # noqa: E402
from src.models.validacao import domicilio, particoes, validar_cv  # noqa: E402

pd.set_option("display.width", 200)
PROC = ROOT / "data" / "processed"
NOME_FINAL = "Regressão Logística final"
NOME_DIA = "Regressão Logística com dia da semana (C 0,03)"

# %% [markdown]
# ## Dados e validação cruzada dos três modelos de referência
# O baseline, o modelo final (a configuração congelada) e a Regressão Logística com dia da
# semana, que o adendo de 20/11 do README declara como comparação exploratória no teste.

# %%
grid = pd.read_parquet(PROC / "grid.parquet")
treino = grid[grid["split"] == "treino"].reset_index(drop=True)
X, y, w = treino[FEATURES], treino["y"].to_numpy(), treino["peso_semana"].to_numpy()
FOLDS = particoes(y, domicilio(treino))
prevalencia = (y * w).sum() / w.sum()

base = validar_cv(baseline(), X, y, w, FOLDS, peso_ajuste=w)
final = validar_cv(construir(CONFIG), X, y, w, FOLDS)
dia = validar_cv(regressao_logistica(0.03), X, y, w, FOLDS)
print(f"Treino: {len(treino):,} linhas, {y.sum()} positivos; prevalência ponderada "
      f"{prevalencia:.4f}")


def fmt(v: float, casas: int = 4) -> str:
    return f"{v:.{casas}f}"


# %% [markdown]
# ## 1. PR-AUC média dos folds × agregada
# O protocolo reporta a média da PR-AUC dos 50 folds (cerca de 105 positivos em cada). No
# teste, a PR-AUC é calculada uma vez sobre todas as linhas. Aqui, a versão agregada é a PR-AUC
# de todas as previsões out-of-fold de cada repetição, com a média das 10 repetições.

# %%
linhas = []
for nome, res in [("baseline", base), (NOME_FINAL, final), (NOME_DIA, dia)]:
    agregada = [prauc(y, res.oof[:, k], w) for k in range(res.oof.shape[1])]
    captura = [captura_topo(y, res.oof[:, k], w) for k in range(res.oof.shape[1])]
    linhas.append({"modelo": nome, "PR-AUC média dos folds": res.prauc.mean(),
                   "PR-AUC agregada": np.mean(agregada),
                   "agregada, mín–máx": f"{min(agregada):.4f}–{max(agregada):.4f}",
                   "vezes a prevalência (agregada)": np.mean(agregada) / prevalencia,
                   "captura agregada": np.mean(captura)})
t1 = pd.DataFrame(linhas)
print(t1.round(4).to_string(index=False))
print(f"\nFinal ÷ baseline: {t1.iloc[1, 1] / t1.iloc[0, 1]:.2f} na média dos folds, "
      f"{t1.iloc[1, 2] / t1.iloc[0, 2]:.2f} na agregada")

# Por que a média dos folds é maior: em quintos aleatórios das próprias previsões agregadas
# (que também misturam os 5 modelos de cada repetição), a PR-AUC volta a subir. Logo, o que
# pesa é o tamanho da amostra de cada cálculo, não a mistura de escalas entre modelos.
dom = domicilio(treino).to_numpy()
linhas = []
for rep in range(3):
    s = final.oof[:, rep]
    media_folds = np.mean([prauc(y[va], s[va], w[va]) for _, va in FOLDS[5 * rep: 5 * rep + 5]])
    quintos = []
    for semente in range(20):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=500 + semente)
        quintos += [prauc(y[va], s[va], w[va]) for _, va in cv.split(np.zeros(len(y)), y, dom)]
    linhas.append({"repetição": rep, "agregada": prauc(y, s, w),
                   "média dos folds originais": media_folds,
                   "média em quintos aleatórios": np.mean(quintos)})
print()
print(pd.DataFrame(linhas).round(4).to_string(index=False))

# %% [markdown]
# ## 2. Origem móvel dentro do treino
# Treina com os meses anteriores e valida nos seguintes, com domicílios inteiros (mês do
# domicílio = o menor mês das suas UCs). Intervalo de 95% da diferença de PR-AUC final −
# baseline por bootstrap pareado de UPAs (300 reamostragens).

# %%
mes_dom = treino.groupby(CHAVE_DOM)["ordem_mes"].transform("min").to_numpy()
ORIGENS = [("jul–set → out–nov", 2, [3, 4]), ("jul–nov → dez–jan", 4, [5, 6]),
           ("jul–jan → fev", 6, [7])]
CANDIDATOS = {"baseline": (baseline(), True), NOME_FINAL: (construir(CONFIG), False),
              NOME_DIA: (regressao_logistica(0.03), False)}
rng = np.random.default_rng(0)
upa = treino["COD_UPA"].to_numpy()


def ic_diferenca(yv, pa, pb, wv, upas_v, gerador, reps: int = 300) -> tuple[float, float]:
    """IC de 95% de PR-AUC(pa) − PR-AUC(pb) por bootstrap pareado de UPAs."""
    unicas = np.unique(upas_v)
    indices = {u: np.where(upas_v == u)[0] for u in unicas}
    difs = []
    for _ in range(reps):
        ii = np.concatenate([indices[u] for u in gerador.choice(unicas, size=len(unicas))])
        if yv[ii].sum() == 0:
            continue
        difs.append(prauc(yv[ii], pa[ii], wv[ii]) - prauc(yv[ii], pb[ii], wv[ii]))
    lo, hi = np.percentile(difs, [2.5, 97.5])
    return float(lo), float(hi)


linhas = []
for rotulo, ate, val in ORIGENS:
    tr, va = np.where(mes_dom <= ate)[0], np.where(np.isin(mes_dom, val))[0]
    prev = {}
    for nome, (modelo, com_peso) in CANDIDATOS.items():
        extra = {"modelo__sample_weight": w[tr] / w[tr].mean()} if com_peso else {}
        prev[nome] = clone(modelo).fit(X.iloc[tr], y[tr], **extra).predict_proba(X.iloc[va])[:, 1]
    lo, hi = ic_diferenca(y[va], prev[NOME_FINAL], prev["baseline"], w[va], upa[va], rng)
    linha = {"janela": rotulo, "positivos": int(y[va].sum())}
    for nome in CANDIDATOS:
        linha[f"PR-AUC {nome}"] = fmt(prauc(y[va], prev[nome], w[va]))
    for nome in CANDIDATOS:
        linha[f"captura {nome}"] = f"{captura_topo(y[va], prev[nome], w[va]):.1%}"
    linha["IC final − baseline"] = f"[{lo:+.4f}; {hi:+.4f}]"
    linhas.append(linha)
print(pd.DataFrame(linhas).T.to_string(header=False))

# %% [markdown]
# ## 3. Generalização por região
# Treina com as outras 4 regiões do treino e avalia na região que ficou de fora. O modelo nunca
# viu a região avaliada: a coluna dela no one-hot fica zerada no ajuste.

# %%
regiao = treino["regiao"].astype(str).to_numpy()
rng_regiao = np.random.default_rng(1)
linhas = []
for nome_regiao in REGIOES.values():
    tr, va = np.where(regiao != nome_regiao)[0], np.where(regiao == nome_regiao)[0]
    p_base = (clone(baseline()).fit(X.iloc[tr], y[tr], modelo__sample_weight=w[tr] / w[tr].mean())
              .predict_proba(X.iloc[va])[:, 1])
    p_final = clone(construir(CONFIG)).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
    lo, hi = ic_diferenca(y[va], p_final, p_base, w[va], upa[va], rng_regiao)
    linhas.append({"região avaliada": nome_regiao, "positivos": int(y[va].sum()),
                   "taxa": f"{(y[va] * w[va]).sum() / w[va].sum():.3%}",
                   "PR-AUC baseline": fmt(prauc(y[va], p_base, w[va])),
                   "PR-AUC final": fmt(prauc(y[va], p_final, w[va])),
                   "captura baseline": f"{captura_topo(y[va], p_base, w[va]):.1%}",
                   "captura final": f"{captura_topo(y[va], p_final, w[va]):.1%}",
                   "IC final − baseline": f"[{lo:+.4f}; {hi:+.4f}]"})
print(pd.DataFrame(linhas).to_string(index=False))

# %% [markdown]
# ## 4. Partições por UPA em vez de domicílio
# A UPA é o conjunto de domicílios vizinhos sorteado pelo IBGE. Se vizinhos parecidos ficassem
# dos dois lados das partições, a validação por domicílio ficaria otimista.

# %%
FOLDS_UPA = particoes(y, treino["COD_UPA"])
base_upa = validar_cv(baseline(), X, y, w, FOLDS_UPA, peso_ajuste=w)
final_upa = validar_cv(construir(CONFIG), X, y, w, FOLDS_UPA)
print(f"PR-AUC média dos folds, por domicílio × por UPA: baseline {base.prauc.mean():.4f} × "
      f"{base_upa.prauc.mean():.4f}; final {final.prauc.mean():.4f} × {final_upa.prauc.mean():.4f}")
# Contagem estrutural, só com as chaves (sem rótulo): UPAs com domicílios no treino e no teste
lados = grid[grid["split"] != "sem_mes"].groupby("COD_UPA")["split"].nunique()
print(f"UPAs com domicílios no treino e no teste: {(lados > 1).sum():,} de {lados.size:,}")

# %% [markdown]
# ## 5. Calibração por segmento
# Previsões out-of-fold do modelo final com a sigmoide ajustada fora de cada fold (como no
# notebook 04). Observado ÷ previsto acima de 1 quer dizer que o modelo subestima o segmento.

# %%
REP = final.oof.shape[1]
cal_final = calibrar_fora_do_fold(final.oof, y, w, FOLDS).ravel(order="F")
cal_dia = calibrar_fora_do_fold(dia.oof, y, w, FOLDS).ravel(order="F")
y_rep, w_rep = np.tile(y, REP), np.tile(w, REP)
linhas = []
for col in ["periodo", "dia_semana", "regiao", "tipo_area", "faixa_idade", "sexo", "faixa_renda",
            "classe_abep"]:
    grupo = np.tile(treino[col].astype(str).to_numpy(), REP)
    t = confiabilidade(cal_final, y_rep, w_rep, grupo, copias=REP)
    for nivel, lin in t.iterrows():
        razao = float(lin["observada / prevista"])
        linhas.append({"segmento": col, "nível": nivel, "ocasiões": int(lin["ocasioes"]),
                       "prevista": f"{lin['prevista']:.3%}",
                       "observada": f"{lin['observada']:.3%}",
                       "observada ÷ prevista": round(razao, 2),
                       # o alerta usa a razão sem arredondar
                       "viés > 30% (30+ ocasiões)": "sim" if lin["ocasioes"] >= 30
                       and (razao > 1.3 or razao < 1 / 1.3) else ""})
seg = pd.DataFrame(linhas)
print(seg.to_string(index=False))

dias_cat = pd.Categorical(np.tile(treino["dia_semana"].astype(str).to_numpy(), REP), DIAS)
t_fin = confiabilidade(cal_final, y_rep, w_rep, dias_cat, copias=REP)
t_dia = confiabilidade(cal_dia, y_rep, w_rep, dias_cat, copias=REP)
por_dia = pd.DataFrame({"ocasiões": t_fin["ocasioes"],
                        "observada": t_fin["observada"].map("{:.3%}".format),
                        "prevista final": t_fin["prevista"].map("{:.3%}".format),
                        "erro final": (t_fin["prevista"] / t_fin["observada"] - 1).round(2),
                        "prevista com dia da semana": t_dia["prevista"].map("{:.3%}".format),
                        "erro com dia da semana":
                            (t_dia["prevista"] / t_dia["observada"] - 1).round(2)})
print("\nPor dia da semana (erro = previsto ÷ observado − 1):")
print(por_dia.to_string())
print(f"Erro absoluto médio por dia: final {por_dia['erro final'].abs().mean():.0%}, "
      f"com dia da semana {por_dia['erro com dia da semana'].abs().mean():.0%}")

# %% [markdown]
# ## 6. Tamanho efetivo da amostra e métricas sem peso
# O `peso_semana` varia muito (o fator do sábado é cerca de 3), então os 523 positivos valem
# menos nas métricas ponderadas. O tamanho efetivo de Kish é (Σ peso)² ÷ Σ peso².

# %%
w_pos = w[y == 1]
print(f"Tamanho efetivo dos {len(w_pos)} positivos: {w_pos.sum() ** 2 / (w_pos ** 2).sum():.0f}")
print(f"PR-AUC por fold, final: mín {final.prauc.min():.4f}, mediana "
      f"{np.median(final.prauc):.4f}, máx {final.prauc.max():.4f} "
      f"(desvio-padrão {final.prauc.std(ddof=1):.4f})")
sem_peso = {nome: np.mean([average_precision_score(y[va], res.oof[va, i // 5])
                           for i, (_, va) in enumerate(FOLDS)])
            for nome, res in [("baseline", base), (NOME_FINAL, final)]}
print(f"PR-AUC sem peso (média dos folds): baseline {sem_peso['baseline']:.4f}, "
      f"final {sem_peso[NOME_FINAL]:.4f}")

# %% [markdown]
# ## 7. Taxa por mês da coleta
# O mês não é feature (Data Card, seção 3). IC de 95% por bootstrap de UPAs (200 reamostragens).

# %%
rng_mes = np.random.default_rng(0)
MESES = ["jul/17", "ago/17", "set/17", "out/17", "nov/17", "dez/17", "jan/18", "fev/18"]
linhas = []
for m, nome_mes in enumerate(MESES):
    d = treino[treino["ordem_mes"] == m]
    indices = d.groupby("COD_UPA").indices
    unicas = d["COD_UPA"].unique()
    yd, wd = d["y"].to_numpy(), d["peso_semana"].to_numpy()
    boot = []
    for _ in range(200):
        ii = np.concatenate([indices[u] for u in rng_mes.choice(unicas, len(unicas))])
        boot.append((yd[ii] * wd[ii]).sum() / wd[ii].sum())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    linhas.append({"mês": nome_mes, "positivos": int(yd.sum()),
                   "taxa": f"{(yd * wd).sum() / wd.sum():.3%}", "IC 95%": f"[{lo:.3%}; {hi:.3%}]"})
print(pd.DataFrame(linhas).to_string(index=False))

# %% [markdown]
# ## 8. Definição do alvo: bebidas no limite
# A lista de refrigerantes inclui água tônica, bebida tipo H2O e refrigerante de quinino. Quantos
# positivos do treino dependem delas?

# %%
cad = cadastro_produtos().set_index("codigo")["descricao"]
cods_rf = codigos_refrigerante()
limite = {c for c in cods_rf
          if any(t in str(cad.get(c, "")) for t in ("AGUA TONICA", "H2O", "QUININO"))}
consumo = pd.read_parquet(PROC / "pof_consumo.parquet",
                          columns=CHAVE_PESSOA + ["QUADRO", "V9001", "hora"])
consumo["periodo"] = periodo_do_dia(consumo["hora"])
chave = CHAVE_PESSOA + ["QUADRO", "periodo"]
oc = (consumo.assign(pz=consumo["V9001"].isin(codigos_pizza()),
                     rf=consumo["V9001"].isin(cods_rf),
                     rf_sem=consumo["V9001"].isin(cods_rf - limite))
      .groupby(chave)[["pz", "rf", "rf_sem"]].max().reset_index()
      .merge(treino[chave].astype({"periodo": str}), on=chave, how="inner"))
print(f"Bebidas no limite: {', '.join(sorted(cad[c] for c in limite))}")
print(f"Positivos do treino: {int((oc['pz'] & oc['rf']).sum())} com a definição atual, "
      f"{int((oc['pz'] & oc['rf_sem']).sum())} sem as bebidas no limite")

# %% [markdown]
# ## 9. Calibração com validação cruzada aninhada
# Em cada um dos 50 folds, uma validação cruzada interna (5 partes, por domicílio) gera
# previsões out-of-fold só com o treino do fold, e a sigmoide é ajustada nelas. Depois, o
# modelo é ajustado no treino do fold inteiro, com o C da validação ou com C × 4/5, e
# avaliado na parte de fora. É o mesmo procedimento do modelo final, uma camada abaixo.


# %%
def aninhado(i: int, tr: np.ndarray, va: np.ndarray) -> tuple[np.ndarray, dict]:
    interno = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1000 + i)
    oof = np.empty(len(tr))
    for itr, iva in interno.split(np.zeros(len(tr)), y[tr], dom[tr]):
        m = clone(construir(CONFIG)).fit(X.iloc[tr[itr]], y[tr[itr]])
        oof[iva] = m.predict_proba(X.iloc[tr[iva]])[:, 1]
    a, b = ajustar_sigmoide(oof, y[tr], w[tr])
    bruto = construir(CONFIG).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
    ajustado = (construir(config_final(CONFIG)).fit(X.iloc[tr], y[tr])
                .predict_proba(X.iloc[va])[:, 1])
    return va, {"sem sigmoide": bruto, "sigmoide + C da validação": aplicar_sigmoide(bruto, a, b),
                "sigmoide + C × 4/5 (modelo final)": aplicar_sigmoide(ajustado, a, b)}


saidas = Parallel(n_jobs=-1)(delayed(aninhado)(i, tr, va) for i, (tr, va) in enumerate(FOLDS))
idx = np.concatenate([va for va, _ in saidas])
linhas = []
for versao in saidas[0][1]:
    prev = np.concatenate([s[versao] for _, s in saidas])
    linhas.append({"versão": versao,
                   "log loss ponderado (média dos folds)": np.mean(
                       [log_loss(y[va], s[versao], sample_weight=w[va], labels=[0, 1])
                        for va, s in saidas]),
                   "média prevista": f"{np.average(prev, weights=w[idx]):.3%}",
                   "taxa observada": f"{np.average(y[idx], weights=w[idx]):.3%}"})
print(pd.DataFrame(linhas).round(5).to_string(index=False))
