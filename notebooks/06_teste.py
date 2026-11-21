# %% [markdown]
# # 06 – Avaliação no teste (mar–jul/2018), uma única vez
#
# O modelo congelado em `models/` (commit 46c0a6e) é avaliado uma única vez no período de
# teste, com as regras fixadas antes no README (Metodologia: "Avaliação no teste" e adendo de
# 20/11/2026). Este notebook foi escrito e ensaiado antes da rodada no teste, e nada no modelo
# muda depois dela.
#
# 1. Métricas no teste, ao lado do baseline
# 2. Incerteza: bootstrap pareado de UPAs e a regra para declarar o modelo melhor
# 3. Classes `alta`, `media` e `baixa` e matriz de confusão
# 4. Calibração
# 5. Segmentos e canal
# 6. Resultado por mês (adendo)
# 7. Comparação exploratória com o dia da semana (adendo)
# 8. Erros
# 9. Explicações: SHAP por fator
# 10. Checagem sem peso
#
# Rodar: `python notebooks/06_teste.py` (cerca de 2 minutos). Grava as tabelas em
# `reports/teste_tabelas.md` e as figuras em `reports/figures/teste_*.png`.
# Pré-requisitos: a grade e `models/explicacao.json` (`python -m src.models.explicar`).
#
# Ensaio: `ENSAIO=1 python notebooks/06_teste.py` roda o mesmo código num falso teste dentro do
# treino. Ele treina com jul–jan e avalia em fev, como a última janela da origem móvel do
# notebook 05, e grava numa pasta temporária (ou em `SAIDA`). Serve para achar erros no código
# sem olhar o teste.

# %%
import os
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

from sklearn.metrics import precision_recall_curve, roc_auc_score  # noqa: E402

from src.features.build_grid import DIAS, PERIODOS, REGIOES  # noqa: E402
from src.ingestion.pof import CHAVE_DOM  # noqa: E402
from src.models import artefato, explicar  # noqa: E402
from src.models.avaliacao import (  # noqa: E402
    MINIMO_POSITIVOS,
    ajustar_baseline,
    bootstrap_upa,
    calibracao_por_faixa,
    captura_por_local,
    intervalo,
    matriz_confusao,
    mes_do_domicilio,
    metricas,
    nome_mes,
    perfil,
    por_segmento,
    receita,
)
from src.models.calibracao import confiabilidade  # noqa: E402
from src.models.explicar import NOME_FATOR, Explicador, referencia  # noqa: E402
from src.models.metricas import captura_topo, prauc  # noqa: E402
from src.models.preparo import FEATURES, UFS  # noqa: E402
from src.models.treinar import CONFIG  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

ENSAIO = os.environ.get("ENSAIO") == "1"
if ENSAIO:
    SAIDA = Path(os.environ.get("SAIDA") or tempfile.mkdtemp(prefix="ensaio_06_"))
else:
    SAIDA = ROOT / "reports"
FIG = SAIDA / "figures"
FIG.mkdir(parents=True, exist_ok=True)
TABELAS = SAIDA / "teste_tabelas.md"

# Comparação exploratória do adendo de 20/11: dia da semana e interação período × dia, C = 0,03
CONFIG_DIA = {**CONFIG, "C": 0.03, "dia": "dia_semana"}
# Referência da validação cruzada comparável ao teste: PR-AUC e captura agregadas das
# previsões out-of-fold, média das 10 repetições (notebook 05, seção 1; README, adendo)
REFERENCIA_CV = {"final": (0.0276, 0.669), "baseline": (0.0143, 0.536), "dia": (0.0292, 0.673)}
NOMES = {"final": "modelo final", "baseline": "baseline", "dia": "LR com dia da semana"}

SUPERFICIE = "#fcfcfb"
TINTA, TINTA_2, TINTA_MUDA = "#0b0b0b", "#52514e", "#898781"
GRADE = "#e1e0d9"
AZUL, LARANJA = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlecolor": TINTA, "axes.labelcolor": TINTA_2, "axes.edgecolor": GRADE,
    "axes.facecolor": SUPERFICIE, "figure.facecolor": SUPERFICIE,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": TINTA_MUDA, "ytick.color": TINTA_2,
    "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.6,
    "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": SUPERFICIE,
})

ROTULOS = {
    "periodo": {"manha": "Manhã", "almoco": "Almoço", "tarde": "Tarde", "noite": "Noite"},
    "tipo_area": {"capital": "Capital", "regiao_metropolitana": "Região metropolitana",
                  "interior_urbano": "Interior urbano", "rural": "Rural"},
    "faixa_idade": {"10_17": "10 a 17 anos", "18_29": "18 a 29 anos", "30_44": "30 a 44 anos",
                    "45_59": "45 a 59 anos", "60_mais": "60 anos ou mais"},
    "sexo": {"feminino": "Feminino", "masculino": "Masculino"},
    "local_pizza": {"casa": "Casa", "estabelecimento": "Estabelecimento",
                    "outro_fora": "Outro lugar fora"},
    "fim_de_semana": {0: "Segunda a quinta", 1: "Sexta a domingo"},
}
TITULOS = {"periodo": "Período", "regiao": "Região", "tipo_area": "Tipo de área",
           "classe_abep": "Classe ABEP", "faixa_idade": "Faixa de idade", "sexo": "Sexo",
           "fim_de_semana": "Dia", "local_pizza": "Local da pizza"}


def pct(v, casas: int = 1) -> str:
    return "—" if pd.isna(v) else f"{v * 100:.{casas}f}%".replace(".", ",")


def num(v, casas: int = 4) -> str:
    return "—" if pd.isna(v) else f"{v:.{casas}f}".replace(".", ",")


def inteiro(v) -> str:
    return f"{int(v):,}".replace(",", ".")


def sinal(v, casas: int = 4) -> str:
    return "—" if pd.isna(v) else f"{v:+.{casas}f}".replace(".", ",")


def pct_sinal(v) -> str:
    """Erro relativo com sinal: +52% = previsto 52% acima do observado."""
    return "—" if pd.isna(v) else f"{v * 100:+.0f}%"


def pp(v, casas: int = 1) -> str:
    """Diferença entre duas frações, em pontos percentuais."""
    return "—" if pd.isna(v) else f"{v * 100:+.{casas}f}".replace(".", ",") + " p.p."


def rotulo(coluna: str, valor) -> str:
    return ROTULOS.get(coluna, {}).get(valor, str(valor))


md = [f"# Avaliação no teste{' (ENSAIO: jul–jan → fev, dentro do treino)' if ENSAIO else ''}",
      "", "Gerado por `notebooks/06_teste.py`. Métricas ponderadas por `peso_semana`; "
      "intervalos de 95% por bootstrap pareado de UPAs.", ""]


def registrar(titulo: str, tabela: pd.DataFrame, nota: str | None = None) -> None:
    """Imprime a tabela e a acrescenta, já formatada, ao markdown de saída."""
    print(f"\n{titulo}")
    print(tabela.to_string(index=False))
    md.extend([f"## {titulo}", ""])
    if nota:
        md.extend([nota, ""])
    colunas = [str(c) for c in tabela.columns]
    md.append("| " + " | ".join(colunas) + " |")
    md.append("|" + "---|" * len(colunas))
    for _, linha in tabela.iterrows():
        md.append("| " + " | ".join(str(v) for v in linha) + " |")
    md.append("")


# %% [markdown]
# ## Dados e modelos
# - **Modelo final:** o arquivo congelado, carregado com a checagem do hash.
# - **Baseline:** taxa ponderada de cada célula dia × período do treino.
# - **LR com dia da semana:** a comparação exploratória do adendo, com a mesma receita do modelo
#   final. A conferência abaixo mostra que essa receita, aplicada à configuração congelada,
#   reproduz o arquivo de `models/`.

# %%
grid = pd.read_parquet(ROOT / "data" / "processed" / "grid.parquet")
treino_todo = grid[grid["split"] == "treino"].reset_index(drop=True)
if ENSAIO:
    mes_dom = treino_todo.groupby(CHAVE_DOM)["ordem_mes"].transform("min")
    treino = treino_todo[mes_dom <= 6].reset_index(drop=True)
    teste = treino_todo[mes_dom == 7].reset_index(drop=True)
else:
    treino = treino_todo
    teste = grid[grid["split"] == "teste"].reset_index(drop=True)
del grid

if ENSAIO:
    final, cortes_final = receita(CONFIG, treino)
    explicador = Explicador(final, referencia(final, treino[FEATURES], treino["peso_semana"]))
    meta = None
    print("ENSAIO: modelo final refeito com jul–jan pela mesma receita; teste = fev (treino)")
else:
    final, meta = artefato.carregar()
    cortes_final = meta["classification"]["cuts"]
    explicador = explicar.carregar(final, meta)
    refeito, cortes_refeito = receita(CONFIG, treino)
    diferenca = float(np.abs(refeito.predict_proba(treino[FEATURES])[:, 1]
                             - final.predict_proba(treino[FEATURES])[:, 1]).max())
    mesmos_cortes = all(np.isclose(cortes_refeito[c], cortes_final[c], rtol=1e-9, atol=0)
                        for c in cortes_final)
    print(f"Modelo: {meta['model_name']} {meta['model_version']} "
          f"(SHA-256 {meta['model_file_sha256'][:12]}…), treinado em {meta['trained_at']}")
    print(f"Conferência da receita: diferença máxima do score no treino {diferenca:.1e}; "
          f"mesmos cortes: {mesmos_cortes}")
    assert diferenca < 1e-10 and mesmos_cortes, "a receita não reproduz o modelo congelado"

base, corte_base = ajustar_baseline(treino)
dia, cortes_dia = receita(CONFIG_DIA, treino)

Xte, yte, wte = teste[FEATURES], teste["y"].to_numpy(), teste["peso_semana"].to_numpy()
upa = teste["COD_UPA"].to_numpy()
scores = {"final": final.predict_proba(Xte)[:, 1], "baseline": base.predict_proba(Xte)[:, 1],
          "dia": dia.predict_proba(Xte)[:, 1]}
cortes = {"final": cortes_final["alta"], "baseline": corte_base, "dia": cortes_dia["alta"]}
classes = {"final": artefato.classificar(scores["final"], cortes_final),
           "baseline": np.where(scores["baseline"] >= corte_base, "alta", "baixa")}
prevalencia = float(np.average(yte, weights=wte))
taxa_treino = float(np.average(treino["y"], weights=treino["peso_semana"]))
domicilios_com_ocasiao = teste.loc[teste["y"] == 1, CHAVE_DOM].drop_duplicates().shape[0]

teste["regiao"] = pd.Categorical(teste["regiao"].astype(str), list(REGIOES.values()))
teste["dia_semana"] = pd.Categorical(teste["dia_semana"].astype(str), DIAS)
teste["periodo"] = pd.Categorical(teste["periodo"].astype(str), PERIODOS)

print(f"\nTeste: {len(teste):,} linhas, {int(yte.sum())} ocasiões em "
      f"{domicilios_com_ocasiao} domicílios, {teste['COD_UPA'].nunique():,} UPAs; "
      f"taxa ponderada {prevalencia:.3%} (treino: {taxa_treino:.3%})")
print(f"Cortes de alta: final {cortes['final']:.4%} (média {cortes_final['media']:.4%}), "
      f"baseline {corte_base:.4%}, LR com dia {cortes['dia']:.4%}")

# %% [markdown]
# ## 1. Métricas no teste, ao lado do baseline
# A PR-AUC é calculada uma vez sobre todas as linhas do teste. A referência comparável da
# validação cruzada é a agregada das previsões out-of-fold (0,0276 no modelo final e 0,0143 no
# baseline), não a média dos folds. Precisão, recall e F1 usam o corte de `alta` de cada modelo,
# fixado no treino. A prevalência sobe de 0,30% no treino para 0,41% no teste, e o score não é
# recalibrado.

# %%
res = {nome: metricas(yte, scores[nome], wte, cortes[nome]) for nome in ("final", "baseline")}
ordem_metricas = list(res["final"])
t1 = pd.DataFrame({
    "métrica": ordem_metricas,
    "modelo final": [res["final"][m] for m in ordem_metricas],
    "baseline": [res["baseline"][m] for m in ordem_metricas],
})
formato = {"PR-AUC": num, "ROC-AUC": lambda v: num(v, 3), "Brier": lambda v: num(v, 5),
           "taxa prevista": lambda v: pct(v, 3), "taxa observada": lambda v: pct(v, 3)}
for coluna in ("modelo final", "baseline"):
    t1[coluna] = [formato.get(m, pct)(v) for m, v in zip(t1["métrica"], t1[coluna], strict=True)]
t1.loc[len(t1)] = ["PR-AUC ÷ prevalência",
                   num(res["final"]["PR-AUC"] / prevalencia, 1) + "×",
                   num(res["baseline"]["PR-AUC"] / prevalencia, 1) + "×"]
t1.loc[len(t1)] = ["referência da validação cruzada: PR-AUC agregada",
                   num(REFERENCIA_CV["final"][0]), num(REFERENCIA_CV["baseline"][0])]
t1.loc[len(t1)] = ["referência da validação cruzada: captura agregada no top 10%",
                   pct(REFERENCIA_CV["final"][1]), pct(REFERENCIA_CV["baseline"][1])]
registrar("1. Métricas no teste", t1,
          f"{inteiro(len(teste))} linhas, {int(yte.sum())} ocasiões, taxa ponderada "
          f"{pct(prevalencia, 3)}. Cortes de `alta` fixados no treino: modelo final "
          f"{pct(cortes['final'], 4)}; baseline {pct(corte_base, 4)} (P90 do baseline no treino).")

# %% [markdown]
# ## 2. Incerteza: bootstrap pareado de UPAs
# 1.000 reamostragens das UPAs do teste, com reposição, as mesmas para os três modelos. Pela
# regra do README, o modelo só é declarado melhor que o baseline se o intervalo de 95% da
# diferença de PR-AUC não incluir zero.

# %%
boot = bootstrap_upa(yte, scores, wte, upa, semente=0)
linhas = []
for nome in ("final", "baseline", "dia"):
    lo_p, hi_p = intervalo(boot[f"PR-AUC {nome}"])
    lo_c, hi_c = intervalo(boot[f"captura {nome}"])
    linhas.append({"modelo": NOMES[nome],
                   "PR-AUC": num(prauc(yte, scores[nome], wte)),
                   "IC 95% PR-AUC": f"{num(lo_p)} a {num(hi_p)}",
                   "captura no top 10%": pct(captura_topo(yte, scores[nome], wte)),
                   "IC 95% captura": f"{pct(lo_c)} a {pct(hi_c)}"})
dif_pr = boot["PR-AUC final"] - boot["PR-AUC baseline"]
dif_cap = boot["captura final"] - boot["captura baseline"]
lo_d, hi_d = intervalo(dif_pr)
lo_dc, hi_dc = intervalo(dif_cap)
melhor = lo_d > 0
linhas.append({"modelo": "diferença final − baseline",
               "PR-AUC": sinal(prauc(yte, scores["final"], wte)
                               - prauc(yte, scores["baseline"], wte)),
               "IC 95% PR-AUC": f"{sinal(lo_d)} a {sinal(hi_d)}",
               "captura no top 10%": pp(captura_topo(yte, scores["final"], wte)
                                        - captura_topo(yte, scores["baseline"], wte)),
               "IC 95% captura": f"{pp(lo_dc)} a {pp(hi_dc)}"})
t2 = pd.DataFrame(linhas)
registrar("2. Intervalos de 95% (bootstrap pareado de UPAs, 1.000 reamostragens)", t2,
          f"Modelo final melhor que o baseline pela regra do README (IC da diferença de PR-AUC "
          f"acima de zero): **{'sim' if melhor else 'não'}**. Reamostragens com a diferença "
          f"positiva: {pct((dif_pr > 0).mean())}.")

# %% [markdown]
# ## 3. Classes `alta`, `media` e `baixa`
# Na validação cruzada, as classes tinham 10%, 20% e 70% do peso e 67,0%, 27,4% e 5,6% das
# ocasiões. Aqui, o mesmo corte aplicado ao teste.

# %%
oof_peso = meta["classification"]["weight_share_oof"] if meta else {}
oof_cap = meta["classification"]["occasion_capture_oof"] if meta else {}
linhas = []
for c in ("alta", "media", "baixa"):
    m = classes["final"] == c
    taxa_c = float((yte * wte)[m].sum() / wte[m].sum()) if m.any() else np.nan
    linhas.append({"classe": c, "% do peso": pct(wte[m].sum() / wte.sum()),
                   "% do peso na validação": pct(oof_peso.get(c, np.nan)),
                   "% das ocasiões": pct((yte * wte)[m].sum() / (yte * wte).sum()),
                   "% das ocasiões na validação": pct(oof_cap.get(c, np.nan)),
                   "ocasiões (linhas)": int(yte[m].sum()),
                   "taxa observada": pct(taxa_c, 2),
                   "vezes a taxa do teste": num(taxa_c / prevalencia, 1) + "×"})
registrar("3. Classes do modelo final no teste", pd.DataFrame(linhas))

for nome in ("final", "baseline"):
    cm = matriz_confusao(yte, scores[nome], wte, cortes[nome])
    cm["% do peso"] = [pct(v, 2) for v in cm["% do peso"]]
    cm["linhas"] = [inteiro(v) for v in cm["linhas"]]
    registrar(f"3b. Matriz de confusão no corte de alta: {NOMES[nome]}", cm)

# %% [markdown]
# ## 4. Calibração
# O score não é recalibrado com o teste. Com a prevalência maior no teste, o esperado é que o
# score fique, em média, abaixo da taxa observada; a pergunta é se a ordem e as faixas se
# mantêm.

# %%
brier_constante = float(np.average((yte - taxa_treino) ** 2, weights=wte))
linhas = []
for nome in ("final", "baseline"):
    prevista = res[nome]["taxa prevista"]
    linhas.append({"modelo": NOMES[nome], "taxa prevista": pct(prevista, 3),
                   "taxa observada": pct(prevalencia, 3),
                   "observada ÷ prevista": num(prevalencia / prevista, 2),
                   "Brier": num(res[nome]["Brier"], 6)})
linhas.append({"modelo": "taxa constante do treino", "taxa prevista": pct(taxa_treino, 3),
               "taxa observada": pct(prevalencia, 3),
               "observada ÷ prevista": num(prevalencia / taxa_treino, 2),
               "Brier": num(brier_constante, 6)})
registrar("4. Calibração geral", pd.DataFrame(linhas))

decis = calibracao_por_faixa(scores["final"], yte, wte, faixas=10)
t4 = decis.reset_index()
t4_fmt = pd.DataFrame({"decil do score": t4["faixa"],
                       "% do peso": [pct(v) for v in t4["% do peso"]],
                       "ocasiões": t4["ocasiões"],
                       "taxa prevista": [pct(v, 3) for v in t4["taxa prevista"]],
                       "taxa observada": [pct(v, 3) for v in t4["taxa observada"]],
                       "observada ÷ prevista": [num(v, 2) for v in t4["observada ÷ prevista"]]})
registrar("4b. Calibração por decil do score (modelo final)", t4_fmt)

eixo_pct = FuncFormatter(lambda v, _: pct(v, 0))
eixo_pct_log = FuncFormatter(lambda v, _: pct(v, max(0, int(-np.floor(np.log10(v * 100))))))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
topo = 0.0
for nome, cor in (("final", AZUL), ("baseline", LARANJA)):
    precisao, recall, _ = precision_recall_curve(yte, scores[nome], sample_weight=wte)
    # Em degraus, como no scikit-learn: ligar os pontos por retas inventaria precisões
    ax1.plot(recall, precisao, color=cor, lw=2, drawstyle="steps-post",
             label=f"{NOMES[nome].capitalize()} (PR-AUC {num(res[nome]['PR-AUC'], 3)})")
    topo = max(topo, precisao[recall >= 0.05].max())
ax1.axhline(prevalencia, color=TINTA_MUDA, lw=1, ls="--")
ax1.text(0.55, prevalencia, f" prevalência {pct(prevalencia, 2)}", ha="left", va="bottom",
         color=TINTA_MUDA, fontsize=8.5)
ax1.set_xlabel("Recall (fração das ocasiões encontradas)")
ax1.set_ylabel("Precisão")
ax1.set_ylim(0, topo * 1.15)
ax1.xaxis.set_major_formatter(eixo_pct)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: pct(v, 1)))  # marcas de 0,5% em 0,5%
ax1.set_title("Precisão × recall no teste")
ax1.legend(frameon=False, fontsize=8.5)

validos = decis[decis["taxa observada"] > 0]
ax2.plot(decis["taxa prevista"], decis["taxa prevista"], color=TINTA_MUDA, lw=1, ls="--")
ax2.plot(validos["taxa prevista"], validos["taxa observada"], color=AZUL, lw=2, marker="o",
         ms=8, mec=SUPERFICIE, mew=2, label="Modelo final, por decil do score")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.xaxis.set_major_formatter(eixo_pct_log)
ax2.yaxis.set_major_formatter(eixo_pct_log)
for eixo in (ax2.xaxis, ax2.yaxis):
    eixo.set_minor_formatter(NullFormatter())
ax2.set_xlabel("Taxa prevista (escala log)")
ax2.set_ylabel("Taxa observada (escala log)")
ax2.set_title("Calibração por decil do score")
ax2.text(0.02, 0.96, "linha tracejada: previsto = observado", transform=ax2.transAxes,
         color=TINTA_MUDA, fontsize=8.5, va="top")
ax2.legend(frameon=False, fontsize=8.5, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "teste_1_precisao_calibracao.png")
plt.close(fig)

# %% [markdown]
# ## 5. Segmentos e canal
# Período, região, tipo de área e público (classe ABEP, faixa de idade e sexo). Níveis com
# menos de 30 ocasiões no teste são só descritos. A PR-AUC é calculada dentro do nível; a
# coluna "% em alta" é a fração das ocasiões do nível que cada modelo põe em `alta`.

# %%
sub = {k: scores[k] for k in ("final", "baseline")}
sub_cortes = {k: cortes[k] for k in ("final", "baseline")}
for coluna in ("periodo", "regiao", "tipo_area", "classe_abep", "faixa_idade", "sexo"):
    t = por_segmento(teste[coluna], yte, sub, sub_cortes, wte)
    saida = pd.DataFrame({TITULOS[coluna]: [rotulo(coluna, v) for v in t["nível"]],
                          "linhas": [inteiro(v) for v in t["linhas"]],
                          "ocasiões": t["ocasiões"],
                          "taxa observada": [pct(v, 2) for v in t["taxa observada"]]})
    for c in ("PR-AUC final", "PR-AUC baseline"):
        saida[c] = [num(v) for v in t.get(c, pd.Series(np.nan, index=t.index))]
    for c in ("% em alta final", "% em alta baseline"):
        saida[c] = [pct(v) for v in t.get(c, pd.Series(np.nan, index=t.index))]
    saida["observada ÷ prevista (final)"] = [
        num(v, 2) for v in t.get("observada ÷ prevista", pd.Series(np.nan, index=t.index))]
    registrar(f"5. Segmento: {TITULOS[coluna]}", saida,
              f"Níveis com menos de {MINIMO_POSITIVOS} ocasiões: só descritos (—).")

t = captura_por_local(teste["local_pizza"], yte, {k: classes[k] for k in ("final", "baseline")},
                      wte)
saida = pd.DataFrame({"local da pizza": [rotulo("local_pizza", v) for v in t["local"]],
                      "ocasiões": t["ocasiões"]})
for c in ("% em alta final", "% em alta ou média final", "% em alta baseline"):
    saida[c] = [pct(v) for v in t.get(c, pd.Series(np.nan, index=t.index))]
registrar("5b. Canal: captura das ocasiões de cada local da pizza", saida,
          "O local só existe quando há pizza, então o resultado por canal é a fração das "
          "ocasiões de cada local que fica em cada classe. `Casa` inclui delivery (a POF não "
          "separa).")

# %% [markdown]
# ## 6. Resultado por mês (adendo de 20/11)
# Só descrição: na origem móvel dentro do treino, o ganho sobre o baseline variou entre janelas,
# e o teste tem a Copa do Mundo (jun–jul) e as festas juninas, que o treino não tem. O mês é o
# que decidiu o split do domicílio.

# %%
mes = mes_do_domicilio(teste)
linhas = []
for ordem in sorted(mes.dropna().unique()):
    m = (mes == ordem).to_numpy()
    yy, ww = yte[m], wte[m]
    tem = yy.sum() > 0
    observada = float((yy * ww).sum() / ww.sum())
    prevista = float(np.average(scores["final"][m], weights=ww))
    linhas.append({
        "mês": nome_mes(ordem), "linhas": inteiro(m.sum()), "ocasiões": int(yy.sum()),
        "PR-AUC final": num(prauc(yy, scores["final"][m], ww)) if tem else "—",
        "PR-AUC baseline": num(prauc(yy, scores["baseline"][m], ww)) if tem else "—",
        "captura final": pct(captura_topo(yy, scores["final"][m], ww)) if tem else "—",
        "captura baseline": pct(captura_topo(yy, scores["baseline"][m], ww)) if tem else "—",
        "taxa prevista": pct(prevista, 3), "taxa observada": pct(observada, 3),
        "observada ÷ prevista": num(observada / prevista, 2),
    })
registrar("6. Resultado por mês do teste", pd.DataFrame(linhas),
          "Captura no top 10% do peso de cada mês. Meses com poucas ocasiões têm métricas "
          "instáveis.")

# %% [markdown]
# ## 7. Comparação exploratória com o dia da semana (adendo de 20/11)
# A Regressão Logística com dia da semana (C = 0,03, sem peso no ajuste) recebeu a mesma receita
# do modelo final. Ela entra só como comparação: o modelo da aplicação continua o de `models/`,
# qualquer que seja o resultado.

# %%
dif_dia = boot["PR-AUC dia"] - boot["PR-AUC final"]
lo_x, hi_x = intervalo(dif_dia)
cal_final = confiabilidade(scores["final"], yte, wte, teste["dia_semana"])
cal_dia = confiabilidade(scores["dia"], yte, wte, teste["dia_semana"])


def erro_por_dia(cal: pd.DataFrame) -> pd.Series:
    """Previsto ÷ observado − 1 em cada dia: a mesma medida do notebook 05 (seção 5).

    Um dia sem ocasião no teste não tem razão definida e fica de fora da média.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        erro = cal["prevista"] / cal["observada"] - 1
    return erro.replace([np.inf, -np.inf], np.nan)


erro_final = float(erro_por_dia(cal_final).abs().mean())
erro_dia = float(erro_por_dia(cal_dia).abs().mean())
t7 = pd.DataFrame([
    {"modelo": NOMES[n], "PR-AUC": num(prauc(yte, scores[n], wte)),
     "captura no top 10%": pct(captura_topo(yte, scores[n], wte)),
     "erro médio da calibração por dia": pct(e)}
    for n, e in (("final", erro_final), ("dia", erro_dia))])
registrar("7. Comparação exploratória: dia da semana × fim de semana", t7,
          "Diferença de PR-AUC (dia − final): "
          f"{sinal(prauc(yte, scores['dia'], wte) - prauc(yte, scores['final'], wte))}, "
          f"IC 95% {sinal(lo_x)} a {sinal(hi_x)}. Erro médio da calibração por dia: média de "
          "|prevista ÷ observada − 1| nos dias, a mesma medida do notebook 05 (21% no modelo "
          "final e 16% com dia da semana, no treino). Exploratório: não muda o modelo da "
          "aplicação.")
dias = pd.DataFrame({"dia": cal_final.index.astype(str), "ocasiões": cal_final["ocasioes"],
                     "observada": [pct(v, 3) for v in cal_final["observada"]],
                     "prevista final": [pct(v, 3) for v in cal_final["prevista"]],
                     "erro final": [pct_sinal(v) for v in erro_por_dia(cal_final)],
                     "prevista dia": [pct(v, 3) for v in cal_dia["prevista"]],
                     "erro dia": [pct_sinal(v) for v in erro_por_dia(cal_dia)]})
registrar("7b. Calibração por dia da semana", dias)

# %% [markdown]
# ## 8. Erros
# - **Falsos negativos:** ocasiões que o modelo final pôs em `baixa`.
# - **Falsos positivos:** contextos em `alta` sem ocasião. Com uma taxa de cerca de 2% em
#   `alta`, quase todo contexto em `alta` é um falso positivo; o perfil deles é comparado com o
#   dos verdadeiros positivos.
#
# O perfil usa as dimensões dos segmentos (seção 5) e o fim de semana. O dia atípico fica de
# fora: ele é declarado depois do dia, e um dia com pizza tende a ser chamado de atípico por
# causa dela (Data Card), então o perfil por dia atípico confundiria causa e efeito.

# %%
ocasiao = yte == 1
grupos = {"todas as ocasiões": ocasiao,
          "ocasiões em baixa (FN)": ocasiao & (classes["final"] == "baixa"),
          "alta com ocasião (VP)": ocasiao & (classes["final"] == "alta"),
          "alta sem ocasião (FP)": ~ocasiao & (classes["final"] == "alta")}
fn = grupos["ocasiões em baixa (FN)"]
print(f"Falsos negativos: {int(fn.sum())} ocasiões em baixa, "
      f"{pct(wte[fn].sum() / wte[ocasiao].sum())} do peso das ocasiões")
print(f"Falsos positivos: {pct(1 - res['final']['precisão em alta'])} do peso de alta")
for coluna in ("periodo", "fim_de_semana", "regiao", "tipo_area", "classe_abep", "faixa_idade",
               "sexo", "local_pizza"):
    usados = {k: v for k, v in grupos.items()
              if not (coluna == "local_pizza" and k == "alta sem ocasião (FP)")}
    t = perfil(teste[coluna], usados, wte)
    if "sem informação" in t.index and (t.loc["sem informação"] == 0).all():
        t = t.drop(index="sem informação")  # o local só falta onde não há pizza
    binaria = coluna == "fim_de_semana"
    t.insert(0, TITULOS[coluna],
             [rotulo(coluna, int(float(v)) if binaria else v) for v in t.index])
    for c in usados:
        t[c] = [pct(v) for v in t[c]]
    registrar(f"8. Perfil dos erros: {TITULOS[coluna]}", t.reset_index(drop=True))

# %% [markdown]
# ## 9. Explicações: SHAP por fator
# SHAP exato da Regressão Logística (src.models.explicar), em log-odds do score calibrado, com
# o treino como referência. Global: média ponderada do valor absoluto da contribuição de cada
# fator. Local: os 3 fatores de maior contribuição formam o `top_factors` da API.

# %%
contrib_teste = explicador.contribuicoes(Xte)
contrib_treino = explicador.contribuicoes(treino[FEATURES])
global_teste = contrib_teste.abs().mul(wte, axis=0).sum() / wte.sum()
global_treino = (contrib_treino.abs().mul(treino["peso_semana"].to_numpy(), axis=0).sum()
                 / treino["peso_semana"].sum())
ordem_fatores = global_teste.sort_values(ascending=False).index
top3 = contrib_teste.abs().rank(axis=1, ascending=False, method="first") <= 3
t9 = pd.DataFrame({"fator": [NOME_FATOR[f] for f in ordem_fatores],
                   "|contribuição| média no teste": [
                       num(global_teste[f], 3) for f in ordem_fatores],
                   "no treino": [num(global_treino[f], 3) for f in ordem_fatores],
                   "% das linhas com o fator no top 3": [
                       pct(np.average(top3[f], weights=wte)) for f in ordem_fatores]})
registrar("9. SHAP global (log-odds do score calibrado)", t9)

exemplos = {
    "ocasião em alta com o maior score": np.flatnonzero(grupos["alta com ocasião (VP)"]),
    "ocasião em baixa com o menor score": np.flatnonzero(fn),
    "alta sem ocasião com o maior score": np.flatnonzero(grupos["alta sem ocasião (FP)"]),
}
md.extend(["## 9b. SHAP local: exemplos do teste", ""])
for rotulo_ex, indices in exemplos.items():
    if len(indices) == 0:
        continue
    s = scores["final"][indices]
    i = int(indices[np.argmin(s)] if "menor" in rotulo_ex else indices[np.argmax(s)])
    linha = Xte.iloc[[i]]
    contexto = linha.iloc[0].to_dict()
    contexto["UF"] = UFS[str(contexto["UF"])]
    tops = explicador.top_fatores(linha)[0]
    texto = "; ".join(f"{NOME_FATOR[t['factor']]}: {t['value']} ({t['effect']}, chance × "
                      f"{num(t['odds_multiplier'], 2)})" for t in tops)
    print(f"\n{rotulo_ex}: score {pct(scores['final'][i], 3)} ({classes['final'][i]})")
    print(f"  contexto: {contexto}")
    print(f"  top_factors: {texto}")
    md.extend([f"- **{rotulo_ex.capitalize()}:** score {pct(scores['final'][i], 3)} "
               f"({classes['final'][i]}). Contexto: {contexto}. Fatores: {texto}.", ""])

fig, ax = plt.subplots(figsize=(6.5, 3.4))
valores = global_teste[ordem_fatores][::-1]
ax.barh([NOME_FATOR[f] for f in valores.index], valores.to_numpy(), color=AZUL, height=0.6)
for y_pos, v in enumerate(valores.to_numpy()):
    ax.text(v + valores.max() * 0.01, y_pos, num(v, 2), va="center", fontsize=8.5, color=TINTA_2)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: num(v, 1)))
ax.set_xlabel("Contribuição média, em valor absoluto (log-odds do score calibrado)")
ax.set_title("SHAP global no teste: quanto cada fator move o score")
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(FIG / "teste_2_shap_global.png")
plt.close(fig)

# %% [markdown]
# ## 10. Checagem sem peso
# PR-AUC, captura no top 10% e ROC-AUC sem `peso_semana`, contando cada linha da amostra igual.

# %%
um = np.ones(len(yte))
linhas = [{"modelo": NOMES[n], "PR-AUC sem peso": num(prauc(yte, scores[n], um)),
           "captura no top 10% sem peso": pct(captura_topo(yte, scores[n], um)),
           "ROC-AUC sem peso": num(roc_auc_score(yte, scores[n]), 3)}
          for n in ("final", "baseline")]
registrar("10. Métricas sem peso (checagem)", pd.DataFrame(linhas))

# %%
TABELAS.write_text("\n".join(md) + "\n", encoding="utf-8")
destino = TABELAS if ENSAIO else TABELAS.relative_to(ROOT)
print(f"\n[ok] tabelas em {destino} e figuras em {FIG if ENSAIO else FIG.relative_to(ROOT)}/")
