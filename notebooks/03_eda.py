# %% [markdown]
# # 03 – Análise exploratória (só treino)
#
# Cinco perguntas, todas com taxas ponderadas por `peso_semana` (PESO_FINAL com cada dia da
# semana valendo 1/7) e intervalo de confiança de 95%:
# 1. **Quando** a ocasião acontece (dia × período)? Essa tabela é também o baseline do modelo.
# 2. **Quem** tem mais ocasiões à noite (classe ABEP, tipo de área, região, idade, sexo)?
# 3. **Interação:** o efeito do fim de semana muda com a classe?
# 4. **Marca:** em que classes a pizza vem menos acompanhada de refrigerante e de guaraná?
# 5. **Canal:** onde a pizza é comida (em casa ou fora), por segmento?
#
# Só o treino (jul/2017–fev/2018); o teste não é lido. Os intervalos reamostram UPAs
# (a unidade primária da amostra da POF), com 1.000 reamostragens. Saída: figuras em
# `reports/figures/` e as tabelas de todas elas em `reports/eda_tabelas.md`.
#
# Rodar: `python notebooks/03_eda.py` (menos de 1 minuto).
# Pré-requisito: `python -m src.features.build_grid`.

# %%
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

from src.features.build_grid import DIAS, PERIODOS  # noqa: E402

PROC = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"
TABELAS = ROOT / "reports" / "eda_tabelas.md"
B = 1000  # reamostragens do bootstrap

# Paleta do skill de visualização (modo claro); azul e laranja validados como par de séries
SUPERFICIE = "#fcfcfb"
TINTA, TINTA_2, TINTA_MUDA = "#0b0b0b", "#52514e", "#898781"
GRADE = "#e1e0d9"
AZUL, LARANJA = "#2a78d6", "#eb6834"
RAMPA_AZUL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
              "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

ROTULOS = {
    "periodo": {"manha": "Manhã", "almoco": "Almoço", "tarde": "Tarde", "noite": "Noite"},
    "dia_semana": dict(zip(DIAS, ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"])),
    "tipo_area": {"capital": "Capital", "regiao_metropolitana": "Região metropolitana",
                  "interior_urbano": "Interior urbano", "rural": "Rural"},
    "faixa_idade": {"10_17": "10 a 17 anos", "18_29": "18 a 29 anos", "30_44": "30 a 44 anos",
                    "45_59": "45 a 59 anos", "60_mais": "60 anos ou mais"},
    "sexo": {"feminino": "Feminino", "masculino": "Masculino"},
    "quando": {"seg_qui": "Segunda a quinta", "sex_dom": "Sexta a domingo"},
}
TITULOS = {"classe_abep": "Classe ABEP", "tipo_area": "Tipo de área", "regiao": "Região",
           "faixa_idade": "Idade", "sexo": "Sexo"}

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlecolor": TINTA, "axes.labelcolor": TINTA_2, "axes.edgecolor": GRADE,
    "axes.facecolor": SUPERFICIE, "figure.facecolor": SUPERFICIE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "xtick.color": TINTA_MUDA, "ytick.color": TINTA_2, "ytick.major.size": 0,
    "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": SUPERFICIE,
})


def pct(v: float, casas: int = 2) -> str:
    return f"{v:.{casas}f}%".replace(".", ",")


def milhar(v) -> str:
    return f"{int(v):,}".replace(",", ".")


def minusc(texto: str) -> str:
    """Primeira letra minúscula, sem mexer em siglas ("Classe ABEP" → "classe ABEP")."""
    return texto[0].lower() + texto[1:]


FMT_PCT = FuncFormatter(lambda v, _: f"{v:g}%".replace(".", ","))


def rotulo(col: str, valor) -> str:
    return ROTULOS.get(col, {}).get(valor, str(valor))


# %% [markdown]
# ## Taxa ponderada com intervalo de confiança

# %%
def taxa(d: pd.DataFrame, por: list[str], num, den=None, semente: int = 0):
    """Σ peso_semana·num / Σ peso_semana·den por grupo (em %), com IC 95% por bootstrap de UPAs.

    `n` é o tamanho da base (linhas do denominador) e `casos`, quantas delas têm o evento.
    Devolve também a matriz do bootstrap (reamostragens × grupos), para razões entre grupos.
    """
    num = np.asarray(num, dtype=bool)
    den = np.ones(len(d), dtype=bool) if den is None else np.asarray(den, dtype=bool)
    peso = d["peso_semana"].to_numpy()
    x = d[por].copy()
    x["upa"] = d["COD_UPA"].to_numpy()
    x["wn"], x["wd"] = peso * (num & den), peso * den
    x["n"], x["casos"] = den.astype(int), (num & den).astype(int)

    tot = x.groupby(por, observed=True)[["wn", "wd", "n", "casos"]].sum()
    tot = tot[tot["wd"] > 0]
    por_upa = x.groupby(["upa"] + por, observed=True)[["wn", "wd"]].sum()
    wn = por_upa["wn"].unstack(por, fill_value=0).reindex(columns=tot.index, fill_value=0)
    wd = por_upa["wd"].unstack(por, fill_value=0).reindex(columns=tot.index, fill_value=0)

    rng = np.random.default_rng(semente)
    u = len(wn)
    reamostra = rng.multinomial(u, np.full(u, 1 / u), size=B)  # quantas vezes cada UPA entra
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = (reamostra @ wn.to_numpy()) / (reamostra @ wd.to_numpy()) * 100
    inf, sup = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    out = pd.DataFrame({"taxa": tot["wn"] / tot["wd"] * 100, "ic_inf": inf, "ic_sup": sup,
                        "n": tot["n"], "casos": tot["casos"]}, index=tot.index)
    return out, boot


def razao(r: pd.DataFrame, boot: np.ndarray, a, b) -> tuple[float, float, float]:
    """Razão entre as taxas dos grupos a e b, com IC 95% do mesmo bootstrap."""
    ia, ib = r.index.get_loc(a), r.index.get_loc(b)
    with np.errstate(invalid="ignore", divide="ignore"):
        rb = boot[:, ia] / boot[:, ib]
    inf, sup = np.nanpercentile(rb, [2.5, 97.5])
    return r["taxa"].iloc[ia] / r["taxa"].iloc[ib], inf, sup


# %% [markdown]
# ## Funções de desenho

# %%
def pontos(ax, r: pd.DataFrame, ys: np.ndarray, cor: str = AZUL) -> None:
    """Um ponto por grupo e o IC de 95% como linha."""
    ax.hlines(ys, r["ic_inf"], r["ic_sup"], color=cor, alpha=0.45, lw=2, zorder=2)
    ax.scatter(r["taxa"], ys, s=46, color=cor, edgecolor=SUPERFICIE, linewidth=2, zorder=3)


def eixo_pct(ax, xmax: float) -> None:
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_formatter(FMT_PCT)
    ax.grid(axis="x", color=GRADE, lw=0.8)
    ax.set_axisbelow(True)


def anotar(ax, x: float, y: float, texto: str) -> None:
    ax.annotate(texto, (x, y), xytext=(6, 0), textcoords="offset points", va="center",
                fontsize=8.5, color=TINTA_2)


def titulo(fig, texto: str, sub: str) -> None:
    fig.suptitle(texto, x=0.0, y=1.0, ha="left", va="bottom", fontsize=12.5,
                 fontweight="bold", color=TINTA)
    fig.text(0.0, 0.985, sub, ha="left", va="top", fontsize=9.5, color=TINTA_2)


def secoes(fig, tabelas: dict, rotulo_x: str, referencia: float, casas: int = 2,
           xmax: float | None = None, xticks=None, linhas_sub: int = 1) -> None:
    """Painéis empilhados, um por segmento, com o mesmo eixo x e a média como linha cinza."""
    alturas = [len(r) + 0.8 for r in tabelas.values()]
    gs = {"height_ratios": alturas, "hspace": 0.55, "left": 0.27,
          "top": 0.985 - (0.42 + 0.18 * linhas_sub) / fig.get_figheight()}
    eixos = fig.subplots(len(tabelas), 1, sharex=True, gridspec_kw=gs)
    xmax = xmax or max(r["ic_sup"].max() for r in tabelas.values()) * 1.18
    for ax, (col, r) in zip(eixos, tabelas.items()):
        ys = np.arange(len(r))[::-1]
        ax.axvline(referencia, color=TINTA_MUDA, lw=1, zorder=1)
        pontos(ax, r, ys)
        ax.set_yticks(ys, [rotulo(col, k) for k in r.index])
        ax.set_ylim(-0.6, len(r) - 0.4)
        ax.set_title(TITULOS[col], pad=4)
        topo = r["taxa"].idxmax()
        anotar(ax, r.loc[topo, "ic_sup"], ys[list(r.index).index(topo)],
               pct(r.loc[topo, "taxa"], casas))
        eixo_pct(ax, xmax)
        if xticks is not None:
            ax.set_xticks(xticks)
    eixos[-1].set_xlabel(rotulo_x)


# %% [markdown]
# ## Dados

# %%
grid = pd.read_parquet(PROC / "grid.parquet")
t = grid[grid["split"] == "treino"].reset_index(drop=True)
noite = t[t["periodo"] == "noite"].reset_index(drop=True)
noite["quando"] = np.where(noite["fim_de_semana"] == 1, "sex_dom", "seg_qui")
print(f"Treino: {len(t):,} linhas, {int(t['y'].sum())} ocasiões, {t['COD_UPA'].nunique():,} UPAs")
print(f"À noite: {int(noite['y'].sum())} ocasiões ({noite['y'].sum() / t['y'].sum():.0%})")

md = ["# Tabelas da análise exploratória (só treino)", "",
      "Gerado por `notebooks/03_eda.py`. Taxas ponderadas por `peso_semana`, IC de 95% por "
      "bootstrap de "
      "UPAs. `Base` é o número de linhas do denominador; `Casos`, quantas têm o evento.", ""]


def registrar(titulo_tabela: str, r: pd.DataFrame, cols: list[str]) -> None:
    md.extend([f"## {titulo_tabela}", "", "| Grupo | Taxa | IC 95% | Base | Casos |",
               "|---|---|---|---|---|"])
    for k, row in r.iterrows():
        chave = k if isinstance(k, tuple) else (k,)
        nome = " · ".join(rotulo(c, v) for c, v in zip(cols, chave))
        md.append(f"| {nome} | {pct(row['taxa'])} | {pct(row['ic_inf'])} a {pct(row['ic_sup'])} "
                  f"| {milhar(row['n'])} | {milhar(row['casos'])} |")
    md.append("")
    print(f"\n{titulo_tabela}")
    print(r.round(3).to_string())


# %% [markdown]
# ## 1. Quando: dia × período

# %%
dp, _ = taxa(t, ["periodo", "dia_semana"], t["y"] == 1)
registrar("1. Taxa da ocasião por período e dia (baseline)", dp, ["periodo", "dia_semana"])
mat = dp["taxa"].unstack("dia_semana").reindex(index=PERIODOS, columns=DIAS)

fds, boot_fds = taxa(noite, ["quando"], noite["y"] == 1)
r_fds = razao(fds, boot_fds, "sex_dom", "seg_qui")
registrar("1b. Noite: sexta a domingo × segunda a quinta", fds, ["quando"])
print(f"Noite, sexta a domingo ÷ segunda a quinta: {r_fds[0]:.2f} "
      f"(IC {r_fds[1]:.2f} a {r_fds[2]:.2f})")

fig, ax = plt.subplots(figsize=(7.5, 2.7))
ax.imshow(mat.to_numpy(), cmap=LinearSegmentedColormap.from_list("azul", RAMPA_AZUL),
          vmin=0, aspect="auto")
vmax = np.nanmax(mat.to_numpy())
for i, per in enumerate(PERIODOS):
    for j, dia in enumerate(DIAS):
        v = mat.loc[per, dia]
        ax.text(j, i, pct(v), ha="center", va="center", fontsize=9,
                color="#ffffff" if v > 0.55 * vmax else TINTA)
ax.set_xticks(range(len(DIAS)), [ROTULOS["dia_semana"][d] for d in DIAS])
ax.set_yticks(range(len(PERIODOS)), [ROTULOS["periodo"][p] for p in PERIODOS])
ax.tick_params(length=0, colors=TINTA_2)
ax.spines["bottom"].set_visible(False)
titulo(fig, "A ocasião é da noite e cresce ao longo da semana",
       "Pessoas que comeram pizza e tomaram refrigerante no período (% ponderado, treino). "
       "Cada célula é o score do baseline.")
fig.savefig(FIG / "eda_1_dia_periodo.png")
plt.close(fig)

# %% [markdown]
# ## 2. Quem: perfil, à noite

# %%
media_noite = (noite["y"] * noite["peso_semana"]).sum() / noite["peso_semana"].sum() * 100
perfil = {}
for col in ["classe_abep", "tipo_area", "regiao", "faixa_idade", "sexo"]:
    r, _ = taxa(noite, [col], noite["y"] == 1)
    perfil[col] = r.sort_values("taxa", ascending=False) if col == "regiao" else r
    registrar(f"2. Noite, por {minusc(TITULOS[col])}", perfil[col], [col])

fig = plt.figure(figsize=(7.5, 8.2))
secoes(fig, perfil, "Pizza com refrigerante à noite (% ponderado das noites)", media_noite)
titulo(fig, "Classe, tipo de área e idade separam mais que região e sexo",
       "Taxa noturna da ocasião por segmento, com IC de 95% (treino). "
       f"A linha cinza é a média das noites ({pct(media_noite)}).")
fig.savefig(FIG / "eda_2_perfil.png")
plt.close(fig)

# %% [markdown]
# ## 3. Interação: fim de semana × classe

# %%
it, boot_it = taxa(noite, ["classe_abep", "quando"], noite["y"] == 1)
registrar("3. Noite, por classe e dia", it, ["classe_abep", "quando"])
classes = list(it.index.get_level_values("classe_abep").unique())
lift = {c: razao(it, boot_it, (c, "sex_dom"), (c, "seg_qui")) for c in classes}
md.extend(["## 3b. Sexta a domingo ÷ segunda a quinta, por classe (noite)", "",
           "| Classe | Razão | IC 95% |", "|---|---|---|"])
for c, (v, a, b) in lift.items():
    md.append(f"| {c} | {v:.2f} | {a:.2f} a {b:.2f} |".replace(".", ","))
    print(f"{c}: sexta a domingo ÷ segunda a quinta = {v:.2f} (IC {a:.2f} a {b:.2f})")
md.append("")

fig, ax = plt.subplots(figsize=(7.5, 4.0))
base_y = np.arange(len(classes))[::-1]
for quando, cor, desloc in [("seg_qui", AZUL, 0.17), ("sex_dom", LARANJA, -0.17)]:
    r = it.xs(quando, level="quando").reindex(classes)
    pontos(ax, r, base_y + desloc, cor)
xmax = it["ic_sup"].max() * 1.22
eixo_pct(ax, xmax)
ax.set_yticks(base_y, classes)
for c, y in zip(classes, base_y):
    ax.text(xmax, y, f"×{lift[c][0]:.1f}".replace(".", ","), ha="right", va="center",
            fontsize=9, color=TINTA_2)
ax.text(xmax, len(classes) - 0.35, "sex–dom ÷ seg–qui", ha="right", va="bottom", fontsize=8.5,
        color=TINTA_MUDA)
ax.set_xlabel("Pizza com refrigerante à noite (% ponderado das noites)")
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=AZUL, label="Segunda a quinta"),
                   Line2D([], [], marker="o", ls="", color=LARANJA, label="Sexta a domingo")],
          loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9,
          handletextpad=0.3, borderaxespad=0.2)
titulo(fig, "O salto de fim de semana muda de tamanho entre as classes",
       "Taxa noturna por classe ABEP e dia, com IC de 95% (treino). "
       "À direita, sex–dom ÷ seg–qui.\n"
       "Sem padrão claro por renda e com intervalos largos: um indício de interação, "
       "não uma prova.")
fig.subplots_adjust(top=0.80)
fig.savefig(FIG / "eda_3_classe_fim_de_semana.png")
plt.close(fig)

# %% [markdown]
# ## 4. Marca: pizza, refrigerante e guaraná por classe (noite)

# %%
p_pizza, _ = taxa(noite, ["classe_abep"], noite["pizza"])
p_refri, _ = taxa(noite, ["classe_abep"], noite["y"] == 1, den=noite["pizza"])
p_guar, _ = taxa(noite, ["classe_abep"], noite["y_guarana"] == 1, den=noite["y"] == 1)
g_reg, _ = taxa(noite, ["regiao"], noite["y_guarana"] == 1, den=noite["y"] == 1)
g_reg = g_reg.sort_values("taxa", ascending=False)
registrar("4a. Noites com pizza, por classe", p_pizza, ["classe_abep"])
registrar("4b. Pizzas da noite com refrigerante, por classe", p_refri, ["classe_abep"])
registrar("4c. Ocasiões da noite em que o refrigerante era guaraná, por classe", p_guar,
          ["classe_abep"])
registrar("4d. Ocasiões da noite em que o refrigerante era guaraná, por região", g_reg,
          ["regiao"])

paineis = [(p_pizza, "Comeu pizza", "% das noites", 1),
           (p_refri, "Pizza com refrigerante", "% das pizzas", 0),
           (p_guar, "O refrigerante era guaraná", "% das pizzas com refrigerante", 0)]
fig, eixos = plt.subplots(1, 3, figsize=(7.5, 3.6), sharey=True, gridspec_kw={"wspace": 0.25})
ys = np.arange(len(p_pizza))[::-1]
for ax, (r, nome, unidade, casas) in zip(eixos, paineis):
    r = r.reindex(p_pizza.index)
    pontos(ax, r, ys)
    eixo_pct(ax, min(r["ic_sup"].max() * 1.35, 100))
    ax.set_title(nome, pad=4)
    ax.set_xlabel(unidade)
    for c in ["DE", "A"]:
        i = list(r.index).index(c)
        anotar(ax, r["ic_sup"].iloc[i], ys[i], pct(r["taxa"].iloc[i], casas))
eixos[0].set_yticks(ys, list(p_pizza.index))
titulo(fig, "Quanto mais alta a classe, mais pizza e menos refrigerante junto",
       "Noites do treino, por classe ABEP, com IC de 95%. "
       "Cada painel é uma fatia do anterior: noites → pizzas → pizzas com refrigerante.")
fig.savefig(FIG / "eda_4_pizza_refrigerante_guarana.png")
plt.close(fig)

# %% [markdown]
# ## 5. Canal: onde a pizza é comida (todas as refeições com pizza)

# %%
pz = t[t["pizza"] & t["local_pizza"].notna()].reset_index(drop=True)
em_casa = pz["local_pizza"] == "casa"
media_casa = (em_casa * pz["peso_semana"]).sum() / pz["peso_semana"].sum() * 100
local = {}
for col in ["classe_abep", "tipo_area", "regiao"]:
    r, _ = taxa(pz, [col], em_casa)
    local[col] = r.sort_values("taxa", ascending=False) if col == "regiao" else r
    registrar(f"5. Pizzas comidas em casa, por {minusc(TITULOS[col])}", local[col], [col])

fig = plt.figure(figsize=(7.5, 5.6))
secoes(fig, local, "Pizzas comidas em casa (% ponderado das refeições com pizza)", media_casa,
       casas=0, xmax=112, xticks=range(0, 101, 20), linhas_sub=2)
titulo(fig, "Três em cada quatro pizzas são comidas em casa; a região muda mais que a classe",
       "Refeições com pizza no treino, qualquer período, com IC de 95%. "
       f"A linha cinza é a média ({pct(media_casa, 0)}).\n"
       "Delivery comido em casa conta como casa.")
fig.savefig(FIG / "eda_5_pizza_em_casa.png")
plt.close(fig)

# %%
TABELAS.write_text("\n".join(md), encoding="utf-8")
print(f"\n[ok] figuras em {FIG.relative_to(ROOT)}/ e tabelas em {TABELAS.relative_to(ROOT)}")
