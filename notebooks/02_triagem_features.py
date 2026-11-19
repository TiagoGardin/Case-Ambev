# %% [markdown]
# # 02 – Triagem de features candidatas (só com o treino)
#
# Pergunta: além das 9 features de partida, algum grupo de variáveis candidatas
# melhora a previsão? Esta triagem substitui a versão exploratória de 16/11/2026,
# que media o ganho no período de teste.
#
# Protocolo, fixado antes de rodar:
# - **Dados:** só o treino (jul/2017–fev/2018). O teste não é lido.
# - **Validação cruzada repetida:** `StratifiedGroupKFold` com 5 folds, agrupada
#   por domicílio, repetida 10 vezes com partições diferentes (50 folds). As mesmas
#   partições valem para todas as comparações.
# - **Métrica:** PR-AUC ponderada por `peso_semana` (PESO_FINAL com cada dia da semana
#   valendo 1/7), em cada fold.
# - **Modelos:** LightGBM raso e determinístico, e Regressão Logística simples.
# - **Regra:** um grupo entra se o ganho de PR-AUC for significativo a 5%
#   (unilateral) **nos dois modelos**, pelo teste t corrigido para validação
#   cruzada repetida (Nadeau e Bengio, 2003). Senão é empate, e fica a opção mais
#   simples (sem o grupo; no caso do tipo de área, fica `urbano`).
# - **Mais de um grupo aprovado:** entram em ordem de t médio, e cada um precisa
#   passar de novo sobre o conjunto já atualizado.
# - **Renda depois da classe ABEP:** se a classe entrar, a renda só fica se ainda
#   acrescentar ganho pela mesma regra.
#
# Por que repetir: com uma única partição em 5 folds, a decisão se mostrou
# instável. Mover de lado no split um domicílio sem positivos (20 linhas) trocou
# as partições e inverteu a decisão de um grupo.
#
# Fora por desenho, sem medição: dieta, IMC e insegurança alimentar (dados
# sensíveis), consumo do outro dia de recordatório (o marketing não tem esse dado
# na aplicação) e clima mensal (a POF não tem a data exata).
#
# Rodar: `python notebooks/02_triagem_features.py` (cerca de 10 minutos).
# Pré-requisito: `python -m src.features.build_grid`.

# %%
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

from src.features.classe_abep import (  # noqa: E402
    COD_AUTOMOVEL,
    COD_LAVA_ROUPA,
    COD_MICRO_ONDAS,
    COD_MICROCOMPUTADOR,
    COD_MOTOCICLETA,
)
from src.ingestion.pof import CHAVE_DOM, CHAVE_PESSOA, CHAVE_UC, ler_registro  # noqa: E402
from src.models.validacao import K, REPETICOES, comparar, domicilio, particoes  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
PROC = ROOT / "data" / "processed"

BASE = ["periodo", "dia_semana", "UF", "urbano", "idade", "sexo", "anos_estudo",
        "n_moradores", "renda_pc"]
GRUPOS = {  # nome: (colunas que entram, colunas que saem)
    "Tipo de área no lugar de urbano": (["tipo_area"], ["urbano"]),
    "Classe ABEP": (["classe_abep"], []),
    "Composição da família": (["n_criancas", "n_adolesc", "n_adultos", "n_idosos",
                               "posicao"], []),
    "Bens, finanças e moradia": (["tem_cartao", "uc_tem_cartao", "plano_saude", "tem_conta",
                                  "banheiros", "carro", "computador", "microondas", "ar_cond",
                                  "lava_roupa", "moto", "apartamento", "alugado"], []),
    "Trabalho e estudo": (["trabalha", "estudante"], []),
}
CATEGORICAS = ["periodo", "dia_semana", "UF", "sexo", "tipo_area", "classe_abep", "posicao"]

# Sem amostragem de linhas e colunas: o LightGBM fica determinístico, e a diferença
# entre dois conjuntos de features vem só das features
PARAMS_LGB = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100,
                  reg_lambda=5, deterministic=True, force_col_wise=True, random_state=0,
                  verbose=-1)

COD_AR_CONDICIONADO = 1402101
BENS = {COD_AUTOMOVEL: "carro", COD_MICROCOMPUTADOR: "computador", COD_MICRO_ONDAS: "microondas",
        COD_AR_CONDICIONADO: "ar_cond", COD_LAVA_ROUPA: "lava_roupa", COD_MOTOCICLETA: "moto"}
POSICAO = {1: "referencia", 2: "conjuge", 3: "conjuge", 4: "filho", 5: "filho", 6: "filho"}
LAYOUT_MORADOR_EXTRA = {
    "COD_UPA": (8, 9), "NUM_DOM": (17, 2), "NUM_UC": (19, 1), "COD_INFORMANTE": (20, 2),
    "V0406": (38, 1),  # tem plano ou seguro-saúde: 1 sim
    "V0407": (39, 1),  # teve rendimento ou trabalhou: 1 sim
    "V0409": (41, 2),  # nº de cartões de crédito
    "V0410": (43, 1),  # nº de contas correntes
    "V0415": (49, 1),  # frequenta escola ou creche: 1 sim
}
LAYOUT_DOMICILIO_EXTRA = {
    "COD_UPA": (8, 9), "NUM_DOM": (17, 2),
    "V0201": (19, 1),  # tipo: 2 = apartamento
    "V0217": (47, 1),  # condição de ocupação: 3 = alugado
}


# %% [markdown]
# ## 1. Candidatas
# Variáveis que não fazem parte do pipeline. Saem dos parquets processados e,
# quando a coluna não foi lida no pipeline, direto dos arquivos da POF.

# %%
def contagem(texto: pd.Series) -> pd.Series:
    return pd.to_numeric(texto, errors="coerce").fillna(0)


def juntar_candidatas(g: pd.DataFrame) -> pd.DataFrame:
    mor = pd.read_parquet(PROC / "pof_morador.parquet")

    # Composição da UC, sem empregados domésticos e parentes deles (V0306 18 e 19)
    m = mor[~mor["V0306"].isin([18, 19])]
    idade = m["V0403"]
    faixas = ["n_criancas", "n_adolesc", "n_adultos", "n_idosos"]
    comp = (m.assign(n_criancas=idade < 10, n_adolesc=idade.between(10, 17),
                     n_adultos=idade.between(18, 59), n_idosos=idade >= 60)
             .groupby(CHAVE_UC, as_index=False)[faixas].sum())
    pos = mor[CHAVE_PESSOA].assign(posicao=mor["V0306"].map(POSICAO).fillna("outro"))

    # Trabalho, estudo e finanças da pessoa
    bruto = ler_registro("MORADOR.txt", LAYOUT_MORADOR_EXTRA)
    pes = bruto[CHAVE_PESSOA].assign(
        plano_saude=(bruto["V0406"] == "1").astype(int),
        trabalha=(bruto["V0407"] == "1").astype(int),
        tem_cartao=(contagem(bruto["V0409"]) > 0).astype(int),
        tem_conta=(contagem(bruto["V0410"]) > 0).astype(int),
        estudante=(bruto["V0415"] == "1").astype(int),
    )
    uc_cartao = (pes.groupby(CHAVE_UC, as_index=False)["tem_cartao"].max()
                    .rename(columns={"tem_cartao": "uc_tem_cartao"}))

    # Moradia: tipo, condição de ocupação e banheiros
    dbruto = ler_registro("DOMICILIO.txt", LAYOUT_DOMICILIO_EXTRA)
    dom = dbruto[CHAVE_DOM].assign(apartamento=(dbruto["V0201"] == "2").astype(int),
                                   alugado=(dbruto["V0217"] == "3").astype(int))
    banheiros = (pd.read_parquet(PROC / "pof_domicilio.parquet")[CHAVE_DOM + ["V02111"]]
                   .rename(columns={"V02111": "banheiros"}))

    # Bens duráveis (quantidade na UC)
    inv = pd.read_parquet(PROC / "pof_inventario.parquet")
    inv = inv[inv["V9001"].isin(BENS)]
    bens = (inv.assign(bem=inv["V9001"].map(BENS))
               .pivot_table(index=CHAVE_UC, columns="bem", values="V9005",
                            aggfunc="sum", fill_value=0)
               .reindex(columns=list(BENS.values()), fill_value=0)
               .reset_index())
    bens.columns.name = None

    out = (g.merge(comp, on=CHAVE_UC, how="left")
            .merge(pos, on=CHAVE_PESSOA, how="left")
            .merge(pes, on=CHAVE_PESSOA, how="left")
            .merge(uc_cartao, on=CHAVE_UC, how="left")
            .merge(dom, on=CHAVE_DOM, how="left")
            .merge(banheiros, on=CHAVE_DOM, how="left")
            .merge(bens, on=CHAVE_UC, how="left"))
    out[list(BENS.values())] = out[list(BENS.values())].fillna(0)
    assert len(out) == len(g), "a junção duplicou linhas"
    return out


grid = pd.read_parquet(PROC / "grid.parquet")
treino = grid[grid["split"] == "treino"].reset_index(drop=True)
df = juntar_candidatas(treino)
for c in CATEGORICAS:
    df[c] = df[c].astype(str).astype("category")

novas = [c for entra, _ in GRUPOS.values() for c in entra]
nulos = df[novas].isna().sum()
print(f"Treino: {len(df):,} linhas, {int(df['y'].sum())} positivos")
print("Nulos nas candidatas:", nulos[nulos > 0].to_dict() or "nenhum")


# %% [markdown]
# ## 2. Taxas descritivas nas noites (só treino)

# %%
def taxa_noite(d: pd.DataFrame, col: str) -> pd.DataFrame:
    n = d[d["periodo"] == "noite"].assign(yw=lambda x: x["y"] * x["peso_semana"])
    t = n.groupby(col, observed=True).agg(yw=("yw", "sum"), w=("peso_semana", "sum"),
                                          positivos=("y", "sum"))
    t["taxa noturna (%)"] = (t["yw"] / t["w"] * 100).round(2)
    return t[["taxa noturna (%)", "positivos"]]


for col in ["tipo_area", "classe_abep", "faixa_renda"]:
    t = taxa_noite(treino, col)
    razao = t["taxa noturna (%)"].max() / t["taxa noturna (%)"].min()
    print(f"\n[{col}] maior ÷ menor = {razao:.1f} vezes")
    print(t.to_string())

# %% [markdown]
# ## 3. Validação cruzada repetida

# %%
FOLDS = particoes(df["y"], domicilio(df))  # as mesmas partições da modelagem
pos_fold = [int(df["y"].iloc[va].sum()) for _, va in FOLDS]
print(f"{len(FOLDS)} folds ({REPETICOES} repetições × {K}); "
      f"positivos por fold: {min(pos_fold)} a {max(pos_fold)}")


def prever_lgb(Xtr: pd.DataFrame, ytr: pd.Series, Xva: pd.DataFrame) -> np.ndarray:
    m = lgb.LGBMClassifier(**PARAMS_LGB)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xva)[:, 1]


def prever_lr(Xtr: pd.DataFrame, ytr: pd.Series, Xva: pd.DataFrame) -> np.ndarray:
    """Regressão Logística simples: one-hot, idade e idade², log da renda."""
    A = pd.concat([Xtr, Xva])
    cat = [c for c in A.columns if c in CATEGORICAS]
    for c in A.columns.difference(cat):
        A[c] = A[c].astype(float)
        if A[c].isna().any():
            A[c + "_na"] = A[c].isna().astype(float)
            A[c] = A[c].fillna(A[c].iloc[: len(Xtr)].median())
    if "renda_pc" in A:
        A["renda_pc"] = np.log1p(A["renda_pc"].clip(lower=0))
    if "idade" in A:
        A["idade2"] = A["idade"] ** 2
    A = pd.get_dummies(A, columns=cat, dtype=float)
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))
    m.fit(A.iloc[: len(Xtr)], ytr)
    return m.predict_proba(A.iloc[len(Xtr):])[:, 1]


_cache: dict = {}


def prauc(cols: list[str], algo: str) -> np.ndarray:
    """PR-AUC ponderada por `peso_semana` em cada um dos 50 folds."""
    chave = (tuple(sorted(cols)), algo)
    if chave not in _cache:
        prever = prever_lgb if algo == "lgb" else prever_lr
        r = []
        for tr, va in FOLDS:
            a, b = df.iloc[tr], df.iloc[va]
            p = prever(a[cols], a["y"], b[cols])
            r.append(average_precision_score(b["y"], p, sample_weight=b["peso_semana"]))
        _cache[chave] = np.array(r)
    return _cache[chave]


def avaliar(ref_cols: list[str], cand_cols: list[str]) -> tuple[dict, dict, bool]:
    lg = comparar(prauc(ref_cols, "lgb"), prauc(cand_cols, "lgb"))
    lr = comparar(prauc(ref_cols, "lr"), prauc(cand_cols, "lr"))
    return lg, lr, lg["passa"] and lr["passa"]


def aplicar(cols: list[str], entra: list[str], sai: list[str]) -> list[str]:
    return [c for c in cols if c not in sai] + [c for c in entra if c not in cols]


def resumo(lg: dict, lr: dict) -> str:
    return (f"LightGBM {lg['ganho']:+.4f} (p = {lg['p']:.2f}) | "
            f"LR {lr['ganho']:+.4f} (p = {lr['p']:.2f})")


# %% [markdown]
# ## 4. Cada grupo contra a base

# %%
print(f"Base (9 features): PR-AUC média {prauc(BASE, 'lgb').mean():.4f} no LightGBM, "
      f"{prauc(BASE, 'lr').mean():.4f} na LR")

linhas = []
for nome, (entra, sai) in GRUPOS.items():
    lg, lr, ok = avaliar(BASE, aplicar(BASE, entra, sai))
    linhas.append({"grupo": nome, "ganho LGB": lg["ganho"], "p LGB": lg["p"],
                   "ganho LR": lr["ganho"], "p LR": lr["p"], "entra": ok,
                   "t médio": (lg["t"] + lr["t"]) / 2})
    print(f"  {nome}: {resumo(lg, lr)}", flush=True)
individual = pd.DataFrame(linhas)
print()
print(individual.drop(columns="t médio").round(4).to_string(index=False))

# %% [markdown]
# ## 5. Aprovados entram um de cada vez, sobre o conjunto atualizado

# %%
atual = list(BASE)
aprovados = individual[individual["entra"]].sort_values("t médio", ascending=False)["grupo"]
for nome in aprovados:
    entra, sai = GRUPOS[nome]
    cols = aplicar(atual, entra, sai)
    lg, lr, ok = avaliar(atual, cols)
    print(f"{nome} sobre o conjunto atual: {resumo(lg, lr)} → {'entra' if ok else 'empate'}")
    if ok:
        atual = cols

# %% [markdown]
# ## 6. A renda ainda acrescenta depois da classe ABEP?

# %%
if "classe_abep" in atual:
    sem_renda = [c for c in atual if c != "renda_pc"]
    lg, lr, ok = avaliar(sem_renda, atual)  # ganho de manter a renda
    print(f"Manter a renda junto com a classe: {resumo(lg, lr)} → "
          f"{'a renda fica' if ok else 'empate, a renda sai'}")
    if not ok:
        atual = sem_renda
else:
    print("A classe ABEP não entrou; a renda continua.")

# %% [markdown]
# ## 7. Resultado

# %%
print("Conjunto final:", atual)
if atual != BASE:
    lg, lr, _ = avaliar(BASE, atual)
    print(f"Final contra a base: {resumo(lg, lr)}")
