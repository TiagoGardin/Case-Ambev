"""Fase 1 – Coesão e qualidade dos dados.

Rodar da raiz: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python audit/01_dados.py
Lê data/processed/*.parquet e a documentação da POF; não grava nada fora de audit/outputs/.
Nenhuma métrica de modelo usa o teste; nele só entram contagens e distribuições sem rótulo.
"""

from __future__ import annotations


import pandas as pd
from _comum import PROC, Relatorio, carregar_grid, taxa

from src.features import build_grid as bg
from src.features.mes_entrevista import validar_regra_pagamento
from src.ingestion.codigos_alimentos import (
    cadastro_produtos,
    codigos_pizza,
    codigos_refrigerante,
)
from src.ingestion.pof import CHAVE_DOM, CHAVE_PESSOA, CHAVE_UC, CONDOMINIO, PRESTACAO_IMOVEL
from src.models.preparo import UFS

r = Relatorio("01_dados")
r("# Fase 1 – Dados")
g = carregar_grid()
consumo = pd.read_parquet(PROC / "pof_consumo.parquet")
morador = pd.read_parquet(PROC / "pof_morador.parquet")
CHAVE = CHAVE_PESSOA + ["QUADRO", "periodo"]

# ---------------------------------------------------------------------------
r.secao("1.1 Versões do dataset: o grid salvo é o que o código gera hoje?")
novo = bg.montar_grid()  # em memória; não chama main(), não grava
try:
    pd.testing.assert_frame_equal(g.reset_index(drop=True), novo.reset_index(drop=True))
    r("OK: data/processed/grid.parquet é idêntico ao grid reconstruído em memória "
      f"({len(g):,} linhas × {g.shape[1]} colunas).")
except AssertionError as erro:
    r("DIVERGÊNCIA entre o grid salvo e o reconstruído:", str(erro)[:800])

# ---------------------------------------------------------------------------
r.secao("1.2 Schema, tipos e nulos")
r.tabela(g.dtypes.astype(str).rename("dtype"))
nulos = g.isna().groupby(g["split"]).sum().T
nulos = nulos[nulos.sum(axis=1) > 0]
r("Colunas com nulos, por split (local_pizza é nulo por desenho quando não há pizza):")
r.tabela(nulos)
features = ["periodo", "dia_semana", "UF", "urbano", "idade", "sexo", "anos_estudo",
            "n_moradores", "renda_pc"]
r("Nulos nas 9 features do modelo, por split:")
r.tabela(g[features].isna().groupby(g["split"]).sum())

# ---------------------------------------------------------------------------
r.secao("1.3 Duplicatas e chave pessoa × dia × período")
r(f"Duplicatas na chave {CHAVE}: {g.duplicated(CHAVE).sum()}")
r(f"QUADRO: {sorted(g['QUADRO'].unique())} | períodos: {list(g['periodo'].cat.categories)}")
dias = g.drop_duplicates(CHAVE_PESSOA + ["QUADRO"]).groupby(CHAVE_PESSOA).size()
r(f"Pessoas: {dias.size:,} ({(dias == 2).sum():,} com 2 dias, {(dias == 1).sum():,} com 1)")
r(f"Linhas por pessoa-dia: {g.groupby(CHAVE_PESSOA + ['QUADRO'], observed=True).size().unique()}")
inconsist = (g["y"] != (g["pizza"] & g["refri"]).astype(int)).sum()
r(f"y diferente de pizza & refri: {inconsist}; y_guarana > y: {(g['y_guarana'] > g['y']).sum()}")

# ---------------------------------------------------------------------------
r.secao("1.4 Valores impossíveis e extremos")
desc = g[["idade", "anos_estudo", "n_moradores", "renda_pc", "peso", "peso_semana"]].describe(
    percentiles=[0.001, 0.01, 0.5, 0.99, 0.999]).T
r.tabela(desc.round(2))
r(f"renda_pc = 0: {(g['renda_pc'] == 0).sum():,} linhas "
  f"({g.loc[g['renda_pc'] == 0].drop_duplicates(CHAVE_UC).shape[0]} UCs); "
  f"renda_pc < 0: {(g['renda_pc'] < 0).sum()}; peso ≤ 0: {(g['peso'] <= 0).sum()}")
uf_fora = sorted(set(g["UF"].astype(str)) - set(UFS))
r(f"UF fora das 27: {uf_fora} | UFs distintas: {g['UF'].nunique()} | "
  f"urbano: {sorted(g['urbano'].unique())} | sexo: {sorted(g['sexo'].dropna().unique())}")
r(f"Horas de consumo no registro bruto: min {consumo['hora'].min()}, max {consumo['hora'].max()}")

# ---------------------------------------------------------------------------
r.secao("1.5 Estrutura por split (sem desempenho de modelo)")
por_split = g.groupby("split").agg(
    linhas=("y", "size"), positivos=("y", "sum"),
    pessoas=("COD_INFORMANTE", lambda s: g.loc[s.index, CHAVE_PESSOA].drop_duplicates().shape[0]),
    domicilios=("NUM_DOM", lambda s: g.loc[s.index, CHAVE_DOM].drop_duplicates().shape[0]),
    upas=("COD_UPA", "nunique"))
r.tabela(por_split)
dom_lados = g.groupby(CHAVE_DOM)["split"].nunique()
upa_lados = g[g["split"] != "sem_mes"].groupby("COD_UPA")["split"].nunique()
r(f"Domicílios nos dois lados do split: {(dom_lados > 1).sum()} | "
  f"UPAs com domicílios no treino e no teste: {(upa_lados > 1).sum()} de {upa_lados.size}")

# ---------------------------------------------------------------------------
r.secao("1.6 Continuidade temporal e mês reconstruído da entrevista")
ucs = g.drop_duplicates(CHAVE_UC)
meses = (ucs.groupby(["ordem_mes", "split"], dropna=False).size().unstack(fill_value=0))
r("UCs por mês da coleta (ordem 0 = jul/2017 … 12 = jul/2018) e split:")
r.tabela(meses)
r("Origem do mês (UCs), por split:")
r.tabela(pd.crosstab(ucs["fonte_mes"].fillna("sem estimativa"), ucs["split"]))
upa_meses = ucs.dropna(subset=["ordem_mes"]).groupby("COD_UPA")["ordem_mes"].agg(["min", "max"])
amplitude = upa_meses["max"] - upa_meses["min"]
r(f"Amplitude do mês dentro da UPA: mediana {amplitude.median():.0f}, "
  f"P95 {amplitude.quantile(0.95):.0f}, máx {amplitude.max():.0f} meses; "
  f"UPAs com amplitude > 3 meses: {(amplitude > 3).sum()}")
alug = pd.read_parquet(PROC / "pof_aluguel_estimado.parquet")
pag = pd.read_parquet(PROC / "pof_pagamentos_moradia.parquet")
r("Validação da regra dos inquilinos (src/features/mes_entrevista.py:131), em UCs de mês "
  "conhecido que pagam prestação ou condomínio:")
r.tabela(validar_regra_pagamento(alug, pag, [PRESTACAO_IMOVEL, CONDOMINIO]).round(3), index=False)
inquilinos = ucs[ucs["fonte_mes"].isin(["aluguel_pago", "vizinhos_upa"])]
estimado = g["fonte_mes"].isin(["aluguel_pago", "vizinhos_upa"])
r(f"UCs com mês estimado por aluguel pago ou vizinhos: {len(inquilinos):,} "
  f"({len(inquilinos) / len(ucs):.1%} das UCs); positivos nessas UCs (treino): "
  f"{int(g.loc[(g['split'] == 'treino') & estimado, 'y'].sum())}")

# ---------------------------------------------------------------------------
r.secao("1.7 Definição do alvo: códigos de pizza e refrigerante")
cad = cadastro_produtos().set_index("codigo")["descricao"]
cod_pz, cod_rf = codigos_pizza(), codigos_refrigerante()
freq = consumo["V9001"].value_counts()
SUSPEITOS_PZ = (r"MOLHO|MASSA|BISCOITO|SALGADINHO|SABOR|TEMPERO|ESFIHA|ESFIRRA|P[AÃ]O|"
                r"TORRADA|FERMENTO")
SUSPEITOS_RF = r"\bPO\b|EM PO|XAROPE|REFRESCO|SUCO|MATE|CHA |ENERG|AGUA|ISOT|SODA"
for nome, cods, padrao in [("pizza", cod_pz, SUSPEITOS_PZ), ("refrigerante", cod_rf, SUSPEITOS_RF)]:
    t = pd.DataFrame({"codigo": sorted(cods)})
    t["descricao"] = t["codigo"].map(cad)
    t["registros_no_consumo"] = t["codigo"].map(freq).fillna(0).astype(int)
    t["suspeito"] = t["descricao"].fillna("").str.contains(padrao, regex=True)
    r(f"{nome}: {len(t)} códigos, {int((t['registros_no_consumo'] > 0).sum())} aparecem no consumo "
      f"({int(t['registros_no_consumo'].sum()):,} registros); suspeitos pelo nome: "
      f"{int(t['suspeito'].sum())}")
    r.tabela(t.sort_values("registros_no_consumo", ascending=False), index=False)
sem_cadastro = set(consumo["V9001"].unique()) - set(cad.index)
r(f"Códigos do consumo fora do cadastro: {len(sem_cadastro)} "
  f"({int(consumo['V9001'].isin(sem_cadastro).sum()):,} registros)")
# Itens que mencionam pizza ou refrigerante no cadastro e ficaram FORA das listas
for nome, padrao, cods in [("pizza", r"PIZZA", cod_pz),
                           ("refrigerante", r"REFRIG|GUARANA|COLA|SODA|TUBAINA|FANTA|SPRITE",
                            cod_rf)]:
    fora = cad[cad.str.contains(padrao, na=False, regex=True) & ~cad.index.isin(cods)]
    fora = fora.to_frame("descricao").assign(registros=lambda d: d.index.map(freq).fillna(0)
                                             .astype(int))
    r(f"Cadastro com '{padrao}' fora da lista de {nome}: {len(fora)} códigos, "
      f"{int(fora['registros'].sum()):,} registros no consumo")
    if len(fora):
        r.tabela(fora.sort_values("registros", ascending=False).head(25))

# ---------------------------------------------------------------------------
r.secao("1.8 Madrugada (0–4h) dentro da 'noite' do mesmo dia de recordatório")
itens = consumo.assign(periodo=bg.periodo_do_dia(consumo["hora"]),
                       pizza=consumo["V9001"].isin(cod_pz), refri=consumo["V9001"].isin(cod_rf),
                       madrugada=consumo["hora"] <= 4)
noite = itens[itens["periodo"] == "noite"].assign(
    pz_tarde_noite=lambda d: d["pizza"] & ~d["madrugada"],
    rf_tarde_noite=lambda d: d["refri"] & ~d["madrugada"],
    pz_madr=lambda d: d["pizza"] & d["madrugada"],
    rf_madr=lambda d: d["refri"] & d["madrugada"])
k = CHAVE_PESSOA + ["QUADRO"]
agg = noite.groupby(k)[["pz_tarde_noite", "rf_tarde_noite", "pz_madr", "rf_madr"]].max()
pos = agg[(agg["pz_tarde_noite"] | agg["pz_madr"]) & (agg["rf_tarde_noite"] | agg["rf_madr"])]
so_18_23 = pos[pos["pz_tarde_noite"] & pos["rf_tarde_noite"]]
r(f"Positivos noturnos: {len(pos):,}. Com pizza e refrigerante entre 18h e 23h: {len(so_18_23):,}. "
  f"Dependem de item da madrugada: {len(pos) - len(so_18_23):,} "
  f"({(len(pos) - len(so_18_23)) / len(pos):.1%}).")
r(f"Registros de consumo entre 0h e 4h: {int(itens['madrugada'].sum()):,} "
  f"({itens['madrugada'].mean():.2%} dos itens)")

# ---------------------------------------------------------------------------
r.secao("1.9 Joins da montagem do grid (src/features/build_grid.py:226)")
base = bg.base_pessoa_dia(consumo)
cruz = base.merge(pd.DataFrame({"periodo": bg.PERIODOS}), how="cross")
oc = bg.ocasioes_por_periodo(consumo)
j1 = cruz.merge(oc, on=CHAVE, how="left", indicator=True)
pp = bg.perfil_pessoa(morador)
r(f"pessoa-dias: {len(base):,} → grade × 4: {len(cruz):,} → + ocasiões: {len(j1):,} "
  f"(chaves de ocasião fora da grade: {len(oc) - (j1['_merge'] == 'both').sum()})")
sem_morador = (base[CHAVE_PESSOA].drop_duplicates()
               .merge(pp[CHAVE_PESSOA], how="left", indicator=True)["_merge"]
               .eq("left_only").sum())
r(f"perfil_pessoa: {len(pp):,} linhas, chave única: {not pp.duplicated(CHAVE_PESSOA).any()}; "
  f"pessoas do consumo sem cadastro de morador: {sem_morador}")
uc_tab = g.drop_duplicates(CHAVE_UC)
r(f"UCs no grid: {len(uc_tab):,}; sem mês (sem_mes): {int(uc_tab['ordem_mes'].isna().sum())}; "
  f"sem classe ABEP: {int(uc_tab['classe_abep'].isna().sum())}; "
  f"sem tipo de área: {int(uc_tab['tipo_area'].isna().sum())}")
mor10 = morador[morador["V0403"] >= 10].merge(base[CHAVE_DOM].drop_duplicates(), on=CHAVE_DOM)
com_rec = mor10.merge(base[CHAVE_PESSOA].drop_duplicates(), on=CHAVE_PESSOA, how="left",
                      indicator=True)
r(f"Moradores com 10+ anos nos domicílios do módulo de consumo: {len(mor10):,}; com recordatório: "
  f"{(com_rec['_merge'] == 'both').mean():.1%} (o restante não respondeu; o PESO_FINAL do consumo "
  "é calibrado para o módulo)")

# ---------------------------------------------------------------------------
r.secao("1.10 Unidades: renda per capita × renda disponível per capita do IBGE")
uc_renda = (g.drop_duplicates(CHAVE_UC)[CHAVE_UC + ["renda_pc"]]
            .merge(morador.drop_duplicates(CHAVE_UC)[CHAVE_UC + ["RENDA_DISP_PC"]], on=CHAVE_UC))
ok = uc_renda["RENDA_DISP_PC"] > 0
razao = uc_renda.loc[ok, "renda_pc"] / uc_renda.loc[ok, "RENDA_DISP_PC"]
r(f"renda_pc / RENDA_DISP_PC por UC: mediana {razao.median():.2f}, P10 {razao.quantile(0.1):.2f}, "
  f"P90 {razao.quantile(0.9):.2f}; correlação de Spearman "
  f"{uc_renda.loc[ok, ['renda_pc', 'RENDA_DISP_PC']].corr(method='spearman').iloc[0, 1]:.3f}")
r(f"Mediana da renda_pc (UCs): R$ {uc_renda['renda_pc'].median():,.0f} por mês; salário mínimo "
  f"usado nas faixas: R$ {bg.SALARIO_MINIMO_2018:,.0f}")

# ---------------------------------------------------------------------------
r.secao("1.11 peso_semana: cada dia da semana com 1/7 do peso em cada split")
pdias = g.drop_duplicates(CHAVE_PESSOA + ["QUADRO"])
for s in ["treino", "teste"]:
    d = pdias[pdias["split"] == s]
    amostra = d.groupby("dia_semana", observed=True).size() / len(d)
    fator = (d.groupby("dia_semana", observed=True)["peso_semana"].sum()
             / d.groupby("dia_semana", observed=True)["peso"].sum())
    fatia = d.groupby("dia_semana", observed=True)["peso_semana"].sum() / d["peso_semana"].sum()
    r(f"{s}:")
    r.tabela(pd.DataFrame({"fatia dos dias na amostra": amostra.round(3),
                           "fator do peso": fator.round(2),
                           "fatia com peso_semana": fatia.round(4)}))

# ---------------------------------------------------------------------------
r.secao("1.12 Diferenças de composição entre treino e teste (sem rótulo)")
tt = g[g["split"].isin(["treino", "teste"])]
for col in ["regiao", "urbano", "tipo_area", "faixa_idade", "sexo", "faixa_renda", "classe_abep"]:
    comp = (tt.groupby(["split", col], observed=True)["peso_semana"].sum()
            .groupby(level="split").transform(lambda s: s / s.sum()).unstack(0))
    comp["teste − treino (p.p.)"] = (comp["teste"] - comp["treino"]) * 100
    r(f"{col} (fatia ponderada por peso_semana):")
    r.tabela(comp.round(3))
treino = g[g["split"] == "treino"]
r(f"Taxa ponderada do treino: {taxa(treino['y'], treino['peso_semana']):.4%}. "
  "A do teste não é recalculada aqui.")

r.gravar()
