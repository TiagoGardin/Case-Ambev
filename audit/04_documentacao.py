"""Fase 4 – Documentação × código e reprodutibilidade.

Rodar da raiz: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python audit/04_documentacao.py
Não executa scripts que gravam arquivos: o treino do modelo final é refeito em memória.
"""

from __future__ import annotations

import hashlib
import io
import re
import subprocess
import sys
from importlib import metadata

import joblib
import numpy as np
import pandas as pd
from _comum import ROOT, Relatorio, carregar_grid, taxa

from src.ingestion.pof import CHAVE_DOM, CHAVE_PESSOA
from src.models.artefato import carregar
from src.models.calibracao import ModeloCalibrado, aplicar_sigmoide, ajustar_sigmoide
from src.models.metricas import percentil_ponderado
from src.models.modelos import construir
from src.models.preparo import FEATURES
from src.models.treinar import CONFIG, config_final
from src.models.validacao import domicilio, particoes, validar_cv

r = Relatorio("04_documentacao")
r("# Fase 4 – Documentação × código e reprodutibilidade")
g = carregar_grid()
treino = g[g["split"] == "treino"].reset_index(drop=True)
X, y, w = treino[FEATURES], treino["y"].to_numpy(), treino["peso_semana"].to_numpy()

# ---------------------------------------------------------------------------
r.secao("4.1 O modelo commitado é reproduzível a partir do código?")
modelo, meta = carregar()
folds = particoes(y, domicilio(treino))
cv = validar_cv(construir(CONFIG), X, y, w, folds)
score = cv.oof.ravel(order="F")
y_rep, w_rep = np.tile(y, cv.oof.shape[1]), np.tile(w, cv.oof.shape[1])
a, b = ajustar_sigmoide(score, y_rep, w_rep)
cal = aplicar_sigmoide(score, a, b)
cortes = {"alta": percentil_ponderado(cal, w_rep, 0.90),
          "media": percentil_ponderado(cal, w_rep, 0.70)}
novo = ModeloCalibrado(construir(config_final(CONFIG)).fit(X, y), a, b)
dif = np.abs(novo.predict_proba(X)[:, 1] - modelo.predict_proba(X)[:, 1]).max()
buf = io.BytesIO()
joblib.dump(novo, buf)
r.tabela(pd.DataFrame({
    "item": ["a da sigmoide", "b da sigmoide", "corte alta", "corte media",
             "PR-AUC média na validação cruzada", "maior diferença de score no treino",
             "SHA-256 do arquivo do modelo"],
    "metadados": [meta["calibration"]["a"], meta["calibration"]["b"],
                  meta["classification"]["cuts"]["alta"], meta["classification"]["cuts"]["media"],
                  meta["cross_validation"]["prauc_mean"], "", meta["model_file_sha256"][:16]],
    "reproduzido": [a, b, cortes["alta"], cortes["media"], cv.prauc.mean(), f"{dif:.2e}",
                    hashlib.sha256(buf.getvalue()).hexdigest()[:16]]}), index=False)

# ---------------------------------------------------------------------------
r.secao("4.2 Números citados no README e no Data Card")
pessoas = g[CHAVE_PESSOA].drop_duplicates().shape[0]
tr, te = g[g["split"] == "treino"], g[g["split"] == "teste"]
pos_te_dom = te.loc[te["y"] == 1, CHAVE_DOM].drop_duplicates().shape[0]
noite = tr[tr["periodo"] == "noite"]


def taxa_noite(filtro) -> float:
    d = noite[filtro(noite)]
    return taxa(d["y"], d["peso_semana"])


fds = noite["dia_semana"].astype(str).isin(["Sexta-Feira", "Sabado", "Domingo"])
pizza_tr = tr[tr["pizza"] & tr["local_pizza"].notna()]


def casa(d: pd.DataFrame) -> float:
    return taxa(d["local_pizza"].eq("casa"), d["peso_semana"])


afirm = [
    ("README: pessoas no recordatório", "46.164", f"{pessoas:,}"),
    ("README: linhas e ocasiões", "340.072 e 838", f"{len(g):,} e {int(g['y'].sum())}"),
    ("Data Card: treino (linhas, positivos)", "223.748 e 523",
     f"{len(tr):,} e {int(tr['y'].sum())}"),
    ("Data Card: teste (linhas, positivos)", "116.316 e 315",
     f"{len(te):,} e {int(te['y'].sum())}"),
    ("Data Card: domicílios com positivos no teste", "189", f"{pos_te_dom}"),
    ("README: taxa do treino (peso_semana)", "0,30%", f"{taxa(tr['y'], tr['peso_semana']):.2%}"),
    ("README: taxa do teste (peso_semana)", "0,41%", f"{taxa(te['y'], te['peso_semana']):.2%}"),
    ("README: fatia das ocasiões à noite", "92%",
     f"{(noite['y'] * noite['peso_semana']).sum() / (tr['y'] * tr['peso_semana']).sum():.0%}"),
    ("README: taxa noturna na segunda", "0,50%",
     f"{taxa_noite(lambda d: d['dia_semana'].astype(str).eq('Segunda-Feira')):.2%}"),
    ("README: taxa noturna no sábado", "2,09%",
     f"{taxa_noite(lambda d: d['dia_semana'].astype(str).eq('Sabado')):.2%}"),
    ("README: sexta a domingo ÷ segunda a quinta (noite)", "2,4",
     f"{taxa_noite(lambda d: fds) / taxa_noite(lambda d: ~fds):.1f}"),
    ("README: noite, classe DE e B1", "0,34% e 4,40%",
     f"{taxa_noite(lambda d: d['classe_abep'].astype(str).eq('DE')):.2%} e "
     f"{taxa_noite(lambda d: d['classe_abep'].astype(str).eq('B1')):.2%}"),
    ("README: noite, rural e capital", "0,23% e 1,71%",
     f"{taxa_noite(lambda d: d['tipo_area'].astype(str).eq('rural')):.2%} e "
     f"{taxa_noite(lambda d: d['tipo_area'].astype(str).eq('capital')):.2%}"),
    ("README: noite, 60+ e 18–29", "0,61% e 1,82%",
     f"{taxa_noite(lambda d: d['faixa_idade'].astype(str).eq('60_mais')):.2%} e "
     f"{taxa_noite(lambda d: d['faixa_idade'].astype(str).eq('18_29')):.2%}"),
    ("README: pizzas comidas em casa", "75%", f"{casa(pizza_tr):.0%}"),
    ("README: em casa, Sudeste e Norte", "80% e 56%",
     f"{casa(pizza_tr[pizza_tr['regiao'] == 'Sudeste']):.0%} e "
     f"{casa(pizza_tr[pizza_tr['regiao'] == 'Norte']):.0%}"),
]
r.tabela(pd.DataFrame(afirm, columns=["afirmação", "documentado", "recalculado"]), index=False)

# ---------------------------------------------------------------------------
r.secao("4.3 Ambiente: versões instaladas × requirements.txt")
linhas = []
for linha in (ROOT / "requirements.txt").read_text().splitlines():
    m = re.match(r"^([A-Za-z0-9_.-]+)==(\S+)", linha.strip())
    if not m:
        continue
    try:
        inst = metadata.version(m.group(1))
    except metadata.PackageNotFoundError:
        inst = "não instalado"
    linhas.append({"pacote": m.group(1), "requirements": m.group(2), "instalado": inst,
                   "confere": inst == m.group(2)})
linhas.append({"pacote": "python", "requirements": (ROOT / ".python-version").read_text().strip(),
               "instalado": sys.version.split()[0], "confere": sys.version.startswith(
                   (ROOT / ".python-version").read_text().strip())})
r.tabela(pd.DataFrame(linhas), index=False)
nao_fixados = sorted({"joblib", "numpy"} - {d["pacote"] for d in linhas})
r(f"Importados pelo código e ausentes do requirements.txt (vêm como dependência): {nao_fixados}")

# ---------------------------------------------------------------------------
r.secao("4.4 Caminhos fixos, sementes e testes")
achados = subprocess.run(["grep", "-rnI", "--include=*.py", "-E", r"/Users/|/home/|C:\\\\",
                          "src", "notebooks", "tests"],
                         cwd=ROOT, capture_output=True, text=True).stdout.strip()
r(f"Caminhos absolutos no código: {achados or 'nenhum'}")
sementes = subprocess.run(["grep", "-rnI", "--include=*.py", "-E",
                           r"random_state|default_rng|seed", "src", "notebooks"],
                          cwd=ROOT, capture_output=True, text=True).stdout.strip()
r("Sementes declaradas:")
r("```")
r(sementes)
r("```")
pytest = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=ROOT,
                        capture_output=True, text=True)
r(f"pytest: {pytest.stdout.strip().splitlines()[-1]}")

# ---------------------------------------------------------------------------
r.secao("4.5 Trechos dos documentos usados nas divergências (arquivo:linha)")
for arquivo, padrao in [
    ("README.md", r"única vez|Generalização por região|SHAP|data/external|em construção|"
                  r"interação período × dia|quando \(período, dia da semana\)"),
    ("docs/data_card.md", r"A preencher|triagem|Open-Meteo"),
]:
    saida = subprocess.run(["grep", "-n", "-E", padrao, arquivo], cwd=ROOT, capture_output=True,
                           text=True).stdout.strip()
    r(f"{arquivo}:")
    r("```")
    r("\n".join(lin[:220] for lin in saida.splitlines()))
    r("```")
vazias = [p for p in ["src/api", "src/decision_engine", "app", "data/external"]
          if not [f for f in (ROOT / p).iterdir() if f.name not in {".gitkeep", "__init__.py",
                                                                      "__pycache__"}]]
r(f"Pastas citadas na arquitetura e ainda vazias: {vazias}")

r.gravar()
