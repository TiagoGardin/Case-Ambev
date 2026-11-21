"""Núcleo da API: carrega os artefatos uma vez e pontua um contexto.

Artefatos, todos em models/:
- o modelo congelado e os metadados (src.models.artefato confere o hash do modelo);
- a referência do SHAP por fator (src.models.explicar confere que é do mesmo modelo);
- o contexto observado do Decision Engine (src.decision_engine.contexto), que vem dos
  dados do treino e não depende do modelo.

`pontuar` faz o caminho inteiro de uma requisição: features → score calibrado →
classe → os 3 fatores do SHAP → regra do Decision Engine. Devolve os 6 campos do
contrato da API.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.decision_engine import contexto as contexto_decisao
from src.decision_engine.regras import VERSAO as VERSAO_REGRAS
from src.decision_engine.regras import recomendar
from src.models import artefato, explicar
from src.models.preparo import FEATURES, UFS, preparar

SIGLA_PARA_CODIGO = {sigla: codigo for codigo, sigla in UFS.items()}
AVISOS = [
    "O score é a probabilidade calibrada de pizza com refrigerante no mesmo período, para "
    "uma pessoa num dia e período (POF 2017-2018).",
    "A recomendação vem de regras documentadas (docs/decision_engine.md). Ela indica uma "
    "categoria de ativação coerente com o contexto; não estima efeito causal nem venda "
    "incremental.",
    "Os fatores são o SHAP exato da Regressão Logística, em relação ao contexto médio do treino.",
]


@dataclass(frozen=True)
class Recursos:
    modelo: object
    metadados: dict
    explicador: explicar.Explicador
    contexto: dict


@lru_cache(maxsize=1)
def recursos(pasta: str = str(artefato.PASTA)) -> Recursos:
    """Carrega e confere os artefatos uma vez por processo."""
    modelo, metadados = artefato.carregar(Path(pasta))
    return Recursos(modelo, metadados, explicar.carregar(modelo, metadados, Path(pasta)),
                    contexto_decisao.carregar(Path(pasta)))


def metadados_publicos(meta: dict) -> dict:
    """O que a API devolve sobre o modelo em cada resposta."""
    return {
        "model_name": meta["model_name"],
        "model_version": meta["model_version"],
        "trained_at": meta["trained_at"],
        "algorithm": meta["algorithm"],
        "model_file_sha256": meta["model_file_sha256"],
        "training_period": meta["training_data"]["period"].split(" (")[0],
        "reference_rate": meta["reference_rate"],
        "classification_cuts": meta["classification"]["cuts"],
        "decision_engine_version": VERSAO_REGRAS,
    }


def pontuar(entrada: dict, rec: Recursos) -> dict:
    """Uma requisição já validada → os 6 campos do contrato.

    `entrada` tem as 9 features com a UF em sigla (SP, BA…), `urbano` booleano e, para o
    Decision Engine, `canal` e `associacao_marca` (podem ser None).
    """
    linha = {c: entrada[c] for c in FEATURES if c != "UF"}
    linha["UF"] = SIGLA_PARA_CODIGO[entrada["uf"]]
    linha["urbano"] = int(bool(entrada["urbano"]))
    X = pd.DataFrame([linha])[FEATURES]
    meta = rec.metadados
    score = float(rec.modelo.predict_proba(X)[0, 1])
    classe = str(artefato.classificar([score], meta["classification"]["cuts"])[0])
    top = rec.explicador.top_fatores(X)[0]
    regiao = str(preparar(X)["regiao"].iloc[0])
    decisao = recomendar(score, classe, top, regiao, rec.contexto, meta["reference_rate"],
                         entrada.get("canal"), entrada.get("associacao_marca"))
    return {
        "occasion_score": score,
        "classification": classe,
        "top_factors": top,
        "recommendation_category": decisao.categoria,
        "recommendation_reason": decisao.justificativa,
        "model_metadata": {**metadados_publicos(meta), "decision_rule": decisao.regra,
                           "region": regiao, "notes": AVISOS},
    }
