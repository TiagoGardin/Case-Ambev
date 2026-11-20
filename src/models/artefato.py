"""Modelo final congelado: arquivo, metadados e classificação do score.

- models/occasion_model.joblib: o ModeloCalibrado (preparo + modelo + sigmoide),
  que recebe as 9 features brutas;
- models/model_metadata.json: configuração escolhida na validação cruzada, cortes
  de `alta` e `media`, métricas, versões das bibliotecas e o SHA-256 do arquivo do
  modelo.

A avaliação no teste, a API e o app carregam os dois por `carregar`, que confere o
hash. Os dois arquivos vão para o git: o commit registra o modelo congelado antes
da avaliação no teste.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

PASTA = Path(__file__).resolve().parents[2] / "models"
ARQUIVO_MODELO = "occasion_model.joblib"
ARQUIVO_METADADOS = "model_metadata.json"
CLASSES = ("baixa", "media", "alta")


def classificar(score, cortes: dict) -> np.ndarray:
    """`alta` a partir do corte de alta (P90), `media` a partir do de média (P70), senão `baixa`."""
    s = np.asarray(score, dtype=float)
    return np.where(s >= cortes["alta"], "alta", np.where(s >= cortes["media"], "media", "baixa"))


def sha256(caminho: Path) -> str:
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def salvar(modelo, metadados: dict, pasta: Path = PASTA) -> Path:
    """Grava o modelo e os metadados (com o hash do arquivo do modelo)."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / ARQUIVO_MODELO
    joblib.dump(modelo, caminho)
    metadados = {**metadados, "model_file": ARQUIVO_MODELO, "model_file_sha256": sha256(caminho)}
    (pasta / ARQUIVO_METADADOS).write_text(
        json.dumps(metadados, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return caminho


def carregar(pasta: Path = PASTA):
    """(modelo, metadados). Erro se o arquivo do modelo não for o dos metadados."""
    pasta = Path(pasta)
    metadados = json.loads((pasta / ARQUIVO_METADADOS).read_text(encoding="utf-8"))
    caminho = pasta / metadados["model_file"]
    if sha256(caminho) != metadados["model_file_sha256"]:
        raise ValueError(f"{caminho.name} não confere com o hash de {ARQUIVO_METADADOS}")
    return joblib.load(caminho), metadados
