"""Funções comuns aos scripts da auditoria. Nada aqui altera arquivos do projeto."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "audit" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
PROC = ROOT / "data" / "processed"


class Relatorio:
    """Imprime e acumula o texto de uma fase; grava em audit/outputs/<nome>.md."""

    def __init__(self, nome: str):
        self.nome = nome
        self.linhas: list[str] = []

    def __call__(self, *partes) -> None:
        texto = " ".join(str(p) for p in partes)
        print(texto, flush=True)
        self.linhas.append(texto)

    def secao(self, titulo: str) -> None:
        self("")
        self(f"## {titulo}")

    def tabela(self, df: pd.DataFrame | pd.Series, **kwargs) -> None:
        self("```")
        self(df.to_string(**kwargs))
        self("```")

    def gravar(self) -> Path:
        caminho = OUT / f"{self.nome}.md"
        caminho.write_text("\n".join(self.linhas) + "\n", encoding="utf-8")
        return caminho


def carregar_grid() -> pd.DataFrame:
    return pd.read_parquet(PROC / "grid.parquet")


def taxa(y, peso) -> float:
    y = pd.Series(y).to_numpy(dtype=float)
    peso = pd.Series(peso).to_numpy(dtype=float)
    return float((y * peso).sum() / peso.sum())
