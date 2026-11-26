"""Cliente HTTP da API do Occasion Score. O app só fala com o modelo por aqui.

A URL vem da variável de ambiente API_URL (padrão: http://localhost:8000).
"""

from __future__ import annotations

import os

import requests

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
TEMPO_LIMITE = 5  # segundos


class ApiIndisponivel(RuntimeError):
    """A API não respondeu (fora do ar, porta errada ou tempo esgotado)."""


class ApiErro(RuntimeError):
    """A API respondeu com erro (ex.: 422, entrada fora do domínio do modelo)."""

    def __init__(self, status: int, detalhe):
        super().__init__(f"{status}: {detalhe}")
        self.status = status
        self.detalhe = detalhe


def _requisitar(metodo: str, caminho: str, corpo: dict | None = None) -> dict:
    try:
        r = requests.request(metodo, f"{API_URL}{caminho}", json=corpo, timeout=TEMPO_LIMITE)
    except requests.RequestException as erro:
        raise ApiIndisponivel(str(erro)) from erro
    if r.status_code >= 400:
        try:
            detalhe = r.json().get("detail", r.text)
        except ValueError:
            detalhe = r.text
        raise ApiErro(r.status_code, detalhe)
    return r.json()


def saude() -> dict:
    return _requisitar("GET", "/health")


def modelo() -> dict:
    return _requisitar("GET", "/model")


def regras() -> dict:
    return _requisitar("GET", "/rules")


def oportunidade() -> dict:
    """GET /oportunidade: o Mapa de Oportunidade (públicos-momento e estados)."""
    return _requisitar("GET", "/oportunidade")


def prever(contexto: dict) -> dict:
    """POST /predict com o contexto já no formato da API (os 6 campos voltam como dict)."""
    return _requisitar("POST", "/predict", contexto)
