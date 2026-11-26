"""Coleta dos dados externos do Mapa de Oportunidade (docs/oportunidade.md, parte 2).

Uso:
    python -m src.oportunidade.coleta

Grava em data/external/:
- osm_pizzarias_elementos.csv: cada pizzaria do OpenStreetMap, com a UF e as etiquetas
  usadas pela regra (para auditoria);
- osm_pizzarias_uf.csv: a contagem por UF;
- populacao_uf_censo2022.csv: a população residente do Censo 2022 por UF.

O OpenStreetMap muda todo dia, então uma nova coleta dá números um pouco diferentes.
Os arquivos versionados no git são o retrato usado no projeto; a data e o instante da base
do OpenStreetMap ficam gravados em cada linha. Datas e instantes estão em UTC.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from src.models.preparo import UFS

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "data" / "external"

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# Identificação genérica: a API recusa requisições sem User-Agent
CABECALHO = {"User-Agent": "case-ambev-oportunidade/1.0 (projeto academico)"}

# A consulta de docs/oportunidade.md, uma por UF
CONSULTA = """[out:json][timeout:300];
area["ISO3166-2"="BR-{sigla}"]["admin_level"="4"]->.uf;
(
  nwr(area.uf)["amenity"~"^(restaurant|fast_food)$"]["cuisine"~"pizza",i];
  nwr(area.uf)["amenity"~"^(restaurant|fast_food)$"]["name"~"pizz",i];
);
out tags;"""

# A mesma regra, reaplicada às etiquetas devolvidas (conferência da consulta)
AMENITY = re.compile(r"^(restaurant|fast_food)$")
COZINHA = re.compile(r"pizza", re.IGNORECASE)
NOME = re.compile(r"pizz", re.IGNORECASE)

SIDRA_URL = "https://apisidra.ibge.gov.br/values/t/4709/n3/all/v/93/p/2022"
SIDRA_BRASIL_URL = "https://apisidra.ibge.gov.br/values/t/4709/n1/all/v/93/p/2022"
FONTE_POPULACAO = ("IBGE, Censo Demográfico 2022, SIDRA tabela 4709 "
                   "(População residente), variável 93")


def criterio(tags: dict) -> str:
    """Por que o elemento é pizzaria: 'cozinha', 'nome' ou 'ambos'. Vazio se não for."""
    if not AMENITY.match(tags.get("amenity", "")):
        return ""
    por_cozinha = bool(COZINHA.search(tags.get("cuisine", "")))
    por_nome = bool(NOME.search(tags.get("name", "")))
    if por_cozinha and por_nome:
        return "ambos"
    return "cozinha" if por_cozinha else ("nome" if por_nome else "")


def consultar_overpass(sigla: str, tentativas: int = 5) -> dict:
    """Roda a consulta de uma UF, com espera crescente quando a API está ocupada."""
    for t in range(tentativas):
        r = requests.post(OVERPASS_URL, data={"data": CONSULTA.format(sigla=sigla)},
                          headers=CABECALHO, timeout=400)
        if r.status_code == 200:
            resposta = r.json()
            # A API devolve 200 com uma observação quando a consulta estoura o tempo
            if "remark" in resposta and "runtime error" in resposta["remark"]:
                raise RuntimeError(f"{sigla}: {resposta['remark']}")
            return resposta
        if r.status_code in (429, 502, 503, 504):
            time.sleep(30 * (t + 1))
            continue
        r.raise_for_status()
    raise RuntimeError(f"{sigla}: a API do Overpass não respondeu em {tentativas} tentativas")


def elementos_da_uf(sigla: str, resposta: dict) -> pd.DataFrame:
    linhas = []
    for el in resposta["elements"]:
        tags = el.get("tags", {})
        linhas.append({
            "uf": sigla, "tipo": el["type"], "osm_id": el["id"],
            "amenity": tags.get("amenity", ""), "cuisine": tags.get("cuisine", ""),
            "nome": tags.get("name", ""), "criterio": criterio(tags),
        })
    return pd.DataFrame(linhas, columns=["uf", "tipo", "osm_id", "amenity", "cuisine", "nome",
                                         "criterio"])


def coletar_pizzarias() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Elementos e contagem por UF. Falha se algum elemento não cumprir a regra."""
    data_coleta = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    elementos, contagem = [], []
    for codigo, sigla in UFS.items():
        resposta = consultar_overpass(sigla)
        el = elementos_da_uf(sigla, resposta)
        fora = el[el["criterio"] == ""]
        if len(fora):
            raise ValueError(f"{sigla}: {len(fora)} elementos não cumprem a regra")
        if el.duplicated(["tipo", "osm_id"]).any():
            raise ValueError(f"{sigla}: elemento repetido na resposta")
        elementos.append(el)
        contagem.append({
            "codigo_uf": codigo, "uf": sigla, "pizzarias": len(el),
            "nos": int((el["tipo"] == "node").sum()), "vias": int((el["tipo"] == "way").sum()),
            "relacoes": int((el["tipo"] == "relation").sum()),
            "so_cozinha": int((el["criterio"] == "cozinha").sum()),
            "so_nome": int((el["criterio"] == "nome").sum()),
            "ambos": int((el["criterio"] == "ambos").sum()),
            "osm_base": resposta["osm3s"]["timestamp_osm_base"],
            "areas_base": resposta["osm3s"]["timestamp_areas_base"],
            "data_coleta": data_coleta,
        })
        print(f"  {sigla}: {len(el)} pizzarias")
        time.sleep(3)  # a API pede no máximo 2 consultas simultâneas; aqui é uma por vez
    return pd.concat(elementos, ignore_index=True), pd.DataFrame(contagem)


def ler_sidra(url: str) -> pd.DataFrame:
    """A primeira linha da resposta do SIDRA é o cabeçalho; as demais são os valores."""
    r = requests.get(url, headers=CABECALHO, timeout=60)
    r.raise_for_status()
    dados = r.json()
    return pd.DataFrame(dados[1:])


def coletar_populacao() -> pd.DataFrame:
    """População residente por UF, conferida contra o total do Brasil da mesma tabela."""
    data_coleta = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    uf = ler_sidra(SIDRA_URL)
    brasil = ler_sidra(SIDRA_BRASIL_URL)
    if not (uf["D2N"] == "População residente").all() or not (uf["D3C"] == "2022").all():
        raise ValueError("a tabela do SIDRA não é a população residente de 2022")
    pop = pd.DataFrame({
        "codigo_uf": uf["D1C"], "uf": uf["D1C"].map(UFS), "nome": uf["D1N"],
        "populacao": uf["V"].astype("int64"), "fonte": FONTE_POPULACAO,
        "data_coleta": data_coleta,
    })
    if pop["uf"].isna().any() or set(pop["codigo_uf"]) != set(UFS):
        raise ValueError("as UFs do SIDRA não batem com as do projeto")
    total = int(brasil["V"].iloc[0])
    if pop["populacao"].sum() != total:
        raise ValueError(f"a soma das UFs ({pop['populacao'].sum()}) difere do Brasil ({total})")
    ordem = {c: i for i, c in enumerate(UFS)}
    return pop.sort_values("codigo_uf", key=lambda s: s.map(ordem)).reset_index(drop=True)


def main() -> None:
    EXT.mkdir(parents=True, exist_ok=True)
    print("População (Censo 2022)...")
    pop = coletar_populacao()
    pop.to_csv(EXT / "populacao_uf_censo2022.csv", index=False)
    print(f"  {len(pop)} UFs, {pop['populacao'].sum():,} pessoas")
    print("Pizzarias (OpenStreetMap)...")
    elementos, contagem = coletar_pizzarias()
    elementos.to_csv(EXT / "osm_pizzarias_elementos.csv", index=False)
    contagem.to_csv(EXT / "osm_pizzarias_uf.csv", index=False)
    print(f"  {len(elementos)} pizzarias em {len(contagem)} UFs")


if __name__ == "__main__":
    main()
