"""Ranking do Mapa de Oportunidade: públicos-momento e estados (docs/oportunidade.md).

Uso:
    python -m src.oportunidade.ranking

Aplica o modelo congelado às linhas do treino de adultos e grava models/oportunidade.json.
Nada é treinado e o teste não é usado. As conferências do documento rodam aqui, e o script
falha se alguma não passar, inclusive a de que duas execuções geram o mesmo arquivo.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.decision_engine import contexto as contexto_decisao
from src.decision_engine.regras import recomendar
from src.features.build_grid import FIM_DE_SEMANA, PERIODOS, REGIOES, regiao_da_uf
from src.ingestion.pof import CHAVE_PESSOA
from src.models import artefato
from src.models.preparo import FEATURES, UFS

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "data" / "external"
PROC = ROOT / "data" / "processed"
ARQUIVO = "oportunidade.json"

IDADE_MINIMA = 18
# Faixas de idade do Google Ads; o limite de cima é exclusivo
FAIXAS_ETARIAS = {"18-24": (18, 25), "25-34": (25, 35), "35-44": (35, 45), "45-54": (45, 55),
                  "55-64": (55, 65), "65+": (65, np.inf)}
BLOCOS = {0: "seg_qui", 1: "sex_dom"}
DIAS_DO_BLOCO = {"seg_qui": ["Segunda-Feira", "Terca-Feira", "Quarta-Feira", "Quinta-Feira"],
                 "sex_dom": FIM_DE_SEMANA}
# Programação de horário de cada período (fim exclusivo), como na grade: manhã 5–10h,
# almoço 11–14h, tarde 15–17h e noite 18–4h, com a madrugada no mesmo dia do registro
HORARIOS = {"manha": ["05:00-11:00"], "almoco": ["11:00-15:00"], "tarde": ["15:00-18:00"],
            "noite": ["18:00-24:00", "00:00-05:00"]}
DIMENSOES = ["faixa_etaria", "sexo", "regiao", "periodo", "bloco"]
PUBLICO = ["faixa_etaria", "sexo", "regiao"]
AMOSTRA_MINIMA = 100  # pessoas por público (faixa × sexo × região)
OCASIOES_PARA_RELATO = 10
CASAS = 10  # arredondamento dos números gravados, para o arquivo não variar no último dígito


# --------------------------------------------------------------------------
# Regras puras (testadas sem os dados)
# --------------------------------------------------------------------------

def faixa_etaria(idade: pd.Series) -> pd.Series:
    limites = [v[0] for v in FAIXAS_ETARIAS.values()] + [np.inf]
    return pd.cut(idade, limites, right=False, labels=list(FAIXAS_ETARIAS)).astype(str)


def categoria_do_estado(chance_acima: bool, densidade_acima: bool) -> str:
    """A tabela da regra do canal (docs/oportunidade.md, parte 2)."""
    if not chance_acima:
        return "presenca_de_marca"
    return "ativacao_delivery" if densidade_acima else "combo_consumo_em_casa"


def categorias_por_classe(contexto: dict, taxa_referencia: float) -> dict:
    """Categoria e regra do Decision Engine para cada classe, sem canal e sem associação.

    Usa a própria função do Decision Engine, para o mapa nunca divergir das regras.
    """
    return {classe: (r.categoria, r.regra) for classe in artefato.CLASSES
            for r in [recomendar(taxa_referencia, classe, [], "Sudeste", contexto,
                                 taxa_referencia)]}


def anuncio(faixa: str, sexo: str, regiao: str, periodo: str, bloco: str) -> dict:
    """Parâmetros de anúncio de um público-momento. Não é plano de mídia."""
    ufs = sorted(s for c, s in UFS.items() if REGIOES[c[0]] == regiao)
    return {"faixa_etaria": faixa, "genero": sexo, "localizacao_ufs": ufs,
            "dias": DIAS_DO_BLOCO[bloco], "horarios": HORARIOS[periodo]}


# --------------------------------------------------------------------------
# Dados
# --------------------------------------------------------------------------

def carregar_adultos(modelo) -> pd.DataFrame:
    """Linhas do treino de pessoas com 18 anos ou mais, com o score do modelo congelado."""
    grid = pd.read_parquet(PROC / "grid.parquet")
    adultos = grid[(grid["split"] == "treino") & (grid["idade"] >= IDADE_MINIMA)].copy()
    if not (adultos["split"] == "treino").all() or adultos["idade"].min() < IDADE_MINIMA:
        raise ValueError("só entram linhas do treino de adultos")
    if not (adultos["fim_de_semana"] == adultos["dia_semana"].astype(str)
            .isin(FIM_DE_SEMANA).astype(int)).all():
        raise ValueError("fim_de_semana não corresponde a sexta a domingo")
    adultos["score"] = modelo.predict_proba(adultos[FEATURES])[:, 1]
    adultos["faixa_etaria"] = faixa_etaria(adultos["idade"])
    adultos["bloco"] = adultos["fim_de_semana"].map(BLOCOS)
    adultos["periodo"] = adultos["periodo"].astype(str)
    adultos["UF"] = adultos["UF"].astype(str)
    adultos["ws"] = adultos["peso_semana"] * adultos["score"]
    adultos["wy"] = adultos["peso_semana"] * adultos["y"]
    return adultos.reset_index(drop=True)


def carregar_externos() -> tuple[pd.DataFrame, pd.DataFrame]:
    pizzarias = pd.read_csv(EXT / "osm_pizzarias_uf.csv", dtype={"codigo_uf": str})
    populacao = pd.read_csv(EXT / "populacao_uf_censo2022.csv", dtype={"codigo_uf": str})
    return pizzarias, populacao


# --------------------------------------------------------------------------
# Parte 1: públicos e momentos
# --------------------------------------------------------------------------

def publicos(adultos: pd.DataFrame, cortes: dict, categorias: dict) -> pd.DataFrame:
    total_w, total_ws = adultos["peso_semana"].sum(), adultos["ws"].sum()
    media = total_ws / total_w
    g = adultos.groupby(DIMENSOES, observed=True)
    cel = g.agg(w=("peso_semana", "sum"), ws=("ws", "sum"), wy=("wy", "sum"),
                ocasioes_observadas=("y", "sum")).reset_index()
    cel["chance"] = cel["ws"] / cel["w"]
    cel["comparacao"] = cel["chance"] / media
    cel["parcela_das_ocasioes"] = cel["ws"] / total_ws
    cel["parcela_dos_momentos"] = cel["w"] / total_w
    cel["taxa_observada"] = cel["wy"] / cel["w"]
    cel["prioridade"] = artefato.classificar(cel["chance"], cortes)
    cel["categoria"] = cel["prioridade"].map(lambda c: categorias[c][0])
    cel["regra"] = cel["prioridade"].map(lambda c: categorias[c][1])

    # Pessoas e renda mais comum por público (uma linha por pessoa, peso de expansão)
    pessoas = adultos.drop_duplicates(CHAVE_PESSOA)
    n = pessoas.groupby(PUBLICO, observed=True).size().rename("pessoas_amostra")
    renda = (pessoas.groupby(PUBLICO + ["faixa_renda"], observed=True)["peso"].sum()
             .reset_index().sort_values(PUBLICO + ["peso", "faixa_renda"],
                                        ascending=[True] * len(PUBLICO) + [False, True])
             .drop_duplicates(PUBLICO).set_index(PUBLICO)["faixa_renda"].astype(str)
             .rename("renda_mais_comum"))
    cel = cel.join(n, on=PUBLICO).join(renda, on=PUBLICO)
    cel["amostra_suficiente"] = cel["pessoas_amostra"] >= AMOSTRA_MINIMA

    # Posições só entre os públicos com amostra suficiente; empate desfeito pelas chaves
    cel = cel.sort_values(DIMENSOES).reset_index(drop=True)
    ok = cel["amostra_suficiente"]
    for medida, coluna in [("chance", "posicao_chance"),
                           ("parcela_das_ocasioes", "posicao_parcela")]:
        ordem = cel[ok].sort_values(medida, ascending=False, kind="stable").index
        cel[coluna] = pd.Series(range(1, len(ordem) + 1), index=ordem).astype("Int64")
    return cel


# --------------------------------------------------------------------------
# Parte 2: estados
# --------------------------------------------------------------------------

def estados(adultos: pd.DataFrame, pizzarias: pd.DataFrame, populacao: pd.DataFrame
            ) -> tuple[pd.DataFrame, dict]:
    total_ws = adultos["ws"].sum()
    g = adultos.groupby("UF", observed=True)
    uf = g.agg(w=("peso_semana", "sum"), ws=("ws", "sum")).reset_index()
    uf["pessoas_amostra"] = adultos.drop_duplicates(CHAVE_PESSOA).groupby("UF").size().reindex(
        uf["UF"]).to_numpy()
    uf["chance"] = uf["ws"] / uf["w"]
    uf["parcela_das_ocasioes"] = uf["ws"] / total_ws
    uf = uf.rename(columns={"UF": "codigo_uf"})
    uf["uf"] = uf["codigo_uf"].map(UFS)
    uf["regiao"] = regiao_da_uf(uf["codigo_uf"])
    uf = uf.merge(populacao[["codigo_uf", "nome", "populacao"]], on="codigo_uf", how="outer")
    uf = uf.merge(pizzarias[["codigo_uf", "pizzarias"]], on="codigo_uf", how="outer")
    uf["pizzarias_100mil"] = uf["pizzarias"] / uf["populacao"] * 1e5
    medianas = {"chance": float(uf["chance"].median()),
                "pizzarias_100mil": float(uf["pizzarias_100mil"].median())}
    uf["chance_acima"] = uf["chance"] >= medianas["chance"]
    uf["densidade_acima"] = uf["pizzarias_100mil"] >= medianas["pizzarias_100mil"]
    uf["categoria"] = [categoria_do_estado(c, d)
                       for c, d in zip(uf["chance_acima"], uf["densidade_acima"])]
    uf = uf.sort_values(["chance_acima", "parcela_das_ocasioes", "uf"],
                        ascending=[False, False, True], kind="stable").reset_index(drop=True)
    uf["posicao"] = range(1, len(uf) + 1)
    return uf, medianas


# --------------------------------------------------------------------------
# Montagem, conferências e gravação
# --------------------------------------------------------------------------

def _num(v):
    if isinstance(v, (float, np.floating)):
        return round(float(v), CASAS)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if v is pd.NA:
        return None
    return v


def montar(adultos: pd.DataFrame, pizzarias: pd.DataFrame, populacao: pd.DataFrame,
           metadados: dict, contexto: dict) -> dict:
    cortes = metadados["classification"]["cuts"]
    categorias = categorias_por_classe(contexto, metadados["reference_rate"])
    cel = publicos(adultos, cortes, categorias)
    uf, medianas = estados(adultos, pizzarias, populacao)
    media = adultos["ws"].sum() / adultos["peso_semana"].sum()

    lista_publicos = []
    for r in cel.itertuples(index=False):
        item = {c: _num(getattr(r, c)) for c in [
            *DIMENSOES, "chance", "comparacao", "parcela_das_ocasioes", "parcela_dos_momentos",
            "prioridade", "categoria", "regra", "renda_mais_comum", "pessoas_amostra",
            "amostra_suficiente", "posicao_chance", "posicao_parcela", "ocasioes_observadas",
            "taxa_observada"]}
        item["anuncio"] = anuncio(r.faixa_etaria, r.sexo, r.regiao, r.periodo, r.bloco)
        lista_publicos.append(item)

    lista_estados = [{c: _num(getattr(r, c)) for c in [
        "posicao", "uf", "codigo_uf", "nome", "regiao", "chance", "parcela_das_ocasioes",
        "pizzarias", "populacao", "pizzarias_100mil", "chance_acima", "densidade_acima",
        "categoria", "pessoas_amostra"]} for r in uf.itertuples(index=False)]

    relato_uf = spearmanr(uf["chance"], uf["pizzarias_100mil"])
    rel = cel[cel["ocasioes_observadas"] >= OCASIOES_PARA_RELATO]
    relato_pub = spearmanr(rel["chance"], rel["taxa_observada"])
    return {
        "descricao": "Mapa de Oportunidade: públicos-momento e estados, pela regra de "
                     "docs/oportunidade.md. Não é previsão de resultado de campanha.",
        "regra": "docs/oportunidade.md",
        "modelo": {"model_version": metadados["model_version"],
                   "model_file_sha256": metadados["model_file_sha256"]},
        "dados": {
            "treino": metadados["training_data"]["period"],
            "idade_minima": IDADE_MINIMA, "peso": "peso_semana",
            "linhas": int(len(adultos)),
            "pessoas": int(adultos.drop_duplicates(CHAVE_PESSOA).shape[0]),
            "ocasioes_observadas": int(adultos["y"].sum()),
        },
        "fontes": {
            "pizzarias": {"arquivo": "data/external/osm_pizzarias_uf.csv",
                          "fonte": "© colaboradores do OpenStreetMap (ODbL), API Overpass",
                          "base_osm": [pizzarias["osm_base"].min(), pizzarias["osm_base"].max()],
                          "data_coleta_utc": pizzarias["data_coleta"].iloc[0]},
            "populacao": {"arquivo": "data/external/populacao_uf_censo2022.csv",
                          "fonte": populacao["fonte"].iloc[0],
                          "data_coleta_utc": populacao["data_coleta"].iloc[0]},
        },
        "chance_media_adultos": _num(media),
        "cortes_prioridade": {k: _num(v) for k, v in cortes.items()},
        "amostra_minima_pessoas": AMOSTRA_MINIMA,
        "medianas_estados": {k: _num(v) for k, v in medianas.items()},
        "relato": {
            "spearman_chance_x_pizzarias_estados": _num(relato_uf.statistic),
            "spearman_chance_x_taxa_observada_publicos": _num(relato_pub.statistic),
            "publicos_no_relato": int(len(rel)),
            "nota": "Só reportados; não mudam a regra nem a ordem.",
        },
        "publicos": lista_publicos,
        "estados": lista_estados,
    }


def conferir(resultado: dict) -> None:
    """Conferências 3 a 5 e 7 de docs/oportunidade.md (1 e 2 vêm na carga; 6 no main)."""
    pub, est = resultado["publicos"], resultado["estados"]
    medidas = [i[c] for i in pub + est for c in ("chance", "parcela_das_ocasioes")]
    if any(v is None or not np.isfinite(v) for v in medidas):
        raise ValueError("chance ou parcela ausente ou não finita")
    n_esperado = len(FAIXAS_ETARIAS) * 2 * len(REGIOES) * len(PERIODOS) * len(BLOCOS)
    if len(pub) != n_esperado:
        raise ValueError(f"{len(pub)} públicos-momento, esperados {n_esperado}")
    for nome, itens in [("públicos", pub), ("estados", est)]:
        soma = sum(i["parcela_das_ocasioes"] for i in itens)
        if not np.isclose(soma, 1, atol=1e-8):
            raise ValueError(f"as parcelas dos {nome} somam {soma}")
    media = sum(i["parcela_dos_momentos"] * i["chance"] for i in pub)
    if not np.isclose(media, resultado["chance_media_adultos"], rtol=1e-8):
        raise ValueError("a média das chances dos públicos não reproduz a dos adultos")
    if len(est) != 27 or {e["codigo_uf"] for e in est} != set(UFS):
        raise ValueError("os estados não são as 27 UFs")
    if any(e["populacao"] is None or e["populacao"] <= 0 or e["pizzarias"] is None
           for e in est):
        raise ValueError("UF sem população ou sem contagem de pizzarias")
    for e in est:
        if e["categoria"] != categoria_do_estado(e["chance_acima"], e["densidade_acima"]):
            raise ValueError(f"{e['uf']} fora da regra do canal")


def gerar() -> dict:
    modelo, metadados = artefato.carregar()  # confere o hash do modelo (conferência 2)
    contexto = contexto_decisao.carregar()
    pizzarias, populacao = carregar_externos()
    adultos = carregar_adultos(modelo)  # só treino e adultos (conferência 1)
    resultado = montar(adultos, pizzarias, populacao, metadados, contexto)
    conferir(resultado)
    return resultado


def serializar(resultado: dict) -> str:
    return json.dumps(resultado, ensure_ascii=False, indent=2) + "\n"


def main() -> None:
    primeiro, segundo = serializar(gerar()), serializar(gerar())
    if primeiro != segundo:  # conferência 6
        raise ValueError("duas execuções geraram arquivos diferentes")
    caminho = artefato.PASTA / ARQUIVO
    caminho.write_text(primeiro, encoding="utf-8")
    r = json.loads(primeiro)
    ok = [p for p in r["publicos"] if p["amostra_suficiente"]]
    print(f"{len(r['publicos'])} públicos-momento ({len(ok)} com amostra suficiente), "
          f"{len(r['estados'])} estados. Conferências 1 a 7: ok.")
    print(f"Gravado: {caminho}")


if __name__ == "__main__":
    main()
