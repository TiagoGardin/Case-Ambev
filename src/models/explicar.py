"""Explicação de cada previsão: SHAP exato da Regressão Logística, agrupado em fatores.

O logit da Regressão Logística é uma soma: intercepto + Σ coeficiente × coluna
transformada. Com uma referência fixa (a média das colunas transformadas no
treino), o SHAP de cada coluna é exatamente coeficiente × (valor − média), que é
o SHAP intervencional de um modelo linear. A soma das contribuições mais o logit
da referência dá o logit da previsão.

As colunas que vêm da mesma entrada formam um fator. O período, o fim de semana e
a interação entre os dois viram um fator só, o momento, porque a interação depende
das duas entradas. Nos outros fatores, cada coluna depende de uma entrada só. Como
cada parcela da soma fica inteira dentro de um fator, a soma das contribuições das
colunas do fator é o SHAP exato do fator.

A sigmoide de calibração multiplica o logit por `a` (src.models.calibracao). No
log-odds do score calibrado, a contribuição de cada fator é a × SHAP, e
exp(a × SHAP) é o quanto o fator multiplica a chance da ocasião em relação ao
contexto de referência.

A referência é a média ponderada por `peso_semana` do treino e fica em
models/explicacao.json, gerado só com o treino por `python -m src.models.explicar`.
A avaliação no teste, a API e o app usam o mesmo arquivo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.artefato import PASTA
from src.models.preparo import preparar

ARQUIVO = "explicacao.json"

# Fator → colunas do preparo que entram nele (o modelo final usa todas estas)
FATORES = {
    "momento": ("periodo", "periodo_fds", "fim_de_semana"),
    "regiao": ("regiao",),
    "idade": ("faixa_idade",),
    "sexo": ("sexo",),
    "escolaridade": ("anos_estudo",),
    "moradores": ("n_moradores",),
    "renda": ("log_renda_pc",),
    "area": ("urbano",),
}
# Fator → entradas da API (src.models.preparo.FEATURES) de onde ele sai
ENTRADAS = {
    "momento": ("periodo", "dia_semana"),
    "regiao": ("UF",),
    "idade": ("idade",),
    "sexo": ("sexo",),
    "escolaridade": ("anos_estudo",),
    "moradores": ("n_moradores",),
    "renda": ("renda_pc",),
    "area": ("urbano",),
}
NOME_FATOR = {
    "momento": "momento (período e dia)",
    "regiao": "região",
    "idade": "faixa de idade",
    "sexo": "sexo",
    "escolaridade": "escolaridade",
    "moradores": "moradores no domicílio",
    "renda": "renda per capita",
    "area": "área",
}
ROTULO_PERIODO = {"manha": "manhã", "almoco": "almoço", "tarde": "tarde", "noite": "noite"}
ROTULO_IDADE = {"10_17": "10 a 17 anos", "18_29": "18 a 29 anos", "30_44": "30 a 44 anos",
                "45_59": "45 a 59 anos", "60_mais": "60 anos ou mais"}


def colunas_de_origem(colunas) -> list[str]:
    """Coluna do preparo de onde vem cada coluna transformada, na ordem dos coeficientes."""
    origem = []
    for nome, transformador, cols in colunas.transformers_:
        if nome == "remainder" or isinstance(transformador, str) and transformador == "drop":
            continue
        if hasattr(transformador, "categories_"):
            for coluna, categorias in zip(cols, transformador.categories_, strict=True):
                origem += [coluna] * len(categorias)
        else:
            origem += list(cols)
    return origem


def transformar(pipeline, X: pd.DataFrame) -> np.ndarray:
    """As colunas que entram na Regressão Logística (preparo + one-hot + padronização)."""
    T = pipeline[:-1].transform(X)
    return T.toarray() if hasattr(T, "toarray") else np.asarray(T, dtype=float)


def descrever(fator: str, p: pd.Series) -> str:
    """Valor do fator numa linha já preparada, como o modelo o enxerga."""
    if fator == "momento":
        dias = "sexta a domingo" if p["fim_de_semana"] == 1 else "segunda a quinta"
        return f"{ROTULO_PERIODO[str(p['periodo'])]} de {dias}"
    if fator == "regiao":
        return str(p["regiao"])
    if fator == "idade":
        return ROTULO_IDADE[str(p["faixa_idade"])]
    if fator == "sexo":
        return str(p["sexo"])
    if fator == "escolaridade":
        anos = int(round(p["anos_estudo"]))
        return "16 anos ou mais de estudo" if anos >= 16 else (
            f"{anos} ano de estudo" if anos == 1 else f"{anos} anos de estudo")
    if fator == "moradores":
        n = int(round(p["n_moradores"]))
        return "1 morador" if n == 1 else f"{n} moradores"
    if fator == "renda":
        return f"R$ {p['renda_pc']:,.0f} por pessoa ao mês".replace(",", ".")
    if fator == "area":
        return "urbana" if p["urbano"] == 1 else "rural"
    raise ValueError(f"fator desconhecido: {fator!r}")


@dataclass
class Explicador:
    """SHAP por fator de um ModeloCalibrado cuja última etapa é a Regressão Logística.

    `media` é a referência: a média de cada coluna transformada (models/explicacao.json).
    """

    modelo: object
    media: np.ndarray
    colunas: list[str] = field(init=False)
    origem: list[str] = field(init=False)

    def __post_init__(self) -> None:
        pipeline = self.modelo.pipeline
        lr = pipeline.named_steps["modelo"]
        self.colunas = [str(c) for c in pipeline.named_steps["colunas"].get_feature_names_out()]
        self.origem = colunas_de_origem(pipeline.named_steps["colunas"])
        self.coef = np.asarray(lr.coef_[0], dtype=float)
        self.intercepto = float(lr.intercept_[0])
        self.media = np.asarray(self.media, dtype=float)
        if not len(self.colunas) == len(self.origem) == len(self.coef) == len(self.media):
            raise ValueError("colunas, coeficientes e referência com tamanhos diferentes")
        dentro = {c for cols in FATORES.values() for c in cols}
        if set(self.origem) - dentro:
            raise ValueError(f"colunas fora dos fatores: {sorted(set(self.origem) - dentro)}")
        self._grupo = np.array([[o in FATORES[f] for f in FATORES] for o in self.origem],
                               dtype=float)

    @property
    def log_odds_referencia(self) -> float:
        """Log-odds do score calibrado no contexto de referência."""
        return self.modelo.a * (self.intercepto + float(self.coef @ self.media)) + self.modelo.b

    def contribuicoes(self, X: pd.DataFrame) -> pd.DataFrame:
        """Contribuição de cada fator no log-odds do score calibrado (linhas × fatores).

        A soma de uma linha mais `log_odds_referencia` é o logit do score calibrado.
        """
        T = transformar(self.modelo.pipeline, X)
        shap = ((T - self.media) * self.coef) @ self._grupo
        return pd.DataFrame(self.modelo.a * shap, index=X.index, columns=list(FATORES))

    def top_fatores(self, X: pd.DataFrame, k: int = 3) -> list[list[dict]]:
        """Os k fatores de maior contribuição, em valor absoluto, de cada linha."""
        contrib = self.contribuicoes(X)
        prep = preparar(X)
        saida = []
        for i in range(len(X)):
            c = contrib.iloc[i]
            ordem = c.abs().sort_values(ascending=False, kind="stable").index[:k]
            saida.append([{
                "factor": f,
                "value": descrever(f, prep.iloc[i]),
                "effect": "aumenta" if c[f] > 0 else ("reduz" if c[f] < 0 else "neutro"),
                "odds_multiplier": round(float(np.exp(c[f])), 3),
                "log_odds": round(float(c[f]), 4),
            } for f in ordem])
        return saida


def referencia(modelo, X: pd.DataFrame, peso) -> np.ndarray:
    """Média ponderada de cada coluna transformada: o contexto de referência."""
    return np.average(transformar(modelo.pipeline, X), axis=0, weights=np.asarray(peso, float))


def salvar(explicador: Explicador, metadados: dict, origem: str, pasta: Path = PASTA) -> Path:
    caminho = Path(pasta) / ARQUIVO
    conteudo = {
        "description": "Referência do SHAP por fator (src.models.explicar): média de cada "
                       "coluna transformada da Regressão Logística",
        "reference": origem,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_file_sha256": metadados["model_file_sha256"],
        "factors": {f: list(cols) for f, cols in FATORES.items()},
        "columns": explicador.colunas,
        "mean": [float(v) for v in explicador.media],
        "log_odds_reference": explicador.log_odds_referencia,
    }
    caminho.write_text(json.dumps(conteudo, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    return caminho


def carregar(modelo, metadados: dict, pasta: Path = PASTA) -> Explicador:
    """O Explicador do modelo congelado. Erro se a referência for de outro modelo."""
    conteudo = json.loads((Path(pasta) / ARQUIVO).read_text(encoding="utf-8"))
    if conteudo["model_file_sha256"] != metadados["model_file_sha256"]:
        raise ValueError(f"{ARQUIVO} foi gerado para outro arquivo de modelo")
    explicador = Explicador(modelo, np.asarray(conteudo["mean"]))
    if explicador.colunas != conteudo["columns"]:
        raise ValueError(f"as colunas de {ARQUIVO} não são as do modelo")
    return explicador


def main() -> None:
    from src.models import artefato
    from src.models.calibracao import logit
    from src.models.treinar import carregar_treino

    treino, X, y, peso = carregar_treino()
    modelo, metadados = artefato.carregar()
    explicador = Explicador(modelo, referencia(modelo, X, peso))

    # Conferência: referência + soma das contribuições = logit do score calibrado
    contrib = explicador.contribuicoes(X)
    reconstruido = explicador.log_odds_referencia + contrib.sum(axis=1).to_numpy()
    erro = float(np.abs(reconstruido - logit(modelo.predict_proba(X)[:, 1])).max())
    if erro > 1e-8:
        raise ValueError(f"as contribuições não somam o logit do score (erro {erro:.2e})")

    caminho = salvar(explicador, metadados,
                     "treino (jul/2017 a fev/2018), média ponderada por peso_semana, "
                     f"{len(X):,} linhas".replace(",", "."))
    importancia = contrib.abs().mul(peso, axis=0).sum() / peso.sum()
    print(f"Referência: log-odds {explicador.log_odds_referencia:.3f} "
          f"(score {1 / (1 + np.exp(-explicador.log_odds_referencia)):.4%}); "
          f"erro máximo da soma: {erro:.1e}")
    print("Importância no treino (média ponderada do |log-odds|):")
    print(importancia.sort_values(ascending=False).round(3).to_string())
    print(f"Gravado: {caminho}")


if __name__ == "__main__":
    main()
