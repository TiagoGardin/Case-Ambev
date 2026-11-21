"""Contexto observado para o Decision Engine, calculado só com o treino.

Para as ocasiões de pizza + refrigerante (y = 1) do treino, no Brasil e em cada
região: onde a pizza foi comida (casa, estabelecimento ou outro lugar fora) e em
quantas o refrigerante foi guaraná. Tudo ponderado por `peso_semana`, com intervalo
de 95% por bootstrap de UPAs (1.000 reamostragens, semente 0) e o tamanho efetivo
da amostra (Kish).

Uma ocasião do treino (Sudeste, 0,08% do peso) não tem o local registrado. Ela entra
no total sem local, então os três locais somam 99,8% no Sudeste e 99,9% no Brasil.

Limites que o Decision Engine respeita:
- "casa" inclui delivery: a POF não separa pizza pedida de pizza feita ou comprada;
- "guaraná" é a categoria (qualquer marca), não a Guaraná Antarctica;
- por região há de 66 a 171 ocasiões (de 17 a 101 efetivas), e os intervalos são
  largos. Nenhuma região fica abaixo da participação nacional do guaraná com
  segurança, então a associação à marca não sai daqui: ela é informada por quem usa
  a API, e estas participações aparecem só como contexto na justificativa.

Gera models/contexto_decisao.json: `python -m src.decision_engine.contexto`.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_grid import REGIOES
from src.models.artefato import PASTA

ARQUIVO = "contexto_decisao.json"
REPETICOES, SEMENTE = 1000, 0
MEDIDAS = {
    "casa": lambda d: (d["local_pizza"] == "casa").to_numpy(dtype=float),
    "estabelecimento": lambda d: (d["local_pizza"] == "estabelecimento").to_numpy(dtype=float),
    "outro_fora": lambda d: (d["local_pizza"] == "outro_fora").to_numpy(dtype=float),
    "guarana": lambda d: d["y_guarana"].to_numpy(dtype=float),
}


def resumo(ocasioes: pd.DataFrame, contagens: np.ndarray, idx_upa: np.ndarray) -> dict:
    """Participações ponderadas, intervalos e tamanho efetivo de um grupo de ocasiões.

    `contagens` (reamostragens × UPAs) vem do sorteio de todas as UPAs com ocasião, e
    `idx_upa` é a UPA de cada linha do grupo, então os grupos usam as mesmas
    reamostragens.
    """
    w = ocasioes["peso_semana"].to_numpy(dtype=float)
    pesos = contagens[:, idx_upa] * w
    saida = {"ocasioes": int(len(ocasioes)),
             "ocasioes_efetivas": round(float(w.sum() ** 2 / (w ** 2).sum()), 1)}
    for nome, medida in MEDIDAS.items():
        x = medida(ocasioes)
        boot = (pesos * x).sum(axis=1) / pesos.sum(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        saida[nome] = {"participacao": float(np.average(x, weights=w)),
                       "ic95": [float(lo), float(hi)]}
    return saida


def calcular(treino: pd.DataFrame) -> dict:
    """Contexto do Brasil e de cada região, a partir das ocasiões do treino."""
    ocasioes = treino[treino["y"] == 1].reset_index(drop=True)
    unicas, idx = np.unique(ocasioes["COD_UPA"].to_numpy(), return_inverse=True)
    rng = np.random.default_rng(SEMENTE)
    contagens = np.stack([np.bincount(rng.integers(0, len(unicas), len(unicas)),
                                      minlength=len(unicas)) for _ in range(REPETICOES)])
    regiao = ocasioes["regiao"].astype(str).to_numpy()
    return {
        "brasil": resumo(ocasioes, contagens, idx),
        "regioes": {r: resumo(ocasioes[regiao == r], contagens, idx[regiao == r])
                    for r in REGIOES.values()},
    }


def salvar(contexto: dict, origem: str, pasta: Path = PASTA) -> Path:
    caminho = Path(pasta) / ARQUIVO
    conteudo = {
        "description": "Contexto observado do Decision Engine: local da pizza e participação "
                       "do guaraná nas ocasiões de pizza + refrigerante",
        "source": origem,
        "weights": "peso_semana",
        "interval": f"95%, bootstrap de UPAs ({REPETICOES} reamostragens, semente {SEMENTE})",
        "notes": ["casa inclui delivery: a POF não separa",
                  "guarana é a categoria (qualquer marca), não a Guaraná Antarctica"],
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        **contexto,
    }
    caminho.write_text(json.dumps(conteudo, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    return caminho


def carregar(pasta: Path = PASTA) -> dict:
    return json.loads((Path(pasta) / ARQUIVO).read_text(encoding="utf-8"))


def main() -> None:
    from src.models.treinar import carregar_treino

    treino, *_ = carregar_treino()
    contexto = calcular(treino)
    caminho = salvar(contexto, "treino (jul/2017 a fev/2018), ocasiões de pizza + refrigerante")
    for nome, c in [("Brasil", contexto["brasil"]), *contexto["regioes"].items()]:
        partes = [f"{m} {c[m]['participacao']:.0%} [{c[m]['ic95'][0]:.0%}; {c[m]['ic95'][1]:.0%}]"
                  for m in MEDIDAS]
        print(f"{nome:<13} {c['ocasioes']:>3} ocasiões ({c['ocasioes_efetivas']:>5} efetivas): "
              + " | ".join(partes))
    print(f"Gravado: {caminho}")


if __name__ == "__main__":
    main()
