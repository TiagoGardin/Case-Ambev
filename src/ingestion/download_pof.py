"""Download e extração dos microdados da POF 2017-2018 (IBGE).

Uso:
    python -m src.ingestion.download_pof

Baixa os três pacotes do FTP do IBGE para data/raw/pof/ e extrai os arquivos.
Se um arquivo já existir, não baixa de novo (idempotente).
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import requests

BASE_URL = (
    "https://ftp.ibge.gov.br/Orcamentos_Familiares/"
    "Pesquisa_de_Orcamentos_Familiares_2017_2018/Microdados/"
)

# Versões fixadas: o IBGE publica novas versões com outra data no nome.
# Mudar aqui é uma decisão consciente, registrada no Data Card.
PACOTES = {
    "dados": "Dados_20230713.zip",
    "documentacao": "Documentacao_20230713.zip",
    "programas": "Programas_de_Leitura_20230713.zip",
}

# Do pacote de dados, só precisamos destes registros (o zip inteiro tem ~1,3 GB descompactado).
REGISTROS_NECESSARIOS = {"CONSUMO_ALIMENTAR.txt", "MORADOR.txt", "ALUGUEL_ESTIMADO.txt"}

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "pof"


def nome_seguro(nome: str) -> str:
    """Troca caracteres fora do ASCII básico por '_'.

    Os zips do IBGE gravam nomes com acento num encoding que o macOS não aceita
    (o `unzip` falha com 'write error'). Normalizar o nome evita o problema.
    """
    return re.sub(r"[^A-Za-z0-9._/ -]", "_", nome)


def baixar(nome_arquivo: str, destino: Path) -> Path:
    caminho = destino / nome_arquivo
    if caminho.exists():
        print(f"[ok] {nome_arquivo} já existe, pulando download")
        return caminho
    print(f"[..] baixando {nome_arquivo}")
    with requests.get(BASE_URL + nome_arquivo, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(caminho, "wb") as f:
            for bloco in r.iter_content(chunk_size=1 << 20):
                f.write(bloco)
    return caminho


def extrair(zip_path: Path, destino: Path, apenas: set[str] | None = None) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            if apenas and Path(info.filename).name not in apenas:
                continue
            alvo = destino / nome_seguro(info.filename)
            if alvo.exists():
                continue
            alvo.parent.mkdir(parents=True, exist_ok=True)
            alvo.write_bytes(z.read(info))
            print(f"     extraído: {alvo.relative_to(RAW_DIR)}")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zips = {k: baixar(v, RAW_DIR) for k, v in PACOTES.items()}
    extrair(zips["dados"], RAW_DIR / "dados", apenas=REGISTROS_NECESSARIOS)
    extrair(zips["documentacao"], RAW_DIR / "documentacao")
    extrair(zips["programas"], RAW_DIR / "programas")


if __name__ == "__main__":
    main()
