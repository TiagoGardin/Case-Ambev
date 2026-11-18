"""Testes da leitura e do download da POF.

Os de download usam um zip criado na hora. O dos códigos de alimentos lê a
documentação do IBGE e é pulado se ela não tiver sido baixada.
"""

import zipfile

import pytest

from src.ingestion.codigos_alimentos import (
    ARQ_CADASTRO,
    ARQ_GRUPOS,
    codigos_guarana,
    codigos_pizza,
    codigos_refrigerante,
)
from src.ingestion.download_pof import extrair, nome_seguro
from src.ingestion.pof import _colspecs


def test_colspecs_converte_para_intervalos_zero_based():
    assert _colspecs({"UF": (1, 2), "COD_UPA": (8, 9)}) == [(0, 2), (7, 16)]


def test_nome_seguro_troca_acentos():
    assert nome_seguro("Dicionários de váriaveis.xls") == "Dicion_rios de v_riaveis.xls"
    assert nome_seguro("Dados/MORADOR.txt") == "Dados/MORADOR.txt"


def test_extrair_respeita_a_lista_e_nao_deixa_parcial(tmp_path):
    pacote = tmp_path / "pacote.zip"
    with zipfile.ZipFile(pacote, "w") as z:
        z.writestr("Dados/MORADOR.txt", "linha 1\n")
        z.writestr("Dados/OUTRO.txt", "não deveria sair\n")
    destino = tmp_path / "dados"

    extrair(pacote, destino, apenas={"MORADOR.txt"})
    extrair(pacote, destino, apenas={"MORADOR.txt"})  # de novo: nada muda

    arquivos = sorted(p.relative_to(destino).as_posix() for p in destino.rglob("*") if p.is_file())
    assert arquivos == ["Dados/MORADOR.txt"]
    assert (destino / "Dados" / "MORADOR.txt").read_text() == "linha 1\n"


@pytest.mark.skipif(not (ARQ_GRUPOS.exists() and ARQ_CADASTRO.exists()),
                    reason="documentação da POF não baixada")
def test_codigos_da_ocasiao():
    pizza, refri, guarana = codigos_pizza(), codigos_refrigerante(), codigos_guarana()
    assert (len(pizza), len(refri), len(guarana)) == (19, 34, 6)
    assert guarana <= refri  # toda ocasião pizza + guaraná também é pizza + refri
    assert 8201902 in refri  # Tubaína light entra pelo nome
    assert 8200308 not in refri  # refresco de guaraná natural não é refrigerante
