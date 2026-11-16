# Notas – POF 2017–2018

Anotações da exploração da documentação e da leitura dos microdados.

## Arquivos baixados
Fonte: `https://ftp.ibge.gov.br/Orcamentos_Familiares/Pesquisa_de_Orcamentos_Familiares_2017_2018/Microdados/`
Download automatizado em `src/ingestion/download_pof.py`.

| Arquivo | Versão | Data do download |
|---|---|---|
| Dados_20230713.zip | 13/07/2023 | 16/11/2026 |
| Documentacao_20230713.zip | 13/07/2023 | 16/11/2026 |
| Programas_de_Leitura_20230713.zip | 13/07/2023 | 16/11/2026 |

Registros usados: `CONSUMO_ALIMENTAR.txt`, `MORADOR.txt`, `ALUGUEL_ESTIMADO.txt`.

## Variáveis do registro Consumo Alimentar
Posições copiadas do **dicionário** (`Dicionários de variáveis.xls`, aba Consumo Alimentar).

| Variável | Posição inicial | Tamanho | Significado | Valores |
|---|---|---|---|---|
| UF | 1 | 2 | Unidade da Federação | 11–53 (código IBGE) |
| TIPO_SITUACAO_REG | 7 | 1 | Situação do domicílio | 1 urbano, 2 rural |
| COD_UPA | 8 | 9 | Unidade primária de amostragem | UF(2) + seq(6) + DV(1) |
| NUM_DOM | 17 | 2 | Número do domicílio na UPA | |
| NUM_UC | 19 | 1 | Unidade de consumo no domicílio | |
| COD_INFORMANTE | 20 | 2 | Número da pessoa | |
| QUADRO | 22 | 2 | Dia do recordatório | 72 = 1º dia, 73 = 2º dia |
| V9001 | 32 | 7 | Código do alimento | ver cadastro |
| V9015 | 39 | 3 | Horário do consumo | "0H" a "23H" |
| V9017 | 44 | 1 | Ocasião de consumo | 1 café, 2 almoço, 3 lanche, 4 jantar, 5 ceia, 6 outra, 9 n.d. |
| V9018 | 45 | 1 | Local da refeição | 1 em casa, 2 merenda, 3 restaurante a quilo, 4 lanchonete/fast food, 5 restaurante, 6 ambulante, 7 fora outros, 9 n.d. |
| PESO_FINAL | 504 | 15 | Peso amostral calibrado | usar em toda taxa/estimativa |
| RENDA_TOTAL | 519 | 10 | Renda bruta mensal da UC (R$) | |
| DIA_SEMANA | 529 | 15 | Dia da semana do consumo | "Domingo" … "Sabado" |
| DIA_ATIPICO | 544 | 1 | Dia atípico na alimentação | 1 sim, 2 não |

Chave de uma pessoa: `COD_UPA + NUM_DOM + NUM_UC + COD_INFORMANTE`.
Chave de uma linha: pessoa + `QUADRO + SEQ`.

### Outros registros
| Registro | Variável | Posição | Tamanho | Uso |
|---|---|---|---|---|
| Morador | V0403 | 33 | 3 | Idade |
| Morador | V0404 | 36 | 1 | Sexo (1 homem, 2 mulher) |
| Morador | V0405 | 37 | 1 | Cor ou raça |
| Morador | ANOS_ESTUDO | 76 | 2 | Escolaridade |
| Morador | RENDA_DISP_PC | 117 | 20 | Renda disponível per capita (pode ser negativa) |
| Aluguel Estimado | V9010 | 41 | 2 | Mês anterior à abertura da entrevista → mês da entrevista = V9010 + 1 |

## Códigos de alimentos
Regra: **subgrupo oficial do IBGE** (`Classificação dos grupos de Consumo Alimentar.xlsx`) **∪ busca por nome no cadastro**, menos exclusões explícitas. Lista completa: `python -m src.ingestion.codigos_alimentos`.

| Grupo | Nº de códigos | Regra | Observação |
|---|---|---|---|
| Pizza | 19 | Subgrupo "Pizzas" ∪ nome contém "PIZZA" | Inclui calzone e pizza doce |
| Refrigerante | 33 | Subgrupos "Refrigerantes" e "Refrigerantes diet/light" ∪ nomes de refrigerante/marca | Excluídos: refresco de guaraná natural, água gaseificada com sumo, mate, ice tea, Tampico |
| Guaraná | 6 | Refrigerantes com "GUARANA" no nome | Tradicional, light e diet. **Não identifica marca** |

## Resultados da verificação (`notebooks/01_verificacao_pof.py`)
| Medida | Valor |
|---|---|
| Linhas do Consumo Alimentar | 1.175.390 |
| Pessoas | 46.164 |
| Pessoa-dias | 85.018 |
| Refeições (pessoa × dia × V9017) | 363.412 |
| Refeições com pizza | 1.724 (0,63% ponderado) |
| Refeições com pizza + refri | 821 (0,28%) |
| Refeições com pizza + guaraná | 187 (0,06%) |
| Participação do guaraná nas ocasiões pizza + refri | 22,8% |

## Armadilhas encontradas
1. **Encoding dos nomes no zip:** os nomes com acento falham no `unzip` do macOS. Resolvido extraindo com `zipfile` e normalizando os nomes.
2. **Script em R do IBGE com larguras erradas:** as últimas colunas do Consumo Alimentar (PESO_FINAL, RENDA_TOTAL, DIA_SEMANA) saem deslocadas. Resolvido usando as posições do dicionário.
3. **Itens com nome de marca:** Coca-Cola, Fanta, Sukita e Tubaína têm códigos próprios. Uma busca só por "REFRIGERANTE" perde cerca de 190 ocasiões.
4. **Sem data da entrevista:** só há o mês aproximado (via Aluguel Estimado), sem ano, para cerca de 86% das UCs.
5. **Poucos sábados:** o recordatório cobre o dia anterior à entrevista, e há pouca entrevista no domingo. Usar `PESO_FINAL` nas taxas.

## Dúvidas
- Julho aparece em 2017 e 2018: como tratar no split temporal?
- Usar `RENDA_TOTAL / nº de moradores` ou `RENDA_DISP_PC` como proxy de classe?
