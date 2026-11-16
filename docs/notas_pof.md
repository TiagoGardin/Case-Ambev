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
4. **Sem data da entrevista:** reconstruídos ano e mês via Aluguel Estimado (`V9010 + 1` + `DEFLATOR` para separar os dois julhos), para cerca de 86% das UCs. Ver decisão 1.
5. **Poucos sábados:** o recordatório cobre o dia anterior à entrevista, e há pouca entrevista no domingo. Usar `PESO_FINAL` nas taxas.

## Decisões

### 1. Ano e mês da entrevista e split temporal
**Problema:** a coleta foi de jul/2017 a jul/2018. `V9010 + 1` dá o mês, mas julho aparece nos dois anos.

**Solução:** os outros meses ocorrem uma única vez (ago–dez → 2017, jan–jun → 2018). Para julho, o `DEFLATOR` do Aluguel Estimado leva valores a preços de 15/01/2018: é > 1 para valores de 2017 (corrige para cima) e < 1 para 2018.

**Evidência:**
- Em julho, a distribuição do `DEFLATOR` é bimodal, sem nenhum valor entre 0,985 e 1,0.
- Todas as 1.836 UPAs com entrevista em julho caem num único ano.
- Os meses vizinhos confirmam o padrão: junho/2018 tem média de 0,983, agosto/2017 tem 1,013.

**Resultado:**
| Mês | UCs |
|---|---|
| Julho/2017 | 2.974 |
| Julho/2018 | 1.109 (fim da coleta) |

Coluna `ordem_mes`: 0 = jul/2017 … 12 = jul/2018.

**Split temporal:**
| Conjunto | Período | `ordem_mes` | Parcela das UCs |
|---|---|---|---|
| Treino | jul/2017 a fev/2018 | 0–7 | ~65% |
| Teste | mar/2018 a jul/2018 | 8–12 | ~35% |
| Sem mês | — | — | ~14% das pessoas (fica fora do split temporal; usada no split por região) |

### 2. Renda (proxy de classe)
**Decisão:** renda per capita = `RENDA_TOTAL` da UC ÷ nº de moradores da UC, em faixas de salário mínimo de 2018 (R$ 954).

| Faixa | Renda per capita |
|---|---|
| 1 | até ½ SM |
| 2 | ½ a 1 SM |
| 3 | 1 a 2 SM |
| 4 | 2 a 5 SM |
| 5 | acima de 5 SM |

**Por quê:**
- `RENDA_DISP_PC` pode ser negativa (por causa das deduções), o que é difícil de interpretar.
- Faixas de SM são legíveis para o negócio (aproximam "classe social").
- Faixas também são uma entrada simples para a API, ao contrário de um valor em reais de 2018.

**Limitação:** não é a classificação ABEP (Critério Brasil) usada pelo mercado. É apenas uma aproximação de classe.
