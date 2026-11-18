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

Registros usados: `CONSUMO_ALIMENTAR.txt`, `MORADOR.txt`, `ALUGUEL_ESTIMADO.txt`, `DESPESA_COLETIVA.txt` (quadro 10), `DOMICILIO.txt`, `INVENTARIO.txt`.

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
| Morador | ESTRATO_POF | 3 | 4 | Estrato geográfico → tipo de área (decisão 4) |
| Morador | V0306 | 22 | 2 | Condição na UC (1 = pessoa de referência) |
| Morador | V0403 | 33 | 3 | Idade |
| Morador | V0404 | 36 | 1 | Sexo (1 homem, 2 mulher) |
| Morador | V0405 | 37 | 1 | Cor ou raça (lida, não usada) |
| Morador | ANOS_ESTUDO | 76 | 2 | Anos de estudo (fundamental de 9 anos: completo = 9) |
| Morador | NIVEL_INSTRUCAO | 116 | 1 | 1 sem instrução … 7 superior completo |
| Morador | RENDA_DISP_PC | 117 | 20 | Renda disponível per capita (pode ser negativa) |
| Aluguel Estimado | V9010 | 41 | 2 | Mês anterior à abertura da entrevista → mês da entrevista = V9010 + 1 |
| Aluguel Estimado | DEFLATOR | 45 | 12 | Leva valores a preços de 15/01/2018 (separa os dois julhos) |
| Despesa Coletiva | QUADRO | 20 | 2 | 10 = despesas com moradia |
| Despesa Coletiva | V9001 | 24 | 7 | 1000101 aluguel, 1000301 prestação, 1000501 condomínio |
| Despesa Coletiva | V9010 | 47 | 2 | Último mês em que a despesa foi paga |
| Despesa Coletiva | V9011 | 49 | 2 | Nº de meses pagos no período de referência |
| Domicílio | V0207 | 26 | 1 | Abastecimento de água (1 = rede geral) |
| Domicílio | V02111 | 34 | 2 | Banheiros de uso exclusivo (com chuveiro e vaso) |
| Domicílio | V0220 | 49 | 1 | Rua pavimentada (1 sim, 2 não) |
| Inventário | V9001 | 24 | 7 | Código do bem durável (quadro 14) |
| Inventário | V9005 | 31 | 2 | Quantidade do bem |

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
4. **Sem data da entrevista:** ano e mês reconstruídos via Aluguel Estimado (`V9010 + 1`, com o `DEFLATOR` separando os dois julhos). Inquilinos não têm esse registro e ficavam todos fora do split; o mês deles vem do aluguel pago e dos vizinhos da UPA. Ver decisões 1 e 3.
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

**Split temporal** (já com os inquilinos da decisão 3):
| Conjunto | Período | `ordem_mes` | Linhas da grade | Positivos |
|---|---|---|---|---|
| Treino | jul/2017 a fev/2018 | 0–7 | 223.768 (65,8%) | 523 |
| Teste | mar/2018 a jul/2018 | 8–12 | 116.296 (34,2%) | 315 |
| Sem mês | — | — | 8 (1 pessoa) | 0 |

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

**Atualização (17/11):** a classe ABEP (decisão 5) virou a feature principal de classe social. A renda fica como alternativa a comparar nos experimentos.

### 3. Mês da entrevista dos inquilinos
**Problema:** quem mora de aluguel não tem o registro de Aluguel Estimado. Na primeira versão, 99% das pessoas "sem mês" eram inquilinos, e treino e teste tinham **zero** inquilinos (cerca de 16% da população, com taxa da ocasião maior que a média).

**Solução** (`src/features/mes_entrevista.py`), em três níveis:
1. Com Aluguel Estimado: regra da decisão 1.
2. Inquilino com aluguel pago: mês = último mês pago (`V9010` do quadro 10) + 1, a mesma lógica do Aluguel Estimado. O ano vem dos vizinhos: UCs da mesma UPA são entrevistadas na mesma época (em geral dentro de 2–3 meses). Se o mês do pagamento ficar a mais de 2 meses da mediana dos vizinhos, usa a mediana.
3. Sem aluguel pago: mediana dos vizinhos da UPA.

**Validação:** como o mês real dos inquilinos é desconhecido, a regra foi aplicada a UCs com mês conhecido que pagam prestação ou condomínio (mesmo tipo de despesa mensal), fingindo não saber o mês, e sem usar a própria UC na mediana dos vizinhos.

| Teste | Regra | Erro ≤ 1 mês | Mesmo lado do corte treino/teste |
|---|---|---|---|
| Prestação ou condomínio (3.943 UCs) | Pagamento + vizinhos | 96,1% | 97,0% |
| Prestação ou condomínio | Só vizinhos | 81,5% | 86,7% |
| Aluguel pago, casos atípicos com os dois registros (227 UCs) | Pagamento + vizinhos | 85,5% | 91,6% |

**Resultado:** 2.695 UCs pelo aluguel pago, 451 pelos vizinhos e 1 sem mês.

### 4. Tipo de área
**Decisão:** `ESTRATO_POF` mapeado pela tabela oficial `Estratos POF 2017-2018.xls` em capital, região metropolitana, interior urbano e rural (`src/features/tipo_area.py`). Substitui `urbano` como feature.

**Por quê:** `urbano` esconde uma diferença grande. Na triagem, a taxa noturna da ocasião foi 1,80% na capital, 1,44% na RM, 1,02% no interior urbano e 0,18% no rural. O "rural" do estrato coincide 100% com a situação do domicílio, então a nova variável só detalha o urbano.

### 5. Classe social pelo Critério Brasil (ABEP)
**Decisão:** pontos do Critério Brasil (versão de 01/06/2019) calculados por UC (`src/features/classe_abep.py`):

| Componente | Fonte na POF |
|---|---|
| Automóvel, microcomputador, lava-louça, geladeira, freezer, lava-roupa, DVD, micro-ondas, moto | Inventário (quadro 14); a geladeira duplex conta também como freezer (regra da ABEP) |
| Banheiros | Domicílio (`V02111`) |
| Água encanada, rua pavimentada | Domicílio (`V0207`, `V0220`) |
| Instrução do chefe | Pessoa de referência (`V0306 = 1`): `NIVEL_INSTRUCAO`, com Fundamental I completo = 5 anos de estudo |

Cortes: DE 0–16, C2 17–22, C1 23–28, B2 29–37, B1 38–44, A 45+.

**Por quê:**
- Separa a ocasião melhor que a renda: a taxa noturna vai de 0,34% (DE) a 2,57% (B1), uma diferença de 7,6 vezes (nas faixas de renda, 3,3 vezes).
- É a classificação do mercado e do TAPI ("Classe A").
- A distribuição ponderada fica próxima da oficial (DE 32,2% contra 28,3%; A 2,1% contra 2,5%).

**Aproximações:** faltam empregados domésticos e secadora; a POF só conta banheiro com chuveiro; não separa automóvel profissional; a pessoa de referência faz o papel de chefe da família.
