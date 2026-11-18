# Data Card – Occasion Intelligence Pizza + Guaraná

Seções exigidas pelo TAPI (Seção 4). Itens marcados com _A preencher_ ainda estão em aberto.

## 1. Unidade de análise
Pessoa × dia de recordatório × período do dia (manhã 5–10h, almoço 11–14h, tarde 15–17h, noite 18–4h).

Cada pessoa da POF relatou tudo o que comeu em 1 ou 2 dias não consecutivos. A grade cruza cada pessoa-dia com os 4 períodos, para que os períodos **sem** a ocasião também apareçam como exemplos negativos.

## 2. Target
- **Definição:** `y = 1` se a pessoa consumiu pizza **e** refrigerante no mesmo período do mesmo dia.
- **Tipo:** ground truth (consumo declarado em recordatório de 24h, POF 2017–2018).
- **Tamanho:** 838 casos positivos em 340.072 linhas (0,31% ponderado), em 817 pessoas de 522 domicílios.
- **Códigos de alimentos:** classificação oficial do IBGE (subgrupos "Pizzas", "Refrigerantes", "Refrigerantes diet/light") mais busca pelo nome no cadastro, menos exclusões explícitas (mate, ice tea, Tampico, refresco natural). Lista em `src/ingestion/codigos_alimentos.py`.
- **Medida de marca (não é target):** participação do refrigerante de guaraná dentro das ocasiões positivas.
- **Por que não pizza + guaraná como target:** são só 196 casos; com o split temporal sobrariam 55 no teste, pouco para avaliar com estabilidade. A participação do guaraná é analisada de forma descritiva, por segmento.

## 3. Instante de previsão, features e anti-leakage
**Instante de previsão:** início do dia. Toda feature precisa ser conhecida antes de a pessoa comer, e também precisa ser algo que um time de marketing consiga informar sobre um contexto ou público.

**Features do modelo**
| Grupo | Features |
|---|---|
| Quando | `periodo`, `dia_semana` |
| Onde | `UF` (ou `regiao`, na regressão logística), `urbano` |
| Quem | `idade` (ou `faixa_idade`), `sexo`, `anos_estudo`, `n_moradores`, `renda_pc` |

Escolhidas pela triagem com validação cruzada no treino (abaixo). `tipo_area` e `classe_abep` continuam na base para os segmentos e para o Decision Engine.

**Alternativas a comparar nos experimentos:** `faixa_renda` no lugar de `renda_pc`; `fim_de_semana` no lugar de `dia_semana`.

**Excluídas por leakage**
- `V9017` (tipo de refeição), `V9018` (local da refeição) e outros alimentos do mesmo período: são observados junto com o consumo. O local da pizza fica na base só como coluna de análise (`local_pizza`): mede o desempenho por canal e dá as taxas de consumo em casa por segmento que o Decision Engine usa.
- `dia_atipico`: perguntado depois do dia ("foi um dia atípico?"); um dia com pizza tende a ser declarado atípico justamente por causa dela.

**Excluídas pelo desenho da validação**
- `mes_entrevista`, `estacao`: o treino vai de julho a fevereiro e o teste de março a julho, então o outono nunca aparece no treino. Com só 12 meses, o efeito do mês também se confunde com quais regiões foram entrevistadas em cada época.
- `ordem_mes`: só define o split.

**Triagem de candidatas** ([notebooks/02_triagem_features.py](../notebooks/02_triagem_features.py), 18/11/2026)

Cada grupo foi somado às 9 features de partida (`periodo`, `dia_semana`, `UF`, `urbano`, `idade`, `sexo`, `anos_estudo`, `n_moradores`, `renda_pc`); o tipo de área entrou no lugar de `urbano`. A avaliação usou só o treino: validação cruzada em 5 folds agrupada por domicílio, repetida 10 vezes (50 folds), com PR-AUC ponderada, num LightGBM raso e numa Regressão Logística simples. Um grupo entraria se o ganho fosse significativo a 5% nos dois modelos, pelo teste t corrigido para validação cruzada repetida (Nadeau e Bengio, 2003); senão, fica a opção mais simples. Nenhum entrou. A PR-AUC da base foi 0,025 no LightGBM e 0,036 na Regressão Logística.

| Candidata | Ganho de PR-AUC no LightGBM (p) | Ganho na Regressão Logística (p) | Decisão |
|---|---|---|---|
| Tipo de área no lugar de `urbano` | −0,0024 (0,83) | −0,0036 (0,75) | Não entra; fica para segmentos |
| Classe ABEP | +0,0004 (0,45) | −0,0014 (0,63) | Não entra; fica para segmentos e Decision Engine |
| Composição da família (crianças, adolescentes, adultos, idosos, posição na família) | −0,0008 (0,65) | −0,0018 (0,77) | Não entra |
| Bens, cartão de crédito, conta, plano de saúde, apartamento, aluguel | +0,0040 (0,15) | −0,0062 (0,81) | Não entra |
| Trabalho ou rendimento, frequência à escola | 0,0000 (0,50) | −0,0003 (0,63) | Não entra |

Versões anteriores, substituídas por esta: uma exploratória, de 16/11/2026, mediu os ganhos no período de teste; outra, de 18/11, usou uma única partição em 5 folds e se mostrou instável, porque uma mudança irrelevante no split inverteu a decisão de um grupo.

**Fora por desenho** (não medidas)
| Candidata | Motivo |
|---|---|
| Dieta, IMC, insegurança alimentar | Dados sensíveis (seção 9) |
| Refrigerante consumido no outro dia de recordatório | Só existe para o 2º dia (metade da base) e exigiria, na aplicação, conhecer o consumo individual de ontem, que o marketing não tem |
| Clima mensal por UF | A POF não tem a data exata, e a média do mês se confunde com a UF. Chuva e feriado do dia entram no Decision Engine, não no modelo |

## 4. Fontes e versões
| Fonte | Papel | Versão / data de download | Granularidade | Chave de integração |
|---|---|---|---|---|
| POF 2017–2018 – microdados (IBGE) | Ground truth + features | Pacote de 13/07/2023, baixado em 16/11/2026 | Pessoa × item consumido; morador; UC; domicílio | Chaves da POF (abaixo) |
| Critério Brasil – ABEP | Tabela de pontos e cortes da classe social | Versão válida a partir de 01/06/2019 | Domicílio | — |
| Open-Meteo / BrasilAPI | Chuva e feriado do dia no Decision Engine (não entram no modelo) | _A preencher_ | Município × dia | Município + data |
| Vigitel 2006–2024, Censo 2022, OSM | Opcionais: contexto recente para o Mapa de Oportunidade | _A preencher_ | UF / capital | UF |

Registros da POF usados: Consumo Alimentar, Morador, Aluguel Estimado, Despesa Coletiva (quadro 10), Domicílio e Inventário.

## 5. Regras de integração
- **Chaves:** domicílio = `COD_UPA + NUM_DOM`; unidade de consumo (UC) = domicílio + `NUM_UC`; pessoa = UC + `COD_INFORMANTE`.
- **Mês da entrevista** (por UC; usado só no split). A POF pública não tem a data, então ela é reconstruída em três níveis (coluna `fonte_mes`):

  | Nível | Quem | Regra | UCs |
  |---|---|---|---|
  | `aluguel_estimado` | Imóvel próprio ou cedido | Mês = `V9010 + 1` do Aluguel Estimado; julho de 2017 ou 2018 pelo `DEFLATOR` (> 1 → 2017) | 16.999 (84,4%) |
  | `aluguel_pago` | Inquilinos | Mês = último aluguel pago + 1; ano e coerência pela mediana dos vizinhos da mesma UPA (se distante mais de 2 meses, usa a mediana) | 2.695 (13,4%) |
  | `vizinhos_upa` | Inquilinos sem aluguel pago ou com pagamento incoerente | Mediana do mês das UCs da mesma UPA | 451 (2,2%) |
  | — | Sem pagamento e sem vizinhos | Fica fora do split temporal | 1 |

  **Validação da regra dos inquilinos:** aplicada a 3.943 UCs com mês conhecido que pagam prestação ou condomínio, fingindo não saber o mês: erro de no máximo 1 mês em 96,1% e mesmo lado do corte treino/teste em 97,0% (só vizinhos: 81,5% e 86,7%). Reproduzível em `notebooks/01_verificacao_pof.py`, seção 6.
- **Split temporal:** treino com entrevistas de jul/2017 a fev/2018 (223.748 linhas, 523 positivos); teste de mar/2018 a jul/2018 (116.316 linhas, 315 positivos). O lado do corte é decidido por domicílio, porque as UCs de um domicílio são entrevistadas na mesma visita e a família divide a pizza. Vale o mês da UC com a fonte mais confiável (aluguel estimado, depois aluguel pago, depois vizinhos) e, em empate, o mais cedo.
- **Tipo de área:** `ESTRATO_POF` (registro Morador) mapeado pela tabela oficial "Estratos POF 2017-2018.xls" em capital, região metropolitana, interior urbano e rural. O "rural" coincide 100% com a situação do domicílio.
- **Classe ABEP:** pontos do Critério Brasil por UC, somando bens do Inventário, banheiros, água e pavimentação do Domicílio e a instrução da pessoa de referência (Morador). A distribuição ponderada fica próxima da oficial:

  | | DE | C2 | C1 | B2 | B1 | A |
  |---|---|---|---|---|---|---|
  | POF (calculada) | 32,2% | 25,4% | 20,2% | 15,8% | 4,4% | 2,1% |
  | ABEP 2019 (oficial) | 28,3% | 26,8% | 21,5% | 16,5% | 4,4% | 2,5% |
- **Renda:** `RENDA_TOTAL` da UC ÷ nº de moradores (`renda_pc`, feature); em 5 faixas de salário mínimo de 2018, R$ 954 (`faixa_renda`, alternativa).

## 6. Dicionário de variáveis
Base `data/processed/grid.parquet` (gerada por `src/features/build_grid.py`): 340.072 linhas = 85.018 pessoa-dias × 4 períodos.

| Variável | Origem | Tipo | Papel | Descrição |
|---|---|---|---|---|
| `y` | Consumo (V9001) | 0/1 | **Target** | Pizza e refrigerante no mesmo período do dia |
| `y_guarana` | Consumo (V9001) | 0/1 | Análise | Pizza e refrigerante de guaraná no mesmo período |
| `pizza`, `refri`, `guarana` | Consumo (V9001) | bool | Análise | Consumo de cada item no período |
| `local_pizza` | Consumo (V9018) | cat | Análise (leakage) | Onde a pizza do período foi comida: `casa`, `estabelecimento` (restaurante, a quilo, lanchonete) ou `outro_fora` (merenda, ambulante, outros); nulo se não houve pizza |
| `peso` | Consumo (PESO_FINAL) | float | Ponderação | Peso amostral; usar em taxas e métricas populacionais |
| `split` | Derivada | cat | Validação | `treino`, `teste`, `sem_mes` |
| `ano_entrevista`, `mes_entrevista`, `ordem_mes` | Aluguel Estimado / Despesa Coletiva | float | Validação | Ano e mês estimados da entrevista; ordem 0 = jul/2017 … 12 = jul/2018 |
| `fonte_mes` | Derivada | cat | Validação | Como o mês foi obtido (seção 5) |
| `estacao` | Derivada do mês | cat | Análise | Estação do ano (hemisfério sul) |
| `periodo` | Consumo (V9015) | cat | **Feature** | manhã, almoço, tarde, noite |
| `dia_semana` | Consumo (DIA_SEMANA) | cat | **Feature** | Dia da semana do recordatório |
| `UF`, `regiao` | Consumo (UF) | cat | **Feature** | Unidade da Federação e macrorregião |
| `tipo_area` | Morador (ESTRATO_POF) + tabela de estratos | cat | Segmento | capital, regiao_metropolitana, interior_urbano, rural |
| `idade`, `faixa_idade` | Morador (V0403) | int / cat | **Feature** | Idade e faixa etária (10–17, 18–29, 30–44, 45–59, 60+) |
| `sexo` | Morador (V0404) | cat | **Feature** | masculino / feminino |
| `anos_estudo` | Morador (ANOS_ESTUDO) | int | **Feature** | Anos de estudo (fundamental de 9 anos) |
| `n_moradores` | Morador (contagem por UC) | int | **Feature** | Tamanho da unidade de consumo |
| `abep_pontos`, `classe_abep` | Inventário + Domicílio + Morador | int / cat | Segmento | Pontos e classe do Critério Brasil (DE, C2, C1, B2, B1, A); também usada no Decision Engine |
| `renda_pc` | Consumo (RENDA_TOTAL) ÷ n_moradores | float | **Feature** | Renda per capita da UC |
| `faixa_renda` | Derivada de `renda_pc` | cat | Alternativa | Faixa de renda per capita em salários mínimos de 2018 |
| `urbano` | Consumo (TIPO_SITUACAO_REG) | 0/1 | **Feature** | Domicílio urbano (o rural coincide com o de `tipo_area`) |
| `fim_de_semana` | Derivada | 0/1 | Alternativa | Sexta, sábado ou domingo |
| `dia_atipico` | Consumo (DIA_ATIPICO) | 0/1 | Análise (leakage) | Pessoa declarou que o dia foi atípico |

## 7. População coberta e excluída
- **Coberta:** 46.164 pessoas de 10 anos ou mais, em 20.112 domicílios particulares (subamostra do módulo de consumo alimentar da POF), com 85.018 dias de recordatório: 38.854 pessoas responderam os 2 dias e 7.310 só 1.
- **Excluída pela própria pesquisa:** menores de 10 anos (fora do módulo de consumo) e domicílios coletivos.
- **Fora do split temporal:** 1 pessoa (UC sem pagamento de moradia e sem vizinhos com mês conhecido).
- **Representatividade:** usar `peso` em taxas e métricas populacionais. Os sábados são sub-representados na amostra (o recordatório é do dia anterior e há poucas entrevistas aos domingos), e o peso corrige isso.

## 8. Vieses e limitações
- **Dados de 2017–2018:** anteriores à pandemia e ao crescimento do delivery.
- **Consumo autodeclarado:** sujeito a esquecimento e a subdeclaração de alimentos vistos como pouco saudáveis.
- **Sem marca:** a POF não identifica marca; "guaraná" inclui Guaraná Antarctica, Kuat e marcas regionais.
- **Sem delivery:** a POF registra só onde a pizza foi comida. Pizza pedida por delivery e comida em casa conta como `casa`, então uma regra de delivery no Decision Engine é premissa de negócio, não dado.
- **Mês da entrevista estimado:** erro de até 1 mês em cerca de 96% dos casos; nos inquilinos a estimativa é menos precisa que nos demais.
- **Classe ABEP aproximada:**
  - Faltam empregados domésticos e secadora (subestima um pouco as classes altas).
  - A POF conta como banheiro só o que tem chuveiro e vaso.
  - Não separa automóvel de uso profissional.
  - Usa a pessoa de referência como chefe da família.
- **Positivos agrupados:** pizza é compartilhada, então os casos se concentram em poucos domicílios (os 315 positivos do teste estão em 189 domicílios). Validação cruzada e intervalos de confiança precisam agrupar por domicílio.
- **Poucos positivos:** 838 casos no total. As métricas de segmentos pequenos (ex.: classe A, com cerca de 20 ocasiões) têm incerteza alta.

## 9. Licença e LGPD
- **Fonte:** microdados públicos do IBGE, citar como "IBGE, Pesquisa de Orçamentos Familiares 2017-2018".
- **Anonimização:** a base não tem nome, endereço nem documento. As chaves são códigos internos da pesquisa, e não há tentativa de reidentificação.
- **Variáveis sensíveis:** cor/raça, dieta, peso/altura e insegurança alimentar não são usadas como feature.
- **Resultados:** o modelo e a aplicação trabalham com perfis e contextos agregados, não com pessoas identificáveis.
