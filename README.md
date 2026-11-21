# Occasion Intelligence – Pizza + Guaraná

Projeto do case Inteli Academy x Ambev (TAPI). Estima a propensão da ocasião **Pizza + Refrigerante** por contexto (público, região, dia, período) e transforma o **Occasion Score** em uma recomendação de ativação para Guaraná Antarctica.

> Status: em construção

## Arquitetura

```
fontes (POF 2017-2018 do IBGE; Critério Brasil da ABEP)
  → src/ingestion      (download e leitura dos microdados)
  → src/features       (mês da entrevista, tipo de área, classe ABEP,
                        grade pessoa × dia × período com target e features)
  → src/models         (baseline, candidatos, avaliação, SHAP)
  → src/decision_engine(regras documentadas)
  → src/api            (FastAPI)
  → app/               (Streamlit)
```

## Estrutura do repositório

| Pasta | Conteúdo |
|---|---|
| `data/raw` | Microdados originais (não versionados) |
| `data/external` | Fontes de contexto externas (ainda não usadas) |
| `data/processed` | Bases tratadas (`.parquet`) |
| `notebooks/` | EDA e experimentos |
| `src/` | Código do pipeline |
| `models/` | Modelo final congelado antes do teste e seus metadados, a referência do SHAP e o contexto do Decision Engine (versionados) |
| `reports/` | Gráficos para README e vídeo (`figures/`) e tabelas da análise exploratória |
| `docs/` | Data Card, notas e regras do Decision Engine |
| `tests/` | Testes (pytest) |

## Dados

- **Fonte do target:** POF 2017–2018 (IBGE), recordatório alimentar de 24h de 46.164 pessoas.
- **Unidade de análise:** pessoa × dia × período (340.072 linhas, 838 ocasiões de pizza + refrigerante).
- **Features:** quando (período, dia da semana), onde (UF, urbano ou rural) e quem (idade, sexo, escolaridade, tamanho da família, renda per capita), escolhidas por triagem com validação cruzada no treino. O modelo final usa o dia da semana na forma de fim de semana (sexta a domingo). Tipo de área e classe ABEP ficam para os segmentos e para o Decision Engine.
- **Validação:** split temporal pelo mês estimado da entrevista (treino jul/2017–fev/2018, teste mar–jul/2018).

Detalhes, decisões e limitações: [docs/data_card.md](docs/data_card.md) e [docs/notas_pof.md](docs/notas_pof.md).

## Análise exploratória

Só com o treino, com taxas ponderadas por `peso_semana` (o peso amostral com cada dia da semana valendo 1/7; ver Data Card, seção 7) e IC de 95% por bootstrap de UPAs ([notebooks/03_eda.py](notebooks/03_eda.py); todas as tabelas em [reports/eda_tabelas.md](reports/eda_tabelas.md)).

1. **A ocasião é da noite e cresce ao longo da semana.** A noite concentra 92% dos casos (481 de 523; 94% com o peso). A taxa noturna vai de 0,50% na segunda a 2,09% no sábado, e sexta a domingo têm 2,4 vezes a taxa de segunda a quinta (IC 1,7 a 3,4). Essa tabela é o baseline do modelo.

   ![Heatmap da taxa da ocasião por dia e período: perto de zero de manhã, no almoço e à tarde; à noite, de 0,50% na segunda a 2,09% no sábado](reports/figures/eda_1_dia_periodo.png)

2. **Classe, tipo de área e idade separam a ocasião mais que região e sexo.** À noite, a taxa vai de 0,34% na classe DE a 4,40% na B1, de 0,23% no rural a 1,71% na capital e de 0,61% aos 60 anos ou mais a 1,82% aos 18–29. Mesmo assim, tipo de área e classe não melhoraram o modelo além de `urbano`, `UF`, renda e escolaridade (ver *Escolha das features*, em Metodologia), e por isso entram como segmentos e no Decision Engine.

   ![Taxa noturna da ocasião por classe ABEP, tipo de área, região, idade e sexo, com intervalos de confiança](reports/figures/eda_2_perfil.png)

3. **O salto de fim de semana muda de tamanho entre as classes.** Sexta a domingo multiplicam a taxa noturna por 3,9 na DE (IC 1,1 a 11,7) e na C2 (IC 1,7 a 8,2) e por 3,2 na B1 (IC 1,1 a 12,3), contra 1,5 na B2 (IC 0,6 a 3,2). Não há um padrão claro por renda, e os intervalos são largos: é um indício de interação, não uma prova. É o tipo de efeito que uma regra fixa como "ativar nas noites de sexta" não captura.

   ![Taxa noturna por classe, de segunda a quinta e de sexta a domingo, com a razão entre as duas por classe](reports/figures/eda_3_classe_fim_de_semana.png)

4. **Quanto mais alta a classe, mais pizza e menos refrigerante junto.** A pizza aparece em 0,5% das noites na DE e em 7,7% na A (10,1% na B1), mas vem com refrigerante em 63% das vezes na DE e em 20% na A (IC 5% a 65%). A participação do guaraná nessas ocasiões não segue a classe (de 24% a 33%, com intervalos sobrepostos); por região, vai de 14% no Norte a 31% no Centro-Oeste, também com intervalos largos. Isso sugere que, nas classes altas, a oportunidade é associar o refrigerante à pizza (awareness), mais do que disputar marca.

   ![Por classe: noites com pizza, pizzas com refrigerante e ocasiões com guaraná, com intervalos de confiança](reports/figures/eda_4_pizza_refrigerante_guarana.png)

5. **Três em cada quatro pizzas são comidas em casa, e a região muda mais que a classe.** São 75% no total, com o delivery contado como casa, porque a POF não o separa. A fatia vai de 80% no Sudeste a 56% no Norte (IC 36% a 76%); por classe e tipo de área não há padrão claro. Isso sustenta o consumo em casa como regra padrão do Decision Engine, com a região como modulador.

   ![Fatia das pizzas comidas em casa por classe ABEP, tipo de área e região, com intervalos de confiança](reports/figures/eda_5_pizza_em_casa.png)

## Metodologia

Protocolo fixado antes da avaliação final no período de teste (mar–jul/2018). O histórico do git registra a data desta seção.

### Occasion Score
- **Definição:** probabilidade estimada de a pessoa comer pizza e tomar refrigerante no período, dado o contexto e o perfil: `P(y = 1 | período, dia da semana, UF, urbano, idade, sexo, escolaridade, tamanho da família, renda per capita)`. O instante de previsão é o início do dia ([Data Card](docs/data_card.md), seção 3).
- **Escala:** de 0 a 1, calibrada. Como a taxa média é baixa (0,30% no treino, ponderada), a aplicação mostra também quantas vezes o score supera essa média.
- **`classification`:** `alta` para os 10% de linhas (pessoa × dia × período) com maior score, `media` para os 20% seguintes e `baixa` para o resto. Os cortes são os percentis 90 e 70 do score, ponderados por `peso_semana`, nas previsões *out-of-fold* do treino. Eles ficam fixos daí em diante e vão no `model_metadata`.

### Métrica principal e custo dos erros
- **Falso positivo:** ativação gasta num contexto sem ocasião. **Falso negativo:** ocasião de pizza + refrigerante que a ativação não alcançou.
- Com orçamento limitado, a marca ativa só o topo do ranking, e os dois custos aparecem ali: a precisão do topo mede o desperdício (FP) e o recall mede as ocasiões perdidas (FN).
- **Principal: PR-AUC** (average precision), que resume precisão × recall em todos os cortes. Um score sem sinal tem PR-AUC igual à prevalência, então o ganho se lê como múltiplo dela.
- **Leitura de negócio: captura no top 10%**, a fração das ocasiões que cai nos 10% de linhas com maior score.
- **Por que não acurácia nem ROC-AUC:** com 99,7% de negativos, as duas ficam altas mesmo com um topo ruim. O ROC-AUC entra como complementar.

### Baseline e candidatos
- **Baseline:** taxa ponderada de `y` por dia da semana × período no treino (28 células), usada diretamente como score. É o "sexta à noite é bom para pizza" em forma de número.
- **Regressão Logística:** categóricas em one-hot, numéricas padronizadas (anos de estudo, moradores e log da renda per capita) e regularização L2; `regiao` no lugar de `UF` (27 categorias são muitas para 523 positivos), `faixa_idade` no lugar de `idade` (o efeito da idade não é linear) e interação período × dia da semana, para conter o baseline. Na seleção, o fim de semana entrou no lugar do dia da semana (ver Resultados).
- **LightGBM raso:** poucas folhas e `min_child_samples` alto, porque o treino tem só 523 positivos.
- **Preparo único:** todo modelo começa pelo mesmo preparo de features ([src/models/preparo.py](src/models/preparo.py)), que a API e o app também usam. Ele não aprende nada dos dados, porque faixas e regiões são fixas, e recusa valores que o modelo não conhece.

### Validação
1. **Validação cruzada repetida no treino** (jul/2017–fev/2018): `StratifiedGroupKFold` com 5 folds, agrupado por domicílio para que a mesma família (que divide a pizza) não fique dos dois lados, e repetido 10 vezes com partições diferentes. Com só 523 positivos, uma única partição é ruidosa demais para decidir. A PR-AUC ponderada decide:
   - o modelo final (Regressão Logística ou LightGBM);
   - as alternativas do Data Card: dia da semana ou fim de semana, renda contínua ou em faixas;
   - hiperparâmetros, uso do `peso_semana` no ajuste e calibração (sigmoide sobre as previsões *out-of-fold*, se a curva de confiabilidade pedir).

   Entre duas opções, a mais complexa só vence se o ganho for significativo a 5% no teste t corrigido para validação cruzada repetida (Nadeau e Bengio, 2003); senão, fica a mais simples. Os hiperparâmetros seguem só a maior PR-AUC média.
2. **Teste temporal** (mar–jul/2018): o modelo escolhido, treinado em todo o treino, é avaliado **uma única vez**, ao lado do baseline. Nada é reajustado depois; qualquer mudança posterior é reportada como exploratória. A aplicação usa esse mesmo modelo.
3. **Generalização por região** (complementar): dentro do treino, ajustar com 4 regiões e avaliar na 5ª.

**Escolha das features.** Uma triagem com essa validação cruzada repetida, só no treino ([notebooks/02_triagem_features.py](notebooks/02_triagem_features.py)), testou cinco grupos de candidatas: tipo de área, classe ABEP, composição da família, bens e finanças, trabalho e estudo. Nenhum trouxe ganho significativo, nem no LightGBM nem na Regressão Logística, então ficaram as 9 features de partida ([Data Card](docs/data_card.md), seção 3). Ela substituiu duas versões anteriores. Uma, exploratória e de 16/11, mediu no período de teste o ganho das candidatas sobre um LightGBM com as mesmas 9 features e indicou incluir tipo de área e classe ABEP, que esta triagem descartou. A outra usou uma única partição e se mostrou instável (ver o adendo de 20/11, abaixo). As features não mudam depois do teste.

### Avaliação no teste
- **Métricas, sempre ao lado do baseline:** PR-AUC; captura no top 5%, 10% e 20%; ROC-AUC; precisão, recall, F1 e matriz de confusão no corte de `alta`; Brier e curva de calibração.
- **Peso:** métricas ponderadas por `peso_semana`, o PESO_FINAL ajustado para cada dia da semana valer 1/7. O PESO_FINAL expande pessoas, não dias, e a amostra tem poucas sextas e sábados (9% e 5% dos dias). A versão sem peso fica como checagem.
- **Incerteza:** intervalos de 95% por bootstrap de UPAs (1.000 reamostragens), como na análise exploratória. A UPA é a unidade primária da amostra e contém os domicílios, onde os positivos se concentram (os 315 do teste estão em 189 domicílios). O modelo só é declarado melhor que o baseline se o intervalo da diferença de PR-AUC (bootstrap pareado) não incluir zero.
- **Mudança de prevalência:** a taxa sobe de 0,30% no treino para 0,41% no teste. A comparação com o baseline é sempre no mesmo conjunto, e o score não é recalibrado com o teste.
- **Segmentos:** período, região, tipo de área e público (classe ABEP, faixa etária, sexo). Para canal, usa-se o local onde a pizza foi comida (`local_pizza`: casa, estabelecimento ou outro lugar fora). Como o local só existe para quem comeu pizza, o resultado por canal é a captura das ocasiões de cada local. Segmentos com menos de 30 positivos no teste são só descritos, sem métrica.
- **Erros:** perfil das ocasiões que ficaram em `baixa` (falsos negativos) e dos contextos em `alta` sem ocasião (falsos positivos).
- **Explicações:** SHAP global e local. Os 3 fatores de maior contribuição de cada previsão formam o `top_factors` da API.

### Adendo de 20/11/2026 (antes de abrir o teste)
Acrescentado depois da auditoria dos modelos e antes de qualquer avaliação no teste. Não muda o modelo nem as regras acima: só detalha como o resultado do teste será lido.
- **Referência para comparar com o teste:** no teste, a PR-AUC é calculada uma vez sobre todas as linhas. A referência comparável da validação cruzada é a PR-AUC agregada das previsões out-of-fold: 0,0276 no modelo final e 0,0143 no baseline ([notebooks/05_robustez.py](notebooks/05_robustez.py), seção 1). A média dos folds (0,0371 e 0,0160) sai enviesada para cima, porque cada fold tem cerca de 105 positivos. A regra para declarar o modelo melhor que o baseline não muda.
- **Resultado por mês:** PR-AUC, captura e taxa prevista × observada em cada mês do teste (março a julho), só como descrição. O motivo: dentro do treino, na validação com origem móvel, o ganho sobre o baseline variou entre janelas. Além disso, o período de teste tem a Copa do Mundo e as festas juninas, que o treino não tem.
- **Comparação exploratória com o dia da semana:** a Regressão Logística com dia da semana e interação período × dia (C = 0,03, sem peso no ajuste) entra no relatório do teste só como comparação.
  - Ela recebe a mesma calibração do modelo escolhido: sigmoide ponderada sobre as previsões out-of-fold das mesmas 50 partições e ajuste final em todo o treino com C × 4/5.
  - São comparadas a PR-AUC agregada, a captura e a calibração por dia da semana.
  - O modelo da aplicação continua o de [models/](models/), qualquer que seja o resultado.
- **Leitura anterior do período de teste:** em 16/11, uma triagem exploratória, feita fora do repositório, mediu no período de teste o ROC-AUC nas noites de um LightGBM com as mesmas 9 features (0,814) e o ganho de grupos de candidatas.
  - As inclusões decididas ali (tipo de área e classe ABEP) foram descartadas pela triagem atual, feita só no treino.
  - Nenhuma escolha do modelo final usa o teste, mas o período de teste não é inédito.

## Resultados

### Seleção do modelo (validação cruzada no treino)

Tudo no treino, com as 50 partições e as regras da Metodologia ([notebooks/04_modelagem.py](notebooks/04_modelagem.py)). PR-AUC e captura são ponderadas por `peso_semana`. A prevalência, que é a PR-AUC de um score sem sinal, é 0,0030. Cada passo parte do vencedor do anterior, e a opção mais complexa só vence com ganho significativo.

| Passo | Mais simples (PR-AUC) | Mais complexa (PR-AUC) | Ganho da mais complexa (p) | Fica |
|---|---|---|---|---|
| Modelo | baseline dia × período (0,0160) | Regressão Logística (0,0386) | +0,0226 (0,006) | Regressão Logística |
| Dia | fim de semana (0,0371) | dia da semana (0,0386) | +0,0015 (0,40) | fim de semana |
| Renda | contínua (0,0371) | 5 faixas (0,0306) | −0,0065 (0,76) | contínua |
| Peso no ajuste | sem peso (0,0371) | `peso_semana` (0,0414) | +0,0043 (0,22) | sem peso |
| Algoritmo | Regressão Logística (0,0371) | LightGBM raso, melhor de 6 (0,0284) | −0,0087 (0,82) | Regressão Logística |

- **Quais opções contam como mais simples** ficou escrito no notebook antes de rodar. Para as alternativas do Data Card e para o LightGBM, isso segue o número de parâmetros.
- **O peso no ajuste** era a decisão sem uma ordem óbvia. O ajuste sem peso foi tratado como o mais simples. O ajuste com peso teve a maior PR-AUC média, mas oscilou mais entre os folds (desvio-padrão de 0,0245 contra 0,0188), e o ganho não foi significativo. Tratado como hiperparâmetro (maior média), ele teria vencido.
- **Modelo escolhido:** Regressão Logística com regularização L2 (C = 0,003), ajustada sem peso. Usa:
  - período, fim de semana (sexta a domingo) e a interação período × fim de semana;
  - região, urbano, faixa etária e sexo;
  - anos de estudo, moradores e log da renda per capita.
- **Desempenho na validação cruzada:** PR-AUC 0,0371, 12,3 vezes a prevalência, contra 0,0160 do baseline (ganho +0,0211, p = 0,013, à frente em 48 de 50 folds). Captura no top 10%: 66,9%, contra 54,2% do baseline.
- **PR-AUC agregada:** a PR-AUC acima é a média dos 50 folds, como manda o protocolo. Calculada sobre todas as previsões out-of-fold de cada repetição, ela é 0,0276 no modelo final (9,1 vezes a prevalência) e 0,0143 no baseline.
  - Com ela, o modelo fica 1,9 vezes à frente do baseline, e não 2,3.
  - A diferença vem do tamanho da amostra de cada cálculo: com cerca de 105 positivos por fold, a PR-AUC sai enviesada para cima ([notebooks/05_robustez.py](notebooks/05_robustez.py), seção 1).
  - A captura no top 10% quase não muda: 66,9% contra 53,6% na versão agregada.

**Calibração.** A regularização forte deixa o score achatado em torno da média. Nas previsões out-of-fold:
- **Sem sigmoide:**
  - a média ficava abaixo da taxa ponderada (0,259% contra 0,302%);
  - no 1% de maior score, a previsão era 1,18% contra 4,45% observados;
  - na metade de baixo, 0,13% contra 0,02%.
- **Com sigmoide:** a sigmoide ponderada (a = 2,50, b = +8,23) estica o score sem mudar a ordem. Medida com a sigmoide ajustada fora de cada fold:
  - reduziu o log loss ponderado de 0,01796 para 0,01667 (p < 0,001);
  - acertou o nível geral (0,304% previsto contra 0,302%).
- **Desvios que ficam:**
  - no topo: no 1% de maior score, 5,7% previsto contra 4,4% observado; entre 5% e 10%, 1,1% contra 1,6%;
  - por dia, porque o modelo junta sexta com sábado e domingo: a sexta fica acima (0,47% previsto contra 0,31%) e a quinta abaixo (0,18% contra 0,28%).

**Modelo final.** O modelo final foi ajustado em todo o treino com C × 4/5 = 0,0024 ([src/models/treinar.py](src/models/treinar.py)).
- **Por quê:** no scikit-learn, o mesmo C regulariza relativamente menos quando há mais linhas. Cada modelo da validação cruzada usou 4/5 do treino, e a sigmoide foi ajustada nas previsões deles.
- **O efeito:** com o C da validação, o modelo final saía menos achatado, e o score calibrado ficava 16% acima da taxa do treino (0,351% contra 0,302%). Com C × 4/5, fica em 0,302%.
- **Quando foi feito:** tudo isso foi detectado e corrigido só com o treino, antes do teste. A validação cruzada aninhada confirmou a correção fora da amostra: sem ela, o score calibrado sairia em 0,403% para uma taxa de 0,302%; com ela, sai em 0,302% ([notebooks/05_robustez.py](notebooks/05_robustez.py), seção 9).
- **Reprodução:** um novo treino reproduz a sigmoide, os cortes e os scores exatamente, mas não o arquivo byte a byte. O SHA-256 dos metadados confere a integridade do arquivo, não a reprodução.

O modelo, a sigmoide, os cortes e as métricas estão em [models/](models/).

**Classificação.** Os cortes são fixos e vão no `model_metadata`. Os números abaixo são das previsões out-of-fold calibradas do treino:

| Classe | Score | Contextos | Ocasiões capturadas | Taxa observada | Vezes a média |
|---|---|---|---|---|---|
| `alta` | a partir de 0,777% | 10% | 67,0% | 2,03% | 6,7 |
| `media` | de 0,144% a 0,777% | 20% | 27,4% | 0,41% | 1,4 |
| `baixa` | abaixo de 0,144% | 70% | 5,6% | 0,02% | 0,1 |

### Robustez (checagens no treino)

As checagens foram feitas depois da seleção e da auditoria de 20/11, antes do teste ([notebooks/05_robustez.py](notebooks/05_robustez.py)). Elas descrevem o modelo escolhido e não mudam nada nele.

**Ao longo do tempo.** A validação cruzada da seleção usa partições aleatórias. Treinando com os meses anteriores e validando nos seguintes, o ganho sobre o baseline variou:

| Janela (treina → valida) | Positivos | PR-AUC baseline | PR-AUC final | Captura baseline | Captura final | Diferença de PR-AUC (IC 95%) |
|---|---|---|---|---|---|---|
| jul–set → out–nov | 151 | 0,0124 | 0,0291 | 45,4% | 62,3% | +0,0026 a +0,0580 |
| jul–nov → dez–jan | 105 | 0,0128 | 0,0151 | 55,1% | 42,1% | −0,0071 a +0,0269 |
| jul–jan → fev | 67 | 0,0250 | 0,0382 | 71,9% | 74,4% | −0,0028 a +0,0550 |

- O ganho só foi significativo na primeira janela.
- Em dez–jan, a captura do modelo ficou abaixo da do baseline.
- A Regressão Logística com dia da semana ficou à frente do modelo final nas três janelas, com PR-AUC de 0,0303, 0,0174 e 0,0490. Por isso ela entra como comparação exploratória no teste (adendo de 20/11).

**Em regiões não vistas.** Treinando com 4 regiões e avaliando na 5ª, o modelo supera o baseline em todas, com o intervalo da diferença acima de zero:

| Região avaliada | Positivos | PR-AUC baseline | PR-AUC final | Captura baseline | Captura final | Diferença de PR-AUC (IC 95%) |
|---|---|---|---|---|---|---|
| Norte | 66 | 0,0097 | 0,0190 | 49,7% | 67,8% | +0,0019 a +0,0326 |
| Nordeste | 171 | 0,0112 | 0,0263 | 57,0% | 74,8% | +0,0052 a +0,0457 |
| Sudeste | 113 | 0,0189 | 0,0349 | 61,3% | 62,4% | +0,0008 a +0,0618 |
| Sul | 95 | 0,0169 | 0,0309 | 38,8% | 61,2% | +0,0050 a +0,0417 |
| Centro-Oeste | 78 | 0,0109 | 0,0203 | 44,6% | 52,5% | +0,0015 a +0,0451 |

**Outras checagens:**
- **Vizinhança:** com partições por UPA em vez de domicílio, a PR-AUC média dos folds fica em 0,0359 no modelo final e 0,0163 no baseline, contra 0,0371 e 0,0160 por domicílio. Não há otimismo por vizinhos parecidos.
- **Poucos positivos efetivos:** como o `peso_semana` varia muito, os 523 positivos equivalem a 138 nas métricas ponderadas (tamanho efetivo de Kish).
  - A PR-AUC do modelo final vai de 0,0116 a 0,0905 entre os folds.
  - Sem peso, a conclusão é a mesma: 0,0248 contra 0,0116 do baseline.
  - Diferenças entre candidatos do tamanho das vistas na seleção (até 0,009) ficam abaixo do que o protocolo consegue separar.
- **Calibração por segmento:** com a sigmoide ajustada fora de cada fold, o nível geral bate, mas alguns segmentos com 30 ou mais ocasiões se afastam mais de 30%. A razão observado ÷ previsto fica em:
  - subestimados: 10–17 anos 1,78; B1 1,54; quinta 1,52; C2 1,30;
  - superestimados: renda acima de 5 salários mínimos 0,55; rural 0,63; sexta 0,66; 45–59 anos 0,69; B2 0,73; Centro-Oeste 0,74.

  Com o dia da semana, o erro médio da calibração por dia cairia de 21% para 16%.
- **Taxa por mês:** no treino, janeiro tem taxa bem menor que os outros meses: 0,133% (IC 0,072% a 0,191%), contra 0,28% a 0,39%. O mês não é feature.

### Avaliação no teste

_A preencher. O modelo de [models/](models/) é avaliado uma única vez no teste (mar–jul/2018), conforme a Metodologia e o adendo de 20/11._

## Decision Engine e API

### Explicação de cada previsão (`top_factors`)
O `top_factors` traz os 3 fatores que mais movem o score de cada contexto, calculados pelo SHAP exato da Regressão Logística ([src/models/explicar.py](src/models/explicar.py)).
- **Fórmula:** num modelo linear, o SHAP de cada coluna é coeficiente × (valor − referência). A referência é a média do treino, ponderada por `peso_semana`.
- **Fatores:** as colunas que vêm da mesma entrada formam 8 fatores: momento (período e fim de semana, com a interação), região, faixa de idade, sexo, escolaridade, moradores, renda e área.
- **Leitura:** cada fator mostra a contribuição no log-odds do score calibrado e quanto ele multiplica a chance da ocasião em relação ao contexto médio do treino.
- **Conferência:** os testes comparam o resultado com o valor de Shapley calculado pela definição e com a biblioteca `shap`.
- **Importância no treino:** média do |log-odds| de 1,03 no momento, 0,56 na escolaridade e 0,36 na renda. Os outros fatores ficam abaixo de 0,2.

### Decision Engine
Regras documentadas transformam o score numa categoria de ativação, com justificativa ([docs/decision_engine.md](docs/decision_engine.md)). Não existe histórico de ações e resultados para treinar um segundo modelo, por isso as regras são explícitas e auditáveis.

Elas combinam três coisas:
- a classe do score;
- o canal (delivery, varejo ou estabelecimento), informado por quem usa a API;
- a associação atual à marca no segmento, também informada por quem usa a API.

As justificativas citam os fatores do SHAP e o contexto observado na POF: onde a pizza foi comida e quantas ocasiões tiveram guaraná, por região e com intervalo. Nenhuma justificativa afirma efeito causal.

| Regra | Quando se aplica | Categoria |
|---|---|---|
| R1 | Classe `baixa` | `sem_ativacao_dedicada` |
| R2 | Classe `alta` ou `media` e associação à marca baixa | `awareness_associacao` |
| R3 | Classe `alta` e canal delivery | `ativacao_delivery` |
| R4 | Classe `alta` e canal estabelecimento | `ativacao_ponto_de_consumo` |
| R5 | Classe `alta` e canal varejo ou não informado | `combo_consumo_em_casa` |
| R6 | Classe `media` | `presenca_de_marca` |

A associação à marca é informada, e não calculada, por dois motivos:
- **Incerteza:** a participação do guaraná por região tem intervalos largos. No Norte, por exemplo, é de 16%, com intervalo de 3% a 38%, e nenhuma região fica abaixo da média nacional com segurança.
- **Categoria, não marca:** a POF registra o refrigerante de guaraná de qualquer marca, não a Guaraná Antarctica.

A casa inclui delivery, porque a POF não os separa. A casa reúne 65% das ocasiões no Brasil e é o local mais comum em todas as regiões.

### API
A API foi feita em FastAPI, com validação em Pydantic ([src/api/main.py](src/api/main.py)). Ao subir, ela carrega o modelo, a referência do SHAP e o contexto. Também confere o hash do modelo e se a referência do SHAP é desse mesmo modelo.

**Endpoints:**
- `POST /predict` devolve os 6 campos do contrato.
- `GET /health`, `GET /model` e `GET /rules` dão o estado do serviço, os metadados e as regras.
- `/docs` traz a documentação interativa.

**Entrada:**
- **Quando:** `periodo` e `dia_semana`.
- **Onde:** `uf` e `urbano`.
- **Quem:** `idade`, `sexo`, `anos_estudo`, `n_moradores` e `renda_pc`.
- **Opcionais:** `canal` e `associacao_marca`, que vão para o Decision Engine, e `model_version`. Se a versão pedida não for a carregada, a API devolve 409.

Exemplo de domingo à noite em SP, com canal delivery:

| Campo | Valor |
|---|---|
| `occasion_score` | 0,0288 (2,88%, 9,5 vezes a taxa média do treino) |
| `classification` | `alta` |
| `top_factors` | momento: noite de sexta a domingo (chance × 13,2); renda: R$ 4.000 por pessoa ao mês (× 1,8); escolaridade: 12 anos (× 1,6) |
| `recommendation_category` | `ativacao_delivery` (regra R3) |
| `recommendation_reason` | Texto com o score, a regra, o contexto observado na região e os fatores |
| `model_metadata` | Versão, data de treino, hash do arquivo, cortes das classes, regra aplicada e avisos de uso |

O × 13,2 do momento compara com o contexto médio do treino, que mistura todos os períodos e dias.

## Limitações

- **Ganho instável no tempo:** na validação com origem móvel dentro do treino, o ganho sobre o baseline só foi significativo em uma das três janelas. Em dez–jan, a captura do modelo ficou abaixo da do baseline (Resultados, Robustez). É o teste que vai mostrar se o ganho se mantém.
- **Calibração por dia:** o modelo usa fim de semana (sexta a domingo), e não o dia da semana, porque o dia não trouxe ganho significativo de PR-AUC. Por isso, ele dá a mesma probabilidade para sexta e domingo e para segunda e quinta.
  - Na análise exploratória, porém, a noite de quinta (1,05%) se parece mais com a de sexta (1,15%) do que com as de segunda a quarta (0,50% a 0,66%).
  - Resultado: a quinta fica 34% abaixo do observado e a sexta, 52% acima.
- **Calibração por segmento e no topo:** a sigmoide tem dois parâmetros e acerta o nível geral, não cada segmento ou faixa do ranking.
  - Ficam vieses acima de 30% em alguns públicos, como 10–17 anos (subestimados) e renda acima de 5 salários mínimos (superestimados).
  - No 1% de maior score, a taxa fica superestimada.
  - Nesses casos, o score vale como ranking, mas não como probabilidade exata.
- **Poucos positivos efetivos:** os 523 positivos do treino equivalem a 138 nas métricas ponderadas. Métricas de segmentos pequenos têm incerteza alta.
- **Sazonalidade e eventos:** o mês fica fora do modelo, porque a coleta tem um único ano e o treino não cobre o outono.
  - No treino, janeiro teve taxa bem menor que os outros meses.
  - O período de teste tem a Copa do Mundo de 2018 e as festas juninas, e a taxa sobe de 0,30% para 0,41%.
  - A POF não traz a data do recordatório, então os feriados não são identificáveis.
- **Período de teste já lido:** uma triagem exploratória de 16/11 mediu no período de teste o desempenho de um modelo com as mesmas features. As decisões tiradas dali foram revertidas e o modelo final não depende do teste, mas o período não é inédito (Metodologia, adendo de 20/11).

_A completar com a avaliação no teste._

## Como executar

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Pipeline de dados

```bash
python -m src.ingestion.download_pof     # baixa e extrai a POF 2017-2018 (~150 MB)
python -m src.ingestion.pof              # lê os arquivos de largura fixa → data/processed/*.parquet
python notebooks/01_verificacao_pof.py   # sanidade da leitura, contagem da ocasião, validação do mês
python -m src.features.build_grid        # grade pessoa × dia × período → data/processed/grid.parquet
python notebooks/02_triagem_features.py  # triagem de features com validação cruzada, só no treino
python notebooks/03_eda.py               # análise exploratória (só treino) → reports/figures/
python notebooks/04_modelagem.py         # seleção do modelo na validação cruzada, só no treino
python -m src.models.treinar             # modelo final → models/ (congelado antes do teste)
python notebooks/05_robustez.py          # checagens de robustez do modelo escolhido, só no treino
python -m src.models.explicar            # referência do SHAP por fator → models/explicacao.json (só treino)
python -m src.decision_engine.contexto   # contexto do Decision Engine → models/contexto_decisao.json (só treino)
python notebooks/06_teste.py             # avaliação única no teste → reports/teste_tabelas.md
```

O modelo, a referência do SHAP e o contexto do Decision Engine estão versionados em `models/`, então a API e o app rodam sem os microdados.

### Subir a API

```bash
uvicorn src.api.main:app --reload        # documentação interativa em http://localhost:8000/docs
```

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "periodo": "noite", "dia_semana": "Domingo", "uf": "SP", "urbano": true, "idade": 25,
  "sexo": "feminino", "anos_estudo": 12, "n_moradores": 3, "renda_pc": 4000,
  "canal": "delivery"}'
```

### Testes

```bash
python -m pytest
```

_Comandos do app a preencher._
