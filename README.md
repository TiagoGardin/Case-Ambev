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
| `data/external` | Fontes de contexto (Vigitel, Censo, OSM, clima) |
| `data/processed` | Bases tratadas (`.parquet`) |
| `notebooks/` | EDA e experimentos |
| `src/` | Código do pipeline |
| `models/` | Modelos treinados (não versionados) |
| `reports/figures/` | Gráficos para README e vídeo |
| `docs/` | Data Card, notas e regras do Decision Engine |
| `tests/` | Testes (pytest) |

## Dados

- **Fonte do target:** POF 2017–2018 (IBGE), recordatório alimentar de 24h de 46.164 pessoas.
- **Unidade de análise:** pessoa × dia × período (340.072 linhas, 838 ocasiões de pizza + refrigerante).
- **Features:** quando (período, dia da semana), onde (UF, urbano ou rural) e quem (idade, sexo, escolaridade, tamanho da família, renda per capita), escolhidas por triagem com validação cruzada no treino. Tipo de área e classe ABEP ficam para os segmentos e para o Decision Engine.
- **Validação:** split temporal pelo mês estimado da entrevista (treino jul/2017–fev/2018, teste mar–jul/2018).

Detalhes, decisões e limitações: [docs/data_card.md](docs/data_card.md) e [docs/notas_pof.md](docs/notas_pof.md).

## Metodologia

Protocolo fixado antes da avaliação final no período de teste (mar–jul/2018). O histórico do git registra a data desta seção.

### Occasion Score
- **Definição:** probabilidade estimada de a pessoa comer pizza e tomar refrigerante no período, dado o contexto e o perfil: `P(y = 1 | período, dia da semana, UF, urbano, idade, sexo, escolaridade, tamanho da família, renda per capita)`. O instante de previsão é o início do dia ([Data Card](docs/data_card.md), seção 3).
- **Escala:** de 0 a 1, calibrada. Como a taxa média é baixa (0,28% no treino, ponderada), a aplicação mostra também quantas vezes o score supera essa média.
- **`classification`:** `alta` para os 10% de linhas (pessoa × dia × período) com maior score, `media` para os 20% seguintes e `baixa` para o resto. Os cortes são os percentis 90 e 70 do score, ponderados por `peso`, nas previsões *out-of-fold* do treino. Eles ficam fixos daí em diante e vão no `model_metadata`.

### Métrica principal e custo dos erros
- **Falso positivo:** ativação gasta num contexto sem ocasião. **Falso negativo:** ocasião de pizza + refrigerante que a ativação não alcançou.
- Com orçamento limitado, a marca ativa só o topo do ranking, e os dois custos aparecem ali: a precisão do topo mede o desperdício (FP) e o recall mede as ocasiões perdidas (FN).
- **Principal: PR-AUC** (average precision), que resume precisão × recall em todos os cortes. Um score sem sinal tem PR-AUC igual à prevalência, então o ganho se lê como múltiplo dela.
- **Leitura de negócio: captura no top 10%**, a fração das ocasiões que cai nos 10% de linhas com maior score.
- **Por que não acurácia nem ROC-AUC:** com 99,7% de negativos, as duas ficam altas mesmo com um topo ruim. O ROC-AUC entra como complementar.

### Baseline e candidatos
- **Baseline:** taxa ponderada de `y` por dia da semana × período no treino (28 células), usada diretamente como score. É o "sexta à noite é bom para pizza" em forma de número.
- **Regressão Logística:** features one-hot com regularização L2; `regiao` no lugar de `UF` (27 categorias são muitas para 523 positivos), `faixa_idade` no lugar de `idade` (o efeito da idade não é linear) e interação período × dia da semana, para conter o baseline.
- **LightGBM raso:** poucas folhas e `min_child_samples` alto, porque o treino tem só 523 positivos.

### Validação
1. **Validação cruzada repetida no treino** (jul/2017–fev/2018): `StratifiedGroupKFold` com 5 folds, agrupado por domicílio para que a mesma família (que divide a pizza) não fique dos dois lados, e repetido 10 vezes com partições diferentes. Com só 523 positivos, uma única partição é ruidosa demais para decidir. A PR-AUC ponderada decide:
   - o modelo final (Regressão Logística ou LightGBM);
   - as alternativas do Data Card: dia da semana ou fim de semana, renda contínua ou em faixas;
   - hiperparâmetros, uso do `peso` no ajuste e calibração (sigmoide sobre as previsões *out-of-fold*, se a curva de confiabilidade pedir).

   Entre duas opções, a mais complexa só vence se o ganho for significativo a 5% no teste t corrigido para validação cruzada repetida (Nadeau e Bengio, 2003); senão, fica a mais simples. Os hiperparâmetros seguem só a maior PR-AUC média.
2. **Teste temporal** (mar–jul/2018): o modelo escolhido, treinado em todo o treino, é avaliado **uma única vez**, ao lado do baseline. Nada é reajustado depois; qualquer mudança posterior é reportada como exploratória. A aplicação usa esse mesmo modelo.
3. **Generalização por região** (complementar): dentro do treino, ajustar com 4 regiões e avaliar na 5ª.

**Escolha das features.** Uma triagem com essa validação cruzada repetida, só no treino ([notebooks/02_triagem_features.py](notebooks/02_triagem_features.py)), testou cinco grupos de candidatas: tipo de área, classe ABEP, composição da família, bens e finanças, trabalho e estudo. Nenhum trouxe ganho significativo, nem no LightGBM nem na Regressão Logística, então ficaram as 9 features de partida ([Data Card](docs/data_card.md), seção 3). Versões anteriores da triagem, uma medida no período de teste e outra com uma única partição, foram substituídas por esta. As features não mudam depois do teste.

### Avaliação no teste
- **Métricas, sempre ao lado do baseline:** PR-AUC; captura no top 5%, 10% e 20%; ROC-AUC; precisão, recall, F1 e matriz de confusão no corte de `alta`; Brier e curva de calibração.
- **Peso amostral:** métricas ponderadas por `peso`, para representar a população. A versão sem peso fica como checagem.
- **Incerteza:** intervalos de 95% por bootstrap de domicílios (1.000 reamostragens), porque os 315 positivos do teste estão em 189 domicílios. O modelo só é declarado melhor que o baseline se o intervalo da diferença de PR-AUC (bootstrap pareado) não incluir zero.
- **Mudança de prevalência:** a taxa sobe de 0,28% no treino para 0,36% no teste. A comparação com o baseline é sempre no mesmo conjunto, e o score não é recalibrado com o teste.
- **Segmentos:** período, região, tipo de área e público (classe ABEP, faixa etária, sexo). Para canal, usa-se o local onde a pizza foi comida (`local_pizza`: casa, estabelecimento ou outro lugar fora). Como o local só existe para quem comeu pizza, o resultado por canal é a captura das ocasiões de cada local. Segmentos com menos de 30 positivos no teste são só descritos, sem métrica.
- **Erros:** perfil das ocasiões que ficaram em `baixa` (falsos negativos) e dos contextos em `alta` sem ocasião (falsos positivos).
- **Explicações:** SHAP global e local. Os 3 fatores de maior contribuição de cada previsão formam o `top_factors` da API.

## Resultados

_A preencher._

## Limitações

_A preencher._

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
```

### Testes

```bash
python -m pytest
```

_Comandos de modelos, API e app a preencher._
