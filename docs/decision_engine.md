# Decision Engine

O Decision Engine transforma o Occasion Score numa categoria de ativação, sempre com uma justificativa. Ele funciona por regras escritas, não por um segundo modelo. O código está em [src/decision_engine/regras.py](../src/decision_engine/regras.py) e os testes em [tests/test_decision_engine.py](../tests/test_decision_engine.py).

## Por que regras e não um modelo

Para aprender qual ação funciona em cada contexto, um modelo precisaria de um histórico de contexto → ação → resultado. A POF registra o que as pessoas comeram, mas não as ações de marketing nem o resultado delas. Sem esse histórico, um segundo modelo supervisionado não teria o que aprender. As regras, por outro lado, são explícitas, auditáveis e fáceis de mudar quando o negócio decidir outra coisa.

O que as regras **não** fazem: não estimam efeito causal, venda incremental nem retorno de uma ação. O score diz onde a ocasião costuma acontecer. Ele não diz quanto uma ação vai vender. As justificativas seguem essa linha, e um teste confere que nenhuma delas promete resultado. Elas são escritas para quem decide marketing: falam em chance e prioridade, e não em score e classe ("classe alta" se confunde com classe social), e trazem o intervalo do contexto observado como "faixa provável".

## Entradas

| Entrada | De onde vem | Para que serve |
|---|---|---|
| Score e classe (`alta`, `media`, `baixa`) | Modelo congelado ([models/](../models/)) | Se vale ativar e com que prioridade |
| Os 3 fatores principais (`top_factors`) | SHAP exato por fator ([src/models/explicar.py](../src/models/explicar.py)) | Justificativa |
| Região | UF da requisição | Contexto observado da região |
| Canal (opcional): `delivery`, `varejo` ou `estabelecimento` | Quem usa a API | Por onde ativar |
| Associação atual à marca (opcional): `baixa` ou `normal` | Quem usa a API (pesquisa de marca) | Conversão ou construção de associação |
| Contexto observado | [models/contexto_decisao.json](../models/contexto_decisao.json), calculado só com o treino | Justificativa |

## Categorias

| Categoria | O que significa |
|---|---|
| `ativacao_delivery` | Combo de pizza com Guaraná em aplicativos e pizzarias com entrega |
| `combo_consumo_em_casa` | Promoção ou combo para consumo em casa, no varejo e no delivery |
| `ativacao_ponto_de_consumo` | Ativação em pizzarias, restaurantes e bares |
| `awareness_associacao` | Awareness e construção da associação entre Guaraná e pizza |
| `presenca_de_marca` | Presença de baixo custo (mídia contínua, material de ponto de venda), sem promoção dedicada |
| `sem_ativacao_dedicada` | O contexto só entra em ações amplas |

## Regras, na ordem em que são avaliadas

A primeira regra que se aplica decide.

| Regra | Quando se aplica | Categoria | Por quê |
|---|---|---|---|
| R1 | Classe `baixa` | `sem_ativacao_dedicada` | Os 70% de contextos de menor score reúnem só 5,6% das ocasiões (validação cruzada). Investir neles rende pouco, qualquer que seja o canal. |
| R2 | Classe `alta` ou `media` e associação à marca informada como baixa | `awareness_associacao` | A ocasião acontece, mas a marca não está associada a ela no segmento. Primeiro é preciso construir a associação (exemplo do TAPI). |
| R3 | Classe `alta` e canal `delivery` | `ativacao_delivery` | Chance alta e canal de entrega disponível (exemplo do TAPI). |
| R4 | Classe `alta` e canal `estabelecimento` | `ativacao_ponto_de_consumo` | Chance alta e ponto de consumo definido. |
| R5 | Classe `alta` e canal `varejo` ou não informado | `combo_consumo_em_casa` | Sem canal informado, a regra segue o local mais comum da ocasião: a casa, com 65% das ocasiões no Brasil e o maior percentual em todas as regiões (exemplo do TAPI: consumo doméstico recorrente). |
| R6 | Classe `media` | `presenca_de_marca` | A classe média tem 20% dos contextos e 27,4% das ocasiões. A chance é moderada, então a ação é de baixo custo e vem depois da classe alta na prioridade. |

Os cortes das classes são os do modelo congelado. `alta` vale a partir de 0,7765% e `media` a partir de 0,1436%, que são os percentis 90 e 70 do score calibrado. Os percentis foram calculados nas previsões fora da amostra da validação cruzada do treino, ponderadas por `peso_semana`. Na validação cruzada, a classe `alta` reuniu 67,0% das ocasiões em 10% dos contextos.

## Contexto observado (treino)

Contexto calculado sobre as ocasiões de pizza com refrigerante do treino (jul/2017 a fev/2018), ponderadas por `peso_semana`. Entre parênteses, o intervalo de 95% por bootstrap de UPAs (1.000 reamostragens). "Efetivas" é o tamanho efetivo da amostra (Kish), que considera a variação dos pesos.

| Região | Ocasiões | Efetivas | Em casa | Em estabelecimento | Outro lugar fora | Com guaraná |
|---|---|---|---|---|---|---|
| Brasil | 523 | 138 | 65% (54–75) | 14% (10–21) | 21% (11–32) | 28% (19–38) |
| Norte | 66 | 17 | 51% (28–78) | 30% (5–55) | 19% (7–36) | 16% (3–38) |
| Nordeste | 171 | 101 | 55% (43–66) | 15% (8–22) | 30% (19–42) | 25% (15–36) |
| Sudeste | 113 | 40 | 68% (47–87) | 14% (6–27) | 17% (0–40) | 31% (13–47) |
| Sul | 95 | 56 | 74% (59–85) | 10% (3–21) | 16% (6–27) | 26% (11–41) |
| Centro-Oeste | 78 | 46 | 54% (34–75) | 12% (4–24) | 34% (12–54) | 35% (14–54) |

Uma ocasião do Sudeste não tem o local registrado, por isso ali os três locais somam 99,8%. A tabela é gerada por `python -m src.decision_engine.contexto`.

## Por que a associação à marca é informada, e não calculada

O plano era tirar a associação à marca da POF, pela participação do guaraná nas ocasiões de cada região. Com os intervalos acima, isso não se sustenta. O Norte tem a menor participação (16%), mas o intervalo vai de 3% a 38%, com 17 ocasiões efetivas. Nenhuma região fica abaixo da participação nacional com segurança. Além disso, a POF registra a categoria (refrigerante de guaraná de qualquer marca), não a Guaraná Antarctica.

Por isso a associação entra como informação de quem usa a API, que pode vir de pesquisa de marca. A participação do guaraná na região aparece só como contexto na justificativa da regra R2.

## Limites

- **Casa inclui delivery:** a POF não separa pizza pedida de pizza feita ou comprada pronta. Por isso o delivery depende do canal informado.
- **Canal e associação são informados por quem usa a API:** a qualidade da recomendação depende dessas duas entradas. Quando faltam, as regras usam o score e o contexto observado.
- **Regras são hipóteses de negócio:** quando houver histórico de ações e resultados, por exemplo testes A/B, as regras podem ser avaliadas e substituídas por um modelo de decisão.

## Como mudar uma regra

Altere `REGRAS`, `CATEGORIAS` e `recomendar` em [src/decision_engine/regras.py](../src/decision_engine/regras.py) e esta página. Os testes conferem quatro coisas:
- toda combinação de classe, canal e associação cai numa regra documentada;
- cada regra devolve a sua categoria;
- nenhuma justificativa promete resultado de vendas;
- nenhuma justificativa usa jargão de dados (score, classe, IC 95%, treino, POF, SHAP).
