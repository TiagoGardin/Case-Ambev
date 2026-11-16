# Data Card – Occasion Intelligence Pizza + Guaraná

Seções exigidas pelo TAPI (Seção 4). Itens marcados com _A preencher_ ainda estão em aberto.

## 1. Unidade de análise
Pessoa × dia de recordatório × período do dia (manhã, almoço, tarde, noite).

## 2. Target
- **Definição:** `y = 1` se a pessoa consumiu pizza **e** refrigerante no mesmo período do mesmo dia.
- **Tipo:** ground truth (consumo declarado em recordatório de 24h, POF 2017–2018).
- **Medida de marca (não é target):** participação do refrigerante de guaraná dentro das ocasiões positivas.
- **Por que não pizza + guaraná como target:** _A preencher (número de casos positivos)._

## 3. Instante de previsão e anti-leakage
- **Instante de previsão:** início do dia.
- **Features permitidas:** _A preencher._
- **Excluídas por leakage:** local da refeição (`V9018`), outros alimentos do mesmo período. _Completar._

## 4. Fontes e versões
| Fonte | Papel | Versão / data de download | Granularidade | Chave de integração |
|---|---|---|---|---|
| POF 2017–2018 – Consumo Alimentar (IBGE) | Ground truth + features de público | _A preencher_ | Pessoa × item consumido | — |
| Vigitel 2006–2024 (Min. Saúde) | Contexto (hábito recente por capital) | _A preencher_ | Pessoa × ano, capitais | UF |
| Censo 2022 / SIDRA (IBGE) | Perfil territorial | _A preencher_ | UF | UF |
| OpenStreetMap / Overpass | Oferta de pizzarias | _A preencher_ | Ponto | UF |
| Open-Meteo | Clima mensal | _A preencher_ | UF × mês | UF + mês |

## 5. Regras de integração
- **Pessoa:** registros da POF ligados pela chave `COD_UPA + NUM_DOM + NUM_UC + COD_INFORMANTE`.
- **Ano/mês da entrevista:** vêm do registro Aluguel Estimado, por unidade de consumo (`COD_UPA + NUM_DOM + NUM_UC`). Mês = `V9010 + 1`. O ano de julho é separado pelo `DEFLATOR` (> 1 → 2017, < 1 → 2018). Detalhes em [notas_pof.md](notas_pof.md#decisões).
- **Validação temporal:** treino com entrevistas de jul/2017 a fev/2018, teste de mar/2018 a jul/2018. Pessoas sem mês (~14%) ficam fora do split temporal.
- **Renda:** `RENDA_TOTAL` da UC ÷ nº de moradores, em 5 faixas de salário mínimo de 2018 (R$ 954).
- _Fontes externas (Vigitel, Censo, OSM, clima): a preencher._

## 6. Dicionário de variáveis
| Variável | Origem | Tipo | Descrição |
|---|---|---|---|
| _A preencher_ | | | |

## 7. População coberta e excluída
_A preencher (ex.: pessoas de 10+ anos, domicílios particulares, sem informação de mês)._

## 8. Vieses e limitações
- Dados de 2017–2018 (pré-pandemia e pré-boom de delivery).
- Consumo autodeclarado.
- POF não identifica marca ("guaraná" ≠ Guaraná Antarctica).
- _Completar._

## 9. Licença e LGPD
_A preencher._
