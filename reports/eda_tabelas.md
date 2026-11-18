# Tabelas da análise exploratória (só treino)

Gerado por `notebooks/03_eda.py`. Taxas ponderadas por `peso_semana`, IC de 95% por bootstrap de UPAs. `Base` é o número de linhas do denominador; `Casos`, quantas têm o evento.

## 1. Taxa da ocasião por período e dia (baseline)

| Grupo | Taxa | IC 95% | Base | Casos |
|---|---|---|---|---|
| Manhã · Seg | 0,01% | 0,00% a 0,03% | 8.592 | 1 |
| Manhã · Ter | 0,03% | 0,00% a 0,07% | 9.879 | 5 |
| Manhã · Qua | 0,01% | 0,00% a 0,03% | 9.115 | 1 |
| Manhã · Qui | 0,02% | 0,00% a 0,04% | 9.978 | 2 |
| Manhã · Sex | 0,02% | 0,00% a 0,06% | 4.870 | 2 |
| Manhã · Sáb | 0,00% | 0,00% a 0,00% | 2.560 | 0 |
| Manhã · Dom | 0,01% | 0,00% a 0,02% | 10.943 | 2 |
| Almoço · Seg | 0,12% | 0,00% a 0,29% | 8.592 | 4 |
| Almoço · Ter | 0,00% | 0,00% a 0,00% | 9.879 | 0 |
| Almoço · Qua | 0,00% | 0,00% a 0,00% | 9.115 | 0 |
| Almoço · Qui | 0,03% | 0,00% a 0,08% | 9.978 | 2 |
| Almoço · Sex | 0,05% | 0,00% a 0,15% | 4.870 | 2 |
| Almoço · Sáb | 0,00% | 0,00% a 0,00% | 2.560 | 0 |
| Almoço · Dom | 0,03% | 0,00% a 0,08% | 10.943 | 4 |
| Tarde · Seg | 0,00% | 0,00% a 0,01% | 8.592 | 2 |
| Tarde · Ter | 0,08% | 0,00% a 0,24% | 9.879 | 1 |
| Tarde · Qua | 0,01% | 0,00% a 0,03% | 9.115 | 1 |
| Tarde · Qui | 0,02% | 0,00% a 0,05% | 9.978 | 4 |
| Tarde · Sex | 0,01% | 0,00% a 0,02% | 4.870 | 1 |
| Tarde · Sáb | 0,01% | 0,00% a 0,04% | 2.560 | 1 |
| Tarde · Dom | 0,04% | 0,01% a 0,08% | 10.943 | 7 |
| Noite · Seg | 0,50% | 0,29% a 0,75% | 8.592 | 48 |
| Noite · Ter | 0,61% | 0,36% a 0,90% | 9.879 | 55 |
| Noite · Qua | 0,66% | 0,36% a 1,01% | 9.115 | 50 |
| Noite · Qui | 1,05% | 0,58% a 1,61% | 9.978 | 73 |
| Noite · Sex | 1,15% | 0,67% a 1,76% | 4.870 | 54 |
| Noite · Sáb | 2,09% | 1,13% a 3,38% | 2.560 | 34 |
| Noite · Dom | 1,91% | 1,24% a 2,70% | 10.943 | 167 |

## 1b. Noite: sexta a domingo × segunda a quinta

| Grupo | Taxa | IC 95% | Base | Casos |
|---|---|---|---|---|
| Segunda a quinta | 0,71% | 0,53% a 0,90% | 37.564 | 226 |
| Sexta a domingo | 1,72% | 1,27% a 2,25% | 18.373 | 255 |

## 2. Noite, por classe ABEP

| Grupo | Taxa | IC 95% | Base | Casos |
|---|---|---|---|---|
| DE | 0,34% | 0,15% a 0,63% | 22.471 | 68 |
| C2 | 0,96% | 0,57% a 1,45% | 14.006 | 110 |
| C1 | 1,45% | 1,04% a 1,95% | 9.623 | 135 |
| B2 | 1,56% | 0,98% a 2,28% | 7.071 | 108 |
| B1 | 4,40% | 1,66% a 7,33% | 1.954 | 46 |
| A | 1,53% | 0,58% a 2,85% | 812 | 14 |

## 2. Noite, por tipo de área

| Grupo | Taxa | IC 95% | Base | Casos |
|---|---|---|---|---|
| Capital | 1,71% | 1,27% a 2,21% | 11.598 | 187 |
| Região metropolitana | 1,50% | 0,72% a 2,56% | 5.996 | 67 |
| Interior urbano | 1,00% | 0,67% a 1,44% | 24.873 | 194 |
| Rural | 0,23% | 0,11% a 0,38% | 13.470 | 33 |

## 2. Noite, por região

| Grupo | Taxa | IC 95% | Base | Casos |
|---|---|---|---|---|
| Sul | 1,59% | 1,03% a 2,17% | 7.390 | 82 |
| Sudeste | 1,29% | 0,80% a 1,90% | 13.868 | 107 |
| Centro-Oeste | 1,04% | 0,64% a 1,52% | 6.865 | 75 |
| Norte | 0,85% | 0,43% a 1,36% | 8.418 | 58 |
| Nordeste | 0,78% | 0,58% a 0,99% | 19.396 | 159 |

## 2. Noite, por idade

| Grupo | Taxa | IC 95% | Base | Casos |
|---|---|---|---|---|
| 10 a 17 anos | 1,08% | 0,75% a 1,46% | 8.303 | 90 |
| 18 a 29 anos | 1,82% | 1,16% a 2,57% | 10.710 | 133 |
| 30 a 44 anos | 1,33% | 0,92% a 1,82% | 14.376 | 158 |
| 45 a 59 anos | 0,73% | 0,50% a 1,01% | 12.376 | 76 |
| 60 anos ou mais | 0,61% | 0,24% a 1,07% | 10.172 | 24 |

## 2. Noite, por sexo

| Grupo | Taxa | IC 95% | Base | Casos |
|---|---|---|---|---|
| Feminino | 0,96% | 0,73% a 1,22% | 29.940 | 244 |
| Masculino | 1,34% | 1,00% a 1,76% | 25.997 | 237 |

## 3. Noite, por classe e dia

| Grupo | Taxa | IC 95% | Base | Casos |
|---|---|---|---|---|
| DE · Segunda a quinta | 0,15% | 0,07% a 0,26% | 15.115 | 28 |
| DE · Sexta a domingo | 0,60% | 0,18% a 1,28% | 7.356 | 40 |
| C2 · Segunda a quinta | 0,44% | 0,26% a 0,69% | 9.521 | 49 |
| C2 · Sexta a domingo | 1,71% | 0,81% a 2,90% | 4.485 | 61 |
| C1 · Segunda a quinta | 1,07% | 0,71% a 1,51% | 6.372 | 70 |
| C1 · Sexta a domingo | 1,94% | 1,12% a 2,96% | 3.251 | 65 |
| B2 · Segunda a quinta | 1,30% | 0,64% a 2,14% | 4.756 | 54 |
| B2 · Sexta a domingo | 1,88% | 0,88% a 3,06% | 2.315 | 54 |
| B1 · Segunda a quinta | 2,20% | 0,50% a 4,43% | 1.269 | 19 |
| B1 · Sexta a domingo | 7,05% | 2,08% a 12,13% | 685 | 27 |
| A · Segunda a quinta | 1,24% | 0,26% a 2,58% | 531 | 6 |
| A · Sexta a domingo | 1,92% | 0,32% a 4,32% | 281 | 8 |

## 3b. Sexta a domingo ÷ segunda a quinta, por classe (noite)

| Classe | Razão | IC 95% |
|---|---|---|
| DE | 3,92 | 1,06 a 11,72 |
| C2 | 3,85 | 1,66 a 8,19 |
| C1 | 1,81 | 0,92 a 3,27 |
| B2 | 1,45 | 0,61 a 3,17 |
| B1 | 3,21 | 1,08 a 12,29 |
| A | 1,56 | 0,23 a 8,79 |

## 4a. Noites com pizza, por classe

| Grupo | Taxa | IC 95% | Base | Casos |
|---|---|---|---|---|
| DE | 0,55% | 0,32% a 0,86% | 22.471 | 111 |
| C2 | 1,52% | 1,00% a 2,19% | 14.006 | 179 |
| C1 | 3,13% | 2,12% a 4,42% | 9.623 | 250 |
| B2 | 3,56% | 2,59% a 4,76% | 7.071 | 231 |
| B1 | 10,09% | 4,27% a 17,16% | 1.954 | 99 |
| A | 7,70% | 2,27% a 16,11% | 812 | 43 |

## 4b. Pizzas da noite com refrigerante, por classe

| Grupo | Taxa | IC 95% | Base | Casos |
|---|---|---|---|---|
| DE | 62,61% | 35,68% a 80,82% | 111 | 68 |
| C2 | 63,13% | 52,25% a 72,45% | 179 | 110 |
| C1 | 46,31% | 30,96% a 63,27% | 250 | 135 |
| B2 | 43,88% | 33,38% a 53,90% | 231 | 108 |
| B1 | 43,61% | 30,18% a 60,57% | 99 | 46 |
| A | 19,93% | 5,33% a 64,95% | 43 | 14 |

## 4c. Ocasiões da noite em que o refrigerante era guaraná, por classe

| Grupo | Taxa | IC 95% | Base | Casos |
|---|---|---|---|---|
| DE | 32,83% | 9,37% a 45,91% | 68 | 17 |
| C2 | 28,40% | 12,95% a 48,37% | 110 | 33 |
| C1 | 23,86% | 11,33% a 39,87% | 135 | 35 |
| B2 | 23,90% | 7,45% a 44,09% | 108 | 22 |
| B1 | 29,81% | 4,61% a 64,47% | 46 | 15 |
| A | 25,51% | 2,46% a 64,39% | 14 | 5 |

## 4d. Ocasiões da noite em que o refrigerante era guaraná, por região

| Grupo | Taxa | IC 95% | Base | Casos |
|---|---|---|---|---|
| Centro-Oeste | 31,27% | 12,32% a 51,41% | 75 | 19 |
| Sudeste | 29,58% | 12,31% a 47,48% | 107 | 36 |
| Nordeste | 25,56% | 15,11% a 37,64% | 159 | 42 |
| Sul | 24,03% | 9,56% a 41,84% | 82 | 21 |
| Norte | 13,68% | 1,90% a 35,23% | 58 | 9 |

## 5. Pizzas comidas em casa, por classe ABEP

| Grupo | Taxa | IC 95% | Base | Casos |
|---|---|---|---|---|
| DE | 76,70% | 61,14% a 87,20% | 147 | 99 |
| C2 | 64,87% | 53,50% a 76,94% | 224 | 143 |
| C1 | 77,99% | 66,00% a 86,36% | 301 | 222 |
| B2 | 70,78% | 58,28% a 80,62% | 262 | 180 |
| B1 | 78,09% | 66,61% a 93,09% | 106 | 88 |
| A | 87,97% | 56,58% a 97,17% | 49 | 39 |

## 5. Pizzas comidas em casa, por tipo de área

| Grupo | Taxa | IC 95% | Base | Casos |
|---|---|---|---|---|
| Capital | 72,53% | 61,52% a 81,25% | 367 | 262 |
| Região metropolitana | 79,45% | 67,76% a 90,20% | 147 | 116 |
| Interior urbano | 73,21% | 63,60% a 81,80% | 472 | 329 |
| Rural | 73,20% | 56,01% a 86,27% | 103 | 64 |

## 5. Pizzas comidas em casa, por região

| Grupo | Taxa | IC 95% | Base | Casos |
|---|---|---|---|---|
| Sudeste | 79,76% | 70,94% a 87,81% | 281 | 229 |
| Sul | 72,95% | 62,76% a 82,33% | 217 | 162 |
| Nordeste | 70,78% | 56,44% a 81,57% | 303 | 195 |
| Centro-Oeste | 61,52% | 48,74% a 74,42% | 170 | 111 |
| Norte | 55,96% | 36,31% a 75,97% | 118 | 74 |
