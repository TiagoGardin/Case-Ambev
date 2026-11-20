# Fase 1 – Dados

## 1.1 Versões do dataset: o grid salvo é o que o código gera hoje?
OK: data/processed/grid.parquet é idêntico ao grid reconstruído em memória (340,072 linhas × 36 colunas).

## 1.2 Schema, tipos e nulos
```
COD_UPA                str
NUM_DOM                str
NUM_UC                 str
COD_INFORMANTE         str
QUADRO                 str
peso               float64
peso_semana        float64
split                  str
ano_entrevista     float64
mes_entrevista     float64
ordem_mes          float64
fonte_mes              str
estacao                str
periodo           category
dia_semana        category
UF                     str
regiao                 str
urbano               int64
idade                int64
faixa_idade       category
sexo                   str
anos_estudo          int64
n_moradores          int64
renda_pc           float64
faixa_renda       category
fim_de_semana        int64
tipo_area         category
abep_pontos          int64
classe_abep       category
dia_atipico          int64
y                    int64
y_guarana            int64
pizza                 bool
refri                 bool
guarana               bool
local_pizza       category
```
Colunas com nulos, por split (local_pizza é nulo por desenho quando não há pizza):
```
split           sem_mes   teste  treino
ano_entrevista        8       0       0
mes_entrevista        8       0       0
ordem_mes             8       0       0
fonte_mes             8       0       0
estacao               8       0       0
local_pizza           8  115690  222659
```
Nulos nas 9 features do modelo, por split:
```
         periodo  dia_semana  UF  urbano  idade  sexo  anos_estudo  n_moradores  renda_pc
split                                                                                    
sem_mes        0           0   0       0      0     0            0            0         0
teste          0           0   0       0      0     0            0            0         0
treino         0           0   0       0      0     0            0            0         0
```

## 1.3 Duplicatas e chave pessoa × dia × período
Duplicatas na chave ['COD_UPA', 'NUM_DOM', 'NUM_UC', 'COD_INFORMANTE', 'QUADRO', 'periodo']: 0
QUADRO: ['72', '73'] | períodos: ['manha', 'almoco', 'tarde', 'noite']
Pessoas: 46,164 (38,854 com 2 dias, 7,310 com 1)
Linhas por pessoa-dia: [4]
y diferente de pizza & refri: 0; y_guarana > y: 0

## 1.4 Valores impossíveis e extremos
```
                count     mean      std    min   0.1%      1%      50%       99%     99.9%        max
idade        340072.0    40.22    19.45  10.00  10.00   10.00    39.00     85.00     94.00     105.00
anos_estudo  340072.0     8.37     4.59   0.00   0.00    0.00     9.00     16.00     16.00      16.00
n_moradores  340072.0     3.65     1.75   1.00   1.00    1.00     3.00      9.00     12.00      16.00
renda_pc     340072.0  1579.06  2694.13   0.00  30.02  106.70  1004.13  10690.32  27390.06  235192.43
peso         340072.0  3840.24  4697.27   1.66  39.25  232.65  2498.26  23086.27  49118.14  107836.43
peso_semana  340072.0  3840.15  5923.36   1.21  35.15  197.54  2294.28  26091.57  68248.36  291863.97
```
renda_pc = 0: 36 linhas (3 UCs); renda_pc < 0: 0; peso ≤ 0: 0
UF fora das 27: [] | UFs distintas: 27 | urbano: [np.int64(0), np.int64(1)] | sexo: ['feminino', 'masculino']
Horas de consumo no registro bruto: min 0, max 23

## 1.5 Estrutura por split (sem desempenho de modelo)
```
         linhas  positivos  pessoas  domicilios  upas
split                                                
sem_mes       8          0        1           1     1
teste    116316        315    15816        6987  2470
treino   223748        523    30347       13124  3978
```
Domicílios nos dois lados do split: 0 | UPAs com domicílios no treino e no teste: 1044 de 5404

## 1.6 Continuidade temporal e mês reconstruído da entrevista
UCs por mês da coleta (ordem 0 = jul/2017 … 12 = jul/2018) e split:
```
split      sem_mes  teste  treino
ordem_mes                        
0.0              0      0    1180
1.0              0      0    1960
2.0              0      0    1611
3.0              0      0    1819
4.0              0      0    1679
5.0              0      0    1567
6.0              0      1    1817
7.0              0      0    1517
8.0              0   1550       0
9.0              0   1620       0
10.0             0   1801       0
11.0             0   1576       0
12.0             0    447       0
NaN              1      0       0
```
Origem do mês (UCs), por split:
```
split             sem_mes  teste  treino
fonte_mes                               
aluguel_estimado        0   5901   11098
aluguel_pago            0    941    1754
sem estimativa          1      0       0
vizinhos_upa            0    153     298
```
Amplitude do mês dentro da UPA: mediana 2, P95 3, máx 9 meses; UPAs com amplitude > 3 meses: 138
Validação da regra dos inquilinos (src/features/mes_entrevista.py:131), em UCs de mês conhecido que pagam prestação ou condomínio:
```
                             regra    n  exato  erro ≤ 1 mês  erro > 3 meses  mesmo lado do corte treino/teste
pagamento + vizinhos (regra usada) 3943  0.745         0.961           0.001                             0.970
         só vizinhos (alternativa) 3936  0.277         0.815           0.001                             0.867
```
UCs com mês estimado por aluguel pago ou vizinhos: 3,146 (15.6% das UCs); positivos nessas UCs (treino): 89

## 1.7 Definição do alvo: códigos de pizza e refrigerante
pizza: 19 códigos, 16 aparecem no consumo (1,788 registros); suspeitos pelo nome: 1
```
 codigo                                        descricao  registros_no_consumo  suspeito
8500903                                            PIZZA                  1099     False
8500914                                  PIZZA CALABREZA                   285     False
8500932                     PIZZA DE FRANGO COM CATUPIRY                   107     False
8500915                       PIZZA MUCARELA / MUSSARELA                    79     False
8500917                                 PIZZA PORTUGUESA                    70     False
8500927                                 PIZZA DE QUEIJOS                    42     False
8500916                                   PIZZA PRESUNTO                    29     False
8500936 PIZZA MARGUERITA (MUCARELA, TOMATE E MANJERICAO)                    16     False
8500934          PIZZA DE PEIXE (ATUM, SALMAO, SARDINHA)                    13     False
8500938                     PIZZA DOCE DE QUALQUER SABOR                    13      True
8500910                                          CALZONE                    13     False
8500924                               PIZZA DE PEPPERONI                     8     False
8500928                       PIZZA DE LOMBINHO DEFUMADO                     6     False
8500935        PIZZA DE MUCARELA DE BUFALA E TOMATE SECO                     3     False
8500937                                PIZZA VEGETARIANA                     3     False
8500930                                 PIZZA NAPOLITANA                     2     False
8500939      PIZZA HAWAIANA (QUEIJO, PRESUNTO E ABACAXI)                     0     False
8500929                PIZZA DE MUCARELA E PEITO DE PERU                     0     False
8508017                           PIZZA PORTUGUESA LIGHT                     0     False
```
refrigerante: 34 códigos, 28 aparecem no consumo (16,214 registros); suspeitos pelo nome: 5
```
 codigo                           descricao  registros_no_consumo  suspeito
8200101    REFRIGERANTE DE COLA TRADICIONAL                  5465     False
8200301 REFRIGERANTE DE GUARANA TRADICIONAL                  4330     False
8200102               COCA COLA TRADICIONAL                  3041     False
8203501       REFRIGERANTE NAO ESPECIFICADO                  1159     False
8200211 REFRIGERANTE DE LARANJA TRADICIONAL                   918     False
8200411   REFRIGERANTE DE LIMAO TRADICIONAL                   237     False
8200507     REFRIGERANTE DE UVA TRADICIONAL                   237     False
8200302                 GUARANA TRADICIONAL                   206     False
8200202           FANTA LARANJA TRADICIONAL                   125     False
8200801           REFRIGERANTE DE COLA DIET                    88     False
8200702                     COCA COLA LIGHT                    70     False
8200407     SPRITE REFRIGERANTE TRADICIONAL                    67     False
8201802                 TUBAINA TRADICIONAL                    56     False
8207401        REFRIGERANTE DE GUARANA DIET                    36     False
8201017       BEBIDA REFRIGERANTE, TIPO H2O                    31     False
8200505               FANTA UVA TRADICIONAL                    24     False
8200701          REFRIGERANTE DE COLA LIGHT                    19     False
8204902             AGUA TONICA TRADICIONAL                    18      True
8207501         REFRIGERANTE DE LIMAO LIGHT                    17     False
8207402                        GUARANA DIET                    16     False
8201601       REFRIGERANTE DE GUARANA LIGHT                    15     False
8207601          REFRIGERANTE DE LIMAO DIET                    14     False
8212902                    AGUA TONICA DIET                    12      True
8207701    REFRIGERANTE DE UVA DIET / LIGHT                     7     False
8201602                       GUARANA LIGHT                     3     False
8201511        REFRIGERANTE DE LARANJA DIET                     1     False
8200208                  SUKITA TRADICIONAL                     1     False
8209002                   AGUA TONICA LIGHT                     1      True
8201902                       TUBAINA LIGHT                     0     False
8205402   PARAGUAI REFRIGERANTE TRADICIONAL                     0      True
8201502                 FANTA LARANJA LIGHT                     0     False
8207705                     FANTA UVA LIGHT                     0     False
8209102         PARAGUAI REFRIGERANTE LIGHT                     0      True
8212906   REFRIGERANTE DE QUININO DIETETICO                     0     False
```
Códigos do consumo fora do cadastro: 0 (0 registros)
Cadastro com 'PIZZA' fora da lista de pizza: 0 códigos, 0 registros no consumo
Cadastro com 'REFRIG|GUARANA|COLA|SODA|TUBAINA|FANTA|SPRITE' fora da lista de refrigerante: 44 códigos, 8,266 registros no consumo
```
                                                    descricao  registros
codigo                                                                  
7903102                                    LEITE ACHOCOLATADO       3569
8003801                                     BOLO DE CHOCOLATE       1493
6900821                                    ACHOCOLATADO EM PO       1166
7903110                                   BEBIDA ACHOCOLATADA        453
8200308                           REFRESCO DE GUARANA NATURAL        253
6900701                                  TABLETE DE CHOCOLATE        208
6900702                                    BARRA DE CHOCOLATE        198
6900710                          CHOCOLATE, TABLETE, AO LEITE        127
8507601                                             CHOCOLATE        112
8013001             BOLO DE CHOCOLATE COM RECHEIO E COBERTURA         96
6900801                     CHOCOLATE EM PO DE QUALQUER MARCA         94
6900703                   CHOCOLATE, TABLETE NAO ESPECIFICADO         86
8507802                            ACAI COM GUARANA E GRANOLA         78
6905702                                         GUARANA EM PO         49
8002309                      SOLDA/ SODA/ BOLACHA DE RAPADURA         43
6900714                       CHOCOLATE, TABLETE, MEIO AMARGO         40
8012702  BOLO DE CENOURA COM RECHEIO E COBERTURA DE CHOCOLATE         36
6915302                                    PUDIM DE CHOCOLATE         25
6900712                            CHOCOLATE, TABLETE, BRANCO         18
6915902                                    CALDA DE CHOCOLATE         18
6900711                CHOCOLATE, TABLETE, AO LEITE, RECHEADO         17
6500615                                   MINGAU DE CHOCOLATE         13
6910321                               ACHOCOLATADO EM PO DIET          9
8006103                         CREPE / PANQUECA DE CHOCOLATE          8
7906802                               LEITE ACHOCOLATADO DIET          8
```

## 1.8 Madrugada (0–4h) dentro da 'noite' do mesmo dia de recordatório
Positivos noturnos: 767. Com pizza e refrigerante entre 18h e 23h: 766. Dependem de item da madrugada: 1 (0.1%).
Registros de consumo entre 0h e 4h: 6,003 (0.51% dos itens)

## 1.9 Joins da montagem do grid (src/features/build_grid.py:226)
pessoa-dias: 85,018 → grade × 4: 340,072 → + ocasiões: 340,072 (chaves de ocasião fora da grade: 0)
perfil_pessoa: 178,431 linhas, chave única: True; pessoas do consumo sem cadastro de morador: 0
UCs no grid: 20,146; sem mês (sem_mes): 1; sem classe ABEP: 0; sem tipo de área: 0
Moradores com 10+ anos nos domicílios do módulo de consumo: 53,409; com recordatório: 86.4% (o restante não respondeu; o PESO_FINAL do consumo é calibrado para o módulo)

## 1.10 Unidades: renda per capita × renda disponível per capita do IBGE
renda_pc / RENDA_DISP_PC por UC: mediana 1.00, P10 0.77, P90 1.22; correlação de Spearman 0.962
Mediana da renda_pc (UCs): R$ 1,146 por mês; salário mínimo usado nas faixas: R$ 954

## 1.11 peso_semana: cada dia da semana com 1/7 do peso em cada split
treino:
```
               fatia dos dias na amostra  fator do peso  fatia com peso_semana
dia_semana                                                                    
Segunda-Feira                      0.154           0.96                 0.1429
Terca-Feira                        0.177           0.79                 0.1429
Quarta-Feira                       0.163           0.92                 0.1429
Quinta-Feira                       0.178           0.78                 0.1429
Sexta-Feira                        0.087           1.60                 0.1429
Sabado                             0.046           2.95                 0.1429
Domingo                            0.196           0.74                 0.1429
```
teste:
```
               fatia dos dias na amostra  fator do peso  fatia com peso_semana
dia_semana                                                                    
Segunda-Feira                      0.158           0.90                 0.1429
Terca-Feira                        0.165           0.90                 0.1429
Quarta-Feira                       0.178           0.83                 0.1429
Quinta-Feira                       0.166           0.86                 0.1429
Sexta-Feira                        0.095           1.44                 0.1429
Sabado                             0.045           2.94                 0.1429
Domingo                            0.194           0.73                 0.1429
```

## 1.12 Diferenças de composição entre treino e teste (sem rótulo)
regiao (fatia ponderada por peso_semana):
```
split         teste  treino  teste − treino (p.p.)
regiao                                            
Centro-Oeste  0.073   0.073                 -0.073
Nordeste      0.263   0.264                 -0.154
Norte         0.090   0.089                  0.081
Sudeste       0.425   0.434                 -0.860
Sul           0.150   0.140                  1.006
```
urbano (fatia ponderada por peso_semana):
```
split   teste  treino  teste − treino (p.p.)
urbano                                      
0       0.135   0.149                 -1.366
1       0.865   0.851                  1.366
```
tipo_area (fatia ponderada por peso_semana):
```
split                 teste  treino  teste − treino (p.p.)
tipo_area                                                 
capital               0.235   0.237                 -0.208
regiao_metropolitana  0.151   0.166                 -1.468
interior_urbano       0.478   0.448                  3.042
rural                 0.135   0.149                 -1.366
```
faixa_idade (fatia ponderada por peso_semana):
```
split        teste  treino  teste − treino (p.p.)
faixa_idade                                      
10_17        0.136   0.143                 -0.622
18_29        0.206   0.207                 -0.088
30_44        0.260   0.258                  0.210
45_59        0.223   0.217                  0.593
60_mais      0.174   0.175                 -0.094
```
sexo (fatia ponderada por peso_semana):
```
split      teste  treino  teste − treino (p.p.)
sexo                                           
feminino    0.52   0.525                 -0.487
masculino   0.48   0.475                  0.487
```
faixa_renda (fatia ponderada por peso_semana):
```
split        teste  treino  teste − treino (p.p.)
faixa_renda                                      
ate_0.5sm    0.157   0.181                 -2.371
0.5_1sm      0.250   0.236                  1.404
1_2sm        0.333   0.308                  2.464
2_5sm        0.188   0.206                 -1.832
acima_5sm    0.072   0.068                  0.335
```
classe_abep (fatia ponderada por peso_semana):
```
split        teste  treino  teste − treino (p.p.)
classe_abep                                      
DE           0.305   0.320                 -1.514
C2           0.251   0.244                  0.741
C1           0.215   0.197                  1.768
B2           0.164   0.169                 -0.458
B1           0.042   0.048                 -0.641
A            0.022   0.021                  0.104
```
Taxa ponderada do treino: 0.3025%. A do teste não é recalculada aqui.
