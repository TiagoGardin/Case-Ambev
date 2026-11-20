# Fase 3 – Modelagem (só treino)

## 3.1 Reprodução das métricas de validação cruzada citadas no README
```
                         modelo  PR-AUC reproduzida  PR-AUC no README  captura reproduzida  captura no README
                       baseline              0.0160            0.0160               0.5419              0.542
      Regressão Logística final              0.0371            0.0371               0.6687              0.669
LightGBM (16 folhas, mín. 3000)              0.0284            0.0284               0.6332                NaN
```

## 3.2 Overfitting: dentro da amostra × validação cruzada
```
                              modelo  PR-AUC no próprio treino  PR-AUC na validação cruzada  captura no treino  captura na validação
Regressão Logística final (artefato)                    0.0289                       0.0371             0.6756                0.6687
                LightGBM (16 folhas)                    0.1477                       0.0284             0.9098                0.6332
```

## 3.3 Instabilidade entre folds e tamanho efetivo da amostra
```
                  count    mean     std     min     10%     50%     90%     max
baseline           50.0  0.0160  0.0045  0.0071  0.0092  0.0164  0.0213  0.0297
final              50.0  0.0371  0.0188  0.0116  0.0175  0.0315  0.0633  0.0905
final − baseline   50.0  0.0211  0.0176 -0.0031  0.0047  0.0147  0.0469  0.0721
```
Final à frente em 48 de 50 folds; teste t corrigido: ganho +0.0211, t = 2.30, p = 0.013
Tamanho efetivo (Kish) dos 523 positivos com peso_semana: 138 (26%); de todas as linhas: 62,563 (28%)
```
  modelo  PR-AUC sem peso (média dos folds)  PR-AUC ponderada
baseline                             0.0116            0.0160
   final                             0.0248            0.0371
```

## 3.4 Custo da regra conservadora: o caminho 'maior média' teria dado quanto?
Regressão Logística com dia da semana e peso_semana no ajuste (melhor C = 0.01): PR-AUC 0.0370, captura 66.7%; contra o final: -0.0001 (t = -0.01, p = 0.50)

## 3.5 Viés de calibração por segmento (previsões out-of-fold, sigmoide de fora do fold)
```
   segmento                nível  ocasiões prevista observada  observada / prevista     alerta
    periodo               almoco      12.0   0.063%    0.033%                  0.52           
    periodo                manha      13.0   0.063%    0.013%                  0.21           
    periodo                noite     481.0   1.023%    1.140%                  1.11           
    periodo                tarde      17.0   0.065%    0.024%                  0.38           
 dia_semana              Domingo     180.0   0.439%    0.497%                  1.13           
 dia_semana         Quarta-Feira      52.0   0.190%    0.170%                  0.90           
 dia_semana         Quinta-Feira      81.0   0.184%    0.280%                  1.52 viés > 30%
 dia_semana               Sabado      35.0   0.465%    0.525%                  1.13           
 dia_semana        Segunda-Feira      55.0   0.193%    0.157%                  0.81           
 dia_semana          Sexta-Feira      59.0   0.470%    0.308%                  0.66 viés > 30%
 dia_semana          Terca-Feira      61.0   0.184%    0.180%                  0.98           
     regiao         Centro-Oeste      78.0   0.389%    0.289%                  0.74 viés > 30%
     regiao             Nordeste     171.0   0.230%    0.206%                  0.90           
     regiao                Norte      66.0   0.243%    0.222%                  0.92           
     regiao              Sudeste     113.0   0.326%    0.335%                  1.03           
     regiao                  Sul      95.0   0.368%    0.443%                  1.21           
     urbano                    0      41.0   0.107%    0.067%                  0.63 viés > 30%
     urbano                    1     482.0   0.338%    0.344%                  1.02           
  tipo_area              capital     197.0   0.438%    0.444%                  1.01           
  tipo_area      interior_urbano     213.0   0.289%    0.266%                  0.92           
  tipo_area regiao_metropolitana      72.0   0.327%    0.409%                  1.25           
  tipo_area                rural      41.0   0.107%    0.067%                  0.63 viés > 30%
faixa_idade                10_17      97.0   0.165%    0.294%                  1.78 viés > 30%
faixa_idade                18_29     147.0   0.403%    0.484%                  1.20           
faixa_idade                30_44     170.0   0.394%    0.350%                  0.89           
faixa_idade                45_59      83.0   0.280%    0.194%                  0.69 viés > 30%
faixa_idade              60_mais      26.0   0.195%    0.159%                  0.81           
       sexo             feminino     270.0   0.290%    0.253%                  0.87           
       sexo            masculino     253.0   0.318%    0.357%                  1.12           
faixa_renda              0.5_1sm     108.0   0.168%    0.206%                  1.22           
faixa_renda                1_2sm     161.0   0.254%    0.296%                  1.17           
faixa_renda                2_5sm     159.0   0.468%    0.488%                  1.04           
faixa_renda            acima_5sm      41.0   1.044%    0.571%                  0.55 viés > 30%
faixa_renda            ate_0.5sm      54.0   0.099%    0.127%                  1.28           
classe_abep                    A      16.0   0.966%    0.416%                  0.43           
classe_abep                   B1      49.0   0.729%    1.121%                  1.54 viés > 30%
classe_abep                   B2     112.0   0.546%    0.397%                  0.73 viés > 30%
classe_abep                   C1     150.0   0.338%    0.385%                  1.14           
classe_abep                   C2     121.0   0.210%    0.274%                  1.30 viés > 30%
classe_abep                   DE      75.0   0.117%    0.093%                  0.79           
```

## 3.6 Análise de erro com os cortes de produção (sigmoide e cortes do artefato)
```
     segmento                nível  ocasiões ocasiões em alta ocasiões perdidas (baixa) fatia do segmento em alta taxa dentro de alta
      periodo               almoco        12             0.0%                     82.7%                      0.0%               0.00%
      periodo                manha        13             0.0%                     85.8%                      0.0%               0.00%
      periodo                noite       481            71.1%                      0.9%                     40.0%               2.03%
      periodo                tarde        17             0.0%                     80.4%                      0.0%               0.00%
fim_de_semana                    0       249            44.9%                     11.3%                      6.3%               1.41%
fim_de_semana                    1       274            80.0%                      2.3%                     15.0%               2.37%
       regiao         Centro-Oeste        78            55.4%                     12.6%                     11.9%               1.35%
       regiao             Nordeste       171            58.7%                      6.9%                      7.4%               1.64%
       regiao                Norte        66            63.4%                      3.1%                      8.7%               1.63%
       regiao              Sudeste       113            69.0%                      3.4%                     10.7%               2.16%
       regiao                  Sul        95            74.5%                      8.1%                     12.7%               2.60%
       urbano                    0        41            41.6%                     17.8%                      2.6%               1.08%
       urbano                    1       482            67.8%                      5.2%                     11.3%               2.06%
    tipo_area              capital       197            74.2%                      2.8%                     13.4%               2.45%
    tipo_area      interior_urbano       213            59.6%                      4.9%                     10.1%               1.58%
    tipo_area regiao_metropolitana        72            72.3%                      9.7%                     11.6%               2.55%
    tipo_area                rural        41            41.6%                     17.8%                      2.6%               1.08%
  faixa_idade                10_17        97            52.0%                     11.0%                      5.5%               2.78%
  faixa_idade                18_29       147            76.6%                      5.4%                     14.5%               2.56%
  faixa_idade                30_44       170            66.8%                      3.5%                     12.7%               1.84%
  faixa_idade                45_59        83            56.2%                      5.9%                      9.0%               1.21%
  faixa_idade              60_mais        26            71.7%                      4.8%                      5.6%               2.01%
         sexo             feminino       270            63.0%                      6.4%                      9.7%               1.65%
         sexo            masculino       253            70.0%                      5.0%                     10.3%               2.41%
  faixa_renda              0.5_1sm       108            41.4%                     12.8%                      6.0%               1.42%
  faixa_renda                1_2sm       161            51.5%                      6.3%                     10.0%               1.53%
  faixa_renda                2_5sm       159            85.8%                      3.7%                     17.1%               2.45%
  faixa_renda            acima_5sm        41            96.2%                      0.0%                     23.4%               2.35%
  faixa_renda            ate_0.5sm        54            50.2%                      5.7%                      2.1%               2.97%
  classe_abep                    A        16            83.4%                      0.0%                     23.0%               1.51%
  classe_abep                   B1        49            81.8%                      1.4%                     21.4%               4.28%
  classe_abep                   B2       112            82.0%                      0.9%                     17.9%               1.82%
  classe_abep                   C1       150            66.0%                      6.6%                     12.9%               1.97%
  classe_abep                   C2       121            47.0%                     11.4%                      7.7%               1.67%
  classe_abep                   DE        75            48.5%                     10.2%                      3.2%               1.41%
```
Geral: ocasiões em alta 67.0%, em media 27.4%, em baixa 5.6%

## 3.7 Sazonalidade e eventos dentro do treino (o mês não é feature)
```
   mês  positivos taxa ponderada    IC 95% (UPAs) fatia Sudeste
jul/17         47         0.352% [0.184%; 0.579%]         43.1%
ago/17         88         0.297% [0.192%; 0.407%]         39.8%
set/17         65         0.367% [0.196%; 0.582%]         41.4%
out/17         88         0.277% [0.173%; 0.389%]         42.5%
nov/17         63         0.309% [0.130%; 0.603%]         43.6%
dez/17         56         0.385% [0.175%; 0.674%]         45.7%
jan/18         49         0.133% [0.072%; 0.191%]         41.7%
fev/18         67         0.322% [0.127%; 0.570%]         49.3%
```
Eventos: o fim de ano (dez/17) e o Carnaval (9 a 14/02/2018) estão no treino. A Copa do Mundo (14/06 a 15/07/2018) e as festas juninas caem no período de teste, sem equivalente no treino. A POF não traz a data exata do recordatório, só o dia da semana e o mês estimado, então feriados não são identificáveis.

## 3.8 Calibração por dia: modelo final (fim de semana) × Regressão Logística com dia da semana
```
               ocasiões observada prevista final prevista dia da semana  erro final  erro dia da semana
grupo                                                                                                  
Segunda-Feira        55    0.157%         0.193%                 0.201%        0.23                0.28
Terca-Feira          61    0.180%         0.184%                 0.191%        0.03                0.06
Quarta-Feira         52    0.170%         0.190%                 0.185%        0.11                0.09
Quinta-Feira         81    0.280%         0.184%                 0.250%       -0.34               -0.11
Sexta-Feira          59    0.308%         0.470%                 0.357%        0.52                0.16
Sabado               35    0.525%         0.465%                 0.369%       -0.11               -0.30
Domingo             180    0.497%         0.439%                 0.570%       -0.12                0.15
```
Erro absoluto médio por dia: final 21%, dia da semana 16%. PR-AUC: final 0.0371, dia da semana 0.0386.
