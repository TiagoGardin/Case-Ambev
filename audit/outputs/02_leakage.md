# Fase 2 – Leakage (só treino)

## 2.1 Target leakage: força de cada coluna sozinha (taxa por categoria/decil, 50 folds)
```
     coluna          papel  PR-AUC sozinha  captura top 10%
    periodo        feature          0.0110           0.3761
classe_abep fora do modelo          0.0062           0.2132
anos_estudo        feature          0.0048           0.2011
   renda_pc        feature          0.0044           0.1662
 dia_semana        feature          0.0042           0.1118
dia_atipico fora do modelo          0.0040           0.2136
      idade        feature          0.0040           0.1332
  tipo_area fora do modelo          0.0037           0.1096
         UF        feature          0.0037           0.1406
     urbano        feature          0.0034           0.1135
       sexo        feature          0.0034           0.1177
n_moradores        feature          0.0031           0.0838
```
Referência: prevalência 0,0030; baseline dia × período 0,0160; modelo final 0,0371 (README).
dia_atipico (excluída por leakage no Data Card): taxa da ocasião por valor, treino:
```
               linhas  positivos  taxa ponderada
dia_atipico                                     
0            199552.0      388.0        0.002562
1             24196.0      135.0        0.006474
```
Colunas marcadoras do alvo na grade (pizza, refri, guarana, y_guarana, local_pizza) e dia_atipico não entram em FEATURES (src/models/preparo.py:31); preparar() devolve só COLUNAS_PREPARADAS, derivadas das 9 features.

## 2.2 Grupos: partições por UPA (vizinhança) em vez de domicílio
```
                    modelo  PR-AUC (domicílio)  PR-AUC (UPA)  diferença  captura (domicílio)  captura (UPA)
                  baseline              0.0160        0.0163     0.0003               0.5419         0.5478
 Regressão Logística final              0.0371        0.0359    -0.0012               0.6687         0.6698
LightGBM (melhor da grade)              0.0284        0.0297     0.0013               0.6332         0.6311
```

## 2.3 Temporal: origem móvel dentro do treino (treina no passado, valida no mês seguinte)
```
           origem  positivos na validação                                       modelo             PR-AUC captura top 10%
jul–set → out–nov                     151                                     baseline             0.0124        0.454287
jul–set → out–nov                     151                    Regressão Logística final             0.0291        0.622951
jul–set → out–nov                     151   Regressão Logística dia da semana (C 0,03)             0.0303        0.813515
jul–set → out–nov                     151                   LightGBM (melhor da grade)             0.0225         0.68656
jul–set → out–nov                     151 final − baseline (IC 95%, bootstrap de UPAs) [+0.0026; +0.0580]                
jul–nov → dez–jan                     105                                     baseline             0.0128        0.550893
jul–nov → dez–jan                     105                    Regressão Logística final             0.0151        0.421214
jul–nov → dez–jan                     105   Regressão Logística dia da semana (C 0,03)             0.0174        0.425119
jul–nov → dez–jan                     105                   LightGBM (melhor da grade)             0.0134        0.409408
jul–nov → dez–jan                     105 final − baseline (IC 95%, bootstrap de UPAs) [-0.0071; +0.0269]                
    jul–jan → fev                      67                                     baseline             0.0250          0.7189
    jul–jan → fev                      67                    Regressão Logística final             0.0382        0.744178
    jul–jan → fev                      67   Regressão Logística dia da semana (C 0,03)             0.0490        0.789982
    jul–jan → fev                      67                   LightGBM (melhor da grade)             0.0329        0.731077
    jul–jan → fev                      67 final − baseline (IC 95%, bootstrap de UPAs) [-0.0028; +0.0550]                
```
Comparar com a validação cruzada aleatória agrupada (README): baseline 0,0160, final 0,0371, Regressão Logística com dia da semana 0,0386, LightGBM 0,0284.
UCs do treino com mês a mais de 3 meses da mediana da UPA, por origem do mês (candidatas a mês estimado errado):
```
                  longe    UCs  fração
fonte_mes                             
aluguel_estimado     16  11098  0.0014
aluguel_pago          0   1754  0.0000
vizinhos_upa          0    298  0.0000
```
UCs em lado do corte diferente do da mediana da sua UPA e a mais de 3 meses dela: 13 de 20,145 (limite superior grosseiro de UCs no split errado)

## 2.4 Calibração com validação cruzada aninhada (valida também o C × 4/5 fora da amostra)
```
                                   versão  log loss ponderado (média dos folds)  média prevista  taxa observada
                             sem sigmoide                               0.01796         0.00259         0.00302
                sigmoide + C da validação                               0.01681         0.00403         0.00302
sigmoide + C × 4/5 (como no modelo final)                               0.01673         0.00302         0.00302
```
Comparar com o notebook 04 (sigmoide de fora do fold, sem aninhar): log loss 0,01796 sem sigmoide e 0,01667 com; média prevista 0,304% contra 0,302% observada.

## 2.5 Sensibilidade da definição do alvo: bebidas no limite (água tônica e H2O)
Códigos no limite: [8201017, 8204902, 8209002, 8212902, 8212906] (BEBIDA REFRIGERANTE, TIPO H2O, AGUA TONICA TRADICIONAL, AGUA TONICA LIGHT, AGUA TONICA DIET, REFRIGERANTE DE QUININO DIETETICO)
Positivos do treino: 523 com a definição atual; 521 sem as bebidas no limite.
