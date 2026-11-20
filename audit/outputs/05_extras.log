# Checagens extras (só treino)

## 5.1 PR-AUC média dos folds × PR-AUC agregada das previsões out-of-fold
```
                                    modelo  PR-AUC média dos folds  PR-AUC agregada (média das 10 repetições) mín–máx agregada  captura agregada
                                  baseline                  0.0160                                     0.0143    0.0138–0.0148            0.5362
                 Regressão Logística final                  0.0371                                     0.0276    0.0264–0.0290            0.6689
Regressão Logística dia da semana (C 0,03)                  0.0386                                     0.0292    0.0273–0.0319            0.6728
           LightGBM (16 folhas, mín. 3000)                  0.0284                                     0.0231    0.0203–0.0270            0.6356
```
Razão final ÷ baseline: 2.31 na média dos folds; 1.92 agregada.

## 5.2 Por que a média dos folds é maior: amostra pequena ou mistura de escalas?
```
 repetição  agregada (5 modelos misturados)  média dos folds originais (sem mistura)  média em quintos aleatórios das previsões agregadas
         0                           0.0274                                   0.0377                                               0.0355
         1                           0.0264                                   0.0411                                               0.0332
         2                           0.0280                                   0.0377                                               0.0362
```
Se a média em quintos aleatórios (que também mistura os 5 modelos) fica perto da média dos folds, a diferença vem do tamanho da amostra, não da mistura de escalas.
