# Fase 4 – Documentação × código e reprodutibilidade

## 4.1 O modelo commitado é reproduzível a partir do código?
```
                              item        metadados      reproduzido
                     a da sigmoide         2.499379         2.499379
                     b da sigmoide         8.229964         8.229964
                        corte alta         0.007765         0.007765
                       corte media         0.001436         0.001436
 PR-AUC média na validação cruzada         0.037105         0.037105
maior diferença de score no treino                          0.00e+00
      SHA-256 do arquivo do modelo 12a7f3f80b1228c3 c8d60861eb34fbcc
```

## 4.2 Números citados no README e no Data Card
```
                                         afirmação   documentado   recalculado
                   README: pessoas no recordatório        46.164        46,164
                         README: linhas e ocasiões 340.072 e 838 340,072 e 838
             Data Card: treino (linhas, positivos) 223.748 e 523 223,748 e 523
              Data Card: teste (linhas, positivos) 116.316 e 315 116,316 e 315
      Data Card: domicílios com positivos no teste           189           189
              README: taxa do treino (peso_semana)         0,30%         0.30%
               README: taxa do teste (peso_semana)         0,41%         0.41%
                README: fatia das ocasiões à noite           92%           94%
                   README: taxa noturna na segunda         0,50%         0.50%
                    README: taxa noturna no sábado         2,09%         2.09%
README: sexta a domingo ÷ segunda a quinta (noite)           2,4           2.4
                     README: noite, classe DE e B1 0,34% e 4,40% 0.34% e 4.40%
                    README: noite, rural e capital 0,23% e 1,71% 0.23% e 1.71%
                        README: noite, 60+ e 18–29 0,61% e 1,82% 0.61% e 1.82%
                    README: pizzas comidas em casa           75%           75%
                  README: em casa, Sudeste e Norte     80% e 56%     80% e 56%
```

## 4.3 Ambiente: versões instaladas × requirements.txt
```
      pacote requirements instalado  confere
      pandas        3.1.1     3.1.1     True
     pyarrow       26.0.1    26.0.1     True
    openpyxl        3.1.5     3.1.5     True
        xlrd        2.0.2     2.0.2     True
    requests       2.34.2    2.34.2     True
scikit-learn        1.9.1     1.9.1     True
       scipy       1.18.1    1.18.1     True
    lightgbm        4.7.0     4.7.0     True
        shap       0.54.1    0.54.1     True
  matplotlib       3.11.2    3.11.2     True
     fastapi      0.162.4   0.162.4     True
     uvicorn       0.54.0    0.54.0     True
    pydantic       2.14.3    2.14.3     True
   streamlit       1.67.0    1.67.0     True
      pytest        9.2.0     9.2.0     True
      python         3.12   3.12.14     True
```
Importados pelo código e ausentes do requirements.txt (vêm como dependência): ['joblib', 'numpy']

## 4.4 Caminhos fixos, sementes e testes
Caminhos absolutos no código: nenhum
Sementes declaradas:
```
src/models/validacao.py:42:            for tr, va in StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=r)
src/models/modelos.py:62:              "deterministic": True, "force_col_wise": True, "random_state": 0, "n_jobs": 1,
notebooks/03_eda.py:120:    rng = np.random.default_rng(semente)
notebooks/02_triagem_features.py:83:                  reg_lambda=5, deterministic=True, force_col_wise=True, random_state=0,
```
pytest: 62 passed, 5 warnings in 4.07s

## 4.5 Trechos dos documentos usados nas divergências (arquivo:linha)
README.md:
```
5:> Status: em construção
14:  → src/models         (baseline, candidatos, avaliação, SHAP)
25:| `data/external` | Fontes de contexto (Vigitel, Censo, OSM, clima) |
38:- **Features:** quando (período, dia da semana), onde (UF, urbano ou rural) e quem (idade, sexo, escolaridade, tamanho da família, renda per capita), escolhidas por triagem com validação cruzada no treino. Tipo de áre
85:- **Regressão Logística:** categóricas em one-hot, numéricas padronizadas (anos de estudo, moradores e log da renda per capita) e regularização L2; `regiao` no lugar de `UF` (27 categorias são muitas para 523 positivo
96:2. **Teste temporal** (mar–jul/2018): o modelo escolhido, treinado em todo o treino, é avaliado **uma única vez**, ao lado do baseline. Nada é reajustado depois; qualquer mudança posterior é reportada como exploratóri
97:3. **Generalização por região** (complementar): dentro do treino, ajustar com 4 regiões e avaliar na 5ª.
108:- **Explicações:** SHAP global e local. Os 3 fatores de maior contribuição de cada previsão formam o `top_factors` da API.
161:_A preencher. O teste (mar–jul/2018) é aberto uma única vez, com o modelo de [models/](models/)._
```
docs/data_card.md:
```
3:Seções exigidas pelo TAPI (Seção 4). Itens marcados com _A preencher_ ainda estão em aberto.
28:Escolhidas pela triagem com validação cruzada no treino (abaixo). `tipo_area` e `classe_abep` continuam na base para os segmentos e para o Decision Engine.
40:**Triagem de candidatas** ([notebooks/02_triagem_features.py](../notebooks/02_triagem_features.py), 18/11/2026)
66:| Open-Meteo / BrasilAPI | Chuva e feriado do dia no Decision Engine (não entram no modelo) | _A preencher_ | Município × dia | Município + data |
67:| Vigitel 2006–2024, Censo 2022, OSM | Opcionais: contexto recente para o Mapa de Oportunidade | _A preencher_ | UF / capital | UF |
```
Pastas citadas na arquitetura e ainda vazias: ['src/api', 'src/decision_engine', 'app', 'data/external']
