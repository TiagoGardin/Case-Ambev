# Fase 0 – Inventário

Estado auditado: commit `46c0a6e` (20/11/2026 18:40), árvore limpa. São 51 arquivos versionados.

## Artefatos

| Tipo | Arquivos |
|---|---|
| Ingestão | `src/ingestion/download_pof.py` (download e extração), `src/ingestion/pof.py` (leitura de largura fixa → 6 parquets), `src/ingestion/codigos_alimentos.py` (códigos de pizza e refrigerante) |
| Features | `src/features/mes_entrevista.py` (mês reconstruído), `tipo_area.py`, `classe_abep.py`, `build_grid.py` (grade, alvo, split, `peso_semana`) |
| Modelagem | `src/models/preparo.py`, `modelos.py`, `validacao.py`, `metricas.py`, `calibracao.py`, `artefato.py`, `treinar.py` |
| Notebooks (scripts `.py`) | `01_verificacao_pof.py` (sanidade, mês), `02_triagem_features.py` (triagem, só treino), `03_eda.py` (só treino; grava figuras e tabelas), `04_modelagem.py` (seleção, só treino) |
| Dados | `data/raw` (zips do IBGE, não versionados), `data/processed/pof_*.parquet` (6), `data/processed/grid.parquet` (340.072 × 36) |
| Modelo salvo | `models/occasion_model.joblib` (6.269 bytes) e `models/model_metadata.json` |
| Documentos | `README.md`, `docs/data_card.md`, `docs/notas_pof.md`, `reports/eda_tabelas.md`, 5 figuras |
| Testes | `tests/test_ingestion.py`, `tests/test_features.py`, `tests/test_models.py` (62 testes) |
| Vazios (pendentes) | `src/api/`, `src/decision_engine/`, `app/`, `data/external/` |

## Modelos

| ID | Modelo | Algoritmo | Onde | Dados | Features | Validação | Métricas reportadas | Versão | Status |
|---|---|---|---|---|---|---|---|---|---|
| M1 | Baseline dia × período | taxa ponderada (`peso_semana`) em 28 células | `src/models/modelos.py` (`BaselineDiaPeriodo`) | treino | período × dia | CV 5 folds × 10 por domicílio | PR-AUC 0,0160; captura no top 10% 54,2% | c70e865 (18/11) | ativo (referência) |
| M2 | Regressão Logística de referência | LR L2, C = 0,03 | `modelos.py` (`regressao_logistica`, `colunas_lr("dia_semana")`) | treino | período, dia, período × dia, região, faixa etária, sexo, anos de estudo, moradores, log renda, urbano | mesma CV | 0,0386; 67,2% | c70e865 | descartado (a versão mais simples ficou) |
| **M3** | **Modelo final** | LR L2, C = 0,003, sigmoide ponderada (a = 2,499, b = 8,230); ajuste final com C = 0,0024 | `src/models/treinar.py`, `models/occasion_model.joblib` | treino | período, fim de semana, período × fim de semana, região, faixa etária, sexo, anos de estudo, moradores, log renda, urbano | mesma CV; teste pendente | 0,0371; 66,9% | 46c0a6e (20/11) | **ativo, congelado** |
| M4 | LR com renda em faixas | LR L2, C = 0,1 | `colunas_lr(..., "faixas")` | treino | M3 com faixas de renda | mesma CV | 0,0306; 67,4% | 46c0a6e | descartado |
| M5 | LR com `peso_semana` no ajuste | LR L2, C = 0,003, ponderada | notebook 04, seção 8 | treino | as de M3 | mesma CV | 0,0414; 65,3% | 46c0a6e | descartado pela regra "mais simples" |
| M6 | LightGBM raso (grade de 6) | 300 árvores, taxa 0,03, λ = 5; 4, 8 ou 16 folhas × 300 ou 3.000 linhas por folha | `modelos.py` (`lightgbm_raso`) | treino | as 9 brutas | mesma CV | melhor: 0,0284; 63,3% | 46c0a6e | descartado |
| M7 | Triagem: LightGBM | 300 árvores, 15 folhas, 100 linhas por folha | `notebooks/02_triagem_features.py` (`PARAMS_LGB`) | treino | 9 + grupos candidatos | mesma CV | base 0,0259 | a463344 (18/11) | auxiliar (seleção de features) |
| M8 | Triagem: LR simples | LR C = 0,05, idade e idade², log renda, one-hot por `get_dummies` | notebook 02 (`prever_lr`) | treino | 9 + grupos candidatos | mesma CV | base 0,0356 | a463344 | auxiliar |
| M9 | Triagem exploratória de 16/11 | LightGBM e LR | rascunho fora do repositório (log da sessão de 16/11) | treino, avaliada no **teste** (noites) | as mesmas 9 + 7 grupos | **teste temporal** | ROC-AUC nas noites 0,814 → 0,829 com tipo de área e ABEP | não versionado | substituído em 18/11 (M7/M8) |
| M10 | Triagem com partição única | LightGBM e LR | versão intermediária do notebook 02 | treino | idem | 1 partição | — | não versionado | substituído (instável, notebook 02, linhas 25–27) |

## Fluxo

```
IBGE (zips) → download_pof.py → data/raw/pof/
  → pof.py → data/processed/pof_{consumo,morador,aluguel_estimado,pagamentos_moradia,domicilio,inventario}.parquet
  → codigos_alimentos.py (alvo) + mes_entrevista.py (split) + tipo_area.py + classe_abep.py
  → build_grid.py → grid.parquet (alvo y, 9 features, split por domicílio, peso_semana)
  → 01 (sanidade) | 02 (triagem, treino) | 03 (EDA, treino → reports/)
  → src/models: preparo (sem estado) → pipelines → validação cruzada → calibração
  → 04_modelagem.py (seleção, confere CONFIG) → treinar.py → models/ (congelado)
  → [pendente] avaliação no teste, SHAP/top_factors, Decision Engine, API, app
```

## Etapas soltas ou duplicadas
- **Dois códigos de modelo:** a triagem (notebook 02) tem uma Regressão Logística e um LightGBM próprios, com outra especificação da LR (idade e idade², `get_dummies`, C fixo). As features foram triadas com modelos diferentes dos da seleção (M2–M6).
- **Contexto externo prometido e não implementado:** `data/external/` está vazia. Open-Meteo e BrasilAPI estão "_A preencher_" no Data Card.
- **Camadas pendentes:** `src/api/`, `src/decision_engine/` e `app/` só têm `__init__.py` ou `.gitkeep`. O README está marcado "em construção".
- **Mês reconstruído:** é usado só no split e não aparece como feature. A regra é validada no notebook 01 e em `mes_entrevista.validar_regra_pagamento`, a mesma função nos dois lugares, sem duplicação de lógica.
