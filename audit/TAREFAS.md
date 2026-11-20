# Auditoria dos modelos – lista de tarefas

Escopo: todos os modelos do projeto até o commit `46c0a6e` (20/11/2026).

Regras:
- Nenhum arquivo existente é alterado. Os scripts ficam em `audit/` e os resultados em `audit/outputs/`.
- Scripts do projeto que gravam arquivos (`src/models/treinar.py`, `src/features/build_grid.py`, `notebooks/03_eda.py`) não são executados. A lógica deles é reexecutada em memória.
- O período de teste (mar–jul/2018) não é usado para medir nenhum modelo. Ele ainda não foi aberto pelo protocolo do README, e abri-lo aqui gastaria a avaliação única. Nele só entram checagens estruturais, sem desempenho: chaves, sobreposição de domicílios, duplicatas e distribuição das features.

## Fases

- [x] **Fase 0 – Inventário** (`outputs/00_inventario.md`)
  - Arquivos, dados, modelos salvos e documentos.
  - Tabela dos modelos.
  - Fluxo do dado bruto até a previsão, com as etapas soltas.
- [x] **Fase 1 – Dados** (`01_dados.py` → `outputs/01_dados.md`)
  - Fonte, período e definição do alvo; schema e tipos; nulos.
  - Duplicatas na chave pessoa × dia × período.
  - Valores impossíveis.
  - Continuidade temporal, com o mês da entrevista reconstruído.
  - Filtros, joins e unidades.
- [x] **Fase 2 – Leakage** (`02_leakage.py` → `outputs/02_leakage.md`)
  - Temporal, features do futuro, pré-processamento dentro do fold e alvo (feature dominante).
  - Grupos (domicílio e UPA).
  - Seleção feita olhando o teste, inclusive no histórico do git.
  - Duplicatas entre treino e teste.
  - Reexecução sem o vazamento para cada suspeita.
- [x] **Fase 3 – Modelagem** (`03_modelagem.py`, `05_extras.py` → `outputs/`)
  - Baseline e adequação da validação, com origem móvel dentro do treino.
  - Métricas e viés por segmento; comparação justa.
  - Overfitting e instabilidade entre folds.
  - Efeitos de bebidas: sazonalidade, eventos, clima e preço.
  - Análise de erro.
- [x] **Fase 4 – Documentação × código** (`04_documentacao.py`, reprodução dos notebooks 04 e 02 → `outputs/`)
  - Divergências entre o que os documentos dizem e o que o código faz.
  - Reprodução das métricas reportadas.
  - Seeds, versões, caminhos e ordem de execução.
- [x] **Fase 5 – Relatório** `audit/RELATORIO_AUDITORIA.md`
