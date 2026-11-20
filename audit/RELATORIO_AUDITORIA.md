# Relatório de auditoria dos modelos

**Escopo:** todos os modelos do projeto até o commit `46c0a6e` (20/11/2026, 18:40). Lista de tarefas em [TAREFAS.md](TAREFAS.md); scripts em `audit/`, saídas completas em [outputs/](outputs/).

**Regras seguidas:**
- Nenhum arquivo existente foi alterado.
- Scripts do projeto que gravam arquivos não foram executados; a lógica deles foi refeita em memória.
- **O período de teste (mar–jul/2018) não foi usado para medir nenhum modelo.** O protocolo do README prevê uma avaliação única, e a auditoria não pode gastá-la.
- No teste, só entraram checagens estruturais (chaves, sobreposição, composição) e a conferência de contagens que o Data Card já publicava.

**O roteiro é de previsão de vendas; o projeto é outro.** O projeto é uma classificação binária na POF 2017–18, com uma linha por pessoa × dia × período e alvo "pizza e refrigerante no mesmo período". A tradução dos itens:

| Item do roteiro | Equivalente aqui |
|---|---|
| Chave SKU × PDV × data | pessoa × dia × período |
| Cadastro de SKU | códigos de alimentos da POF |
| Semana ISO × comercial | dia da semana do recordatório |
| Hectolitros, preço com imposto, promoções, canibalização, MAPE e WAPE | não se aplicam; viram PR-AUC, captura e viés de calibração por segmento |

## 1. Resumo executivo

1. **O modelo atual é confiável para o uso proposto, com ressalvas.** Trata-se da Regressão Logística congelada em `models/`. Não há vazamento que invalide resultados e nenhum achado é Crítico.
2. **Os dados se sustentam.** O grid é reproduzido idêntico, sem nulos nas features, sem duplicatas na chave e sem perda nos joins. O alvo é rastreável: as bebidas no limite da definição mudam 2 de 523 positivos.
3. **As métricas reportadas se reproduzem exatamente:** baseline 0,0160, modelo final 0,0371, captura de 54,2% contra 66,9%. O modelo salvo é refeito com score idêntico.
4. **Dois achados Altos mudam a leitura dos números.**
   - A PR-AUC reportada é a média por fold, que fica 26% acima da PR-AUC agregada (0,0371 contra 0,0276). A vantagem sobre o baseline cai de 2,3 para 1,9 vezes, e o teste vai medir a agregada.
   - O ganho sobre o baseline não é estável no tempo. Com origem móvel dentro do treino, ele só é significativo em out–nov. Em dez–jan, a captura do modelo (42%) fica abaixo da do baseline (55%).
5. **A captura no top 10% (cerca de 67% contra 54%) é a métrica mais robusta** e se sustenta em todas as checagens.
6. **Calibração.** A calibração geral foi validada fora da amostra, com validação cruzada aninhada, incluindo a correção C × 4/5. Restam vieses acima de 30% em segmentos: quinta e sexta, 10–17 anos, renda acima de 5 salários mínimos, rural.
7. **O teste já foi lido uma vez.** Em 16/11, uma triagem exploratória mediu nele um modelo com as mesmas 9 features. As decisões tiradas daquilo foram revertidas, e o modelo final não depende do teste. Mas a frase "aberto uma única vez" precisa ser corrigida.
8. **Recomendação:** ajustar a documentação, com os itens 1 a 4 do plano, antes de abrir o teste. Não mudar o modelo agora.

## 2. Achados

Severidades: Crítico invalida os resultados reportados; Alto muda a conclusão ou a leitura dos números; Médio é risco real, mas contido ou já parcialmente documentado; Baixo é ajuste de clareza ou de higiene.

| ID | Modelo | Categoria | Severidade | Evidência | Impacto medido | Recomendação |
|---|---|---|---|---|---|---|
| A1 | M3 (e a comparação com M1) | modelagem / documentação | **Alto** | A PR-AUC é calculada por fold (`src/models/validacao.py:96`) e reportada como média (`README.md:130`). [outputs/05_extras.md](outputs/05_extras.md), seções 5.1 e 5.2 | PR-AUC agregada fora da amostra: final 0,0276 (9,1 vezes a prevalência, não 12,3), baseline 0,0143; razão de 1,92 em vez de 2,31. A causa principal é o tamanho da amostra: com cerca de 105 positivos, a PR-AUC sai enviesada para cima (em quintos aleatórios, 0,033–0,036; agregada, 0,026–0,028). A captura não muda (66,9%) | Reportar a PR-AUC agregada ao lado da média dos folds e usá-la como referência para o teste. Destacar a captura, que é estável. Escrever isso no README antes de abrir o teste |
| A2 | M3 × M1 | modelagem (validação) | **Alto** | Partições aleatórias agrupadas, sem ordem temporal (`src/models/validacao.py:42`). [outputs/02_leakage.md](outputs/02_leakage.md), seção 2.3 | Origem móvel dentro do treino, diferença de PR-AUC final − baseline com IC de 95% por UPA: out–nov [+0,003; +0,058]; dez–jan [−0,007; +0,027], com captura de 42% contra 55% do baseline; fev [−0,003; +0,055]. A razão final ÷ baseline vai de 1,2 a 2,3 conforme a janela | Documentar no README e em Limitações. Na avaliação do teste, mostrar o resultado por mês (descritivo). Não mudar o modelo por isso. Em seleções futuras, usar blocos temporais |
| M1 | M9 → M3 | leakage (seleção) | Médio | `git show a4a9372:docs/data_card.md`, linha 38: "triagem exploratória de 16/11/2026, ganho medido em ROC-AUC nas noites". Log da sessão de 16/11, fora do repositório. `README.md:96`, `:99`, `:161` | Em 16/11, um LightGBM com exatamente as 9 features atuais foi medido no teste: ROC-AUC de 0,814 nas noites, captura de 40%. Dali se decidiu incluir tipo de área e ABEP. Em 18/11 tudo foi refeito só com o treino e essas inclusões caíram. As exclusões que não foram retestadas têm motivo não estatístico: dado sensível, indisponível na hora da previsão ou desenho do split. O modelo final não depende do teste, mas o teste não é inédito | Descrever com precisão no README e no Data Card o que foi medido no teste e o que foi revertido. Trocar "aberto uma única vez" por "avaliado uma única vez com o modelo final" e citar a leitura exploratória de 16/11 |
| M2 | M3 | modelagem | Médio | Fim de semana fixado como sexta a domingo (`src/features/build_grid.py:52`). Na EDA, a noite de quinta tem 1,05% e a de sexta 1,15% (`reports/eda_tabelas.md:33-34`). [outputs/03_modelagem.md](outputs/03_modelagem.md), seção 3.8; 02_leakage, seção 2.3 | Na validação aleatória, o dia da semana fica +0,0015 à frente (p = 0,40), então ficou a opção mais simples. Calibração por dia: quinta −34%, sexta +52% (erro médio de 21%, contra 16% com dia da semana). Na origem móvel, o dia da semana ganha nas 3 janelas (PR-AUC de 0,0303, 0,0174 e 0,0490, contra 0,0291, 0,0151 e 0,0382) | Manter o modelo congelado, como manda o protocolo. **Antes de abrir o teste,** declarar a Regressão Logística com dia da semana como comparação exploratória no relatório do teste. Registrar o viés de quinta e sexta em Limitações |
| M3 | M3 | modelagem (calibração) | Médio | [outputs/03_modelagem.md](outputs/03_modelagem.md), seção 3.5 (sigmoide de fora do fold) | Observado ÷ previsto em segmentos com 30 ou mais ocasiões: quinta 1,52; sexta 0,66; Centro-Oeste 0,74; rural 0,63; 10–17 anos 1,78; 45–59 anos 0,69; renda acima de 5 SM 0,55; B1 1,54; B2 0,73; C2 1,30. O ranking não é afetado | Listar em Limitações e alertar no app, onde aparece "vezes a média". Hipóteses a testar depois do teste: o log da renda força uma subida contínua que não existe no topo; a escolaridade de quem tem 10–17 anos reflete a idade |
| M4 | M1–M6 | modelagem (variância) | Médio | [outputs/03_modelagem.md](outputs/03_modelagem.md), seção 3.3 | O tamanho efetivo (Kish) dos 523 positivos com `peso_semana` é de 138 (26%). A PR-AUC do final varia de 0,012 a 0,091 entre folds (desvio-padrão de 0,019). As diferenças entre candidatos (no máximo 0,009) ficam abaixo da resolução do protocolo. Sem peso, a conclusão é a mesma: 0,0248 contra 0,0116 | Reportar o tamanho efetivo e as métricas sem peso. Ler as diferenças entre candidatos como inconclusivas, não como empate provado |
| M5 | M3 | modelagem (sazonalidade e eventos) | Médio | [outputs/03_modelagem.md](outputs/03_modelagem.md), seção 3.7. O mês fica fora do modelo por desenho (Data Card, seção 3) | No treino, janeiro tem taxa de 0,133% (IC 0,072% a 0,191%), contra 0,28% a 0,39% nos outros meses. A Copa de 2018 e as festas juninas só existem no período de teste, onde a taxa sobe de 0,30% para 0,41%. Feriados não são identificáveis | Registrar em Limitações e mostrar o teste por mês. Não recalibrar com o teste, como diz o protocolo |
| M6 | — | documentação | Médio | `README.md:97` (generalização por região), `:14` e `:108` (SHAP), `:25` (`data/external`); `docs/data_card.md:66-67` (Open-Meteo e BrasilAPI "A preencher"); pastas `src/api`, `src/decision_engine`, `app` e `data/external` vazias | A generalização por região é pré-registrada no README e sugerida pelo TAPI, mas não foi feita. As demais pendências são esperadas para um projeto em construção | Fazer a generalização por região ou marcá-la como não feita. Ajustar a linha de `data/external` e as fontes "A preencher" ao que de fato será entregue |
| B1 | — | dados (alvo) | Baixo | `src/ingestion/codigos_alimentos.py:22` e `:73-92`; 01_dados, seção 1.7; 02_leakage, seção 2.5 | Água tônica (tradicional, diet e light), H2O e "quinino" contam como refrigerante, porque estão no subgrupo oficial ou no cadastro: 5 códigos e 62 registros. Tirá-los muda os positivos do treino de 523 para 521 | Registrar a escolha no Data Card |
| B2 | M3 | dados (grupos) | Baixo | 01_dados, seção 1.5; 02_leakage, seção 2.2 | 1.044 de 5.404 UPAs têm domicílios no treino e no teste (vizinhos de lados opostos do corte). Partições por UPA em vez de domicílio mudam a PR-AUC do final em −0,0012 | Documentar. O bootstrap por UPA já trata a dependência dentro do teste |
| B3 | — | dados (split) | Baixo | `src/features/mes_entrevista.py:41`; 01_dados, seção 1.6; 02_leakage, seção 2.3 | O mês de 15,6% das UCs é estimado por aluguel pago ou pelos vizinhos, com 97,0% no lado certo do corte na validação. Só 13 UCs ficam a mais de 3 meses da mediana da UPA e do outro lado do corte | Nenhuma ação: já está documentado no Data Card |
| B4 | M2–M6 | modelagem | Baixo | `notebooks/04_modelagem.py:183`; README, Resultados; 03_modelagem, seção 3.4 | A regra de qual opção de peso é "mais simples" foi decidida em 20/11, no notebook, e não no README pré-registrado. O caminho "maior média" (dia da semana com peso) dá 0,0370 contra 0,0371 (p = 0,50): a regra conservadora não custou nada | Nenhuma ação: já está divulgado no README |
| B5 | — | documentação | Baixo | `README.md:47`; 04_documentacao, seção 4.2 | "A noite concentra 92% dos casos" é a contagem sem peso (481 de 523). Ponderado, como a seção diz usar, dá 94% | Dizer "dos casos (sem peso)" ou usar 94% |
| B6 | M3 | documentação | Baixo | `README.md:38` e `:85` | O texto de Dados e o da Regressão Logística em Metodologia falam em dia da semana e em período × dia da semana. O modelo final usa o fim de semana derivado do dia, o que só aparece em Resultados | Alinhar o texto: o dia da semana é a entrada, o fim de semana é a variável do modelo |
| B7 | — | reprodutibilidade | Baixo | `requirements.txt`; 04_documentacao, seção 4.3 | `joblib` e `numpy` são importados e não estão fixados; vêm como dependência. As versões instaladas conferem com as 15 fixadas | Fixar os dois |
| B8 | M3 | reprodutibilidade | Baixo | 04_documentacao, seção 4.1 | Refazer o treino reproduz a sigmoide, os cortes e os scores (diferença máxima de 0), mas o arquivo gerado tem outro SHA-256 | Documentar que o hash confere integridade, não a reprodução byte a byte |
| B9 | M7, M8 | modelagem | Baixo | `notebooks/02_triagem_features.py:82-84`, `:216-228` | A triagem usou especificações diferentes das da seleção: LR com C fixo, idade e idade² e `get_dummies`; LightGBM com 15 folhas | Documentar. Opcionalmente, refazer a triagem com os pipelines de `src/models` |
| B10 | M8 | leakage (código) | Baixo | `notebooks/02_triagem_features.py:216` e `:227` | O one-hot da triagem é montado sobre treino e validação juntos. Só o conjunto de categorias vaza, sem rótulo; a imputação usa a mediana do treino. Não há vazamento de alvo | Opcional: trocar pelo `preparar` |

### O que se sustenta (verificações sem achado)

**Dados e alvo**
- O grid salvo é idêntico ao reconstruído.
- 46.164 pessoas; 0 nulos nas 9 features; 0 duplicatas na chave; cada pessoa-dia tem 4 linhas.
- `y` é igual a pizza e refrigerante no mesmo período.
- Os joins não perdem nem duplicam linhas. A renda per capita bate com a renda disponível do IBGE (razão mediana de 1,00, Spearman de 0,96). O `peso_semana` dá 1/7 a cada dia em cada split.
- A madrugada (0–4h) contada como noite afeta 1 de 767 positivos noturnos.
- Treino e teste diferem em no máximo 3 pontos percentuais de composição.

**Vazamento**
- Nenhuma feature sozinha chega perto do modelo: a melhor é o período, com PR-AUC de 0,0110.
- A `dia_atipico`, excluída, tem taxa 2,5 vezes maior nos dias atípicos, o que confirma que excluí-la estava certo.
- O preparo não aprende nada dos dados; padronização e one-hot ficam dentro do pipeline e são ajustados em cada fold.
- Nenhum notebook atual lê o teste.

**Calibração**
- A sigmoide e a correção C × 4/5 foram validadas com validação cruzada aninhada: previsto 0,302% contra 0,302% observado. Sem a correção, a previsão seria 0,403%.
- O otimismo da avaliação "fora do fold" é desprezível: log loss de 0,01673 contra 0,01667.

**Escolha do algoritmo**
- O LightGBM descartado tem overfitting forte: PR-AUC de 0,148 no próprio treino contra 0,028 na validação. A Regressão Logística não tem: 0,029 contra 0,037.
- A rejeição do LightGBM também se mantém na origem móvel e nas partições por UPA.

**Reprodutibilidade**
- O notebook 04 reproduz exatamente a rodada que está no README.
- 15 de 16 números citados no README e no Data Card batem; a exceção é o B5.
- As versões instaladas conferem com o `requirements.txt`. Não há caminhos absolutos no código. As sementes estão fixadas. Os 62 testes passam.

## 3. Inventário dos modelos

Detalhes, fluxo e etapas soltas em [outputs/00_inventario.md](outputs/00_inventario.md).

| ID | Modelo | Algoritmo | Onde | Features | Validação | Métricas reportadas | Status |
|---|---|---|---|---|---|---|---|
| M1 | Baseline dia × período | taxa ponderada em 28 células | `src/models/modelos.py` | período × dia | CV 5 × 10 por domicílio | 0,0160; captura 54,2% | ativo (referência) |
| M2 | Regressão Logística de referência | LR L2, C = 0,03 | `modelos.py` | período, dia, período × dia, região, faixa etária, sexo, escolaridade, moradores, log renda, urbano | CV | 0,0386; 67,2% | descartado |
| **M3** | **Modelo final** | LR L2, C = 0,003, sigmoide (a = 2,50, b = 8,23); ajuste final com C = 0,0024 | `src/models/treinar.py`, `models/` | como M2, com fim de semana no lugar do dia | CV; teste pendente | 0,0371; 66,9% | **ativo, congelado** |
| M4 | LR com renda em faixas | LR L2, C = 0,1 | notebook 04 | M3 com faixas de renda | CV | 0,0306; 67,4% | descartado |
| M5 | LR com peso no ajuste | LR L2 ponderada, C = 0,003 | notebook 04 | as de M3 | CV | 0,0414; 65,3% | descartado (regra) |
| M6 | LightGBM raso (6 configurações) | 300 árvores, 4/8/16 folhas × 300/3.000 linhas por folha | `modelos.py` | as 9 brutas | CV | melhor 0,0284; 63,3% | descartado |
| M7 | Triagem: LightGBM | 300 árvores, 15 folhas | notebook 02 | 9 + candidatas | CV | base 0,0259 | auxiliar |
| M8 | Triagem: LR simples | LR C = 0,05 | notebook 02 | 9 + candidatas | CV | base 0,0356 | auxiliar |
| M9 | Triagem exploratória de 16/11 | LightGBM e LR | fora do repositório | 9 + 7 grupos | **teste** (noites) | ROC-AUC 0,814 → 0,829 | substituída |
| M10 | Triagem com partição única | LightGBM e LR | versão intermediária do notebook 02 | idem | 1 partição | — | substituída (instável) |

## 4. Métricas reportadas × reproduzidas × corrigidas

Tudo no treino. Nenhum vazamento foi confirmado, então não há uma versão "sem vazamento" diferente da reproduzida. A coluna da direita traz a métrica corrigida para comparação com o teste e as checagens de robustez.

| Métrica | Baseline reportado | Baseline reproduzido | Final reportado | Final reproduzido | Corrigida / robustez |
|---|---|---|---|---|---|
| PR-AUC, média dos 50 folds | 0,0160 | 0,0160 | 0,0371 | 0,0371 | — |
| **PR-AUC agregada (fora da amostra, média das 10 repetições)** | — | **0,0143** | — | **0,0276** | razão 1,92 (não 2,31); é a métrica comparável à do teste |
| Vezes a prevalência (0,003025) | 5,3 | 4,7 (agregada) | 12,3 | 9,1 (agregada) | — |
| Captura no top 10% | 54,2% | 54,2% (53,6% agregada) | 66,9% | 66,9% (66,9% agregada) | estável |
| PR-AUC com partições por UPA | — | 0,0163 | — | 0,0359 | sem vazamento por vizinhança |
| PR-AUC sem peso | — | 0,0116 | — | 0,0248 | mesma conclusão |
| Origem móvel: out–nov / dez–jan / fev | — | 0,0124 / 0,0128 / 0,0250 | — | 0,0291 / 0,0151 / 0,0382 | ganho significativo só em out–nov |
| Captura na origem móvel | — | 45% / 55% / 72% | — | 62% / 42% / 74% | dez–jan abaixo do baseline |
| Calibração: média prevista × observada | — | — | 0,304% × 0,302% | 0,302% × 0,302% (aninhada) | sem C × 4/5, 0,403% |
| Log loss ponderado | — | — | 0,01667 | 0,01673 (aninhada) | otimismo desprezível |
| LightGBM (M6): média dos folds / agregada | — | — | 0,0284 | 0,0284 / 0,0231 | overfitting: 0,148 no próprio treino |
| Triagem (M7/M8): base e ganhos dos grupos | Data Card, seção 3 | idênticos (seção 4.1) | — | — | — |

### 4.1 Reprodução da triagem (notebook 02)

A reexecução completa ([outputs/repro_nb02.log](outputs/repro_nb02.log)) bate em todos os números com o Data Card (`docs/data_card.md:46-50`). O notebook 04 também foi reexecutado e reproduz a rodada citada no README linha a linha ([outputs/repro_nb04.log](outputs/repro_nb04.log)).

| Grupo | LightGBM: ganho (p) | LR: ganho (p) | Data Card |
|---|---|---|---|
| Base (9 features) | 0,0259 | 0,0356 | igual |
| Tipo de área no lugar de urbano | −0,0027 (0,83) | −0,0029 (0,77) | igual |
| Classe ABEP | +0,0005 (0,43) | +0,0014 (0,40) | igual |
| Composição da família | −0,0004 (0,57) | −0,0026 (0,81) | igual |
| Bens, finanças e moradia | +0,0080 (0,13) | +0,0013 (0,45) | igual |
| Trabalho e estudo | +0,0003 (0,42) | −0,0003 (0,63) | igual |

Os tempos de execução dobraram na auditoria (18 e 19 minutos, contra os "cerca de 10" e "cerca de 9" dos cabeçalhos) porque os scripts rodaram em paralelo, disputando o processador.

## 5. Plano de correção priorizado

Itens 1 a 4 antes de abrir o teste. Nenhum deles muda o modelo congelado.

| # | Achado | O que fazer | Esforço |
|---|---|---|---|
| 1 | A1 | README (Resultados e Avaliação no teste): reportar a PR-AUC agregada (final 0,0276 contra 0,0143 do baseline), destacar a captura e declarar que o teste será comparado com a PR-AUC agregada | 30 min |
| 2 | A2 | README: incluir a checagem com origem móvel e dizer que o ganho variou entre janelas; no notebook do teste, mostrar o resultado por mês | 30–45 min |
| 3 | M1 | README e Data Card: descrever com precisão a leitura exploratória do teste em 16/11 e o que foi revertido; corrigir "aberto uma única vez" | 15 min |
| 4 | M2 | README: pré-declarar a Regressão Logística com dia da semana como comparação exploratória no teste, sem trocar o modelo | 15 min + 30 min no notebook do teste |
| 5 | M3, M4, M5 | Limitações: vieses de calibração por segmento, tamanho efetivo de 138 positivos, janeiro e eventos do período de teste | 30 min |
| 6 | M6 | Generalização por região (opcional, segunda-feira) ou marcá-la como não feita; ajustar `data/external` e as fontes "A preencher" | 1–2 h ou 10 min |
| 7 | B1–B10 | Ajustes de texto, fixar `joblib` e `numpy`, nota sobre o hash | 30 min |

Total: cerca de 2h30, ou 4h com a generalização por região.

## 6. Itens não verificáveis

| Item | Por quê | O que seria preciso |
|---|---|---|
| Desempenho no período de teste | Não medido de propósito: avaliação única do protocolo | A avaliação planejada (notebook 05) |
| Números exatos da triagem de 16/11 | Código e saídas só existem no log da sessão, fora do repositório | Versionar o rascunho ou aceitar o registro do Data Card (`a4a9372`) e do log |
| Feriados e eventos por data | A POF pública não traz a data do recordatório, só o dia da semana e o mês estimado | Microdados com a data da entrevista (acesso restrito do IBGE) |
| Mês exato da entrevista | Reconstruído por regras; a validação só cobre os inquilinos | A data real da entrevista |
| Canal delivery | A POF registra "em casa" sem separar delivery | Outra fonte, como pesquisa de delivery ou dados de plataforma |
| Preço, promoção e temperatura | Não existem na POF | Fontes externas por UF e mês, como Open-Meteo e preços |
| Viés de não resposta no módulo de consumo | 13,6% dos moradores elegíveis sem recordatório; depende da calibração do `PESO_FINAL` pelo IBGE | Documentação metodológica detalhada dos pesos |
| Causa do SHA-256 diferente | A previsão é idêntica; o motivo exato dentro do pickle não foi investigado | Comparar o conteúdo dos dois arquivos objeto a objeto |

## 7. Situação após as correções (20/11/2026)

As correções foram feitas depois deste relatório e antes de abrir o teste. O modelo congelado (`models/`, `CONFIG` e o código que o gera) não mudou. As referências de arquivo e linha das seções 2 a 6 valem para o commit auditado (`46c0a6e`); o README e o Data Card mudaram depois.

**De onde vêm os números novos.** Os números que entraram no README e no Data Card vêm de um notebook novo do projeto, [notebooks/05_robustez.py](../notebooks/05_robustez.py), só com o treino. Ele reproduz exatamente os resultados desta auditoria, com as mesmas sementes ([outputs/notebook05.log](outputs/notebook05.log)), e acrescenta a generalização por região.

**Erratas deste relatório:**
- **Prevalência:** com a prevalência exata (0,003025), a PR-AUC agregada é 9,1 vezes a prevalência no modelo final e 4,7 no baseline. A primeira versão dizia 9,2 e 4,8. A tabela da seção 4 e o achado A1 já estão corrigidos.
- **Classe C2:** a razão observado ÷ previsto é 1,3035, acima do limite de 30%. A primeira versão do notebook 05 arredondava antes de comparar com o limite e não marcava a C2; isso foi corrigido.

| ID | Situação | O que foi feito |
|---|---|---|
| A1 | Corrigido | README: PR-AUC agregada ao lado da média dos folds (0,0276 contra 0,0143; 1,9 vezes), com a causa e a captura agregada. Adendo de 20/11 em Metodologia: a referência para o teste é a PR-AUC agregada |
| A2 | Corrigido | README, Resultados: subseção Robustez com a tabela da origem móvel. Limitações: ganho instável no tempo. Adendo: resultado do teste por mês, só descritivo |
| M1 | Corrigido | README (Escolha das features, adendo e Limitações) e Data Card (seção 3) descrevem com precisão a leitura exploratória de 16/11 e o que foi revertido. O texto provisório do teste deixou de dizer que o teste é "aberto uma única vez" |
| M2 | Corrigido, sem trocar o modelo | Adendo: a Regressão Logística com dia da semana fica pré-declarada como comparação exploratória no teste, com calibração e métricas especificadas. Limitações: o viés de quinta (−34%) e sexta (+52%) e a observação da EDA sobre a quinta |
| M3 | Documentado | README: vieses de calibração por segmento (Robustez) e ressalva de leitura do score (Limitações) |
| M4 | Documentado | README: tamanho efetivo de 138 positivos, faixa da PR-AUC entre folds e métricas sem peso |
| M5 | Documentado | README (Robustez e Limitações) e Data Card (seção 8): janeiro, Copa, festas juninas, feriados não identificáveis |
| M6 | Corrigido | Generalização por região feita (notebook 05, seção 3): o modelo supera o baseline nas 5 regiões, com intervalo acima de zero. `data/external` e as fontes do Data Card agora dizem "ainda não usado" ou "ainda não integrado" |
| B1 | Documentado | Data Card, seção 2: bebidas no limite e impacto (523 → 521 positivos) |
| B2 | Documentado | Data Card, seção 5: 1.044 UPAs dos dois lados do corte; PR-AUC por UPA de 0,0359 contra 0,0371 |
| B3 | Sem ação | Já estava documentado no Data Card |
| B4 | Sem ação | Já estava divulgado no README |
| B5 | Corrigido | README: "92% dos casos (481 de 523; 94% com o peso)" |
| B6 | Corrigido | README: Dados e Metodologia dizem que o modelo final usa o fim de semana derivado do dia |
| B7 | Corrigido | `requirements.txt`: `numpy==2.5.4` e `joblib==1.6.0`, as versões do ambiente |
| B8 | Documentado | README, Modelo final: o hash confere integridade, não reprodução byte a byte |
| B9 | Documentado | Data Card, seção 3: especificação dos modelos da triagem |
| B10 | Sem ação, de propósito | O one-hot da triagem não vaza rótulo e não muda nenhuma previsão. Alterar o notebook 02 só arriscaria mexer na tabela documentada, que se reproduz exatamente |
