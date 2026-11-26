# Mapa de Oportunidade

Esta regra foi escrita em 25/11/2026, a partir do commit `8e0d07c`. Ela veio antes de três coisas: o cálculo do ranking, as chances por público e por UF, e a coleta das pizzarias.

Antes de escrevê-la, eu já conhecia os números da análise exploratória, da avaliação no teste e do contexto do Decision Engine ([decision_engine.md](decision_engine.md)).

As definições abaixo não mudam depois de ver os resultados. Se algo precisar mudar por erro, a mudança entra num adendo no fim deste arquivo, com o motivo.

## O que é

É o diferencial "Mapa de Oportunidade" da seção 9 do case. Ele prioriza públicos, momentos e territórios cruzando o Occasion Score com a densidade de pizzarias e o perfil da população.

O case define que "a priorização é uma regra documentada do Decision Engine aplicada ao território, e não uma nova previsão". Por isso:
- **Nada é treinado.** O ranking usa o modelo congelado, e o script confere o hash do arquivo contra `model_file_sha256` em [models/model_metadata.json](../models/model_metadata.json) antes de rodar.
- **O teste não é usado.** Só entram linhas do treino (jul/2017 a fev/2018).
- **A categoria de ativação** sai das mesmas regras e dos mesmos cortes do Decision Engine.

O mapa tem duas partes:
1. **Públicos e momentos:** para quem e quando anunciar.
2. **Estados:** onde e por qual canal.

## Parte 1: públicos e momentos

### Unidade

A unidade é o público-momento: faixa etária × sexo × região × momento.

| Dimensão | Valores | Por quê |
|---|---|---|
| Faixa etária | 18–24, 25–34, 35–44, 45–54, 55–64 e 65+ (idade da POF) | São as faixas de idade do Google Ads |
| Sexo | feminino, masculino | Como na POF |
| Região | as 5 regiões | É a resolução geográfica do modelo |
| Momento | período × bloco de dias: 8 momentos | São as divisões que o modelo usa |

- **Períodos:** manhã (5h–10h), almoço (11h–14h), tarde (15h–17h) e noite (18h–4h).
- **Blocos de dias:** segunda a quinta, e sexta a domingo.
- **Total:** 6 × 2 × 5 × 8 = 480 públicos-momento.

**Menores de 18 ficam de fora.** As faixas de idade das plataformas de anúncio começam em 18 anos, e a publicidade dirigida a crianças e adolescentes tem restrições no Brasil.

**Renda e escolaridade não viram dimensões.** Elas multiplicariam os públicos por 5 ou mais e deixariam a maioria com amostra pequena. A renda aparece só como descrição do público: a faixa de salário mínimo mais comum entre as pessoas dele.

### Medidas

Cada medida usa o score calibrado do modelo em cada linha do treino de pessoas com 18 anos ou mais. A ponderação é por `peso_semana`, o peso da POF equilibrado por dia da semana, o mesmo usado no ajuste do modelo.

| Medida | Definição |
|---|---|
| Chance | Média ponderada do score no público-momento |
| Comparação | Chance ÷ chance média dos adultos |
| Parcela das ocasiões | Σ(peso × score) do público-momento ÷ Σ(peso × score) de todos os adultos |
| Prioridade | Cortes do Decision Engine aplicados à chance: alta a partir de 0,7765%, média a partir de 0,1436% |
| Categoria | Regra do Decision Engine sem canal e sem associação informados: R1, R5 ou R6 |
| Conferência | Pessoas na amostra, ocasiões observadas e taxa observada |

- **Por que a parcela das ocasiões:** ela mostra onde está o volume. Um público grande com chance média pode reunir mais ocasiões que um público pequeno com chance alta.
- **A conferência fica só no arquivo:** ela não ordena nada.

**Amostra mínima.** Um público (faixa × sexo × região) só entra no ranking se tiver pelo menos 100 pessoas na amostra do treino. Abaixo disso, ele aparece como "amostra pequena", sem posição.

### Ordem

São duas ordens, mostradas lado a lado: por chance e por parcela das ocasiões. Elas não são combinadas numa nota única, porque escolher entre eficiência e alcance é decisão de quem planeja a campanha.

### Tradução para anúncio

Para cada público-momento, os parâmetros que as plataformas de anúncio já aceitam:

| Dimensão do público-momento | Parâmetro de anúncio |
|---|---|
| Idade e sexo | Faixa etária e gênero |
| Região | As UFs da região, como localização |
| Momento | Programação de horário nos dias do bloco |

**A noite segue a definição da grade:** das 18h às 24h e da 0h às 5h do mesmo dia. Isso porque, na POF, as horas de 0 a 4 entram na noite do dia em que foram registradas.

**São parâmetros, não um plano de mídia.** Ficam de fora orçamento, lances, escolha de veículo e previsão de resultado.

## Parte 2: estados

### Unidade

A unidade é a UF. A POF é desenhada para estimar por UF, não por cidade, então as cidades ficam de fora.

### Medidas

**Chance da UF.** É a média ponderada do score nas linhas do treino de adultos da UF, em todos os momentos.

O modelo usa a região, não a UF. Por isso, duas UFs da mesma região só diferem pelo perfil da população: idade, sexo, escolaridade, renda, tamanho da família e área urbana.

**Parcela das ocasiões da UF.** Mesma definição da parte 1.

**Pizzarias por 100 mil habitantes.** Usa duas fontes:

- **Pizzarias (OpenStreetMap):**
  - **O que conta:** elementos (nós, vias e relações) dentro da UF que cumprem duas condições:
    - `amenity` igual a `restaurant` ou `fast_food`;
    - "pizza" em `cuisine` ou "pizz" no nome, sem diferenciar maiúsculas.
  - Cada elemento conta uma vez.
  - **Como é coletado:** pela API Overpass, uma consulta por UF, com a data da coleta gravada:

    ```
    [out:json][timeout:300];
    area["ISO3166-2"="BR-SP"]["admin_level"="4"]->.uf;
    (
      nwr(area.uf)["amenity"~"^(restaurant|fast_food)$"]["cuisine"~"pizza",i];
      nwr(area.uf)["amenity"~"^(restaurant|fast_food)$"]["name"~"pizz",i];
    );
    out tags;
    ```

    A primeira versão deste arquivo terminava a consulta com `out ids;`. A coleta usa `out tags;`, que devolve os mesmos elementos e acrescenta as etiquetas, para auditoria. A troca foi feita na coleta, antes de calcular qualquer chance ou ranking.
  - **Código:** [src/oportunidade/coleta.py](../src/oportunidade/coleta.py). Ele reaplica a regra às etiquetas devolvidas e falha se algum elemento não a cumprir.
- **Habitantes (Censo 2022, IBGE):** população residente por UF, com a tabela do SIDRA e a data da coleta registradas.
- **Arquivos:** `data/external/osm_pizzarias_uf.csv`, `data/external/osm_pizzarias_elementos.csv` (cada pizzaria, para auditoria) e `data/external/populacao_uf_censo2022.csv`. Os três vão para o git, porque o OpenStreetMap muda todo dia e uma nova coleta daria números diferentes. A descrição está em [data/external/README.md](../data/external/README.md).

### Regra do canal

A regra compara cada UF com a mediana das 27 UFs. Ficar igual ou acima da mediana conta como "acima".

| Chance | Pizzarias por habitante | Categoria | Por quê |
|---|---|---|---|
| Acima | Acima | `ativacao_delivery` | Chance alta e oferta de pizzarias para apoiar o delivery |
| Acima | Abaixo | `combo_consumo_em_casa` | Chance alta, mas pouca oferta de pizzarias: o caminho é o combo para casa no varejo |
| Abaixo | Qualquer | `presenca_de_marca` | Chance abaixo da mediana: presença de baixo custo, depois dos estados acima na prioridade |

Sobre a primeira linha: a densidade de pizzarias não separa delivery de salão. A regra escolhe o delivery porque a casa, o que inclui o delivery, concentra as ocasiões em todas as regiões ([decision_engine.md](decision_engine.md)).

### Ordem

Primeiro vêm as UFs com chance acima da mediana, depois as outras. Dentro de cada grupo, a ordem é pela parcela das ocasiões.

## Limites

Estes limites também vão para o app, em palavras simples, e para o README.

- **Não mede efeito causal.** O mapa diz onde a ocasião costuma acontecer, não quanto uma ação vende.
- **O ranking orienta prioridade, mas não garante resultado.** No teste, não deu para afirmar que o modelo supera o baseline, porque o intervalo da diferença inclui zero. A ordem dos momentos vem sobretudo do período e do bloco de dias, que o baseline também usa.
- **Os scores são calculados dentro da amostra.** São do próprio treino. A regularização forte do modelo limita o otimismo dessa reutilização, mas não o elimina.
- **As datas das fontes não coincidem.** A POF é de 2017–18, o OpenStreetMap é de 2026 e a população é de 2022.
- **A cobertura do OpenStreetMap varia entre estados.** Poucas pizzarias mapeadas pode ser falta de mapeamento, e não falta de pizzarias. Os aplicativos de entrega não entram na contagem.
- **Pizzaria por habitante mede oferta.** Não mede demanda nem uso de delivery.
- **O alvo é pizza com refrigerante, não Guaraná.** É o mesmo limite do modelo principal.

## Conferências

Escritas agora e rodadas pelo script; o script falha se alguma não passar.

1. Só entram linhas com `split == "treino"` e idade ≥ 18.
2. O hash do modelo bate com o de `models/model_metadata.json`.
3. As parcelas somam 100%, tanto nos públicos-momento quanto nas UFs.
4. A média das chances dos públicos-momento, ponderada pelo peso de cada um, reproduz a chance média dos adultos.
5. São 27 UFs, todas com contagem de pizzarias e população, e nenhuma com zero habitantes.
6. Rodar o script duas vezes gera o mesmo arquivo.
7. Cada UF cai na categoria da tabela da regra do canal.

Dois números só são reportados, sem mudar nada:
- a correlação de Spearman entre chance e pizzarias por habitante, entre as UFs;
- a correlação entre chance e taxa observada, nos públicos-momento com pelo menos 10 ocasiões observadas.

## Saídas

- **Cálculo:** `src/oportunidade/` gera `models/oportunidade.json`, com os dois rankings, as fontes, as datas de coleta e o hash do modelo.
- **API:** `GET /oportunidade` devolve esse arquivo.
- **App:** a aba "Onde e para quem ativar".
  - Usa a linguagem de quem decide marketing e traz os limites acima em palavras simples.
  - Mostra a atribuição "© colaboradores do OpenStreetMap", exigida pela licença ODbL.
- **Testes:** `tests/test_oportunidade.py`, além do teste de jargão da aba.

## Fora do escopo

- Cidades, porque a POF não estima por cidade.
- Influenciadores.
- Orçamento e lances de mídia.
- Previsão de resultado de campanha.
- Renda e escolaridade como dimensões do ranking.
