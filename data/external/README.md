# Dados externos

Estes são os dados externos do Mapa de Oportunidade ([docs/oportunidade.md](../../docs/oportunidade.md)). Eles não entram no modelo.

Os arquivos vão para o git porque o OpenStreetMap muda todo dia: uma nova coleta daria números um pouco diferentes. O retrato abaixo é o usado no projeto. Todas as datas e instantes estão em UTC.

Para coletar de novo:

```
python -m src.oportunidade.coleta
```

O código está em [src/oportunidade/coleta.py](../../src/oportunidade/coleta.py). Os testes de coesão estão em [tests/test_oportunidade.py](../../tests/test_oportunidade.py).

## Arquivos

| Arquivo | Conteúdo | Fonte |
|---|---|---|
| `populacao_uf_censo2022.csv` | População residente por UF | IBGE, Censo Demográfico 2022, [SIDRA tabela 4709](https://sidra.ibge.gov.br/tabela/4709), variável 93 |
| `osm_pizzarias_uf.csv` | Pizzarias por UF, separadas por tipo de elemento e por critério | OpenStreetMap, API Overpass |
| `osm_pizzarias_elementos.csv` | Cada pizzaria, com UF, tipo, id do OpenStreetMap, `amenity`, `cuisine`, nome e critério. Serve para auditoria | OpenStreetMap, API Overpass |

## Pizzarias

Uma pizzaria é um elemento do OpenStreetMap que cumpre duas condições:
- `amenity` é `restaurant` ou `fast_food`;
- tem "pizza" em `cuisine` ou "pizz" no nome, sem diferenciar maiúsculas.

A consulta está em [docs/oportunidade.md](../../docs/oportunidade.md). O código reaplica a regra às etiquetas devolvidas e falha se algum elemento não a cumprir.

O critério de cada pizzaria indica qual condição ela cumpriu:

| Critério | Significado |
|---|---|
| `cozinha` | só a etiqueta de cozinha |
| `nome` | só o nome |
| `ambos` | as duas |

Resumo do retrato:

| Item | Valor |
|---|---|
| Coleta | 26/11/2026, entre 00h01 e 00h46 UTC (noite de 25/11 em Brasília) |
| Pizzarias | 5.708 nas 27 UFs |
| Tipo de elemento | 4.105 nós, 1.591 vias e 12 relações |
| Critério | 2.886 pelos dois, 2.012 só pela cozinha e 810 só pelo nome |
| Pizzarias por 100 mil habitantes, por UF | de 0,48 (AC) a 7,63 (DF); mediana 2,73 |

Duas observações sobre a regra:
- **Cozinha:** 650 das pizzarias que entram só pela cozinha listam pizza junto com outras cozinhas, por exemplo `regional;pizza`. Elas entram, como a regra define.
- **Nome:** a regra pelo nome não pegou falsos positivos na conferência manual. Os casos menos óbvios são grafias como "Pizzarira" e "Pizzéria" e marcas como "PizzaJack".

## Cuidados

- **Cobertura desigual.** A cobertura do OpenStreetMap varia entre estados. Poucas pizzarias mapeadas pode significar falta de mapeamento, e não falta de pizzarias. Por exemplo, Goiás tem 1,22 por 100 mil habitantes e o Piauí, 4,28.
- **Aplicativos de entrega** não entram na contagem.
- **Licença.** Os dados do OpenStreetMap são © colaboradores do OpenStreetMap e estão disponíveis sob a licença [ODbL](https://opendatacommons.org/licenses/odbl/). Os dois arquivos derivados do OpenStreetMap seguem a mesma licença.
