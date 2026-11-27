"""Textos, rótulos, cenários de exemplo e números mostrados no app.

O app é feito para quem decide marketing: as abas principais falam em "chance" e "prioridade"
e deixam o jargão de dados de fora ("classe alta" se confunde com classe social); a aba
"Detalhes técnicos" guarda o resto. Os números da avaliação no teste vêm de
reports/teste_tabelas.md (rodada única de 21/11/2026); tests/test_app.py confere que estes
valores batem com aquele arquivo.
"""

from __future__ import annotations

PERIODOS = {"manha": "Manhã", "almoco": "Almoço", "tarde": "Tarde", "noite": "Noite"}
HORARIOS = {"manha": "5h–10h", "almoco": "11h–14h", "tarde": "15h–17h", "noite": "18h–4h"}
NO_PERIODO = {"manha": "de manhã", "almoco": "no almoço", "tarde": "à tarde", "noite": "à noite"}
DIAS = {"Segunda-Feira": "Seg", "Terca-Feira": "Ter", "Quarta-Feira": "Qua",
        "Quinta-Feira": "Qui", "Sexta-Feira": "Sex", "Sabado": "Sáb", "Domingo": "Dom"}
DIAS_EXTENSO = {"Segunda-Feira": "Segunda", "Terca-Feira": "Terça", "Quarta-Feira": "Quarta",
                "Quinta-Feira": "Quinta", "Sexta-Feira": "Sexta", "Sabado": "Sábado",
                "Domingo": "Domingo"}
# O modelo só distingue segunda a quinta de sexta a domingo; cada bloco usa um dia dele
BLOCOS_DE_DIAS = {"Segunda a quinta": "Quarta-Feira", "Sexta a domingo": "Sabado"}
FIM_DE_SEMANA = ("Sexta-Feira", "Sabado", "Domingo")
SEXOS = {"feminino": "Mulher", "masculino": "Homem"}
AREAS = {True: "Cidade", False: "Zona rural"}
CANAIS = {"nenhum": "Ainda não sei", "delivery": "Delivery", "varejo": "Supermercado",
          "estabelecimento": "Pizzaria ou bar"}
ASSOCIACOES = {"nenhuma": "Não sei", "baixa": "Pouco", "normal": "Sim"}
UFS = {"AC": "Acre", "AL": "Alagoas", "AP": "Amapá", "AM": "Amazonas", "BA": "Bahia",
       "CE": "Ceará", "DF": "Distrito Federal", "ES": "Espírito Santo", "GO": "Goiás",
       "MA": "Maranhão", "MT": "Mato Grosso", "MS": "Mato Grosso do Sul", "MG": "Minas Gerais",
       "PA": "Pará", "PB": "Paraíba", "PR": "Paraná", "PE": "Pernambuco", "PI": "Piauí",
       "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte", "RS": "Rio Grande do Sul",
       "RO": "Rondônia", "RR": "Roraima", "SC": "Santa Catarina", "SP": "São Paulo",
       "SE": "Sergipe", "TO": "Tocantins"}
# O modelo usa a região, não a UF: todas as UFs de uma região recebem a mesma chance
REGIOES = {"Norte": ("AC", "AM", "AP", "PA", "RO", "RR", "TO"),
           "Nordeste": ("AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"),
           "Sudeste": ("ES", "MG", "RJ", "SP"), "Sul": ("PR", "RS", "SC"),
           "Centro-Oeste": ("DF", "GO", "MS", "MT")}
REGIAO_DA_UF = {uf: regiao for regiao, ufs in REGIOES.items() for uf in ufs}
# Escolaridade em níveis; o valor é o número de anos de estudo enviado à API (fundamental
# de 9 anos). Os níveis incompletos usam a mediana ponderada dos anos do nível, no treino.
ESCOLARIDADES = {0: "Sem instrução", 5: "Fundamental incompleto", 9: "Fundamental completo",
                 10: "Médio incompleto", 12: "Médio completo", 14: "Superior incompleto",
                 16: "Superior completo"}
# Renda por pessoa nas faixas de salário mínimo do projeto (src/features/build_grid.py; o
# mínimo de jan/2018 era R$ 954). Pensar em salários mínimos dispensa corrigir a inflação: o
# valor enviado à API é a mediana ponderada da renda por pessoa dentro da faixa, no treino,
# em reais de jan/2018 (a base de preços da POF).
RENDAS = {"ate_0.5sm": ("Até ½ salário mínimo", 323.0),
          "0.5_1sm": ("½ a 1 salário mínimo", 704.0),
          "1_2sm": ("1 a 2 salários mínimos", 1324.0),
          "2_5sm": ("2 a 5 salários mínimos", 2642.0),
          "acima_5sm": ("Mais de 5 salários mínimos", 7355.0)}
SALARIO_MINIMO_2018 = 954.0

CAMPOS = ("periodo", "dia_semana", "uf", "urbano", "idade", "sexo", "anos_estudo",
          "n_moradores", "renda", "canal", "associacao_marca")


def corpo_api(valores: dict) -> dict:
    """Valores do formulário → corpo do POST /predict.

    A faixa de renda vira o valor típico dela; "nenhum"/"nenhuma" viram ausência.
    """
    return {"periodo": valores["periodo"], "dia_semana": valores["dia_semana"],
            "uf": valores["uf"], "urbano": bool(valores["urbano"]),
            "idade": int(valores["idade"]), "sexo": valores["sexo"],
            "anos_estudo": int(valores["anos_estudo"]),
            "n_moradores": int(valores["n_moradores"]),
            "renda_pc": float(RENDAS[valores["renda"]][1]),
            "canal": None if valores["canal"] == "nenhum" else valores["canal"],
            "associacao_marca": (None if valores["associacao_marca"] == "nenhuma"
                                 else valores["associacao_marca"])}


# Exemplos prontos: um para cada regra do Decision Engine (conferidos em tests/test_app.py)
CENARIOS = [
    {"nome": "Domingo à noite · SP · delivery", "regra": "R3",
     "valores": dict(periodo="noite", dia_semana="Domingo", uf="SP", urbano=True, idade=25,
                     sexo="feminino", anos_estudo=12, n_moradores=3, renda="2_5sm",
                     canal="delivery", associacao_marca="nenhuma")},
    {"nome": "Sábado à noite · RS · pizzaria", "regra": "R4",
     "valores": dict(periodo="noite", dia_semana="Sabado", uf="RS", urbano=True, idade=22,
                     sexo="masculino", anos_estudo=14, n_moradores=2, renda="2_5sm",
                     canal="estabelecimento", associacao_marca="nenhuma")},
    {"nome": "Sexta à noite · MG · supermercado", "regra": "R5",
     "valores": dict(periodo="noite", dia_semana="Sexta-Feira", uf="MG", urbano=True, idade=35,
                     sexo="feminino", anos_estudo=16, n_moradores=4, renda="2_5sm",
                     canal="varejo", associacao_marca="nenhuma")},
    {"nome": "Sexta à noite · BA · marca pouco ligada a pizza", "regra": "R2",
     "valores": dict(periodo="noite", dia_semana="Sexta-Feira", uf="BA", urbano=True, idade=30,
                     sexo="masculino", anos_estudo=10, n_moradores=4, renda="1_2sm",
                     canal="nenhum", associacao_marca="baixa")},
    {"nome": "Quarta à noite · PE", "regra": "R6",
     "valores": dict(periodo="noite", dia_semana="Quarta-Feira", uf="PE", urbano=True, idade=40,
                     sexo="feminino", anos_estudo=9, n_moradores=4, renda="0.5_1sm",
                     canal="nenhum", associacao_marca="nenhuma")},
    {"nome": "Terça no almoço · PI · zona rural", "regra": "R1",
     "valores": dict(periodo="almoco", dia_semana="Terca-Feira", uf="PI", urbano=False,
                     idade=50, sexo="masculino", anos_estudo=5, n_moradores=5,
                     renda="ate_0.5sm", canal="nenhum", associacao_marca="nenhuma")},
]

# --------------------------------------------------------------------------
# Para quem decide marketing
# --------------------------------------------------------------------------
# O caminho aceso na árvore de decisão, em palavras
CAMINHO_SIMPLES = {
    "R1": "a prioridade é baixa, então o momento não pede uma ação dedicada",
    "R2": "a prioridade é {prioridade} e o público liga pouco a marca a pizza, então a ação "
          "constrói essa associação antes de pensar em canal",
    "R3": "a prioridade é alta e o canal escolhido é o delivery",
    "R4": "a prioridade é alta e o canal escolhido é pizzaria ou bar",
    "R5": "a prioridade é alta e o canal é o supermercado ou ainda não foi escolhido, então a "
          "ação segue a casa, onde a ocasião mais acontece",
    "R6": "a prioridade é média, então a ação é de presença de marca, de baixo custo",
}
# Quando cada ação vale, na ordem da árvore
QUANDO_SIMPLES = {
    "R1": "a prioridade é baixa.",
    "R2": "a prioridade é alta ou média e o público liga pouco a marca a pizza.",
    "R3": "a prioridade é alta e o canal é o delivery.",
    "R4": "a prioridade é alta e o canal é pizzaria ou bar (inclui restaurantes).",
    "R5": "a prioridade é alta e o canal é o supermercado ou ainda não foi escolhido.",
    "R6": "a prioridade é média.",
}
COMO_FUNCIONA = [
    ("De onde vêm os dados", "Pesquisa do IBGE",
     "Na Pesquisa de Orçamentos Familiares 2017–2018, 46 mil pessoas contaram tudo o que "
     "comeram em um ou dois dias, com horário e local."),
    ("O que medimos", "Pizza com refrigerante",
     "A chance de uma pessoa comer pizza e tomar refrigerante no mesmo período do dia: manhã, "
     "almoço, tarde ou noite. A pesquisa registrou 838 dessas ocasiões."),
    ("Como ler a chance", "Prioridade",
     "Alta: os 10% de momentos com mais chance. Média: os 20% seguintes. Baixa: os outros 70%."),
    ("Como a ação é escolhida", "Regras claras",
     "A ação sai da prioridade, do canal que você escolher e de quanto o público já liga "
     "Guaraná a pizza. Não é uma previsão de vendas."),
]
NAO_FAZ = [
    "Não prevê vendas nem o resultado de uma campanha: mostra onde a ocasião costuma "
    "acontecer.",
    "Não mede a marca Guaraná Antarctica: a pesquisa registra o tipo de refrigerante (guaraná), "
    "não a marca.",
    "Usa dados de 2017–2018, antes da pandemia e do crescimento do delivery. A pesquisa também "
    "não separa a pizza pedida por delivery da pizza feita ou comprada para comer em casa.",
    "Não sabe por qual canal você vai agir nem o quanto o público liga Guaraná a pizza: essas "
    "respostas vêm de você (plano de mídia, pesquisa de marca).",
]
# "Dá para confiar?": os números da avaliação no teste, arredondados (conferidos no teste)
CONFIANCA = {"alta_ocasioes": "69%", "alta_momentos": "10%", "regra_simples": "60%",
             "baixa_ocasioes": "7%", "baixa_momentos": "69%", "acima_do_estimado": "27%"}
CUIDADOS = [
    "<b>Compare contextos entre si.</b> No período de teste, que teve Copa do Mundo e festas "
    "juninas, a chance real ficou 27% acima da estimada. O valor exato muda com a época; a "
    "ordem entre os contextos é o que mais vale.",
    "<b>Adolescentes de 10 a 17 anos:</b> a chance real deles foi cerca de 3 vezes a estimada. "
    "O modelo subestima esse público.",
    "<b>Norte e classes D e E:</b> o modelo põe menos ocasiões desses públicos em prioridade "
    "alta do que a regra simples (46% contra 67% no Norte; 30% contra 59% nas classes D e E), "
    "porque renda e escolaridade puxam a chance deles para baixo.",
    "<b>Renda alta:</b> o modelo valoriza demais a renda alta. Entre os alarmes falsos, há três "
    "vezes mais pessoas das classes A e B1 do que entre os acertos (15% contra 5%).",
    "<b>Manhã, almoço e tarde:</b> nesses períodos a chance estimada fica acima da real, "
    "sobretudo de sexta a domingo. Eles ficam quase sempre em prioridade baixa; para comparar "
    "momentos, a noite é a referência mais segura.",
]

# --------------------------------------------------------------------------
# Para a equipe técnica (aba "Detalhes técnicos")
# --------------------------------------------------------------------------
CAMINHO_DA_REGRA = {
    "R1": "classe baixa",
    "R2": "classe {classe} e associação à marca informada como baixa",
    "R3": "classe alta, associação não informada como baixa e canal delivery",
    "R4": "classe alta, associação não informada como baixa e canal estabelecimento",
    "R5": "classe alta, associação não informada como baixa e canal varejo ou não informado",
    "R6": "classe média e associação não informada como baixa",
}
FATOS = [
    ("Fonte", "POF 2017–2018",
     "Recordatório alimentar de 24h do IBGE: 46.164 pessoas contaram, item a item, o que comeram, "
     "a que horas e onde."),
    ("Unidade de análise", "Pessoa × dia × período",
     "Cada dia vira 4 períodos: manhã, almoço, tarde e noite. São 340.072 linhas."),
    ("Alvo", "Pizza + refrigerante no período",
     "838 ocasiões. No treino, o guaraná aparece em 28% delas, mas é raro demais para ser o "
     "alvo: a participação dele vira contexto do Decision Engine."),
    ("Validação", "Corte no tempo",
     "Escolhas no treino (jul/2017–fev/2018) com validação cruzada 5 × 10 por domicílio. Teste "
     "em mar–jul/2018, avaliado uma vez. Métrica principal: PR-AUC."),
]

# Avaliação única no teste (reports/teste_tabelas.md, seções 1 a 3)
AVALIACAO = {
    "pr_auc": {"modelo": "0,0265", "baseline": "0,0220", "ic_modelo": "0,0190 a 0,0384",
               "ic_baseline": "0,0138 a 0,0344", "cv": "0,0276 / 0,0143"},
    "diferenca_pr_auc": {"valor": "+0,0045", "ic": "-0,0029 a +0,0138", "positivas": "89,8%"},
    "captura_10": {"modelo": "67,8%", "baseline": "57,1%", "ic_modelo": "55,9% a 77,0%",
                   "ic_baseline": "47,0% a 65,8%", "cv": "66,9% / 53,6%"},
    "diferenca_captura": {"valor": "+10,8 p.p.", "ic": "-1,1 p.p. a +19,6 p.p."},
    "roc_auc": {"modelo": "0,890", "baseline": "0,860"},
    "recall_alta": {"modelo": "68,6%", "baseline": "60,0%"},
    "precisao_alta": {"modelo": "2,7%", "baseline": "2,3%"},
    "fracao_alta": {"modelo": "10,4%", "baseline": "10,7%"},
    "alta": {"peso": "10,4%", "ocasioes": "68,6%", "taxa": "2,69%", "vezes": "6,6×"},
    "baixa": {"peso": "69,3%", "ocasioes": "6,9%"},
    "calibracao": {"prevista": "0,320%", "observada": "0,406%", "razao": "1,27"},
}

ERROS = [
    "<b>Adolescentes de 10 a 17 anos:</b> a taxa observada foi 3,24 vezes a prevista, e só 44% "
    "das ocasiões deles ficaram em <code>alta</code> (66% no baseline).",
    "<b>Corte único de <code>alta</code>:</b> o modelo ordena melhor dentro do Norte e da classe "
    "DE, mas põe menos ocasiões deles em <code>alta</code> que o baseline (46% contra 67%; 30% "
    "contra 59%), porque renda e escolaridade puxam o score deles para baixo.",
    "<b>Renda alta:</b> os falsos positivos têm mais classes A e B1 que os acertos (15% contra "
    "5%). O modelo valoriza demais a renda alta.",
    "<b>Fora da noite:</b> das 26 ocasiões que ficaram em <code>baixa</code>, 94,5% (do peso) "
    "estavam fora da noite.",
]
LIMITACOES = [
    "Dados de 2017–2018, antes da pandemia e do crescimento do delivery; a POF não separa "
    "delivery de comida feita em casa.",
    "A POF registra a categoria (refrigerante de guaraná), não a marca Guaraná Antarctica.",
    "Poucas ocasiões efetivas: 138 no treino e 118 no teste, por causa da variação dos pesos.",
    "Fora da noite, o score fica acima do observado no treino (0,063% contra 0,023%): a "
    "regularização forte deixa o efeito do fim de semana, que só existe à noite, passar para o "
    "dia. São só 42 ocasiões de dia no treino (veja o README, Limitações).",
    "O score não é recalibrado: no teste, a taxa observada ficou 27% acima da prevista, junto "
    "com a alta da prevalência (Copa do Mundo e festas juninas caíram no teste).",
    "As regras do Decision Engine são hipóteses de negócio: nenhuma afirma efeito causal.",
]

# --------------------------------------------------------------------------
# Onde e para quem ativar (Mapa de Oportunidade, docs/oportunidade.md)
# --------------------------------------------------------------------------
FAIXAS_ETARIAS = {"18-24": "18 a 24 anos", "25-34": "25 a 34 anos", "35-44": "35 a 44 anos",
                  "45-54": "45 a 54 anos", "55-64": "55 a 64 anos", "65+": "65 anos ou mais"}
GENEROS = {"feminino": "Mulheres", "masculino": "Homens"}
BLOCOS = {"seg_qui": "Segunda a quinta", "sex_dom": "Sexta a domingo"}
# Ação do caminho da decisão quando o canal ainda não foi escolhido (R1, R5, R6)
ACAO_SEM_CANAL = {"R1": "sem ação dedicada", "R5": "combo para consumo em casa",
                  "R6": "presença de marca de baixo custo"}
# Canal de cada estado, com a regra em palavras simples (a ordem é a da regra)
CANAIS_DOS_ESTADOS = {
    "ativacao_delivery": ("Delivery", "Chance na metade de cima dos estados e muitas pizzarias "
                                      "por habitante"),
    "combo_consumo_em_casa": ("Combo para casa no varejo", "Chance na metade de cima, mas "
                                                           "poucas pizzarias por habitante"),
    "presenca_de_marca": ("Presença de marca de baixo custo", "Chance na metade de baixo dos "
                                                              "estados"),
}
FONTES_DO_MAPA = ("Consumo: pesquisa do IBGE (POF 2017–2018), adultos de 18 anos ou mais. "
                  "População: IBGE, Censo Demográfico 2022. Pizzarias: © colaboradores do "
                  "OpenStreetMap, licença ODbL, mapa de novembro de 2026.")


def cuidados_do_mapa(fora_estimado: str, fora_observado: str) -> list[str]:
    """Os limites de docs/oportunidade.md em palavras simples; os números vêm do próprio mapa."""
    return [
        "<b>Mostra onde a ocasião costuma acontecer, não quanto uma ação vai vender.</b> Nenhuma "
        "lista aqui é previsão de resultado de campanha.",
        "<b>Use como orientação de prioridade.</b> No teste, a estimativa não pôde ser declarada "
        "melhor que uma regra simples que só olha o dia e o horário; a ordem dos momentos vem "
        "sobretudo da noite e do fim de semana, que essa regra também enxerga.",
        f"<b>Fora da noite, a estimativa fica acima do que a pesquisa observou:</b> de manhã, no "
        f"almoço e à tarde somam {fora_estimado} das ocasiões estimadas, contra "
        f"{fora_observado} das observadas. Compare volumes principalmente entre momentos da "
        "noite.",
        "<b>O modelo enxerga regiões, não estados.</b> Dentro de uma mesma região, os estados só "
        "diferem pelo perfil da população (idade, renda, escolaridade, tamanho da família).",
        "<b>O mapa de pizzarias é colaborativo e tem cobertura desigual.</b> Poucas pizzarias "
        "mapeadas podem ser falta de mapeamento: Roraima, por exemplo, aparece com mais "
        "pizzarias por habitante que São Paulo. Aplicativos de entrega não entram na conta.",
        "<b>As datas não coincidem:</b> o consumo é de 2017–2018, a população de 2022 e o mapa "
        "de pizzarias de 2026.",
        "<b>Só adultos.</b> Anúncios segmentados por idade começam aos 18 anos, e publicidade "
        "para crianças e adolescentes tem restrições no Brasil.",
    ]
