# Data Card: tabela de modelagem do Occasion Score

Versão da tabela: a do modelo `occasion_score` 1.0.0 (treinado em 21/11/2026). Gerada por
`occasion.ingestao` e `occasion.modelagem.tabela` a partir dos microdados oficiais do IBGE.
A análise exploratória está em [`notebooks/eda_occasion_score.ipynb`](../notebooks/eda_occasion_score.ipynb).

## Resumo

| item | valor |
|---|---|
| Unidade de análise | pessoa × dia de registro × turno |
| Target | `alvo_pizza_refri`: pizza e refrigerante (genérico ou marca) com até 1 hora de diferença no turno |
| Tipo de target | ground truth de consumo declarado (diário alimentar da POF), binário |
| Instante de previsão | antes do turno, com perfil da pessoa, lugar e calendário |
| Fonte do target | POF 2008-09 e POF 2017-18, Consumo Alimentar Pessoal (IBGE) |
| Linhas | 267.612 (2008-09) e 340.072 (2017-18) |
| Pessoas | 34.003 (2008-09) e 46.164 (2017-18), de 10 anos ou mais |
| Positivos | 668 (2008-09) e 811 (2017-18) |
| Taxa ponderada | 0,39% (2008-09) e 0,33% (2017-18) dos turnos |
| Uso | treino em 2008-09 e teste em 2017-18 (validação temporal); o modelo do app é reajustado nas duas |

## Por que esta unidade e este target

Não há base pública recente que registre cada ocasião de pizza com Guaraná, e a Ambev não
cedeu dados (LGPD). A POF é a única fonte pública nacional que registra **o que cada pessoa
comeu e bebeu, em que dia da semana e a que horas**, com peso amostral para expandir à
população. Por isso o target é consumo real declarado, e não um proxy como buscas.

- **Pizza com refrigerante, não só com Guaraná.** O Guaraná sozinho tem poucas ocasiões na
  amostra (185 de 811 em 2017-18), o que não sustenta um modelo. O modelo estima a ocasião
  "pizza + refrigerante", que é onde o Guaraná compete; a fatia de Guaraná dentro dela (21%,
  IC95 15% a 28%, POF 2017-18) entra no Decision Engine.
- **Até 1 hora de diferença.** Escolhida contra o tipo de refeição declarado em 2017-18 como
  árbitro: "mesma hora" perdeu 23 de 812 pares e "mesmo turno" gerou 27 falsos pares; ±1 hora
  ficou com 12 perdidos e 11 falsos ([checagens de 16/11/2026](relatorios/checagens_tabela_modelagem.txt),
  bloco "comparacao com gabarito").
- **Turno.** Manhã (0h às 10h, inclui a madrugada), almoço (11h
  às 14h), tarde (15h às 17h) e noite (18h às 23h). A madrugada tem 2 registros de pizza e 1
  ocasião por edição. Cada pessoa-dia gera os quatro turnos,
  comendo ou não, para o denominador não depender do padrão de refeições.

## Dicionário de variáveis

### Chaves

| coluna | descrição |
|---|---|
| `edicao` | "2008-09" ou "2017-18" |
| `id_pessoa`, `id_domicilio`, `id_setor` | identificadores da POF (setor = unidade primária de amostragem, usada para agrupar a validação e o bootstrap) |
| `estrato` | estrato geográfico da amostra |
| `dia_registro` | 1 ou 2 (a POF registra dois dias não consecutivos por pessoa) |
| `turno` | manha, almoco, tarde, noite |

### Target e auxiliares (nunca features)

| coluna | descrição |
|---|---|
| `alvo_pizza_refri` | 1 se houve pizza e refrigerante com até 1 hora de diferença no turno (o par que cruza a fronteira conta no turno da pizza) |
| `teve_pizza` | houve pizza no turno |
| `teve_guarana` | houve pizza com guaraná no turno (mesma regra de 1 hora) |
| `fora_de_casa` | a pizza do turno foi consumida fora do domicílio (NA sem pizza) |
| `uf` | código da UF (usado para o Trends e os estratos) |
| `renda_sm` | renda mensal do domicílio em salários mínimos da data de referência (R$ 415 em jan/2009; R$ 954 em jan/2018), para classes nos relatórios e no engine |

### Pesos

| coluna | descrição |
|---|---|
| `peso_final` | peso amostral da POF (expansão para a população) |
| `fator_dia` | faz cada dia da semana valer 1/7 do peso; a amostra de 2017-18 tem só 3.862 pessoas-dia de sábado. Usado nas métricas, não no treino |

### Features

| coluna | tipo | no modelo final | descrição |
|---|---|---|---|
| `turno` | categoria | sim | turno do dia |
| `dia_semana` | categoria (1 = segunda ... 7 = domingo) | sim | dia em que o alimento foi consumido (dicionário do IBGE, as duas edições) |
| `regiao` | categoria | sim | Norte, Nordeste, Sudeste, Sul, Centro-Oeste |
| `tipo_area` | categoria | sim | capital, resto da região metropolitana, interior urbano ou rural, pelas tabelas de estratos do IBGE (`data/referencia/estratos_pof.csv`) |
| `faixa_idade` | categoria | sim | 10-19, 20-24, 25-29, 30-34, 35-39, 40-44, 45-59, 60+ |
| `sexo` | categoria (1 = homem, 2 = mulher) | sim | |
| `renda_pct` | número (0 a 100) | sim | percentil ponderado da renda do domicílio dentro da edição |
| `trends_pizza_rank_uf` | número | não (ablação) | posição da UF no interesse por "pizza" no Google Trends, na janela de campo de cada edição |
| `n_moradores` | número | não (ablação) | moradores do domicílio |
| `tem_crianca` | booleano | não (ablação) | domicílio com criança de menos de 12 anos (ECA, art. 2º) |

O porquê de cada feature e os resultados da ablação estão na seção 8 do notebook de EDA e no
[relatório de experimentos](relatorios/experimentos_occasion_score.md).

## Fontes e versões

| fonte | papel | versão | acesso |
|---|---|---|---|
| POF 2017-2018, IBGE | target e features | microdados de 13/07/2023 (`Dados_20230713.zip`) | FTP do IBGE, `Orcamentos_Familiares/Pesquisa_de_Orcamentos_Familiares_2017_2018/Microdados` |
| POF 2008-2009, IBGE | target e features | microdados de 09/10/2023 (`Dados_20231009.zip`) | FTP do IBGE, mesma estrutura |
| Tabelas de estratos da POF, IBGE | `tipo_area` | documentação das duas edições, transcrita em `data/referencia/estratos_pof.csv` | idem |
| Google Trends, termo "pizza" por UF | ablação (fora do modelo) | consultado em 17/11/2026, janelas 19/05/2008 a 31/05/2009 e 26/06/2017 a 09/07/2018; `data/referencia/trends_pizza_uf.csv` | pytrends |

Fontes do Mapa de Oportunidade (não entram na tabela de modelagem): Censo 2022 (IBGE, FTP),
CNPJ da Receita Federal (base de 11/2026), OpenStreetMap (base de 22/11/2026) e malha de UFs do
IBGE (API de malhas, qualidade mínima, consultada em 23/11/2026). Detalhes no
[relatório do mapa](relatorios/mapa_oportunidade.md).

## Regras de integração

- **As duas edições não são a mesma observação.** Cada uma gera a sua tabela com o mesmo
  código; elas só se juntam no treino do modelo do app, com o peso de cada linha dividido pela
  média da edição para as duas terem a mesma importância.
- **Alimentos.** Pizza, refrigerante genérico, refrigerante de marca e guaraná são marcados
  por códigos de alimento da POF (`occasion.ingestao.codigos`, com a descrição do Cadastro de
  Produtos do Consumo Alimentar do IBGE e as edições em que cada código existe; nenhum código
  muda de significado entre edições).
- **Renda.** Convertida em percentil dentro da edição. O salário mínimo subiu cerca de 35%
  em termos reais entre jan/2009 e jan/2018 (IPCA, série do IBGE), então salários mínimos não
  são comparáveis entre edições.
- **Google Trends.** Entra por UF e edição (chave `uf`), na janela de campo de cada POF, como
  posição (não como valor absoluto, que não é comparável entre consultas). Em 2008-09 o Trends
  devolve zero para AP, DF, RR, SE e TO (3,5% do peso), que ficam sem posição.
- **Censo, CNPJ e OSM** só se juntam no Mapa de Oportunidade, por código de município do IBGE,
  aplicando o modelo já treinado; não viram linhas da tabela de modelagem.

## Estratégia anti-leakage

- **Instante de previsão:** antes do turno. Toda feature é conhecida antes: perfil da pessoa,
  lugar e calendário (dia da semana e turno).
- **Fora do modelo:** tudo o que é consumido ou declarado durante ou depois do turno: itens,
  local da refeição, tipo de refeição, `dia_atipico` (só existe em 2017-18 e é declarado depois
  do registro) e dieta. `teve_pizza`, `teve_guarana` e `fora_de_casa` existem só para análise.
- **Validação temporal:** treino em 2008-09 e teste em 2017-18. Escolha de hiperparâmetros por
  validação cruzada agrupada por setor dentro de 2008-09; a escolha entre candidatos usa uma
  metade dos setores de 2017-18 e a outra metade só mede.
- **Calibração sem vazamento:** Platt ajustado em previsões fora da amostra de 2017-18
  (metades de setores), e só então aplicado ao modelo final.
- **Mesmo processamento no treino e na inferência:** a API e o app chamam o mesmo
  `OccasionScore` e o mesmo `Contexto` do Decision Engine; a renda em salários mínimos da
  entrada vira percentil pela distribuição de 2017-18 guardada no modelo.

## População coberta e excluída

- **Coberta:** moradores de 10 anos ou mais de domicílios particulares permanentes, em áreas
  urbanas e rurais de todas as UFs, que responderam ao bloco de Consumo Alimentar Pessoal da
  POF (subamostra), com peso para expandir à população.
- **Excluída:** menores de 10 anos (o bloco não é aplicado a eles), pessoas em domicílios
  coletivos e em situação de rua, e consumo fora dos dois dias de registro.
- **Período:** campo de maio/2008 a maio/2009 e de julho/2017 a julho/2018 (as janelas do
  Trends seguem essas datas). Não há edição
  mais recente com microdados: a POF 2024-25 está em campo desde novembro de 2024, e o IBGE
  anunciou os primeiros resultados para 27/11/2026, data da entrega, sem microdados.

## Vieses e limitações

- **Idade dos dados.** O dado mais recente é de 2017-18, antes da pandemia e do crescimento
  do delivery. A checagem com fontes recentes (Google Trends, divulgações do iFood de 2024 a
  2026, Vigitel) achou o mesmo padrão de noite, fim de semana e sábado no Sul e Sudeste
  (validação de 11/11/2026), mas o nível da taxa pode ter mudado.
- **Sem delivery.** A POF não tem código de entrega: pizza pedida conta como "em casa". O canal
  não é feature e entra só como contexto no app.
- **Sem data.** A POF registra dia da semana e hora, não a data. Clima, feriados e sazonalidade
  mensal não podem entrar no modelo.
- **Amostra pequena de positivos.** 811 ocasiões em 2017-18; segmentos finos (classe A, 26
  ocasiões; sábado, 69) têm intervalos largos. O relatório de avaliação marca os instáveis.
- **Diário declarado.** Esquecimento e subdeclaração de itens são possíveis, e o viés de
  memória não é mensurável com estes dados.
- **Calibração por região.** O modelo final, medido fora da amostra em 2017-18 (`vies.py`),
  subestima o Nordeste (previsto/observado 0,68, IC95 0,54 a 0,85) e superestima o
  Centro-Oeste (1,42, IC95 1,03 a 2,16); o Mapa de Oportunidade corrige essas duas regiões. Na
  validação temporal (treino só em 2008-09) as razões são 0,63 e 1,48.
- **Google Trends** mede interesse de busca, não consumo, e nunca é tratado como target.

## Licença e LGPD

- **Microdados do IBGE:** públicos e desidentificados, baixados do FTP oficial. O conteúdo do
  portal do IBGE é publicado sob Creative Commons Atribuição-SemDerivações 3.0, com exigência
  de citar a fonte; os microdados não trazem um termo próprio no pacote. Citação: "IBGE,
  Pesquisa de Orçamentos Familiares 2008-2009 e 2017-2018, microdados".
- **Google Trends:** dados agregados e normalizados pelo Google; uso sujeito aos termos do
  Google.
- **OpenStreetMap:** © colaboradores do OpenStreetMap, licença ODbL (a cópia versionada em
  `data/referencia/pizzarias_osm_br.csv` mantém essa licença).
- **CNPJ (Receita Federal):** dados abertos publicados pela Receita; o projeto usa só
  estabelecimentos (nome fantasia, CNAE, município, situação) e não guarda CPF de sócios.
  A licença específica do conjunto não foi confirmada nesta entrega.
- **LGPD:** nenhuma base tem dado pessoal identificável. A POF é desidentificada pelo IBGE; o
  repositório não versiona microdados (ficam em `data/raw/`, fora do git) e o app não recebe
  nem guarda dados de pessoas, só um contexto genérico (dia, turno, região, faixa de idade,
  sexo, renda em salários mínimos).
