# Model card: Occasion Score

Português · [English](model.en.md)

Ficha do modelo no formato de *model cards* (Mitchell et al., 2019). A base está descrita no [Data Card](data_card.md), e o sistema em volta do modelo, em [`architecture.md`](architecture.md).

## Detalhes

- **O que é**: probabilidade calibrada, de 0 a 100, de uma pessoa comer pizza com refrigerante num turno (manhã, almoço, tarde ou noite), dado o momento, o lugar e o perfil.
- **Alvo**: pizza e refrigerante, genérico ou de marca, com até 1 hora de diferença no mesmo turno, declarados no diário alimentar da POF. É consumo real, não um proxy.
- **Versão**: 1.1.0, treinada em 26/11/2026 em 607.684 pessoa-turnos com 1.479 ocasiões (POF 2008-09 e 2017-18). Arquivos: `models/occasion_score_v1.1.0.joblib` e `.json`.
- **Arquitetura**:
  - regressão logística com one-hot nas variáveis categóricas e splines no percentil de renda;
  - calibração de Platt em previsões fora da amostra de 2017-18 (metades de setores);
  - correção por grupo: um offset no logit calibrado para Nordeste (+0,48), Centro-Oeste (−0,31) e área rural (−0,60).
- **Features** (7): turno, dia da semana, região, tipo de área (capital, região metropolitana, interior urbano, rural), faixa de idade, sexo e percentil de renda do domicílio.
- **Faixas**: alta (os 10% maiores scores da população de 2017-18, a partir de 1,02%), média (10% a 30%, a partir de 0,098%) e baixa (o resto).
- **Explicação**: SHAP linear, exato para a logística, em logit calibrado. A contribuição de cada feature soma ao logit do score.
- **Incerteza**: IC95 por bootstrap de setores censitários, 200 réplicas que refazem todo o pipeline (logística, Platt e offsets).
- **Licença do código**: nenhuma declarada. Os microdados da POF e do Censo são públicos (IBGE).

## Uso pretendido

- **Para quê**: ordenar contextos (momento, lugar, público) pela chance da ocasião e alimentar o Decision Engine, que escolhe a categoria de ativação.
- **Para quem**: times de marca e trade do Guaraná, pelo simulador, e outros sistemas, pela API.
- **Melhor uso**: comparar contextos entre si. O nível absoluto é de 2017-18 e pode ter mudado desde então.

## Fora do escopo

- Prever vendas, volume ou o efeito de uma ativação. O score diz onde a ocasião acontece, não que uma ação vai criar consumo novo.
- Escolher entre marcas: o alvo é pizza com qualquer refrigerante. A presença do Guaraná entra no Engine pela tabela de associação.
- Datas específicas, clima, preço ou promoção: a POF não tem a data do consumo nem o preço.
- Separar delivery de comida feita em casa: a POF não separa.
- Pessoas com menos de 10 anos (fora do diário alimentar da POF).

## Fatores avaliados

- **Região** e **tipo de área**.
- **Classe de renda** (A a E, por salários mínimos do domicílio).
- **Faixa de idade** e **sexo**.
- **Dia da semana** e **turno**.
- **Edição da POF**, pela validação temporal.

## Métricas

- **Principal: PR-AUC.** A ocasião ocorre em 0,33% dos turnos. Com tantos negativos, o ROC-AUC fica alto mesmo para modelos fracos.
- **Ordenação**: ROC-AUC, lift e recall nos 10% maiores scores.
- **Probabilidade**: Brier, log loss e calibração (média prevista contra observada, por decil e por grupo).
- **Classificação**: precisão, recall e F1 no limiar escolhido em outra metade dos setores, e matriz de confusão.
- Todas ponderadas pelo peso da POF, com cada dia da semana valendo 1/7, e com IC95 por bootstrap pareado de setores.

## Dados de avaliação

- **Validação temporal**: treino na POF 2008-09 (267.612 linhas, 668 ocasiões), teste na POF 2017-18 (340.072 linhas, 811 ocasiões).
- **Escolha e medida separadas**: a escolha entre candidatos usa metade dos setores de 2017-18, e a outra metade só mede.

## Dados de treino

- **Modelo servido**: as duas edições juntas, com a calibração de Platt e a correção por grupo estimadas em previsões fora da amostra de 2017-18.
- **Unidade**: pessoa × dia de registro × turno, de 10 anos ou mais. Detalhes, pesos e regras de integração no [Data Card](data_card.md).

## Resultados

Validação temporal (treino 2008-09, teste 2017-18). Reproduzido por `occasion.modelagem.final.metricas_temporais`.

| Métrica | Baseline (dia × turno) | Logística final |
|---|---:|---:|
| PR-AUC | 0,0182 | **0,0288** |
| ROC-AUC | 0,880 | **0,899** |
| Lift nos 10% maiores | 5,8× | **6,9×** |
| Recall nos 10% maiores | 58% | **69%** |

- **Ganho de PR-AUC sobre o baseline**: +0,011 (IC95 +0,005 a +0,019).
- **Calibração**: média prevista 0,327% contra 0,327% observada; Brier 0,00321.
- **Importância** (queda da PR-AUC ao embaralhar): turno 80%, dia da semana 29%, renda 19%, faixa de idade 13%, região 5%.
- **Faixas**: a alta (10% da população) concentra 69% das ocasiões; a média (20%), 24%; a baixa (70%), 7%.
- **Correção por grupo**: fora da amostra, o score errava o nível do Nordeste (previsto ÷ observado 0,68), do Centro-Oeste (1,42) e da área rural (1,63). Com os offsets, medidos em setores que a correção não viu, as razões vão para 0,99, 1,01 e 1,04. O recall nos 10% maiores sobe de 70,1% para 71,7%, e a PR-AUC empata.
- **Incerteza**: em 2017-18, a meia largura média do IC95 é 48% do score na faixa alta, 62% na média e 97% na baixa. O aviso de faixa incerta aparece em 46% dos casos da faixa alta.

Relatórios: [experimentos](relatorios/experimentos_occasion_score.md), [candidatos](relatorios/candidatos_occasion_score.md), [avaliação](relatorios/avaliacao_occasion_score.md), [viés por grupo](relatorios/vies_por_grupo.md), [dados sintéticos](relatorios/dados_sinteticos.md).

## Considerações éticas

- **Dados pessoais**: a POF é pública e anonimizada pelo IBGE. Nenhum dado de cliente da Ambev foi usado.
- **Viés por grupo**: o modelo nacional subestimava o Nordeste e superestimava o Centro-Oeste e a área rural. A correção é aplicada e medida fora da amostra. A classe D está calibrada, mas tem poucas pessoas na faixa alta nacional, então a API mostra também a posição do score dentro da classe.
- **Público**: o score prioriza onde a ocasião já é frequente (noite, fim de semana) e superestima a classe A (previsto ÷ observado 1,92 na validação temporal), cuja taxa observada fica abaixo das classes B e C. Para não reforçar só o topo da régua, o Engine trata a faixa média das classes B a D como awareness.
- **Sem causalidade**: nenhuma recomendação afirma que uma ação vende mais. Mensagens e mecânicas vêm marcadas como hipóteses.

## Ressalvas e recomendações

- **Dado antigo**: a POF 2017-18 é anterior à pandemia e à expansão do delivery. A checagem com o Google Trends e com divulgações do iFood de 2024 a 2026 confirmou o padrão de noite e fim de semana, mas o nível pode ter mudado. A POF 2024-25 ainda não tem microdados.
- **Amostra pequena**: 811 ocasiões em 2017-18. Segmentos finos (classe A, sábado, rural) têm intervalos largos.
- **Modelo aditivo**: o melhor dia e turno é o mesmo para todo perfil (sábado à noite). Interações (turno × dia, LightGBM, AutoML) foram testadas e não ganharam fora do tempo.
- **Correção por grupo**: escolhida nos dados inteiros. Rodada em só metade dos setores, a regra de significância escolhe grupos diferentes, então os offsets são uma melhora medida, não exata.
- **Recomendação**: usar o score para ordenar e comparar contextos, sempre com o intervalo à vista, e retreinar quando a POF 2024-25 publicar microdados.
