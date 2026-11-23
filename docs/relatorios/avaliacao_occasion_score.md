# Occasion Score: interpretação, faixas, segmentos e erros

Validação temporal: treino POF 2008-09, teste POF 2017-18 (340,072 pessoa-turnos, 811 ocasiões). Pesos: `peso_final × fator_dia`.

## Importância das features

Queda da PR-AUC em 2017-18 ao embaralhar a feature (20 permutações), ao lado da média do |SHAP| no modelo final.

| feature | queda_pr_auc | queda_pct | dp | shap_medio_abs_logit |
|---|---|---|---|---|
| turno | 0.0231 | 79.7985 | 0.0006 | 1.3068 |
| dia_semana | 0.0083 | 28.7753 | 0.0014 | 0.4181 |
| renda_pct | 0.0055 | 19.1335 | 0.0018 | 0.4218 |
| faixa_idade | 0.0037 | 12.7614 | 0.0024 | 0.3890 |
| regiao | 0.0015 | 5.0364 | 0.0018 | 0.1609 |
| tipo_area | 0.0001 | 0.4533 | 0.0052 | 0.3062 |
| sexo | -0.0003 | -0.8981 | 0.0001 | 0.0544 |

## Faixas do score (alta = 10% maiores do peso, média = 10% a 30%)

Cortes na validação temporal: alta ≥ 0.880%, média ≥ 0.158%.

### Modelo

| faixa | populacao_pct | ocasioes_pct | taxa_obs_pct | score_medio_pct | positivos |
|---|---|---|---|---|---|
| alta | 10.001 | 69.081 | 2.257 | 2.185 | 478 |
| media | 20.000 | 24.338 | 0.398 | 0.385 | 263 |
| baixa | 70.000 | 6.581 | 0.031 | 0.045 | 70 |

### Baseline (mesma regra de faixa)

O baseline tem poucos valores distintos: o bloco de empates que cruza o corte entra inteiro, e a faixa alta dele cobre 10.7% do peso, não 10%.

| faixa | populacao_pct | ocasioes_pct | taxa_obs_pct | score_medio_pct | positivos |
|---|---|---|---|---|---|
| alta | 10.714 | 60.915 | 1.858 | 1.875 | 424 |
| media | 19.660 | 32.534 | 0.541 | 0.516 | 324 |
| baixa | 69.626 | 6.551 | 0.031 | 0.036 | 63 |

## Matriz de confusão com a faixa alta como positivo (% da população)

| real | previsto 1 | previsto 0 |
|---|---|---|
| real 1 | 0.226 | 0.101 |
| real 0 | 9.775 | 89.898 |

## Segmentos

`previsto_sobre_obs` acima de 1 = o modelo superestima o segmento.

| segmento | valor | positivos | taxa_obs_pct | score_medio_pct | previsto_sobre_obs | pr_auc_modelo | pr_auc_baseline |
|---|---|---|---|---|---|---|---|
| turno | almoco | 17 | 0.035 | 0.116 | 3.298 | 0.007 | 0.000 |
| turno | manha | 16 | 0.021 | 0.040 | 1.904 | 0.000 | 0.001 |
| turno | noite | 747 | 1.220 | 1.095 | 0.898 | 0.031 | 0.019 |
| turno | tarde | 31 | 0.031 | 0.057 | 1.828 | 0.000 | 0.000 |
| dia_semana | 1 | 85 | 0.209 | 0.176 | 0.841 | 0.031 | 0.007 |
| dia_semana | 2 | 83 | 0.173 | 0.202 | 1.166 | 0.017 | 0.006 |
| dia_semana | 3 | 80 | 0.184 | 0.202 | 1.099 | 0.025 | 0.006 |
| dia_semana | 4 | 114 | 0.262 | 0.175 | 0.666 | 0.017 | 0.009 |
| dia_semana | 5 | 85 | 0.364 | 0.352 | 0.968 | 0.016 | 0.012 |
| dia_semana | 6 | 69 | 0.614 | 0.704 | 1.147 | 0.048 | 0.023 |
| dia_semana | 7 | 295 | 0.482 | 0.479 | 0.994 | 0.032 | 0.017 |
| regiao | Centro-Oeste | 114 | 0.372 | 0.551 | 1.481 | 0.028 | 0.022 |
| regiao | Nordeste | 272 | 0.250 | 0.158 | 0.632 | 0.032 | 0.012 |
| regiao | Norte | 101 | 0.213 | 0.192 | 0.899 | 0.053 | 0.008 |
| regiao | Sudeste | 176 | 0.343 | 0.381 | 1.109 | 0.029 | 0.025 |
| regiao | Sul | 148 | 0.467 | 0.448 | 0.958 | 0.045 | 0.023 |
| tipo_area | capital | 290 | 0.476 | 0.602 | 1.266 | 0.028 | 0.024 |
| tipo_area | interior_urbano | 333 | 0.292 | 0.272 | 0.931 | 0.035 | 0.019 |
| tipo_area | resto_rm | 129 | 0.446 | 0.304 | 0.682 | 0.059 | 0.022 |
| tipo_area | rural | 59 | 0.059 | 0.075 | 1.271 | 0.023 | 0.003 |
| faixa_idade | 10-19 | 199 | 0.476 | 0.438 | 0.920 | 0.054 | 0.040 |
| faixa_idade | 20-24 | 97 | 0.409 | 0.413 | 1.009 | 0.030 | 0.025 |
| faixa_idade | 25-29 | 103 | 0.484 | 0.494 | 1.020 | 0.027 | 0.018 |
| faixa_idade | 30-34 | 82 | 0.352 | 0.404 | 1.149 | 0.028 | 0.018 |
| faixa_idade | 35-39 | 100 | 0.420 | 0.269 | 0.641 | 0.030 | 0.018 |
| faixa_idade | 40-44 | 62 | 0.263 | 0.395 | 1.498 | 0.020 | 0.012 |
| faixa_idade | 45-59 | 125 | 0.233 | 0.299 | 1.284 | 0.051 | 0.014 |
| faixa_idade | 60+ | 43 | 0.146 | 0.087 | 0.597 | 0.078 | 0.012 |
| publico | 20-44 | 444 | 0.387 | 0.393 | 1.016 | 0.024 | 0.017 |
| publico | outras idades | 367 | 0.282 | 0.278 | 0.985 | 0.040 | 0.021 |
| sexo | homem | 391 | 0.369 | 0.329 | 0.891 | 0.036 | 0.021 |
| sexo | mulher | 420 | 0.288 | 0.325 | 1.129 | 0.024 | 0.016 |
| classe | A | 26 | 0.439 | 0.841 | 1.919 | 0.020 | 0.033 |
| classe | B | 112 | 0.675 | 0.663 | 0.983 | 0.058 | 0.065 |
| classe | C | 375 | 0.454 | 0.426 | 0.938 | 0.031 | 0.022 |
| classe | D | 199 | 0.204 | 0.199 | 0.975 | 0.014 | 0.009 |
| classe | E | 99 | 0.136 | 0.127 | 0.932 | 0.022 | 0.012 |

## Perfil dos erros (% ponderado dentro de cada grupo)

Perdidas = ocasiões reais fora da faixa alta; falsos alarmes = faixa alta sem ocasião.

| segmento | valor | ocasioes_reais_pct | perdidas_pct | faixa_alta_pct | falsos_alarmes_pct | n_perdidas |
|---|---|---|---|---|---|---|
| turno | almoco | 2.7 | 6.8 | 1.4 | 1.4 | 15 |
| turno | manha | 1.6 | 5.2 | 0.0 | 0.0 | 16 |
| turno | noite | 93.3 | 80.4 | 98.6 | 98.6 | 271 |
| turno | tarde | 2.4 | 7.7 | 0.0 | 0.0 | 31 |
| dia_semana | 1 | 9.1 | 13.8 | 7.8 | 7.8 | 52 |
| dia_semana | 2 | 7.6 | 13.4 | 9.5 | 9.6 | 53 |
| dia_semana | 3 | 8.0 | 12.6 | 9.7 | 9.8 | 47 |
| dia_semana | 4 | 11.5 | 22.6 | 7.8 | 7.8 | 76 |
| dia_semana | 5 | 15.9 | 22.7 | 17.5 | 17.6 | 34 |
| dia_semana | 6 | 26.8 | 3.5 | 27.1 | 26.9 | 6 |
| dia_semana | 7 | 21.1 | 11.3 | 20.6 | 20.5 | 65 |
| regiao | Centro-Oeste | 8.3 | 5.0 | 11.6 | 11.7 | 34 |
| regiao | Nordeste | 20.1 | 32.7 | 11.1 | 11.0 | 153 |
| regiao | Norte | 5.8 | 9.7 | 5.1 | 5.1 | 53 |
| regiao | Sudeste | 45.2 | 38.7 | 51.5 | 51.6 | 52 |
| regiao | Sul | 20.5 | 13.9 | 20.7 | 20.6 | 41 |
| tipo_area | capital | 34.4 | 18.1 | 41.0 | 41.0 | 77 |
| tipo_area | interior_urbano | 41.0 | 49.0 | 41.6 | 41.7 | 149 |
| tipo_area | resto_rm | 21.9 | 25.9 | 15.9 | 15.8 | 54 |
| tipo_area | rural | 2.6 | 7.0 | 1.5 | 1.5 | 53 |
| faixa_idade | 10-19 | 25.9 | 16.4 | 24.3 | 24.1 | 61 |
| faixa_idade | 20-24 | 10.6 | 7.8 | 11.0 | 11.0 | 34 |
| faixa_idade | 25-29 | 12.5 | 7.7 | 12.9 | 12.9 | 30 |
| faixa_idade | 30-34 | 9.3 | 7.6 | 10.2 | 10.2 | 35 |
| faixa_idade | 35-39 | 11.7 | 17.5 | 7.8 | 7.7 | 59 |
| faixa_idade | 40-44 | 6.6 | 5.5 | 10.1 | 10.2 | 17 |
| faixa_idade | 45-59 | 15.7 | 19.4 | 21.3 | 21.5 | 59 |
| faixa_idade | 60+ | 7.8 | 18.1 | 2.4 | 2.4 | 38 |
| publico | 20-44 | 50.7 | 46.1 | 52.0 | 52.0 | 175 |
| publico | outras idades | 49.3 | 53.9 | 48.0 | 48.0 | 158 |
| sexo | homem | 53.9 | 52.5 | 47.9 | 47.7 | 155 |
| sexo | mulher | 46.1 | 47.5 | 52.1 | 52.3 | 178 |
| classe | A | 4.9 | 3.3 | 8.0 | 8.0 | 6 |
| classe | B | 18.6 | 3.0 | 17.4 | 17.2 | 15 |
| classe | C | 47.9 | 43.3 | 49.4 | 49.4 | 121 |
| classe | D | 19.9 | 38.7 | 19.3 | 19.5 | 123 |
| classe | E | 8.8 | 11.8 | 6.0 | 5.9 | 68 |
| local_pizza | em casa | 69.1 | 71.5 |  |  | 201 |
| local_pizza | fora de casa | 30.8 | 28.5 |  |  | 132 |

## Fatia de guaraná nas ocasiões de pizza com refrigerante (POF 2017-18)

Segmentos com menos de 40 ocasiões ficam marcados como instáveis.

| segmento | valor | ocasioes | com_guarana | fatia_guarana_pct | ic95 | estavel |
|---|---|---|---|---|---|---|
| nacional | Brasil | 811 | 185 | 21.0 | [15.2; 27.6] | True |
| turno | almoco | 17 | 3 | 26.5 | [0.0; 58.5] | False |
| turno | manha | 16 | 5 | 16.8 | [3.3; 59.5] | False |
| turno | noite | 747 | 172 | 20.7 | [14.9; 27.4] | True |
| turno | tarde | 31 | 5 | 30.4 | [0.0; 57.1] | False |
| dia_semana | 1 | 85 | 12 | 15.2 | [5.9; 26.0] | True |
| dia_semana | 2 | 83 | 18 | 31.6 | [12.2; 47.6] | True |
| dia_semana | 3 | 80 | 24 | 28.8 | [15.5; 46.8] | True |
| dia_semana | 4 | 114 | 27 | 31.4 | [12.8; 50.5] | True |
| dia_semana | 5 | 85 | 15 | 17.0 | [5.4; 34.0] | True |
| dia_semana | 6 | 69 | 12 | 14.2 | [2.1; 29.5] | True |
| dia_semana | 7 | 295 | 77 | 23.0 | [14.8; 33.8] | True |
| regiao | Centro-Oeste | 114 | 25 | 20.7 | [8.8; 33.6] | True |
| regiao | Nordeste | 272 | 70 | 21.4 | [14.7; 31.1] | True |
| regiao | Norte | 101 | 20 | 24.0 | [4.8; 43.0] | True |
| regiao | Sudeste | 176 | 42 | 21.5 | [9.8; 36.4] | True |
| regiao | Sul | 148 | 28 | 19.0 | [8.2; 32.4] | True |
| tipo_area | capital | 290 | 61 | 15.8 | [9.5; 23.8] | True |
| tipo_area | interior_urbano | 333 | 90 | 23.6 | [13.9; 35.1] | True |
| tipo_area | resto_rm | 129 | 27 | 25.8 | [8.0; 44.1] | True |
| tipo_area | rural | 59 | 7 | 9.2 | [1.0; 21.6] | True |
| faixa_idade | 10-19 | 199 | 51 | 26.2 | [15.0; 41.8] | True |
| faixa_idade | 20-24 | 97 | 23 | 19.5 | [8.3; 32.2] | True |
| faixa_idade | 25-29 | 103 | 22 | 19.2 | [11.2; 27.1] | True |
| faixa_idade | 30-34 | 82 | 17 | 11.5 | [4.8; 21.6] | True |
| faixa_idade | 35-39 | 100 | 26 | 27.8 | [12.9; 42.4] | True |
| faixa_idade | 40-44 | 62 | 14 | 28.5 | [10.3; 46.3] | True |
| faixa_idade | 45-59 | 125 | 28 | 19.5 | [8.9; 30.5] | True |
| faixa_idade | 60+ | 43 | 4 | 6.9 | [0.3; 19.1] | True |
| publico | 20-44 | 444 | 102 | 21.1 | [15.4; 28.9] | True |
| publico | outras idades | 367 | 83 | 21.0 | [13.3; 31.4] | True |
| sexo | homem | 391 | 83 | 19.1 | [11.9; 27.1] | True |
| sexo | mulher | 420 | 102 | 23.2 | [16.2; 31.5] | True |
| classe | A | 26 | 5 | 8.6 | [0.8; 25.8] | False |
| classe | B | 112 | 22 | 18.9 | [5.3; 38.4] | True |
| classe | C | 375 | 89 | 21.7 | [15.0; 30.1] | True |
| classe | D | 199 | 54 | 25.2 | [14.5; 41.8] | True |
| classe | E | 99 | 15 | 19.5 | [4.5; 43.2] | True |
| local_pizza | em casa | 539 | 126 | 23.5 | [16.0; 32.8] | True |
| local_pizza | fora de casa | 271 | 59 | 15.5 | [9.2; 23.9] | True |
