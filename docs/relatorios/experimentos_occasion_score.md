# Occasion Score: experimentos (treino POF 2008-09, teste POF 2017-18)

Linhas: treino 267,612 (668 positivos), teste 340,072 (811 positivos). Taxa ponderada no teste: 0.327% das pessoa-turno.

## Grade do LightGBM (CV por setor em 2008-09)

| num_leaves | min_child_samples | pr_auc_cv | n_arvores |
|---|---|---|---|
| 31 | 400 | 0.0741 | 54 |
| 31 | 100 | 0.0725 | 65 |
| 15 | 100 | 0.0674 | 82 |
| 15 | 400 | 0.0670 | 88 |
| 7 | 400 | 0.0576 | 195 |
| 7 | 100 | 0.0567 | 134 |

Escolhido: {'num_leaves': 31, 'min_child_samples': 400}, 54 árvores.

## Ablação (ranking em 2017-18; IC95 por bootstrap pareado de setores)

| variante | pr_auc_cv_2008 | pr_auc | roc_auc | taxa_base | lift_top10 | recall_top10 | dif_pr_auc_vs_completo [IC95] | dif_pr_auc_vs_baseline [IC95] |
|---|---|---|---|---|---|---|---|---|
| baseline |  | 0.0182 | 0.8801 | 0.0033 | 5.8054 | 0.5805 | +0.0000 [+0.0000; +0.0000] | +0.0000 [+0.0000; +0.0000] |
| logistica \| completo (10 features) | 0.0490 | 0.0292 | 0.8994 | 0.0033 | 6.8375 | 0.6838 | +0.0000 [+0.0000; +0.0000] | +0.0115 [+0.0057; +0.0192] |
| lgbm \| completo (10 features) | 0.0651 | 0.0270 | 0.8945 | 0.0033 | 6.6858 | 0.6686 | +0.0000 [+0.0000; +0.0000] | +0.0098 [+0.0023; +0.0252] |
| logistica \| sem tem_crianca | 0.0530 | 0.0298 | 0.8984 | 0.0033 | 6.7596 | 0.6760 | +0.0006 [-0.0006; +0.0022] | +0.0121 [+0.0060; +0.0210] |
| lgbm \| sem tem_crianca | 0.0677 | 0.0273 | 0.8973 | 0.0033 | 6.6796 | 0.6680 | -0.0001 [-0.0060; +0.0021] | +0.0098 [+0.0034; +0.0202] |
| logistica \| sem n_moradores | 0.0462 | 0.0286 | 0.8972 | 0.0033 | 6.7702 | 0.6770 | -0.0007 [-0.0029; +0.0015] | +0.0107 [+0.0053; +0.0186] |
| lgbm \| sem n_moradores | 0.0450 | 0.0261 | 0.8941 | 0.0033 | 6.7125 | 0.6713 | -0.0012 [-0.0093; +0.0032] | +0.0086 [+0.0030; +0.0167] |
| logistica \| sem os dois | 0.0466 | 0.0299 | 0.8985 | 0.0033 | 6.8392 | 0.6839 | +0.0007 [-0.0010; +0.0033] | +0.0122 [+0.0060; +0.0215] |
| lgbm \| sem os dois | 0.0568 | 0.0264 | 0.8905 | 0.0033 | 6.9138 | 0.6914 | -0.0010 [-0.0097; +0.0036] | +0.0089 [+0.0026; +0.0182] |
| logistica \| sem os dois, só região | 0.0471 | 0.0289 | 0.8986 | 0.0033 | 6.8383 | 0.6838 | -0.0003 [-0.0028; +0.0024] | +0.0111 [+0.0053; +0.0194] |
| lgbm \| sem os dois, só região | 0.0397 | 0.0242 | 0.8909 | 0.0033 | 6.4136 | 0.6414 | -0.0037 [-0.0174; +0.0027] | +0.0062 [+0.0018; +0.0120] |
| logistica \| sem os dois, só Trends | 0.0482 | 0.0297 | 0.8986 | 0.0033 | 6.7045 | 0.6704 | +0.0005 [-0.0022; +0.0030] | +0.0119 [+0.0063; +0.0206] |
| lgbm \| sem os dois, só Trends | 0.0518 | 0.0287 | 0.8949 | 0.0033 | 6.9402 | 0.6940 | +0.0014 [-0.0051; +0.0067] | +0.0112 [+0.0044; +0.0236] |
| logistica \| sem os dois, sem geografia | 0.0431 | 0.0321 | 0.9009 | 0.0033 | 6.8189 | 0.6819 | +0.0031 [-0.0021; +0.0093] | +0.0146 [+0.0070; +0.0246] |
| lgbm \| sem os dois, sem geografia | 0.0391 | 0.0250 | 0.8932 | 0.0033 | 6.5803 | 0.6580 | -0.0027 [-0.0165; +0.0034] | +0.0072 [+0.0020; +0.0130] |
| logistica \| Trends normalizado (0-1) | 0.0490 | 0.0293 | 0.8999 | 0.0033 | 6.7919 | 0.6792 | +0.0001 [-0.0002; +0.0004] | +0.0116 [+0.0058; +0.0195] |
| lgbm \| Trends normalizado (0-1) | 0.0651 | 0.0253 | 0.8953 | 0.0033 | 6.8646 | 0.6865 | -0.0022 [-0.0116; +0.0016] | +0.0077 [+0.0023; +0.0148] |
| logistica \| dia agrupado (seg-qui, sex, sáb, dom) | 0.0495 | 0.0292 | 0.8999 | 0.0033 | 6.9898 | 0.6990 | +0.0000 [-0.0006; +0.0004] | +0.0115 [+0.0058; +0.0191] |
| lgbm \| dia agrupado (seg-qui, sex, sáb, dom) | 0.0550 | 0.0278 | 0.8948 | 0.0033 | 6.5219 | 0.6522 | +0.0009 [-0.0012; +0.0036] | +0.0107 [+0.0026; +0.0276] |
| logistica \| FINAL | 0.0471 | 0.0289 | 0.8986 | 0.0033 | 6.8383 | 0.6838 | -0.0003 [-0.0028; +0.0024] | +0.0111 [+0.0053; +0.0194] |
| lgbm \| FINAL | 0.0397 | 0.0242 | 0.8909 | 0.0033 | 6.4136 | 0.6414 | -0.0037 [-0.0174; +0.0027] | +0.0062 [+0.0018; +0.0120] |

## Probabilidade (Platt cruzado por setor)

| modelo | brier | log_loss | media_prevista | media_observada |
|---|---|---|---|---|
| baseline calibrado | 0.00322 | 0.01823 | 0.00327 | 0.00327 |
| logística final sem calibrar | 0.00322 | 0.01770 | 0.00359 | 0.00327 |
| logística final calibrada | 0.00321 | 0.01766 | 0.00327 | 0.00327 |

## Classificação (limiar de F1 escolhido na outra metade dos setores)

| modelo | limiar_medio | precision | recall | f1 | vp_pct | fp_pct | fn_pct | vn_pct | positivos_previstos_pct |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.0277 | 0.0232 | 0.1337 | 0.0396 | 0.0437 | 1.8365 | 0.2831 | 97.8367 | 1.8802 |
| logística final | 0.0476 | 0.0601 | 0.1251 | 0.0812 | 0.0409 | 0.6392 | 0.2859 | 99.0340 | 0.6801 |

## Curva de calibração (decis ponderados do score)

| faixa | prevista | observada | positivos |
|---|---|---|---|
| 0 | 0.005 | 0.015 | 6 |
| 1 | 0.012 | 0.012 | 7 |
| 2 | 0.021 | 0.050 | 9 |
| 3 | 0.032 | 0.039 | 9 |
| 4 | 0.049 | 0.015 | 9 |
| 5 | 0.075 | 0.012 | 9 |
| 6 | 0.121 | 0.072 | 21 |
| 7 | 0.226 | 0.125 | 62 |
| 8 | 0.544 | 0.670 | 201 |
| 9 | 2.185 | 2.257 | 478 |

## Segmentos (modelo final calibrado)

| segmento | valor | positivos | taxa_obs_pct | score_medio_pct | pr_auc_modelo | pr_auc_baseline |
|---|---|---|---|---|---|---|
| turno | almoco | 17 | 0.0352 | 0.1163 | 0.0070 | 0.0004 |
| turno | manha | 16 | 0.0210 | 0.0400 | 0.0004 | 0.0008 |
| turno | noite | 747 | 1.2199 | 1.0952 | 0.0306 | 0.0190 |
| turno | tarde | 31 | 0.0310 | 0.0567 | 0.0004 | 0.0004 |
| regiao | Centro-Oeste | 114 | 0.3722 | 0.5511 | 0.0276 | 0.0224 |
| regiao | Nordeste | 272 | 0.2496 | 0.1577 | 0.0319 | 0.0127 |
| regiao | Norte | 101 | 0.2134 | 0.1919 | 0.0526 | 0.0085 |
| regiao | Sudeste | 176 | 0.3432 | 0.3807 | 0.0287 | 0.0228 |
| regiao | Sul | 148 | 0.4675 | 0.4479 | 0.0454 | 0.0238 |
| dia_semana | 1 | 85 | 0.2090 | 0.1758 | 0.0310 | 0.0066 |
| dia_semana | 2 | 83 | 0.1734 | 0.2023 | 0.0170 | 0.0058 |
| dia_semana | 3 | 80 | 0.1835 | 0.2017 | 0.0247 | 0.0070 |
| dia_semana | 4 | 114 | 0.2624 | 0.1748 | 0.0172 | 0.0086 |
| dia_semana | 5 | 85 | 0.3636 | 0.3518 | 0.0158 | 0.0123 |
| dia_semana | 6 | 69 | 0.6139 | 0.7044 | 0.0484 | 0.0238 |
| dia_semana | 7 | 295 | 0.4817 | 0.4785 | 0.0318 | 0.0182 |
| publico | 20-44 | 444 | 0.3867 | 0.3931 | 0.0242 | 0.0173 |
| publico | outras idades | 367 | 0.2819 | 0.2775 | 0.0396 | 0.0206 |
| classe | A | 26 | 0.4385 | 0.8414 | 0.0200 | 0.0304 |
| classe | B | 112 | 0.6752 | 0.6635 | 0.0576 | 0.0575 |
| classe | C | 375 | 0.4541 | 0.4258 | 0.0314 | 0.0225 |
| classe | D | 199 | 0.2045 | 0.1993 | 0.0137 | 0.0089 |
| classe | E | 99 | 0.1358 | 0.1266 | 0.0216 | 0.0090 |

## Canal: recall no top 10% por local da pizza

| local | positivos | recall_no_top10 |
|---|---|---|
| em casa | 539 | 0.6802 |
| fora de casa | 271 | 0.7141 |
| desconhecido | 1 | 1 |

## Coeficientes da logística final

| termo | coef |
|---|---|
| cat__turno_manha | -2.000 |
| cat__faixa_idade_60+ | -1.796 |
| cat__turno_tarde | -1.613 |
| renda__renda_pct_sp_2 | -1.595 |
| cat__tipo_area_rural | -1.585 |
| cat__sexo_1 | -1.387 |
| cat__sexo_2 | -1.339 |
| cat__regiao_Norte | -1.060 |
| renda__renda_pct_sp_1 | -1.050 |
| cat__regiao_Nordeste | -1.038 |
| cat__dia_semana_1 | -0.937 |
| cat__dia_semana_4 | -0.852 |
| cat__turno_almoco | -0.814 |
| cat__dia_semana_2 | -0.746 |
| cat__dia_semana_3 | -0.743 |
| cat__tipo_area_resto_rm | -0.704 |
| cat__tipo_area_interior_urbano | -0.587 |
| cat__faixa_idade_35-39 | -0.531 |
| cat__faixa_idade_45-59 | -0.485 |
| renda__renda_pct_sp_4 | -0.444 |
| cat__regiao_Sudeste | -0.419 |
| renda__renda_pct_sp_3 | -0.271 |
| cat__dia_semana_5 | -0.266 |
| cat__regiao_Centro-Oeste | -0.155 |
| cat__faixa_idade_40-44 | -0.147 |
| cat__faixa_idade_30-34 | -0.146 |
| renda__renda_pct_sp_0 | -0.124 |
| cat__regiao_Sul | -0.053 |
| cat__faixa_idade_20-24 | 0.052 |
| cat__faixa_idade_25-29 | 0.126 |
| cat__tipo_area_capital | 0.150 |
| cat__faixa_idade_10-19 | 0.201 |
| renda__renda_pct_sp_6 | 0.220 |
| cat__dia_semana_7 | 0.252 |
| renda__renda_pct_sp_5 | 0.538 |
| cat__dia_semana_6 | 0.565 |
| cat__turno_noite | 1.702 |
