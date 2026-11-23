# Occasion Score: comparação de candidatos

Treino em 2008-09. Escolha na metade de seleção dos setores de 2017-18; a metade de avaliação só mede. Regra: maior PR-AUC na seleção; se a diferença para um candidato mais simples tem IC95 com zero, fica o mais simples. A CV de 2008-09 aparece só para comparação.

Grade do C da logística (CV 2008-09): C=0.01: 0.0470, C=0.1: 0.0471, C=1.0: 0.0471, C=10.0: 0.0470

AutoML: 900s, melhor estimador `lgbm`, configuração `{'n_estimators': 12, 'num_leaves': 13, 'min_child_samples': 6, 'learning_rate': 0.1297, 'log_max_bin': 4, 'colsample_bytree': 0.9282, 'reg_alpha': 0.001, 'reg_lambda': 0.0289}`.

| candidato | pr_auc_cv_2008 | ep_cv | pr_auc_selecao | pr_auc_avaliacao | recall_top10_avaliacao | dif_vs_logistica_avaliacao [IC95] | dif_vs_baseline_avaliacao [IC95] | pr_auc_2017_completo |
|---|---|---|---|---|---|---|---|---|
| 1. logística (C=1) | 0.0471 | 0.0033 | 0.0316 | 0.0274 | 0.6617 | +0.0000 [+0.0000; +0.0000] | +0.0120 [+0.0037; +0.0250] | 0.0289 |
| 2. logística (C=1.0 por CV) | 0.0471 | 0.0033 | 0.0316 | 0.0274 | 0.6617 | +0.0000 [+0.0000; +0.0000] | +0.0120 [+0.0037; +0.0250] |  |
| 3. logística + turno × dia | 0.0471 | 0.0030 | 0.0327 | 0.0279 | 0.6668 | +0.0007 [-0.0004; +0.0025] | +0.0127 [+0.0041; +0.0267] |  |
| 4. LightGBM regularizado (90 árvores) | 0.0462 | 0.0030 | 0.0288 | 0.0266 | 0.5998 | -0.0008 [-0.0050; +0.0047] | +0.0112 [+0.0021; +0.0254] |  |
| 5. Random Forest | 0.0445 | 0.0052 | 0.0250 | 0.0211 | 0.5894 | -0.0066 [-0.0171; +0.0055] | +0.0054 [-0.0023; +0.0163] |  |
| 6. AutoML FLAML (lgbm; CV = AP interna) | 0.0594 |  | 0.0257 | 0.0231 | 0.6125 | -0.0044 [-0.0150; +0.0048] | +0.0076 [-0.0009; +0.0180] |  |

Maior PR-AUC na seleção: **3. logística + turno × dia**. Escolhido pela regra: **1. logística (C=1)**.

