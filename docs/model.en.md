# Model card: Occasion Score

[Português](model.md) · English

Model sheet in the *model cards* format (Mitchell et al., 2019). The data is described in the [Data Card](data_card.md) (Portuguese), and the system around the model in [`architecture.en.md`](architecture.en.md).

## Details

- **What it is**: calibrated probability, from 0 to 100, that a person eats pizza with soda in a meal period (morning, lunch, afternoon or night), given the moment, the place and the profile.
- **Target**: pizza and soda, generic or branded, within 1 hour of each other in the same period, as declared in the POF food diary. It is real consumption, not a proxy.
- **Version**: 1.1.0, trained on 26/11/2026 on 607,684 person-periods with 1,479 occasions (POF 2008-09 and 2017-18). Files: `models/occasion_score_v1.1.0.joblib` and `.json`.
- **Architecture**:
  - logistic regression with one-hot categorical variables and splines on the income percentile;
  - Platt calibration on out-of-sample 2017-18 predictions (halves of census tracts);
  - per-group correction: a calibrated-logit offset for the Northeast (+0.48), Center-West (−0.31) and rural areas (−0.60).
- **Features** (7): meal period, weekday, region, area type (state capital, metropolitan region, urban interior, rural), age band, sex and household income percentile.
- **Bands**: high (top 10% of scores in the 2017-18 population, from 1.02%), medium (10% to 30%, from 0.098%) and low (the rest).
- **Explanation**: linear SHAP, exact for the logistic model, on the calibrated logit. Feature contributions add up to the score's logit.
- **Uncertainty**: 95% CI by census-tract bootstrap, 200 replicas that redo the whole pipeline (logistic, Platt and offsets).
- **Code license**: none declared. POF and Census microdata are public (IBGE).

## Intended use

- **What for**: ranking contexts (moment, place, audience) by the occasion's likelihood and feeding the Decision Engine, which chooses the activation category.
- **For whom**: Guaraná brand and trade teams, through the simulator, and other systems, through the API.
- **Best use**: comparing contexts with each other. The absolute level is from 2017-18 and may have changed since.

## Out of scope

- Forecasting sales, volume or an activation's effect. The score says where the occasion happens, not that an action will create new consumption.
- Choosing between brands: the target is pizza with any soda. Guaraná's presence enters the Engine through the association table.
- Specific dates, weather, price or promotion: the POF records neither the consumption date nor the price.
- Separating delivery from home-made food: the POF does not separate them.
- People under 10 (outside the POF food diary).

## Factors evaluated

- **Region** and **area type**.
- **Income class** (A to E, by household minimum wages).
- **Age band** and **sex**.
- **Weekday** and **meal period**.
- **POF edition**, through temporal validation.

## Metrics

- **Primary: PR-AUC.** The occasion happens in 0.33% of periods. With so many negatives, ROC-AUC stays high even for weak models.
- **Ranking**: ROC-AUC, lift and recall in the top 10% of scores.
- **Probability**: Brier, log loss and calibration (mean predicted versus observed, by decile and by group).
- **Classification**: precision, recall and F1 at a threshold chosen on another half of the tracts, and the confusion matrix.
- All weighted by the POF weight, with each weekday counting 1/7, and with 95% CIs from a paired census-tract bootstrap.

## Evaluation data

- **Temporal validation**: train on POF 2008-09 (267,612 rows, 668 occasions), test on POF 2017-18 (340,072 rows, 811 occasions).
- **Selection and measurement kept apart**: choosing between candidates uses half of the 2017-18 tracts, and the other half only measures.

## Training data

- **Served model**: both editions together, with Platt calibration and the per-group correction estimated on out-of-sample 2017-18 predictions.
- **Unit**: person × recorded day × meal period, aged 10 or older. Details, weights and integration rules in the [Data Card](data_card.md).

## Results

Temporal validation (train 2008-09, test 2017-18). Reproduced by `occasion.modelagem.final.metricas_temporais`.

| Metric | Baseline (day × period) | Final logistic |
|---|---:|---:|
| PR-AUC | 0.0182 | **0.0288** |
| ROC-AUC | 0.880 | **0.899** |
| Lift in top 10% | 5.8× | **6.9×** |
| Recall in top 10% | 58% | **69%** |

- **PR-AUC gain over the baseline**: +0.011 (95% CI +0.005 to +0.019).
- **Calibration**: mean predicted 0.327% versus 0.327% observed; Brier 0.00321.
- **Importance** (PR-AUC drop when shuffled): meal period 80%, weekday 29%, income 19%, age band 13%, region 5%.
- **Bands**: the high band (10% of the population) holds 69% of occasions; medium (20%), 24%; low (70%), 7%.
- **Per-group correction**: out of sample, the score got the level wrong for the Northeast (predicted ÷ observed 0.68), the Center-West (1.42) and rural areas (1.63). With the offsets, measured on tracts the correction did not see, the ratios become 0.99, 1.01 and 1.04. Recall in the top 10% rises from 70.1% to 71.7%, and PR-AUC ties.
- **Uncertainty**: in 2017-18, the mean half-width of the 95% CI is 48% of the score in the high band, 62% in the medium band and 97% in the low band. The uncertain-band warning appears in 46% of high-band cases.

Reports (Portuguese): [experiments](relatorios/experimentos_occasion_score.md), [candidates](relatorios/candidatos_occasion_score.md), [evaluation](relatorios/avaliacao_occasion_score.md), [group bias](relatorios/vies_por_grupo.md), [synthetic data](relatorios/dados_sinteticos.md).

## Ethical considerations

- **Personal data**: the POF is public and anonymized by IBGE. No Ambev customer data was used.
- **Group bias**: the national model underestimated the Northeast and overestimated the Center-West and rural areas. The correction is applied and measured out of sample. Class D is calibrated but has few people in the national high band, so the API also shows the score's position within the class.
- **Audience**: the score favors where the occasion is already frequent (class A, night, weekend). To avoid reinforcing only that audience, the Engine treats the medium band of classes B to D as awareness.
- **No causality**: no recommendation claims an action sells more. Messages and mechanics are labeled as hypotheses.

## Caveats and recommendations

- **Old data**: POF 2017-18 predates the pandemic and the delivery boom. Checks against Google Trends and iFood releases from 2024 to 2026 confirmed the night and weekend pattern, but the level may have changed. POF 2024-25 has no microdata yet.
- **Small sample**: 811 occasions in 2017-18. Fine segments (class A, Saturday, rural) have wide intervals.
- **Additive model**: the best day and period is the same for every profile (Saturday night). Interactions (period × day, LightGBM, AutoML) were tested and did not win out of time.
- **Per-group correction**: chosen on the full data. Run on only half of the tracts, the significance rule picks different groups, so the offsets are a measured improvement, not an exact one.
- **Recommendation**: use the score to rank and compare contexts, always with the interval in view, and retrain when POF 2024-25 publishes microdata.
