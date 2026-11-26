# Trend Radar

Gerado em 2026-11-26, última semana fechada: 2026-11-15.

Buscas medem interesse, não consumo. O Radar não entra no Occasion Score.

## Fontes

- trends: Google Trends, 'pizza', categoria Comida e bebida, por UF, semanal, 5 anos
- feriados: BrasilAPI, feriados nacionais
- clima: Open-Meteo (ERA5 e previsão de 16 dias), capital da UF
- mapa: Mapa de Oportunidade (Occasion Score no Censo 2022 + CNPJ)

## Validação fora do tempo

Treino até 2024 (2664 UF-semanas), teste de 2025-01-05 a 2026-11-15 (2352). Alvo: desvio % do índice da semana contra a média das 52 semanas anteriores da UF. IC95 do ganho por bootstrap de semanas.

| modelo | MAE (pontos %) | ganho vs baseline | IC95 do ganho | ganho nas semanas com evento | IC95 |
|---|---|---|---|---|---|
| baseline (igual à média do último ano) | 12.63 | 0.0 | [0.0; 0.0] | 0.0 | [0.0; 0.0] |
| calendário | 12.15 | 0.48 | [-0.25; 1.3] | 1.31 | [-0.31; 3.02] |
| calendário + clima | 12.15 | 0.47 | [-0.27; 1.21] | 1.3 | [-0.25; 3.05] |

Semanas com evento: feriado nacional, Carnaval, Natal e Ano-Novo ou Dia da Pizza (24 semanas no teste).

## Coeficientes (modelo final, todas as semanas)

Pontos percentuais de desvio por unidade da variável. IC95 por bootstrap de blocos de 4 semanas (300 reamostras). Anomalia de temperatura em °C e de chuva em mm/dia contra a média 2016-2020 da mesma semana do ano.

| variável | coef | IC95 |
|---|---|---|
| intercepto | 1.29 | [-1.76; 5.31] |
| mes_02 | -8.01 | [-12.85; -4.21] |
| mes_03 | -8.17 | [-13.70; -4.21] |
| mes_04 | -9.34 | [-16.39; -3.09] |
| mes_05 | -7.41 | [-12.48; -3.82] |
| mes_06 | -2.88 | [-11.10; 6.61] |
| mes_07 | 5.11 | [-0.70; 15.21] |
| mes_08 | -0.41 | [-10.41; 15.19] |
| mes_09 | -6.20 | [-14.95; 3.33] |
| mes_10 | -13.59 | [-18.88; -8.72] |
| mes_11 | -15.10 | [-19.84; -9.78] |
| mes_12 | -5.31 | [-11.45; -0.91] |
| feriado | 7.29 | [4.25; 9.38] |
| carnaval | 23.98 | [21.13; 27.32] |
| natal_ano_novo | 18.38 | [13.22; 27.31] |
| dia_da_pizza | 9.80 | [-0.81; 16.19] |
| inicio_mes | 1.00 | [-1.85; 4.15] |
| anom_temp | -0.22 | [-0.93; 0.63] |
| anom_chuva | 0.15 | [0.00; 0.33] |

## Alertas por UF

| UF | nível | variação anual (4 sem.) | faixa de ruído | próxima janela | fatores (4 sem.) |
|---|---|---|---|---|---|
| SP | atencao | -7,1% | ±6.9% | +1,2% | época do ano -13,4%, não explicado pelo calendário e pelo clima -5,6%, semana com feriado nacional +3,6% |
| SC | atencao | -9,6% | ±7.1% | +1,2% | época do ano -13,4%, não explicado pelo calendário e pelo clima -8,5%, semana com feriado nacional +3,6% |
| MS | atencao | -15,6% | ±9.6% | +1,3% | época do ano -13,4%, não explicado pelo calendário e pelo clima -9,1%, semana com feriado nacional +3,6% |
| AL | atencao | -18,6% | ±14.7% | +1,0% | não explicado pelo calendário e pelo clima -24,7%, época do ano -13,4%, semana com feriado nacional +3,6% |
| TO | estavel | +18,2% | ±19.5% | +0,4% | época do ano -13,4%, não explicado pelo calendário e pelo clima +7,2%, semana com feriado nacional +3,6% |
| AM | estavel | +5,9% | ±16.7% | +0,9% | época do ano -13,4%, não explicado pelo calendário e pelo clima +9,7%, semana com feriado nacional +3,6% |
| MA | estavel | +3,2% | ±13.4% | +0,9% | época do ano -13,4%, não explicado pelo calendário e pelo clima -6,2%, semana com feriado nacional +3,6% |
| PI | estavel | +2,0% | ±16.6% | +0,5% | época do ano -13,4%, semana com feriado nacional +3,6%, não explicado pelo calendário e pelo clima -2,1% |
| RO | estavel | -1,6% | ±13.8% | +0,6% | época do ano -13,4%, semana com feriado nacional +3,6% |
| SE | estavel | -3,0% | ±20.0% | +0,8% | época do ano -13,4%, não explicado pelo calendário e pelo clima -5,9%, semana com feriado nacional +3,6% |
| MG | estavel | -3,1% | ±11.1% | +1,5% | época do ano -13,4%, não explicado pelo calendário e pelo clima -6,8%, semana com feriado nacional +3,6% |
| RN | estavel | -3,5% | ±14.3% | +1,0% | época do ano -13,4%, não explicado pelo calendário e pelo clima -4,5%, semana com feriado nacional +3,6% |
| GO | estavel | -4,1% | ±10.9% | +1,0% | época do ano -13,4%, semana com feriado nacional +3,6% |
| PA | estavel | -5,5% | ±14.5% | +0,8% | época do ano -13,4%, semana com feriado nacional +3,6%, não explicado pelo calendário e pelo clima -3,3% |
| PE | estavel | -5,5% | ±12.8% | +1,0% | época do ano -13,4%, não explicado pelo calendário e pelo clima -7,6%, semana com feriado nacional +3,6% |
| MT | estavel | -5,9% | ±11.1% | +1,4% | época do ano -13,4%, semana com feriado nacional +3,6% |
| CE | estavel | -6,1% | ±12.0% | +1,0% | época do ano -13,4%, não explicado pelo calendário e pelo clima -7,6%, semana com feriado nacional +3,6% |
| ES | estavel | -6,1% | ±14.1% | +1,6% | época do ano -13,4%, não explicado pelo calendário e pelo clima -7,0%, semana com feriado nacional +3,6% |
| BA | estavel | -6,2% | ±12.2% | +0,7% | época do ano -13,4%, não explicado pelo calendário e pelo clima -5,9%, semana com feriado nacional +3,6% |
| RJ | estavel | -6,3% | ±11.1% | +1,1% | época do ano -13,4%, não explicado pelo calendário e pelo clima -9,5%, semana com feriado nacional +3,6% |
| PB | estavel | -7,2% | ±13.0% | +1,0% | época do ano -13,4%, não explicado pelo calendário e pelo clima -7,6%, semana com feriado nacional +3,6% |
| PR | estavel | -7,5% | ±8.7% | +1,1% | época do ano -13,4%, não explicado pelo calendário e pelo clima -5,2%, semana com feriado nacional +3,6% |
| RS | estavel | -8,4% | ±10.7% | +0,9% | época do ano -13,4%, não explicado pelo calendário e pelo clima -9,3%, semana com feriado nacional +3,6% |
| DF | estavel | -10,3% | ±10.4% | +0,7% | época do ano -13,4%, não explicado pelo calendário e pelo clima -5,1%, semana com feriado nacional +3,6% |
| AC | sem_volume | – | – | – | o Trends devolve volume insuficiente (zero) em mais de 20% das semanas; a UF não é lida |
| AP | sem_volume | – | – | – | o Trends devolve volume insuficiente (zero) em mais de 20% das semanas; a UF não é lida |
| RR | sem_volume | – | – | – | o Trends devolve volume insuficiente (zero) em mais de 20% das semanas; a UF não é lida |

## Regras do alerta

1. Subindo (ou próxima janela prevista acima da faixa de ruído) e município de potencial alto no Mapa → oportunidade.
2. Subindo sem município de potencial alto → observar.
3. Caindo → atenção.
4. Resto → estável.

## Limitações

- Índice do Trends é relativo dentro de cada UF e amostral (dois pedidos iguais diferem um pouco); a coleta fica versionada.
- "pizza" mistura receita e delivery; a categoria Comida e bebida tira buscas fora de comida (ex.: "fantasia de pizza" no Carnaval de 2025).
- Clima da capital representa a UF inteira (hipótese).
- Só feriados nacionais (BrasilAPI); estaduais e municipais ficam de fora.
- Associação, não causa. O alerta não afirma que uma ativação vende mais.
