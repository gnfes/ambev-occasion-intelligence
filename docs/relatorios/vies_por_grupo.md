# Viés do Occasion Score por grupo e correção (v1.1.0)

Previsões de 2017-18 fora da amostra (340,072 pessoa-turnos,
811 ocasiões), no esquema do modelo final. Antes =
Platt (o score da v1.0.0). Depois = Platt + correção dos grupos servidos.
Platt e offsets ajustados numa metade dos setores e medidos na outra, nos
dois sentidos, numa divisão de setores diferente da usada para escolher os
grupos. Os grupos são os do modelo servido, escolhidos nos dados inteiros:
a medição não é de todo independente da escolha, por isso a robustez em 20
divisões vem logo abaixo. `pso` = previsto ÷ observado (acima de 1 = o modelo
superestima o grupo), IC95 por bootstrap de setores.

## Correção servida no modelo v1.1.0

Regra: grupo de região ou tipo de área com previsto ÷ observado fora da amostra em 2017-18 cujo IC95 (bootstrap por setor) exclui 1 e com ≥ 30 ocasiões, mantido só se a correção estimada numa metade dos setores melhora a calibração do grupo na outra; offset no logit calibrado estimado nas mesmas previsões.

Grupos corrigidos: regiao = Centro-Oeste: -0.312 no logit (×0.73 nas chances); regiao = Nordeste: +0.480 no logit (×1.62 nas chances); tipo_area = rural: -0.601 no logit (×0.55 nas chances). Termo global:
-0.035.

Candidatos com IC95 que exclui 1 nos dados inteiros:
regiao = Centro-Oeste, regiao = Nordeste, tipo_area = rural.

## Validação cruzada da correção

| index | pr_auc | roc_auc | recall_top10 | log_loss | brier |
|---|---|---|---|---|---|
| antes (Platt) | 0.0312 | 0.9030 | 0.7005 | 0.0174 | 0.0032 |
| depois (Platt + correção) | 0.0313 | 0.9053 | 0.7174 | 0.0173 | 0.0032 |

Robustez: o mesmo ajuste em 20 divisões de setores diferentes.

| grupo | pso_antes_mediana | pso_depois_mediana | pso_depois_min | pso_depois_max | divisoes_com_melhora_pct |
|---|---|---|---|---|---|
| regiao = Centro-Oeste | 1.42 | 1.00 | 0.96 | 1.06 | 100 |
| regiao = Nordeste | 0.68 | 1.00 | 0.98 | 1.03 | 100 |
| tipo_area = rural | 1.63 | 1.05 | 0.98 | 1.11 | 100 |

Estabilidade da seleção (a regra de significância rodada em cada metade,
com metade das ocasiões e IC mais largo):

- Só com a metade 0: regiao = Nordeste
- Só com a metade 1: regiao = Centro-Oeste, tipo_area = capital

## Por grupo

### regiao

| valor | positivos | ocasioes_pct | pso_antes | ic_antes | pso_depois | ic_depois | recall_alta_antes_pct | recall_alta_depois_pct | fatia_alta_depois_pct |
|---|---|---|---|---|---|---|---|---|---|
| Centro-Oeste | 114 | 8.32 | 1.42 | [1.03; 2.16] | 1.01 | [0.73; 1.53] | 75.06 | 61.47 | 8.34 |
| Nordeste | 272 | 20.15 | 0.68 | [0.54; 0.85] | 0.99 | [0.79; 1.24] | 50.00 | 66.20 | 18.70 |
| Norte | 101 | 5.84 | 1.30 | [0.89; 2.19] | 1.21 | [0.83; 2.02] | 62.57 | 62.57 | 6.80 |
| Sudeste | 176 | 45.22 | 1.05 | [0.78; 1.42] | 1.00 | [0.75; 1.36] | 73.18 | 72.51 | 46.00 |
| Sul | 148 | 20.47 | 0.99 | [0.77; 1.38] | 0.94 | [0.73; 1.29] | 82.99 | 82.29 | 20.16 |
### tipo_area

| valor | positivos | ocasioes_pct | pso_antes | ic_antes | pso_depois | ic_depois | recall_alta_antes_pct | recall_alta_depois_pct | fatia_alta_depois_pct |
|---|---|---|---|---|---|---|---|---|---|
| capital | 290 | 34.44 | 1.23 | [0.95; 1.61] | 1.24 | [0.96; 1.62] | 84.45 | 84.87 | 41.87 |
| interior_urbano | 333 | 41.03 | 0.89 | [0.70; 1.16] | 0.90 | [0.70; 1.17] | 62.17 | 64.96 | 39.16 |
| resto_rm | 129 | 21.92 | 0.79 | [0.57; 1.17] | 0.79 | [0.57; 1.17] | 68.54 | 71.05 | 18.47 |
| rural | 59 | 2.61 | 1.63 | [1.07; 2.69] | 1.04 | [0.68; 1.74] | 16.66 | 10.89 | 0.51 |
### classe

| valor | positivos | ocasioes_pct | pso_antes | ic_antes | pso_depois | ic_depois | recall_alta_antes_pct | recall_alta_depois_pct | fatia_alta_depois_pct | recall_topo_na_classe_pct |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 26 | 4.89 | 1.59 | [0.89; 3.75] | 1.51 | [0.86; 3.56] | 80.16 | 80.16 | 7.22 | 66.92 |
| B | 112 | 18.56 | 0.93 | [0.64; 1.50] | 0.91 | [0.63; 1.47] | 93.17 | 92.47 | 16.98 | 69.13 |
| C | 375 | 47.88 | 0.99 | [0.80; 1.23] | 0.98 | [0.78; 1.21] | 76.65 | 77.98 | 50.86 | 63.47 |
| D | 199 | 19.91 | 0.94 | [0.70; 1.34] | 0.95 | [0.71; 1.35] | 35.43 | 41.72 | 17.08 | 57.09 |
| E | 99 | 8.77 | 1.09 | [0.68; 2.07] | 1.14 | [0.71; 2.16] | 58.14 | 57.31 | 7.86 | 75.52 |
### faixa_idade

| valor | positivos | ocasioes_pct | pso_antes | ic_antes | pso_depois | ic_depois | recall_alta_antes_pct | recall_alta_depois_pct | fatia_alta_depois_pct |
|---|---|---|---|---|---|---|---|---|---|
| 10-19 | 199 | 25.88 | 0.89 | [0.68; 1.21] | 0.88 | [0.67; 1.20] | 78.21 | 82.33 | 23.91 |
| 20-24 | 97 | 10.62 | 0.96 | [0.69; 1.39] | 0.97 | [0.70; 1.42] | 76.47 | 75.86 | 10.46 |
| 25-29 | 103 | 12.49 | 1.01 | [0.77; 1.38] | 1.00 | [0.77; 1.35] | 73.28 | 73.67 | 12.30 |
| 30-34 | 82 | 9.33 | 1.10 | [0.73; 1.79] | 1.09 | [0.72; 1.77] | 74.12 | 77.20 | 10.05 |
| 35-39 | 100 | 11.67 | 0.85 | [0.63; 1.27] | 0.84 | [0.62; 1.25] | 67.95 | 70.01 | 10.02 |
| 40-44 | 62 | 6.59 | 1.41 | [0.99; 2.29] | 1.41 | [0.98; 2.28] | 70.75 | 72.70 | 9.84 |
| 45-59 | 125 | 15.67 | 1.11 | [0.81; 1.57] | 1.10 | [0.80; 1.54] | 58.24 | 58.11 | 17.81 |
| 60+ | 43 | 7.76 | 1.01 | [0.65; 1.88] | 0.99 | [0.65; 1.85] | 50.44 | 50.44 | 5.61 |
### sexo

| valor | positivos | ocasioes_pct | pso_antes | ic_antes | pso_depois | ic_depois | recall_alta_antes_pct | recall_alta_depois_pct | fatia_alta_depois_pct |
|---|---|---|---|---|---|---|---|---|---|
| homem | 391 | 53.93 | 0.94 | [0.78; 1.16] | 0.93 | [0.77; 1.15] | 74.02 | 74.78 | 50.06 |
| mulher | 420 | 46.07 | 1.09 | [0.89; 1.33] | 1.08 | [0.89; 1.32] | 65.41 | 68.19 | 49.94 |
### publico

| valor | positivos | ocasioes_pct | pso_antes | ic_antes | pso_depois | ic_depois | recall_alta_antes_pct | recall_alta_depois_pct | fatia_alta_depois_pct |
|---|---|---|---|---|---|---|---|---|---|
| 20-44 | 444 | 50.69 | 1.03 | [0.85; 1.28] | 1.03 | [0.85; 1.28] | 72.55 | 73.81 | 52.67 |
| outras | 367 | 49.31 | 0.98 | [0.79; 1.22] | 0.97 | [0.78; 1.21] | 67.49 | 69.62 | 47.33 |

## Como ler

- A correção muda o nível do score dos grupos com viés significativo; a
  ordem dentro de cada grupo não muda, porque o offset é o mesmo para todos
  do grupo.
- Classe de renda não é corrigida. Se a classe D tem `pso` perto de 1 e
  recall baixo na faixa alta, o score está certo e a ocasião é de fato
  menos frequente; a regra alternativa (topo 10% dentro da classe, última
  coluna) mostra quantas ocasiões da classe um alvo por classe alcançaria.
- Grupos com poucas ocasiões (menos de 30) têm IC largo; nenhuma conclusão
  sai deles.
