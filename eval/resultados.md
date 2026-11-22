# Resultados da avaliação

**17 de 17 casos corretos.**

| Caso | Resultado | Obtido | Esperado |
|---|---|---|---|
| efeito por grau recuperado (±0,5) | ok | 9.22 | 9.0 |
| efeito da chuva recuperado (±8) | ok | -33.53 | -35.0 |
| efeito do fim de semana recuperado (±6) | ok | 41.42 | 40.0 |
| multivariado com R² >= 0,85 no holdout | ok | 0.87 | >= 0.85 |
| multivariado erra menos que só temperatura | ok | 12.76 | < 21.35 |
| previsão nunca negativa (10 °C) | ok | 6.4 | >= 0 |
| faltar caro => produzir mais | ok | 1526 | > 1323 |
| roteamento: Quantos sorvetes vou vender amanhã com 31 graus? | ok | ['prever_vendas'] | prever_vendas |
| roteamento: Previsão de vendas para sábado de 27 °C com chuva | ok | ['prever_vendas'] | prever_vendas |
| roteamento: Quanto devo produzir na próxima semana? | ok | ['planejar_producao'] | planejar_producao |
| roteamento: Monte o plano de produção para 30, 31, 29, 26, 24, 28 e 33 graus | ok | ['planejar_producao'] | planejar_producao |
| roteamento: Qual o efeito de cada grau a mais nas vendas? | ok | ['explicar_modelo'] | explicar_modelo |
| roteamento: Quanto a chuva derruba as vendas? | ok | ['explicar_modelo'] | explicar_modelo |
| roteamento: Compare os modelos: qual tem o melhor R²? | ok | ['comparar_modelos'] | comparar_modelos |
| roteamento: Registre o experimento no MLflow | ok | ['registrar_experimento'] | registrar_experimento |
| roteamento: Confirme a ordem de produção para a fábrica | ok | ['confirmar_producao'] | confirmar_producao |
| número inventado é barrado | ok | revisar | revisar |

## Modelos (holdout com os últimos 20% dos dias)

| Modelo | R² | MAE |
|---|---|---|
| so_temperatura | 0.581 | 21.35 |
| multivariado | 0.870 | 12.76 |
