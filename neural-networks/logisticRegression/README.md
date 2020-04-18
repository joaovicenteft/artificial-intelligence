# Rede neural utilizando regressão linear

## Objetivo: Classificar um conjunto de imagens como sendo de gatos ou não. Usa-se o algoritmo de regressão linear como base.


* Utiliza uma transformação linear do tipo Z = Wx + b, onde temos que W é uma matriz de pesos e b é um vetor de bias.
* Posterior a esta transformada, utiliza-se a função sigmoide de maneira a delimitar entre 0 e 1 os valores resultantes da operação.
* Faz-se forward e backward propagation com o objetivo de minimizar a função de perda.

### Executando


```
python3 LR_cat_classification.py lr_utils.py
```

