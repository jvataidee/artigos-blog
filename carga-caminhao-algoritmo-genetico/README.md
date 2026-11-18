# Carga de caminhão com algoritmo genético

Código do [artigo de João Ataíde](https://www.joaoataide.com/blog/otimizacao-de-cargas-com-algoritmo-genetico/).

500 produtos reais do *Brazilian E-Commerce Public Dataset by Olist* (dimensões, peso e preço médio de venda) disputam um baú de 2,4 m x 1,6 m x 1,6 m e 700 kg de carga nominal (premissas do estudo), com regras de caminhão real: folga de 10 cm em cada dimensão, uso de no máximo 90% do peso (630 kg) e caixas acima de 10 kg só no chão. O indivíduo é uma ordem de carregamento, e um decodificador de prateleiras diz o que coube. Compara um algoritmo genético (ordem, cruzamento OX, mutação por troca, elitismo), o genético semeado com a ordem gulosa, a busca aleatória de ordens e três ordens gulosas.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `baixar_dados.py` | Baixa as tabelas de produtos e de itens de pedido (cópia pública do dataset da Olist) para `dados/` |
| `carga.py` | O decodificador, o algoritmo genético, a busca aleatória e as ordens gulosas, com 10 sementes em paralelo. Grava `resultados/` |

## Como rodar

```bash
pip install -r requirements.txt
python baixar_dados.py
python carga.py            # cerca de 7 minutos com 10 sementes
python carga.py --sementes 2 --geracoes 20 --saida teste    # teste rápido
```

## Limites

- O baú, a folga, o limite de 90% do peso e a regra das caixas pesadas são premissas, e não as medidas de um veículo.
- As caixas não giram, e o decodificador de prateleiras aproveita de 57% a 70% do volume.
- O dataset é CC BY-NC-SA 4.0.
