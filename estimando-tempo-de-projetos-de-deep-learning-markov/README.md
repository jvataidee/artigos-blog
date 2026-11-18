# Estimando o prazo de projetos de deep learning com cadeias de Markov

Dados e código do [artigo de João Ataíde](https://www.joaoataide.com/blog/estimando-tempo-de-projetos-de-deep-learning-usando-simulacao-e-cadeias-de-markov/).

Um projeto de deep learning passa por Amostra, Treino, Validação e Produção, e volta de etapa com frequência. Aqui ele é uma cadeia de Markov com Produção como estado final, em que cada transição dura uma semana. O estudo calcula o prazo de forma exata (matriz fundamental e potências da matriz), confere o resultado por simulação e compara cenários em que o retrabalho do Treino diminui. As probabilidades vêm de julgamento de cientistas de dados, e não de medição, e podem ser trocadas pelas da sua equipe em `markov.py`.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `markov.py` | A matriz de transição, o cálculo exato, a simulação e os cenários. Grava `resultados/` |

## Como reproduzir

Com Python 3.10 ou superior, na pasta deste artigo:

```bash
python -m pip install -r requirements.txt
python markov.py                 # 10 mil simulações, alguns segundos
```

## Notas

- **Simulação:** usa semente fixa (42), então os números se repetem. Com outra semente, a média muda na primeira casa decimal. O cálculo exato não depende de semente.
