# Classificando variantes do tanque T-72 em imagens de radar

Dados e código do [artigo de João Ataíde](https://www.joaoataide.com/blog/classificando-imagens-de-radar-de-variacoes-do-tanque-t72/).

Classificação de 5.421 imagens de radar de abertura sintética (SAR) do conjunto MSTAR / IU T-72 variants, com oito variantes do tanque (A04, A05, A07, A10, A32, A62, A63 e A64), usando um ResNet18 e o módulo `arcgis.learn` do ArcGIS Pro. A validação (30% das imagens, estratificada) chega a 97,4% de acurácia, e os erros se concentram entre A04 e A32 e entre A62 e A64.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `preparar_dados.py` | Grava o `map.txt` e atualiza as classes nos arquivos do modelo do ArcGIS |
| `treinar.py` | Separação treino e validação, ResNet18, `lr_find` e 100 épocas com checkpoint |
| `prever.py` | Lê a classe real nos metadados e classifica imagens fora do treino |

## Como reproduzir

Os scripts dependem do ArcGIS Pro com `arcgis.learn` e do conjunto MSTAR / IU T-72 variants, disponível no [SDMS](https://www.sdms.afrl.af.mil/index.php). Os dados não estão neste repositório.

```bash
python preparar_dados.py CAMINHO/T72Variants PASTA_COM_OS_ARQUIVOS_ESRI
python treinar.py CAMINHO/T72Variants
python prever.py CAMINHO/DO/MODELO CAMINHO/T72Variants/teste
```

## Notas

- **Origem dos resultados:** `epocas.csv` vem da tabela de treino registrada na execução do notebook, e `matriz_confusao.csv` foi transcrita da figura da matriz de confusão dessa execução. A soma das linhas dá 1.627 imagens e a diagonal dá 1.585, ou seja, 97,42%, igual à acurácia da época 99. Os scripts deste repositório foram reorganizados a partir do notebook e **não foram reexecutados**, e podem precisar de ajuste de caminhos.
- **Sem teste independente:** a validação também escolheu a melhor época (checkpoint) e é uma fatia aleatória do mesmo conjunto, então a acurácia é otimista. As duas imagens de `prever.py` são uma demonstração.
