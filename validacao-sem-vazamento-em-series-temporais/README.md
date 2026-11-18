# Validação sem vazamento em séries temporais

Dados e código do [artigo de João Ataíde](https://www.joaoataide.com/blog/validacao-sem-vazamento-em-series-temporais/).

Um pixel Sentinel-2 L2A numa área classificada como soja no Mato Grosso (-13,3752; -54,4755), de 2019 a 2025. Os dados vêm do catálogo STAC do Planetary Computer. O estudo é demonstrativo e não estima a magnitude típica do vazamento em outras áreas.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `dados/pixel_soja_s2.csv` | Números digitais de B04 e B08, SCL, data e versão de processamento por cena |
| `dados/ndvi_decendial.csv` | Série arquivada, usada para conferir o pixel central do cubo |
| `dados/cubo_soja_s2.npz` | Janela de 32×32 pixels; pixel da série na posição `[16, 16]` |
| `protocolo.py` | Pré-processamento, entradas causais, avaliação sequencial, MASE e bootstrap |
| `experimento.py` | Resultados principais e figuras 1, 2, 4, 5 e 6 |
| `figura_cubo.py` | Composição e figura 3 do cubo, limitada a 2019–2024 |
| `analise_extra.py` | Sementes pareadas, sensibilidade do bootstrap, fases e relatório de reprodução |
| `verificar_radiometria.py` | Auditoria amostral de XMLs oficiais e pixels COG, com acesso à rede |
| `test_protocolo.py` | Regressões de isolamento temporal, previsão em lote e métricas |
| `baixar_pixel.py`, `baixar_cubo.py` | Nova extração do catálogo, opcional |
| `resultados/` | Tabelas e relatórios produzidos com os dados arquivados |

## Como reproduzir

Na pasta deste artigo, crie um ambiente Python e instale as versões usadas:

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_protocolo
python experimento.py
python figura_cubo.py
python analise_extra.py
```

Para conferir também os trechos e as saídas do artigo no projeto do site:

```bash
python verificar_post.py caminho/para/validacao-sem-vazamento-em-series-temporais.md
```

Esse verificador executa os trechos locais do Markdown informado. O trecho de download é apenas compilado, e a contagem de cenas é conferida contra o CSV arquivado.

Essas análises usam apenas os arquivos locais. As figuras PNG são gravadas em `fig/`, que fica fora do Git. Em ambiente sem interface gráfica, defina `MPLBACKEND=Agg` antes de rodar. A importação dos módulos não executa ajustes, figuras nem avaliações.

Para repetir a conferência radiométrica, que depende da rede:

```bash
python verificar_radiometria.py
```

O relatório `resultados/radiometria.json` compara uma cena de cada uma das sete versões de processamento presentes no arquivo com os XMLs dos produtos e os pixels COG oficiais. As sete amostras coincidiram com os dados arquivados e com a conversão radiométrica aplicada. Isso não substitui uma auditoria de todas as cenas. Os endereços de metadados registrados não incluem tokens de acesso.

Para baixar os dados novamente:

```bash
python baixar_pixel.py dados/pixel_soja_s2.csv
python baixar_cubo.py dados/cubo_soja_s2.npz
```

O catálogo pode mudar com reprocessamentos. Uma extração nova não é necessariamente equivalente aos arquivos arquivados e exige repetir a conferência radiométrica e os resultados.

## Protocolo e limites

O treino usa 2019–2022; a validação, 2023–2024; o teste final, 2025. Antes de construir as entradas, `run_development` corta a série em 2024. Até os protocolos deliberadamente incorretos recebem apenas esse período. As duas condições do KFold usam os mesmos alvos e as mesmas dobras em cada semente. A divisão cronológica usa os mesmos alvos elegíveis de treino nas condições com e sem entradas causais.

As figuras exploratórias, as contagens de cenas e as correlações espaciais usam somente 2019–2024: 216 decêndios, 150 observados, maior lacuna de seis decêndios. As correlações dos 1.024 pixels com o central têm mediana 0,9702 e fração 0,6162 acima de 0,9. Esses pixels não são novas unidades de avaliação do modelo.

A avaliação sequencial ajusta o Random Forest uma vez antes do período avaliado. Cada entrada usa apenas observações anteriores à origem; os parâmetros ficam fixos durante a avaliação. A previsão em lote das entradas causais equivale às chamadas separadas por origem. A hipótese operacional é que o composto do decêndio anterior esteja disponível na abertura do próximo; atrasos de publicação e disponibilidade histórica de versões reprocessadas não são simulados.

A MASE é adaptada à observabilidade irregular: o numerador é o MAE nos alvos observados e o denominador é o MAE da persistência nos pares válidos do treino original. Persistência significa repetir a última observação disponível, não necessariamente o decêndio anterior. A escala permanece fixa, inclusive no teste. MASE abaixo de 1 compara com o erro de referência no treino; a superioridade no período avaliado exige comparar modelo e persistência nesse mesmo período. Ausência de previsão em alvo observado gera erro, em vez de ser descartada silenciosamente.

O bootstrap pareado reamostra blocos da grade completa, mantendo suas lacunas, e seleciona os alvos observados depois. Usa 5.000 reamostragens, semente 0 e blocos de 3, 6 e 12 decêndios. Os modelos não são reajustados em cada reamostragem. A proporção de diferenças negativas é uma estatística das reamostragens, não uma probabilidade calibrada de superioridade.

As sementes principais são 42 para o exemplo e 0 a 9 para a sensibilidade. Nesta última, variam tanto o modelo quanto as dobras do KFold; as duas construções de entradas permanecem pareadas. Os arquivos de proveniência registram versões, escala, linhas elegíveis e hashes dos dados e do código.

## Resultados reproduzidos

| Validação, semente 42 | MASE |
|---|---:|
| Persistência | 0,952 |
| KFold com interpolação global | 0,578 |
| KFold com entradas causais | 1,118 |
| Cronológico com preenchimento prévio | 0,887 |
| Somente previsão causal | 1,243 |
| Treino e previsão causais | 1,148 |
| Treino e previsão causais com idade da observação | 1,070 |

A seleção entre persistência e os dois modelos causais é calculada antes dos resultados de teste e mantém a persistência. Em 2025, ela apresenta MASE 0,910; o RF com idade, reajustado com 2019–2024 e avaliado descritivamente, apresenta 0,874. Os intervalos de diferença das avaliações causais incluem zero nos três comprimentos de bloco. O intervalo do KFold com interpolação global inclui zero quando o bloco sobe para doze decêndios, apesar da vantagem pontual.

Esta revisão verifica o isolamento computacional: substituir todos os valores e as lacunas de 2025 não altera as previsões nem a escala do desenvolvimento. Ela não comprova que o teste permaneceu sem inspeção na elaboração original, já publicada. As comparações adicionais em 2025 são descritivas e não revertem a seleção. Uma avaliação prospectiva independente requer outro período ainda não consultado e decisões registradas antes de sua abertura.

Os erros usam 46 alvos observados na validação e 27 no teste. Não medem diretamente o desempenho durante os períodos inteiramente encobertos. A comparação entre protocolos também muda a composição temporal do treino, portanto não representa uma decomposição causal exata do otimismo.
