# CleanVision em recortes de radar (ICEYE)

Código e resultados do artigo "Quais imagens de radar não deveriam treinar o modelo?", do blog [joaoataide.com](https://joaoataide.com).

Este é um trabalho reorganizado, e não refeito. Os números (280 imagens, 36 de pouca informação, 7 escuras, 3 desfocadas) são os do trabalho original.

## Conteúdo

| Arquivo | Para que serve |
|---|---|
| `limpar_imagens.py` | roda a CleanVision, relata os problemas e, com `--descartar`, move as imagens sinalizadas |
| `mapear_rotulos.py` | gera o `map.txt` de imagem e rótulo para o treino no ArcGIS |

## Limites

- As imagens são da ICEYE e não são públicas, então o repositório não traz os dados.
- Os scripts foram reorganizados a partir do código do trabalho original e não foram executados de novo. O `limpar_imagens.py` move as imagens sinalizadas em vez de apagá-las, o que o código original fazia.
- A CleanVision foi usada com os limiares padrão.

## Como rodar

```bash
pip install -r requirements.txt
python limpar_imagens.py --imagens dados/imagens
```
