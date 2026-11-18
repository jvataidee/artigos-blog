# Embeddings espaciais com H3 para RAG geográfico

Código do artigo [Além da Latitude e Longitude: Criando Embeddings Espaciais para RAG](https://www.joaoataide.com/blog/transformando-coordenadas-em-embeddings-geograficos-com-h3/).

O script baixa pontos de interesse do OpenStreetMap (Overpass API) numa caixa em Recife, converte cada ponto em células H3 de três resoluções (7, 9 e 11) e nos vizinhos próximos, transforma esses tokens em vetores de dimensão fixa com *hashing trick*, e compara a busca só por localização com uma busca híbrida que soma o texto (modelo `all-MiniLM-L6-v2`, do `sentence-transformers`).

| Arquivo | O que faz |
|---|---|
| `embeddings_h3.py` | Coleta os pontos, gera as assinaturas H3, monta os vetores e roda as buscas geográfica e híbrida. |
| `requirements.txt` | Bibliotecas usadas. |

## Como rodar

```bash
pip install -r requirements.txt
python embeddings_h3.py
```

Os mapas (HTML) e as figuras (PNG) são gravados na pasta `saida/`, que não vai para o Git. O script não usa chave de API.
