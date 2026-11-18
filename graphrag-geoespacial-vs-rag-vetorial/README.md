# GraphRAG geoespacial em comparação com RAG vetorial

Código do artigo [GraphRAG geoespacial em comparação com RAG vetorial](https://www.joaoataide.com/blog/graphrag-geoespacial-em-comparacao-com-rag-vetorial/).

O script compara um RAG vetorial e um GraphRAG sobre pontos de interesse ambientais do OpenStreetMap (Overpass API), num raio de 5 km em Recife, com o território dividido em células H3.

| Arquivo | O que faz |
|---|---|
| `graphrag_geoespacial.py` | Coleta os pontos no OSM com cache local, monta as zonas H3 e o grafo, e compara as respostas dos dois métodos. |
| `requirements.txt` | Bibliotecas usadas. |

## Como rodar

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sua-chave"     # PowerShell: $env:OPENAI_API_KEY="sua-chave"
python graphrag_geoespacial.py
```

O script lê a chave da variável de ambiente `OPENAI_API_KEY`, e nenhuma chave fica salva no arquivo. Os mapas (HTML) e as figuras (PNG) são gravados na pasta `saida/`, que não vai para o Git.

Os modelos usados são `text-embedding-3-small` (embeddings) e `gpt-4o-mini` (respostas), que têm custo por uso na API da OpenAI.
