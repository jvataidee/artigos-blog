"""Embeddings espaciais com H3 para RAG geográfico.

Script do artigo https://www.joaoataide.com/blog/transformando-coordenadas-em-embeddings-geograficos-com-h3/
Uso: `pip install -r requirements.txt` e `python embeddings_h3.py`.
Mapas e figuras são gravados na pasta saida/.
"""

from pathlib import Path

import matplotlib.pyplot as plt

OUT_DIR = Path("saida")
OUT_DIR.mkdir(exist_ok=True)
_contador = {"mapa": 0, "fig": 0}


def display(obj):
    """Substitui o display do notebook: mapas viram HTML em saida/, o resto vai para o terminal."""
    if hasattr(obj, "save"):
        _contador["mapa"] += 1
        obj.save(str(OUT_DIR / f"mapa_{_contador['mapa']:02d}.html"))
    elif obj is not None:
        print(obj)


def salvar_fig():
    _contador["fig"] += 1
    plt.savefig(OUT_DIR / f"fig_{_contador['fig']:02d}.png", dpi=150, bbox_inches="tight")
    plt.close()


# # **Geo-Embeddings Multi-Escala com H3 para Recuperação Espacial e Integração com RAG/LLM**

# Recife
SOUTH, WEST, NORTH, EAST = -8.15, -35.02, -7.95, -34.85

import requests, pandas as pd

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

AMENITIES = ["hospital", "school", "pharmacy", "police", "shelter"]

query = f"""
[out:json][timeout:60];
(
  node["amenity"~"^({'|'.join(AMENITIES)})$"]({SOUTH},{WEST},{NORTH},{EAST});
  way["amenity"~"^({'|'.join(AMENITIES)})$"]({SOUTH},{WEST},{NORTH},{EAST});
  relation["amenity"~"^({'|'.join(AMENITIES)})$"]({SOUTH},{WEST},{NORTH},{EAST});
);
out center tags;
"""

resp = requests.post(OVERPASS_URL, data={"data": query})
resp.raise_for_status()
data = resp.json()
display(len(data["elements"]))

rows = []
for el in data["elements"]:
    tags = el.get("tags", {})
    amenity = tags.get("amenity")
    name = tags.get("name", None)

    if el["type"] == "node":
        lat, lon = el.get("lat"), el.get("lon")
    else:
        center = el.get("center", {})
        lat, lon = center.get("lat"), center.get("lon")

    if (lat is None) or (lon is None) or (amenity is None):
        continue

    rows.append({
        "osm_type": el["type"],
        "osm_id": el["id"],
        "name": name,
        "category": amenity,
        "lat": float(lat),
        "lon": float(lon),
    })

df = pd.DataFrame(rows).drop_duplicates(subset=["osm_type","osm_id"])
display(df.head(), df["category"].value_counts())

display(df.head())

display(df["category"].value_counts())

import h3

def add_h3(df: pd.DataFrame, res: int) -> pd.DataFrame:
    out = df.copy()
    out[f"h3_r{res}"] = [
        h3.latlng_to_cell(lat, lon, res) for lat, lon in zip(out["lat"], out["lon"])
    ]
    return out

df = add_h3(df, 7)
df = add_h3(df, 9)
df = add_h3(df, 11)
display(df[[ "lat","lon","h3_r7","h3_r9","h3_r11" ]].head())

# 1. Instale a biblioteca para JavaScript

# 2. Crie visualização com Folium
import folium

m = folium.Map(location=[-8.05, -34.9], zoom_start=13)

# Adicionar pontos
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=5,
        popup=f"{row['name']}<br>{row['category']}",
        color={'hospital':'red', 'school':'blue', 'pharmacy':'green',
               'police':'orange', 'shelter':'purple'}[row['category']],
        fill=True
    ).add_to(m)

# Adicionar hexágonos H3
for h3_cell in df['h3_r9'].unique():
    boundary = h3.cell_to_boundary(h3_cell)
    folium.Polygon(
        locations=boundary,
        color='blue',
        weight=2,
        fill=True,
        fillOpacity=0.1
    ).add_to(m)

display(m.save('recife_h3.html'))

display(m)

# ### 3) Assinatura multi-escala

RESOLUTIONS = (7, 9, 11)

def spatial_signature(lat: float, lon: float, resolutions=RESOLUTIONS):
    return [h3.latlng_to_cell(lat, lon, r) for r in resolutions]

df["sig"] = [spatial_signature(lat, lon) for lat, lon in zip(df["lat"], df["lon"])]
display(df["sig"].head())

# 3.2 Topologia com vizinhos k-ring

def sig_with_neighbors(sig, k=1):
    # inclui o token e seus vizinhos em r9 (exemplo)
    base = sig[1]  # r9
    nbrs = list(h3.grid_disk(base, k))
    return sig + nbrs

df["sig_k1"] = [sig_with_neighbors(s, k=1) for s in df["sig"]]



display(df["sig_k1"].head())

# ### 4) Geo-Embedding vetorial (barato, determinístico e escalável)

import numpy as np, hashlib

def hash_to_idx(token: str, dim: int) -> int:
    h = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(h, 16) % dim

def tokens_to_vec(tokens, dim=1024):
    v = np.zeros(dim, dtype=np.float32)
    for t in tokens:
        v[hash_to_idx(t, dim)] += 1.0
    n = np.linalg.norm(v)
    return v if n == 0 else (v / n)

DIM = 1024
V_geo = np.vstack([tokens_to_vec(toks, dim=DIM) for toks in df["sig_k1"].values])
display(V_geo.shape)

# ### 5) Baseline vs Geo-Embedding (retrieval “perto”)

from sklearn.metrics.pairwise import cosine_similarity

def retrieve_geo(query_lat, query_lon, topk=10):
    q_tokens = sig_with_neighbors(spatial_signature(query_lat, query_lon), k=1)
    q_vec = tokens_to_vec(q_tokens, dim=DIM).reshape(1, -1)
    sims = cosine_similarity(q_vec, V_geo).ravel()
    top = np.argsort(-sims)[:topk]
    return df.iloc[top][["name","category","lat","lon"]].assign(score=sims[top])

display(retrieve_geo(df.iloc[0].lat, df.iloc[0].lon, topk=8))

# ### 6) (Upgrade) Híbrido Geo + Texto para RAG/LLM

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = (df["name"].fillna("") + " " + df["category"].fillna("")).tolist()
V_txt = model.encode(texts, normalize_embeddings=True)

V_hybrid = np.hstack([V_txt, V_geo])  # simples e funciona
display(V_hybrid.shape)

def retrieve_hybrid(query_text, query_lat, query_lon, topk=10):
    q_txt = model.encode([query_text], normalize_embeddings=True)
    q_geo = tokens_to_vec(
        sig_with_neighbors(spatial_signature(query_lat, query_lon), k=1),
        dim=DIM
    ).reshape(1, -1)

    q = np.hstack([q_txt, q_geo])
    sims = cosine_similarity(q, V_hybrid).ravel()
    top = np.argsort(-sims)[:topk]
    return df.iloc[top][["name","category","lat","lon"]].assign(score=sims[top])

display(retrieve_hybrid("hospital", df.iloc[0].lat, df.iloc[0].lon, topk=8))

import numpy as np
import pandas as pd

def _make_query_vector_geo(query_lat: float, query_lon: float) -> np.ndarray:
    q_tokens = sig_with_neighbors(spatial_signature(query_lat, query_lon), k=1)
    q_vec = tokens_to_vec(q_tokens, dim=DIM).reshape(1, -1)
    return q_vec

def _make_query_vector_hybrid(query_text: str, query_lat: float, query_lon: float) -> np.ndarray:
    q_txt = model.encode([query_text], normalize_embeddings=True)
    q_geo = _make_query_vector_geo(query_lat, query_lon)
    q = np.hstack([q_txt, q_geo])
    return q

def retrieve_geo_cli(query_lat: float, query_lon: float, topk: int = 10) -> pd.DataFrame:
    q_vec = _make_query_vector_geo(query_lat, query_lon)
    sims = cosine_similarity(q_vec, V_geo).ravel()
    top = np.argsort(-sims)[:topk]
    out = df.iloc[top][["name", "category", "lat", "lon"]].copy()
    out["score"] = sims[top]
    return out.reset_index(drop=True)

def retrieve_hybrid_cli(query_text: str, query_lat: float, query_lon: float, topk: int = 10) -> pd.DataFrame:
    q = _make_query_vector_hybrid(query_text, query_lat, query_lon)
    sims = cosine_similarity(q, V_hybrid).ravel()
    top = np.argsort(-sims)[:topk]
    out = df.iloc[top][["name", "category", "lat", "lon"]].copy()
    out["score"] = sims[top]
    return out.reset_index(drop=True)

def plot_retrieval(query_lat: float, query_lon: float, results: pd.DataFrame, title: str = ""):
    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # Todos os pontos (contexto)
    ax.scatter(df["lon"].values, df["lat"].values, s=10)

    # Top-k retornado
    ax.scatter(results["lon"].values, results["lat"].values, s=60)

    # Ponto da query
    ax.scatter([query_lon], [query_lat], s=160, marker="X")

    # Linhas da query para cada resultado
    for _, r in results.iterrows():
        ax.plot([query_lon, r["lon"]], [query_lat, r["lat"]], linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True)
    plt.tight_layout()
    salvar_fig()

def _safe_float(prompt: str) -> float:
    while True:
        s = input(prompt).strip().replace(",", ".")
        try:
            return float(s)
        except ValueError:
            print("Valor inválido. Digite um número (ex: -8.05).")

def _safe_int(prompt: str, default: int = 10) -> int:
    s = input(prompt).strip()
    if s == "":
        return default
    try:
        return int(s)
    except ValueError:
        print(f"Valor inválido. Usando default={default}.")
        return default

def interactive_geo_embeddings():
    print("\n=== Geo-Embeddings CLI (H3 + Retrieval) ===")
    print("Comandos: ENTER para usar defaults; 'exit' para sair.\n")

    # defaults: centro aproximado do seu bbox Recife
    default_lat = (SOUTH + NORTH) / 2
    default_lon = (WEST + EAST) / 2

    while True:
        mode = input("Modo [geo/hybrid] (default=hybrid): ").strip().lower()
        if mode == "":
            mode = "hybrid"
        if mode == "exit":
            break
        if mode not in ("geo", "hybrid"):
            print("Modo inválido. Use 'geo' ou 'hybrid'.\n")
            continue

        query_text = ""
        if mode == "hybrid":
            query_text = input("Query de texto (ex: hospital, school, pharmacy): ").strip()
            if query_text.lower() == "exit":
                break
            if query_text == "":
                query_text = "hospital"

        lat_in = input(f"Latitude (default={default_lat:.6f}): ").strip()
        if lat_in.lower() == "exit":
            break
        lon_in = input(f"Longitude (default={default_lon:.6f}): ").strip()
        if lon_in.lower() == "exit":
            break

        query_lat = float(lat_in.replace(",", ".")) if lat_in else float(default_lat)
        query_lon = float(lon_in.replace(",", ".")) if lon_in else float(default_lon)

        topk = _safe_int("TopK (default=10): ", default=10)

        if mode == "geo":
            res = retrieve_geo_cli(query_lat, query_lon, topk=topk)
            print("\n--- TOPK (GEO) ---")
            display(res)  # no Colab/Jupyter
            plot_retrieval(
                query_lat, query_lon, res,
                title=f"Retrieval GEO | lat={query_lat:.6f}, lon={query_lon:.6f} | topk={topk}"
            )
        else:
            res = retrieve_hybrid_cli(query_text, query_lat, query_lon, topk=topk)
            print("\n--- TOPK (HYBRID: TEXTO + GEO) ---")
            display(res)  # no Colab/Jupyter
            plot_retrieval(
                query_lat, query_lon, res,
                title=f"Retrieval HYBRID | '{query_text}' | lat={query_lat:.6f}, lon={query_lon:.6f} | topk={topk}"
            )

        print("\n(rode novamente para outra query; digite 'exit' para sair)\n")

display(interactive_geo_embeddings())

import numpy as np

# -------------------------
# 1) GEO QUERY + RETRIEVAL
# -------------------------
def retrieve_geo(query_lat, query_lon, topk=10):
    q_tokens = sig_with_neighbors(spatial_signature(query_lat, query_lon), k=1)
    q_vec = tokens_to_vec(q_tokens, dim=DIM).reshape(1, -1)

    sims = cosine_similarity(q_vec, V_geo).ravel()
    top = np.argsort(-sims)[:topk]

    out = df.iloc[top][["name", "category", "lat", "lon"]].copy()
    out["score"] = sims[top]
    return out.reset_index(drop=True)

# -------------------------
# 2) PLOT GEO
# -------------------------

def plot_retrieval(query_lat, query_lon, results, title="", margin=0.01):
    """
    Plot com zoom automático na região dos resultados

    Args:
        query_lat, query_lon: coordenadas da query
        results: DataFrame com resultados (deve ter 'lat', 'lon')
        title: título do gráfico
        margin: margem adicional em graus (default: 0.01 ≈ 1km)
    """
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # Calcular bounds da região de interesse
    all_lats = np.concatenate([results["lat"].values, [query_lat]])
    all_lons = np.concatenate([results["lon"].values, [query_lon]])

    lat_min, lat_max = all_lats.min() - margin, all_lats.max() + margin
    lon_min, lon_max = all_lons.min() - margin, all_lons.max() + margin

    # Filtrar pontos de contexto que estão na região visível
    context_mask = (
        (df["lat"] >= lat_min) & (df["lat"] <= lat_max) &
        (df["lon"] >= lon_min) & (df["lon"] <= lon_max)
    )
    context_df = df[context_mask]

    # Pontos de contexto (cinza claro, pequenos)
    ax.scatter(
        context_df["lon"].values,
        context_df["lat"].values,
        s=20,
        c='lightgray',
        alpha=0.5,
        label='Outros pontos na região',
        zorder=1
    )

    # Top-k retornado (azul, médios)
    ax.scatter(
        results["lon"].values,
        results["lat"].values,
        s=100,
        c='royalblue',
        edgecolors='darkblue',
        linewidths=1.5,
        label='Resultados (top-k)',
        zorder=3
    )

    # Ponto da query (vermelho, grande)
    ax.scatter(
        [query_lon],
        [query_lat],
        s=250,
        c='red',
        marker='*',
        edgecolors='darkred',
        linewidths=2,
        label='Query',
        zorder=4
    )

    # Linhas da query para cada resultado
    for _, r in results.iterrows():
        ax.plot(
            [query_lon, r["lon"]],
            [query_lat, r["lat"]],
            'b-',
            alpha=0.3,
            linewidth=1,
            zorder=2
        )

    # Definir limites do plot
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    # Adicionar anotações para os top 3
    for i, row in results.head(3).iterrows():
        ax.annotate(
            f"{i+1}. {row['name'][:20] if pd.notna(row['name']) else row['category']}\n(score: {row['score']:.3f})",
            xy=(row['lon'], row['lat']),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1)
        )

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Aspect ratio igual para lat/lon
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    salvar_fig()


# -------------------------
# 3) RODAR UMA QUERY GEO "DE ANTES"
# -------------------------
query_lat = float(df.iloc[0].lat)
query_lon = float(df.iloc[0].lon)

res_geo = retrieve_geo(query_lat, query_lon, topk=10)
display(res_geo)

display(plot_retrieval(
    query_lat, query_lon, res_geo,
    title=f"Retrieval GEO | lat={query_lat:.6f}, lon={query_lon:.6f} | topk=10"
))

# -------------------------
# 1) HYBRID QUERY + RETRIEVAL
# -------------------------
def retrieve_hybrid(query_text, query_lat, query_lon, topk=10):
    # texto
    q_txt = model.encode([query_text], normalize_embeddings=True)

    # geo
    q_tokens = sig_with_neighbors(spatial_signature(query_lat, query_lon), k=1)
    q_geo = tokens_to_vec(q_tokens, dim=DIM).reshape(1, -1)

    # concat
    q = np.hstack([q_txt, q_geo])

    sims = cosine_similarity(q, V_hybrid).ravel()
    top = np.argsort(-sims)[:topk]

    out = df.iloc[top][["name", "category", "lat", "lon"]].copy()
    out["score"] = sims[top]
    return out.reset_index(drop=True)

# -------------------------
# 2) RODAR UMA QUERY HÍBRIDA "DE ANTES"
# -------------------------
query_text = "hospital"

res_hybrid = retrieve_hybrid(query_text, query_lat, query_lon, topk=10)
display(res_hybrid)

display(plot_retrieval(
    query_lat, query_lon, res_hybrid,
    title=f"Retrieval HYBRID | '{query_text}' | lat={query_lat:.6f}, lon={query_lon:.6f} | topk=10"
))

