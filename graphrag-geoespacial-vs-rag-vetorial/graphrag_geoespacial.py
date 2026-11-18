"""GraphRAG geoespacial em comparação com RAG vetorial.

Script do artigo https://www.joaoataide.com/blog/graphrag-geoespacial-em-comparacao-com-rag-vetorial/
Uso: defina a variável de ambiente OPENAI_API_KEY e rode `python graphrag_geoespacial.py`.
Mapas e figuras são gravados na pasta saida/.
"""

import os, json, time, math
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import h3
import networkx as nx
from tqdm import tqdm

import folium
from folium.plugins import MarkerCluster

from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

from openai import OpenAI

if not os.environ.get("OPENAI_API_KEY"):
    raise SystemExit("Defina a variável de ambiente OPENAI_API_KEY antes de rodar.")
client = OpenAI()

CENTER_LAT, CENTER_LON = -8.0578, -34.8829
RADIUS_M = 5000

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

AMENITIES = ["waste_disposal", "recycling", "fountain", "park", "water_point"]

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

H3_RES = 7
ZONE_RING = 0
MAX_NEAR_M = 500
MAX_NEAR_PER_POI  = 10

ROAD_HIGHWAY_REGEX = "^(primary|secondary|tertiary|residential)$"
ROAD_SAMPLE_STEP = 6
MAX_CONNECTS_PER_ROAD = 20

EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

CACHE_DIR = Path("./data_cache")
CACHE_DIR.mkdir(exist_ok=True)

ICON_MAP = {
    "waste_disposal": "trash",
    "recycling":      "recycle",
    "fountain":       "tint",
    "park":           "leaf",
    "water_point":    "tint",
}

def cache_path(name):
    return CACHE_DIR / f"{name}.json"

def overpass(query, cache_name, url=OVERPASS_URL, timeout=180, force=False):
    p = cache_path(cache_name)
    if p.exists() and not force:
        return json.loads(p.read_text(encoding="utf-8"))

    headers = {"User-Agent": "graphrag-ambiental/1.0"}
    last_err = None
    for attempt in range(1, 6):
        try:
            r = requests.post(url, data={"data": query}, headers=headers, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            p.write_text(json.dumps(data), encoding="utf-8")
            return data
        except Exception as e:
            last_err = e
            time.sleep(2 * attempt)
    raise RuntimeError(f"Overpass falhou: {last_err}")

# ## FETCH OSM


amenities_regex = "|".join(AMENITIES)

q_pois = f"""
[out:json][timeout:120];
(
  node(around:{RADIUS_M},{CENTER_LAT},{CENTER_LON})["amenity"~"^({amenities_regex})$"];
  way (around:{RADIUS_M},{CENTER_LAT},{CENTER_LON})["amenity"~"^({amenities_regex})$"];
  node(around:{RADIUS_M},{CENTER_LAT},{CENTER_LON})["leisure"="park"];
  way (around:{RADIUS_M},{CENTER_LAT},{CENTER_LON})["leisure"="park"];
  node(around:{RADIUS_M},{CENTER_LAT},{CENTER_LON})["amenity"="drinking_water"];
  way (around:{RADIUS_M},{CENTER_LAT},{CENTER_LON})["amenity"="drinking_water"];
);
out tags center;
"""

q_roads = f"""
[out:json][timeout:120];
way(around:{RADIUS_M},{CENTER_LAT},{CENTER_LON})["highway"~"{ROAD_HIGHWAY_REGEX}"];
out tags geom;
"""

data_pois  = overpass(q_pois,  cache_name=f"pois_env_r{RADIUS_M}_{CENTER_LAT}_{CENTER_LON}", force=True)
data_roads = overpass(q_roads, cache_name=f"roads_r{RADIUS_M}_{CENTER_LAT}_{CENTER_LON}")

print(len(data_pois.get("elements", [])), len(data_roads.get("elements", [])))

# ### PARSE — POIs e Roads


def point_to_h3(lat, lon, res=H3_RES):
    return h3.latlng_to_cell(lat, lon, res)


AMENITY_REMAP = {
    "drinking_water": "water_point",
}

def extract_pois(elements):
    rows = []
    seen = set()
    for el in elements:
        tags    = el.get("tags", {})
        amenity = tags.get("amenity") or ("park" if tags.get("leisure") == "park" else None)
        if not amenity:
            continue
        amenity = AMENITY_REMAP.get(amenity, amenity)
        if amenity not in AMENITIES:
            continue

        if el["type"] == "node":
            lat, lon = el["lat"], el["lon"]
        else:
            c = el.get("center")
            if not c:
                continue
            lat, lon = c["lat"], c["lon"]

        osm_id = int(el["id"])
        if osm_id in seen:
            continue
        seen.add(osm_id)

        rows.append({
            "osm_type": el["type"],
            "osm_id":   osm_id,
            "amenity":  amenity,
            "name":     tags.get("name"),
            "lat":      float(lat),
            "lon":      float(lon),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["h3"] = df.apply(lambda r: point_to_h3(r["lat"], r["lon"]), axis=1)
    return df

def extract_roads(elements):
    rows = []
    for el in elements:
        if el.get("type") != "way":
            continue
        geom = el.get("geometry")
        if not geom or len(geom) < 2:
            continue
        mid = geom[len(geom)//2]
        tags = el.get("tags", {})
        rows.append({
            "way_id":  int(el["id"]),
            "highway": tags.get("highway"),
            "name":    tags.get("name"),
            "mid_lat": float(mid["lat"]),
            "mid_lon": float(mid["lon"]),
            "geom":    geom,
        })
    df = pd.DataFrame(rows).drop_duplicates(subset=["way_id"])
    df["h3"] = df.apply(lambda r: point_to_h3(r["mid_lat"], r["mid_lon"]), axis=1)
    return df

df_pois = extract_pois(data_pois["elements"])
df_roads = extract_roads(data_roads["elements"])
print(df_pois.shape, df_roads.shape)

# #### EDA rápida

counts = df_pois["amenity"].value_counts()
display(counts)

plt.figure(figsize=(7, 4))
plt.bar(counts.index, counts.values)
plt.title("POIs Ambientais por amenity")
plt.xticks(rotation=45)
plt.tight_layout()
salvar_fig()

# #### POIs no folium

m = folium.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=14, tiles="cartodbpositron")
cluster = MarkerCluster().add_to(m)

for r in df_pois.itertuples(index=False):
    folium.Marker(
        location=[r.lat, r.lon],
        popup=f"{r.amenity} | {r.name or 'sem nome'}",
        icon=folium.Icon(icon=ICON_MAP.get(r.amenity, "ok-sign"))
    ).add_to(cluster)

display(m)

# ### GRAFO


G = nx.Graph()

for r in df_pois.itertuples(index=False):
    pid = f"poi:{r.osm_id}"
    G.add_node(pid, ntype="POI", name=r.name, amenity=r.amenity,
               lat=r.lat, lon=r.lon, h3=r.h3)
    G.add_edge(pid, f"zone:{r.h3}",    etype="IN_ZONE")
    G.add_edge(pid, f"svc:{r.amenity}", etype="IS_A")

for r in df_roads.itertuples(index=False):
    rid = f"road:{r.way_id}"
    G.add_node(rid, ntype="ROAD", highway=r.highway, name=r.name,
               lat=r.mid_lat, lon=r.mid_lon, h3=r.h3)
    G.add_edge(rid, f"zone:{r.h3}", etype="IN_ZONE")

print(G.number_of_nodes(), G.number_of_edges())

# ### ARESTAS NEAR (POI-POI por distância)


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlon/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

bucket = {}
for r in df_pois.itertuples(index=False):
    bucket.setdefault(r.h3, []).append(r)

added = 0
for r in tqdm(df_pois.itertuples(index=False), total=len(df_pois)):
    src = f"poi:{r.osm_id}"
    candidates = []
    for c in [r.h3] + list(h3.grid_disk(r.h3, 1)):
        candidates.extend(bucket.get(c, []))
    scored = []
    for t in candidates:
        if t.osm_id == r.osm_id:
            continue
        d = haversine_m(r.lat, r.lon, t.lat, t.lon)
        if d <= MAX_NEAR_M:
            scored.append((d, t.osm_id))
    scored.sort(key=lambda x: x[0])
    for d, tid in scored[:MAX_NEAR_PER_POI]:
        G.add_edge(src, f"poi:{tid}", etype="NEAR", dist_m=float(d))
        added += 1

print(added, G.number_of_edges())

# ### ARESTAS CONNECTS (Roads que compartilham interseção)

def quantize(lat, lon, ndigits=5):
    return (round(lat, ndigits), round(lon, ndigits))

pt_to_roads = {}
for r in tqdm(df_roads.itertuples(index=False), total=len(df_roads)):
    for p in r.geom[::ROAD_SAMPLE_STEP]:
        key = quantize(p["lat"], p["lon"])
        pt_to_roads.setdefault(key, set()).add(r.way_id)

connects = 0
for key, roads in pt_to_roads.items():
    roads = list(roads)
    if len(roads) < 2:
        continue
    for i in range(min(len(roads), 10)):
        for j in range(i + 1, min(len(roads), 10)):
            G.add_edge(f"road:{roads[i]}", f"road:{roads[j]}", etype="CONNECTS")
            connects += 1

print(connects)

# ### DOCUMENTOS + EMBEDDINGS dos POIs

def poi_doc(row):
    name = row["name"] if pd.notnull(row["name"]) else "sem nome"
    return (
        f"Ativo de Infraestrutura Ambiental ID {row['osm_id']}: {name}. "
        f"Categoria={row['amenity']}. "
        f"Localização (lat={row['lat']:.6f}, lon={row['lon']:.6f}). "
        f"Zona territorial H3={row['h3']}."
    )

poi_docs = df_pois.apply(poi_doc, axis=1).tolist()
poi_ids  = df_pois["osm_id"].tolist()

def embed_texts(texts, model=EMBED_MODEL, batch_size=128):
    vecs = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        vecs.extend([d.embedding for d in resp.data])
    return np.array(vecs, dtype=np.float32)

poi_emb = embed_texts(poi_docs)
print(poi_emb.shape)



# ### SUMÁRIOS DE ZONA via LLM

def llm_text(prompt, model=LLM_MODEL):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

zone_stats = (
    df_pois.groupby(["h3", "amenity"])
    .size()
    .reset_index(name="n")
    .sort_values(["h3", "n"], ascending=[True, False])
)
zone_total = df_pois.groupby("h3").size().reset_index(name="total")

zone_card = {}
for h, grp in zone_stats.groupby("h3"):
    top_svc = ", ".join([f"{r.amenity}:{int(r.n)}" for r in grp.head(8).itertuples(index=False)])
    total   = int(zone_total.loc[zone_total["h3"] == h, "total"].iloc[0])
    zone_card[h] = (
        f"Zona territorial H3={h} (resolução={H3_RES}). "
        f"Total de ativos ambientais mapeados={total}. "
        f"Principais categorias de infraestrutura ambiental={top_svc}."
    )

dense_zones = zone_total.sort_values("total", ascending=False)["h3"].head(60).tolist()

def summarize_zone(h):
    prompt = (
        f"Gere um sumário executivo curto (2-4 frases) para apoio a decisão pública "
        f"sobre a infraestrutura ambiental desta zona territorial.\n"
        f"Baseie-se APENAS nos dados:\n{zone_card[h]}\n"
        f"Inclua:\n"
        f"(1) Perfil da infraestrutura ambiental local,\n"
        f"(2) Possível vocação territorial (ex.: lazer, gestão de resíduos, suporte hídrico),\n"
        f"(3) Indique que se trata de uma estimativa baseada em dados abertos."
    )
    return llm_text(prompt)

zone_summary = {}
for h in tqdm(dense_zones):
    zone_summary[h] = summarize_zone(h)

def zone_text(h):
    return zone_summary.get(h, zone_card.get(h, f"Zona H3={h}"))

# ### EMBEDDINGS DE ZONA

zone_list = list(zone_card.keys())
zone_docs = [zone_text(h) for h in zone_list]
zone_emb  = normalize(embed_texts(zone_docs))
print(len(zone_list), zone_emb.shape)

# ### QUERY SCHEMA + PARSE

QUERY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "services": {
            "type": "array",
            "items": {"type": "string", "enum": AMENITIES},
        },
        "near_pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "a":        {"type": "string", "enum": AMENITIES},
                    "b":        {"type": "string", "enum": AMENITIES},
                    "radius_m": {"type": "integer", "minimum": 50, "maximum": 5000},
                },
                "required": ["a", "b", "radius_m"],
            },
        },
        "place_hint":  {"type": "string"},
        "top_k_zones": {"type": "integer", "minimum": 1, "maximum": 25},
    },
    "required": ["services", "near_pairs", "place_hint", "top_k_zones"],
}

def parse_query(query):
    prompt = (
        "Extraia estrutura da pergunta no JSON abaixo.\n"
        "Não invente serviços fora do enum.\n\n"
        "REGRAS OBRIGATÓRIAS:\n"
        "- radius_m deve ser entre 50 e 5000. Se não especificado, use 500.\n"
        "- top_k_zones deve ser entre 1 e 25. Se não especificado, use 8.\n"
        "- Se a pergunta mencionar PROXIMIDADE, INTEGRAÇÃO, SINERGIA ou 'perto' "
        "entre dois tipos de serviço, SEMPRE preencha near_pairs com o par correspondente.\n"
        "  Exemplos de mapeamento:\n"
        "    'áreas verdes' ou 'parques' ou 'lazer'  → 'park'\n"
        "    'reciclagem'                             → 'recycling'\n"
        "    'ativos hídricos' ou 'água' ou 'fonte'  → 'water_point' ou 'fountain'\n"
        "    'descarte' ou 'lixo' ou 'resíduos'      → 'waste_disposal'\n"
        "- Se houver dois serviços identificados em contexto de proximidade, "
        "gere um near_pair com esses dois serviços.\n\n"
        f"Schema: {json.dumps(QUERY_SCHEMA)}\n"
        f"Pergunta: {query}\n"
        "Responda SOMENTE com JSON válido, sem markdown."
    )
    raw = llm_text(prompt)
    raw = raw.replace("```json", "").replace("```", "").strip()
    q = json.loads(raw)

    q["top_k_zones"] = max(1, min(25, int(q.get("top_k_zones") or 8)))
    for p in q.get("near_pairs", []):
        p["radius_m"] = max(50, min(5000, int(p.get("radius_m") or 500)))

    # ── Fallback: se services tem ≥2 itens e near_pairs ainda vazio ──────────
    svcs = q.get("services", [])
    if not q.get("near_pairs") and len(svcs) >= 2:
        q["near_pairs"] = [{"a": svcs[0], "b": svcs[1], "radius_m": 500}]

    return q

# ### PIPELINE GraphRAG


def vector_retrieve(query, k=12):
    q_emb = client.embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding
    q_emb = normalize(np.array(q_emb, dtype=np.float32).reshape(1, -1))
    sims  = cosine_similarity(q_emb, poi_emb)[0]
    idx   = np.argsort(-sims)[:k]
    return [(poi_ids[i], poi_docs[i], float(sims[i])) for i in idx]

def baseline_answer(query, k=12):
    top = vector_retrieve(query, k=k)
    ctx = "\n".join([f"- {pid} | score={s:.3f} | {doc}" for pid, doc, s in top])
    prompt = (
        "Você é um assistente GeoAI especializado em diagnóstico de infraestrutura ambiental urbana para apoio a decisões públicas e estratégias ESG. Responda SOMENTE com os POIs fornecidos.\n"
        "Se pedir 'perto', use h3/lat/lon como aproximação e explique.\n"
        "Responda em bullets estruturados.\n"
        "Cite sempre os IDs dos ativos ambientais.\n"
        "Inclua interpretação territorial quando possível.\n"
        f"PERGUNTA:\n{query}\n\nCONTEXTO:\n{ctx}"
    )
    return llm_text(prompt), top

def retrieve_zones(query, topk=10):
    q_emb = client.embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding
    q_emb = normalize(np.array(q_emb, dtype=np.float32).reshape(1, -1))
    sims  = cosine_similarity(q_emb, zone_emb)[0]
    idx   = np.argsort(-sims)[:topk]
    return [(zone_list[i], zone_docs[i], float(sims[i])) for i in idx]

def build_evidence(services, near_pairs, zones_ranked, max_evidence=40, max_ring=2):
    slices = [df_pois[df_pois["h3"].isin(list(h3.grid_disk(h, ZONE_RING)))]
              for h, _, _ in zones_ranked]
    slices = [s for s in slices if not s.empty]
    if not slices:
        return pd.DataFrame(), []
    cand = pd.concat(slices, ignore_index=True).drop_duplicates(subset=["osm_id"])

    if services:
        cand = cand[cand["amenity"].isin(services)]

    evid = []
    if near_pairs:
        # Tenta rings crescentes até achar POIs de ambos os tipos
        ring_df = pd.DataFrame()
        for ring in range(0, max_ring + 1):
            ring_slices = [df_pois[df_pois["h3"].isin(list(h3.grid_disk(h, ring)))]
                           for h, _, _ in zones_ranked]
            ring_slices = [s for s in ring_slices if not s.empty]
            if not ring_slices:
                continue
            candidate = pd.concat(ring_slices, ignore_index=True).drop_duplicates(subset=["osm_id"])

            needed = {p["a"] for p in near_pairs} | {p["b"] for p in near_pairs}
            found  = set(candidate["amenity"].unique())
            if needed.issubset(found):
                ring_df = candidate
                break  # ring suficiente encontrado

        if ring_df.empty:
            # Usa tudo disponível no dataset nas zonas
            ring_df = pd.concat([
                df_pois[df_pois["h3"].isin(list(h3.grid_disk(h, max_ring)))]
                for h, _, _ in zones_ranked
            ], ignore_index=True).drop_duplicates(subset=["osm_id"])

        by_am    = {a: ring_df[ring_df["amenity"] == a].copy() for a in AMENITIES}
        keep_ids = set()

        for p in near_pairs:
            A = by_am.get(p["a"]); B = by_am.get(p["b"])
            if A is None or B is None or len(A) == 0 or len(B) == 0:
                print(f"  near_pair {p['a']}↔{p['b']}: "
                      f"{len(A) if A is not None else 0} tipo-A, "
                      f"{len(B) if B is not None else 0} tipo-B (sem pares possíveis)")
                continue
            B_pts = list(B[["osm_id", "lat", "lon"]].itertuples(index=False))
            for ra in A[["osm_id", "lat", "lon", "name"]].itertuples(index=False):
                dists = [(haversine_m(ra.lat, ra.lon, rb.lat, rb.lon), rb.osm_id)
                         for rb in B_pts
                         if haversine_m(ra.lat, ra.lon, rb.lat, rb.lon) <= p["radius_m"]]
                dists.sort()
                if dists:
                    keep_ids.add(int(ra.osm_id))
                    for d, bid in dists[:5]:
                        keep_ids.add(int(bid))
                        evid.append((int(ra.osm_id), int(bid), float(d),
                                     f"NEAR({p['a']},{p['b']})"))

        cand = ring_df[ring_df["osm_id"].isin(list(keep_ids))].copy()

    return cand.head(max_evidence), evid

def graphrag_answer(query):
    q            = parse_query(query)
    zones_ranked = retrieve_zones(query, topk=int(q["top_k_zones"]))
    df_ev, edges = build_evidence(q["services"], q["near_pairs"], zones_ranked)

    zone_ctx = "\n".join([f"- Zona {h} | score={s:.3f} | {zone_card[h]}"
                          for h, _, s in zones_ranked[:8]])
    poi_ctx  = "\n".join([
        f"- POI {r.osm_id}: {r.amenity} | {r.name or 'sem nome'} | "
        f"lat={r.lat:.6f}, lon={r.lon:.6f} | h3={r.h3}"
        for r in df_ev.itertuples(index=False)
    ]) if not df_ev.empty else "Nenhum POI encontrado nas zonas recuperadas."
    edge_ctx = "\n".join([f"- POI {a} -> POI {b} | {rel} | dist_m={d:.1f}"
                          for a, b, d, rel in edges[:80]])

    prompt = (
        "Você é um assistente GeoAI especializado em diagnóstico de infraestrutura ambiental urbana. "
        "Responda SOMENTE com base no contexto abaixo.\n"
        "As respostas devem apoiar decisão pública ou análise ESG.\n"
        "Não faça inferências além dos dados listados.\n"
        "Se não houver evidência suficiente, indique claramente a limitação.\n\n"
        f"PERGUNTA:\n{query}\n\n"
        f"ZONAS:\n{zone_ctx}\n\n"
        f"POIs:\n{poi_ctx}\n\n"
        f"RELAÇÕES:\n{edge_ctx}"
    )
    return llm_text(prompt), q, zones_ranked, df_ev, edges

# #### VISUALIZAÇÃO — subgrafo no map


def show_subgraph(df_ev, edges, title="Subgrafo GraphRAG Ambiental",
                  center=None, zoom=16, max_nodes=80, max_edges=200):
    if df_ev.empty or "osm_id" not in df_ev.columns:
        print(f"[show_subgraph] Nenhuma evidência para exibir: '{title}'")
        return None

    nodes = df_ev.head(max_nodes).copy()
    nodes["osm_id"] = nodes["osm_id"].astype(int)
    idx = {int(r.osm_id): (float(r.lat), float(r.lon), str(r.amenity), r.name or "sem nome")
           for r in nodes.itertuples(index=False)}

    if center is None:
        lats = [v[0] for v in idx.values()]
        lons = [v[1] for v in idx.values()]
        center = (sum(lats) / len(lats), sum(lons) / len(lons))

    m = folium.Map(location=list(center), zoom_start=zoom, tiles="cartodbpositron")
    folium.map.Marker(
        list(center),
        icon=folium.DivIcon(html=(
            f'<div style="font-size:15px;font-weight:700;background:rgba(255,255,255,.85);'
            f'padding:6px 10px;border-radius:8px;border:1px solid #ddd">{title}</div>'
        ))
    ).add_to(m)

    cl = MarkerCluster().add_to(m)
    for oid, (lat, lon, amenity, name) in idx.items():
        folium.Marker(
            location=[lat, lon],
            popup=f"{amenity} | {name} | POI {oid}",
            icon=folium.Icon(icon=ICON_MAP.get(amenity, "ok-sign"))
        ).add_to(cl)

    drawn = 0
    for a, b, d, rel in edges:
        if drawn >= max_edges:
            break
        a, b = int(a), int(b)
        if a not in idx or b not in idx:
            continue
        lat1, lon1, *_ = idx[a]
        lat2, lon2, *_ = idx[b]
        w = 5 if float(d) <= 150 else 3 if float(d) <= 300 else 2
        folium.PolyLine(
            locations=[[lat1, lon1], [lat2, lon2]],
            weight=w, opacity=0.65,
            tooltip=f"{rel}: {a}→{b} | {float(d):.1f}m"
        ).add_to(m)
        drawn += 1

    m.fit_bounds([[min(v[0] for v in idx.values()), min(v[1] for v in idx.values())],
                  [max(v[0] for v in idx.values()), max(v[1] for v in idx.values())]])
    return m

# #### Baseline vs GraphRAG

TEST_QUERIES = [
    "Há integração entre áreas verdes e infraestrutura de reciclagem neste território?",
    "Quais zonas apresentam ativos hídricos próximos a áreas de lazer (até 800 metros)?",
    "Existem concentrações de infraestrutura de descarte próximas a ativos hídricos?",
    "Quais áreas apresentam sinergia entre parques e reciclagem? Cite evidências territoriais.",
]

rows = []
for q in TEST_QUERIES:
    base_ans, base_top                    = baseline_answer(q, k=14)
    graph_ans, qobj, zones, df_ev, edges  = graphrag_answer(q)
    rows.append({
        "query":            q,
        "baseline_answer":  base_ans,
        "graphrag_answer":  graph_ans,
        "baseline_top_ids": [x[0] for x in base_top[:8]],
        "graphrag_ev_ids":  df_ev["osm_id"].head(12).tolist() if "osm_id" in df_ev.columns else [],
        "zones_top":        [z[0] for z in zones[:5]],
        "edges":            edges,
        "df_ev":            df_ev,
    })

df_compare = pd.DataFrame(rows)

df_compare.head()


for r in rows:
    print("=" * 120)
    print("QUERY:", r["query"])
    print("\n--- BASELINE ---\n", r["baseline_answer"])
    print("\n--- GRAPHRAG ---\n", r["graphrag_answer"])

# ## mapa do último resultado GraphRAG

ans_base, top = baseline_answer(
    "Quais pontos de reciclagem ficam perto de parques?"
)

print(ans_base)

def show_baseline_on_map(top_base, title="Baseline Recuperado"):
    ids = [pid for pid, _, _ in top_base]
    df_base = df_pois[df_pois["osm_id"].isin(ids)].copy()

    m = folium.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=14, tiles="cartodbpositron")

    for r in df_base.itertuples(index=False):
        folium.Marker(
            location=[r.lat, r.lon],
            popup=f"{r.amenity} | {r.name or 'sem nome'} | POI {r.osm_id}",
            icon=folium.Icon(icon=ICON_MAP.get(r.amenity, "ok-sign"))
        ).add_to(m)

    return m

ans_base, top_base = baseline_answer(
    "Quais pontos de reciclagem ficam perto de parques?"
)

display(show_baseline_on_map(top_base, title="Baseline - Reciclagem e Parques"))

ans_g, qobj, zones, df_ev, edges = graphrag_answer(
    "Quais pontos de reciclagem ficam perto de parques?"
)
display(show_subgraph(df_ev, edges, title="Diagnóstico Territorial: Integração entre Reciclagem e Áreas Verdes"))

# ### Avaliação e Compração dos Resultados

JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "groundedness":        {"type": "integer", "minimum": 1, "maximum": 5},
        "constraint_following":{"type": "integer", "minimum": 1, "maximum": 5},
        "notes":               {"type": "string"},
    },
    "required": ["groundedness", "constraint_following", "notes"],
}

def judge(query, answer, evidence_ids):
    prompt = (
        "Avalie a resposta de um assistente geoespacial e retorne JSON válido "
        f"com este schema: {json.dumps(JUDGE_SCHEMA)}\n\n"
        f"Pergunta: {query}\n\n"
        f"Resposta:\n{answer}\n\n"
        f"Evidências (IDs de POIs recuperados):\n{evidence_ids}\n\n"
        "Critérios:\n"
        "- groundedness: 1..5 (resposta parece baseada nas evidências)\n"
        "- constraint_following: 1..5 (cita IDs, respeita 'somente contexto')\n"
        "- notes: 1-3 frases\n"
        "Responda SOMENTE com JSON, sem markdown."
    )
    raw = llm_text(prompt)
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

scores = []
for r in rows:
    s_base  = judge(r["query"], r["baseline_answer"], r["baseline_top_ids"])
    s_graph = judge(r["query"], r["graphrag_answer"],  r["graphrag_ev_ids"])
    scores.append({
        "query":                    r["query"],
        "baseline_groundedness":    s_base["groundedness"],
        "baseline_constraints":     s_base["constraint_following"],
        "baseline_notes":           s_base["notes"],
        "graphrag_groundedness":    s_graph["groundedness"],
        "graphrag_constraints":     s_graph["constraint_following"],
        "graphrag_notes":           s_graph["notes"],
    })

df_scores = pd.DataFrame(scores)
display(df_scores[[
    "query",
    "baseline_groundedness", "baseline_constraints",
    "graphrag_groundedness",  "graphrag_constraints",
]])

# médias por método
print("\nMédias:")
print(f"  Baseline  — groundedness: {df_scores['baseline_groundedness'].mean():.2f} | constraints: {df_scores['baseline_constraints'].mean():.2f}")
print(f"  GraphRAG  — groundedness: {df_scores['graphrag_groundedness'].mean():.2f} | constraints: {df_scores['graphrag_constraints'].mean():.2f}")

# ### RELATIONAL ACCURACY SCORE


def ground_truth_pairs(near_pairs_def, radius_m=500):

    gt = set()
    for u, v, data in G.edges(data=True):
        if data.get("etype") != "NEAR":
            continue
        d = data.get("dist_m", 9999)
        if d > radius_m:
            continue
        # extrai tipo de cada nó
        u_data = G.nodes[u]; v_data = G.nodes[v]
        if u_data.get("ntype") != "POI" or v_data.get("ntype") != "POI":
            continue
        for p in near_pairs_def:
            conds = [
                (u_data.get("amenity") == p["a"] and v_data.get("amenity") == p["b"]),
                (u_data.get("amenity") == p["b"] and v_data.get("amenity") == p["a"]),
            ]
            if any(conds) and d <= p["radius_m"]:
                uid = int(u.split(":")[1]); vid = int(v.split(":")[1])
                gt.add((min(uid, vid), max(uid, vid)))
    return gt


def relational_accuracy(retrieved_edges, gt_pairs):
    """
    Precision  = pares recuperados que estão no GT / total recuperados
    Recall     = pares recuperados que estão no GT / total no GT
    F1         = média harmônica
    """
    if not retrieved_edges or not gt_pairs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "tp": 0, "retrieved": len(retrieved_edges), "gt_size": len(gt_pairs)}

    retrieved_norm = {(min(int(a), int(b)), max(int(a), int(b)))
                      for a, b, *_ in retrieved_edges}
    tp = len(retrieved_norm & gt_pairs)
    precision = tp / len(retrieved_norm) if retrieved_norm else 0.0
    recall    = tp / len(gt_pairs)       if gt_pairs    else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "tp":        tp,
        "retrieved": len(retrieved_norm),
        "gt_size":   len(gt_pairs),
    }

NEAR_PAIRS_DEFAULT = [{"a": a, "b": b, "radius_m": RADIUS_M}
                      for a in AMENITIES for b in AMENITIES if a < b]

ras_rows = []
for q in TEST_QUERIES:
    ans_g, qobj, zones, df_ev, edges = graphrag_answer(q)

    near_pairs = qobj.get("near_pairs") or NEAR_PAIRS_DEFAULT
    gt   = ground_truth_pairs(near_pairs, radius_m=max((p["radius_m"] for p in near_pairs), default=MAX_NEAR_M))
    ras  = relational_accuracy(edges, gt)

    ras_rows.append({"query": q[:60] + "...", **ras})

df_ras = pd.DataFrame(ras_rows)
print("\n=== RELATIONAL ACCURACY SCORE ===")
display(df_ras[["query", "precision", "recall", "f1", "tp", "retrieved", "gt_size"]])

print("\n── Médias ──")
print(df_ras[["precision", "recall", "f1"]].mean().round(4).to_string())



