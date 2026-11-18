"""Carga de um caminhão com um algoritmo genético: quais encomendas levar, e em que ordem empilhar, para maximizar o valor.

Dados: produtos reais do Brazilian E-Commerce Public Dataset by Olist (dimensões, peso e preço médio de venda). Cada
produto sorteado é uma encomenda pendente. O baú tem 2,4 m x 1,6 m x 1,6 m (6,14 m3) e a carga nominal é de 700 kg (premissas do estudo, e não medidas
de um veículo específico). Regras realistas: folga de 10 cm em cada dimensão (o espaço útil é 2,3 x 1,5 x 1,5 m),
uso de no máximo 90% da carga nominal (630 kg) e caixas acima de 10 kg só no chão. Cada caixa é apoiada com a menor
dimensão na vertical.

Decodificador: o indivíduo é uma ordem de carregamento. As caixas entram nessa ordem em prateleiras (camadas, fileiras
e posição na fileira), e a que não cabe ou passaria do peso é deixada de fora. O valor é a soma dos preços das caixas
que entraram. Compara um algoritmo genético (ordem, cruzamento OX, mutação por troca, elitismo), a busca aleatória de
ordens com o mesmo número de avaliações e três ordens gulosas (por valor por volume, por valor e por volume).

Uso: python carga.py [--sementes 10] [--caixas 500] [--saida resultados]
Grava comparacao.csv, convergencia.csv, melhor_carga.csv (genético semeado), melhor_ga_aleatorio.csv, guloso_carga.csv e resumo.json.
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).parent
COMPR, LARG, ALT = 240.0, 160.0, 160.0        # baú, em cm
FOLGA = 10.0                                   # folga operacional em cada dimensão (porta, cintas, ar), em cm
COMPR_U, LARG_U, ALT_U = COMPR - FOLGA, LARG - FOLGA, ALT - FOLGA    # espaço útil de empilhamento
CARGA_NOMINAL_G = 700_000.0                    # 700 kg
CARGA_MAX_G = 0.9 * CARGA_NOMINAL_G            # margem de segurança: só 90% da carga nominal
PESADA_G = 10_000.0                            # caixas acima de 10 kg só vão no chão
SEMENTE_CAIXAS = 42


def carregar(n_caixas: int) -> pd.DataFrame:
    d = pd.read_csv(AQUI / "dados" / "olist_products_dataset.csv")
    itens = pd.read_csv(AQUI / "dados" / "olist_order_items_dataset.csv")
    trad = pd.read_csv(AQUI / "dados" / "product_category_name_translation.csv")
    preco = itens.groupby("product_id").price.mean().rename("preco")
    d = d.merge(preco, on="product_id").merge(trad, on="product_category_name", how="left")
    cols = ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm", "preco"]
    d = d.dropna(subset=cols).copy()
    dims = np.sort(d[["product_length_cm", "product_height_cm", "product_width_cm"]].to_numpy(), axis=1)[:, ::-1]
    d["l"], d["w"], d["h"] = dims[:, 0], dims[:, 1], dims[:, 2]            # maior, médio e menor dimensão
    d = d[(d.l <= COMPR_U) & (d.w <= LARG_U) & (d.h <= ALT_U) & (d.product_weight_g > 0) & (d.preco > 0)]
    d = d.sample(n_caixas, random_state=SEMENTE_CAIXAS).reset_index(drop=True)
    d["peso"] = d.product_weight_g
    d["volume"] = d.l * d.w * d.h
    d["categoria"] = d.product_category_name_english.fillna(d.product_category_name).fillna("outros")
    return d[["product_id", "categoria", "l", "w", "h", "peso", "volume", "preco"]]


def empacotar(ordem, L, W, H, peso, preco, devolver_posicoes: bool = False):
    """Carrega as caixas na ordem dada, em prateleiras. Devolve (valor, volume, peso, n) e, se pedido, as posições."""
    camadas = []                        # cada camada: [z, altura, proximo_y, [fileiras]] ; fileira: [y, largura, x_atual]
    z_prox = 0.0
    valor = vol = pes = 0.0
    n = 0
    pos = []
    for i in ordem:
        l, w, h = L[i], W[i], H[i]
        if pes + peso[i] > CARGA_MAX_G:
            continue
        colocado = False
        pesada = peso[i] > PESADA_G
        for cam in camadas:
            if h > cam[1] or (pesada and cam[0] > 0):      # caixa pesada só na camada do chão
                continue
            for fil in cam[3]:
                if w <= fil[1] and fil[2] + l <= COMPR_U:
                    if devolver_posicoes:
                        pos.append((i, fil[2], fil[0], cam[0]))
                    fil[2] += l
                    colocado = True
                    break
            if colocado:
                break
            if cam[2] + w <= LARG_U:                          # nova fileira nesta camada
                cam[3].append([cam[2], w, l])
                if devolver_posicoes:
                    pos.append((i, 0.0, cam[2], cam[0]))
                cam[2] += w
                colocado = True
                break
        if not colocado and z_prox + h <= ALT_U and (not pesada or z_prox == 0):             # nova camada
            camadas.append([z_prox, h, w, [[0.0, w, l]]])
            if devolver_posicoes:
                pos.append((i, 0.0, 0.0, z_prox))
            z_prox += h
            colocado = True
        if colocado:
            valor += preco[i]
            vol += L[i] * W[i] * H[i]
            pes += peso[i]
            n += 1
    if devolver_posicoes:
        return valor, vol, pes, n, pos
    return valor, vol, pes, n


def cruzamento_ox(p1, p2, rng):
    n = len(p1)
    a, b = sorted(rng.choice(n, 2, replace=False))
    filho = np.full(n, -1)
    filho[a:b] = p1[a:b]
    usados = set(filho[a:b].tolist())
    resto = [g for g in p2 if g not in usados]
    filho[:a] = resto[:a]
    filho[b:] = resto[a:]
    return filho


def genetico(dados, semente, populacao=40, geracoes=100, torneio=3, p_cruz=0.9, p_mut=0.3, elite=2, inicial=None):
    rng = np.random.default_rng(semente)
    L, W, H, peso, preco = dados
    n = len(L)
    pop = [rng.permutation(n) for _ in range(populacao)]
    if inicial is not None:
        pop[0] = np.array(inicial)
    fit = np.array([empacotar(p, L, W, H, peso, preco)[0] for p in pop])
    hist = [float(fit.max())]
    avaliacoes = populacao
    for _ in range(geracoes):
        ordem = np.argsort(-fit)
        novos = [pop[i].copy() for i in ordem[:elite]]
        while len(novos) < populacao:
            def pai():
                idx = rng.choice(populacao, torneio, replace=False)
                return pop[idx[np.argmax(fit[idx])]]
            f = cruzamento_ox(pai(), pai(), rng) if rng.random() < p_cruz else pai().copy()
            if rng.random() < p_mut:
                for _ in range(2):
                    i, j = rng.choice(n, 2, replace=False)
                    f[i], f[j] = f[j], f[i]
            novos.append(f)
        pop = novos
        fit = np.array([empacotar(p, L, W, H, peso, preco)[0] for p in pop])
        avaliacoes += populacao - elite                # os elites mantêm o valor, sem reavaliar o resultado
        hist.append(float(fit.max()))
    melhor = pop[int(np.argmax(fit))]
    return melhor, hist


def aleatoria(dados, semente, avaliacoes):
    rng = np.random.default_rng(semente)
    L, W, H, peso, preco = dados
    melhor, mel_v = None, -1.0
    hist = []
    for _ in range(avaliacoes):
        o = rng.permutation(len(L))
        v = empacotar(o, L, W, H, peso, preco)[0]
        if v > mel_v:
            melhor, mel_v = o, v
        hist.append(mel_v)
    return melhor, hist


def uma_semente(args):
    semente, dados, geracoes, pop, ordem_guloso = args
    melhor_g, hist_g = genetico(dados, semente, pop, geracoes)
    avals = pop + geracoes * (pop - 2)
    melhor_a, hist_a = aleatoria(dados, semente, avals)
    melhor_s, _ = genetico(dados, 100 + semente, pop, geracoes, inicial=ordem_guloso)     # semeado com o guloso
    return semente, melhor_g, hist_g, melhor_a, hist_a, melhor_s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sementes", type=int, default=10)
    ap.add_argument("--caixas", type=int, default=500)
    ap.add_argument("--geracoes", type=int, default=300)
    ap.add_argument("--populacao", type=int, default=60)
    ap.add_argument("--saida", default="resultados")
    args = ap.parse_args()
    saida = AQUI / args.saida
    saida.mkdir(exist_ok=True)

    d = carregar(args.caixas)
    L, W, H = d.l.to_numpy(), d.w.to_numpy(), d.h.to_numpy()
    peso, preco = d.peso.to_numpy(), d.preco.to_numpy()
    dados = (L, W, H, peso, preco)
    vol_bau, n = COMPR * LARG * ALT, len(d)
    print(f"{n} encomendas | volume total {d.volume.sum() / 1e6:.2f} m3 (baú {vol_bau / 1e6:.2f} m3) | peso total {peso.sum() / 1000:.0f} kg (carga nominal {CARGA_NOMINAL_G / 1000:.0f} kg, limite de uso {CARGA_MAX_G / 1000:.0f} kg) | valor total R$ {preco.sum():,.0f}", flush=True)

    # ordens gulosas
    gulosos = {"Guloso por valor por volume": np.argsort(-(preco / (L * W * H))), "Guloso por valor": np.argsort(-preco),
               "Guloso por menor volume": np.argsort(L * W * H)}
    linhas = []
    for nome, o in gulosos.items():
        v, vol, pes, k = empacotar(o, *dados[:3], peso, preco)
        linhas.append({"metodo": nome, "semente": -1, "valor": v, "ocupacao_volume": vol / vol_bau, "ocupacao_peso": pes / CARGA_NOMINAL_G, "caixas": k})
    ordem_g = gulosos["Guloso por valor por volume"]
    _, _, _, _, pos_g = empacotar(ordem_g, L, W, H, peso, preco, True)
    pd.DataFrame(pos_g, columns=["i", "x", "y", "z"]).join(d, on="i").to_csv(saida / "guloso_carga.csv", index=False)

    t0 = time.time()
    with Pool(min(args.sementes, 10)) as pool:
        res = pool.map(uma_semente, [(s, dados, args.geracoes, args.populacao, ordem_g) for s in range(args.sementes)])
    print(f"{args.sementes} sementes em {time.time() - t0:.0f} s", flush=True)

    conv, melhores = [], {}
    for semente, mg, hg, ma, ha, ms in res:
        for nome, o in (("Algoritmo genético", mg), ("Busca aleatória de ordens", ma), ("Algoritmo genético semeado com o guloso", ms)):
            v, vol, pes, k = empacotar(o, L, W, H, peso, preco)
            linhas.append({"metodo": nome, "semente": semente, "valor": v, "ocupacao_volume": vol / vol_bau, "ocupacao_peso": pes / CARGA_NOMINAL_G, "caixas": k})
            if v > melhores.get(nome, (-1.0, None))[0]:
                melhores[nome] = (v, o)
        conv += [{"metodo": "Algoritmo genético", "semente": semente, "geracao": g, "melhor_valor": v} for g, v in enumerate(hg)]
        passo = max(1, len(ha) // len(hg))
        conv += [{"metodo": "Busca aleatória de ordens", "semente": semente, "geracao": g, "melhor_valor": ha[min(g * passo + passo - 1, len(ha) - 1)]} for g in range(len(hg))]
    pd.DataFrame(linhas).to_csv(saida / "comparacao.csv", index=False)
    pd.DataFrame(conv).to_csv(saida / "convergencia.csv", index=False)
    for nome, arq in (("Algoritmo genético", "melhor_ga_aleatorio.csv"), ("Algoritmo genético semeado com o guloso", "melhor_carga.csv")):
        _, _, _, _, pos = empacotar(melhores[nome][1], L, W, H, peso, preco, True)
        pd.DataFrame(pos, columns=["i", "x", "y", "z"]).join(d, on="i").to_csv(saida / arq, index=False)

    tab = pd.read_csv(saida / "comparacao.csv").groupby("metodo")[["valor", "ocupacao_volume", "ocupacao_peso", "caixas"]].agg(["mean", "std"])
    print(tab.round(3).to_string())
    (saida / "resumo.json").write_text(json.dumps({
        "caixas_disponiveis": n, "volume_total_m3": float(d.volume.sum() / 1e6), "volume_bau_m3": vol_bau / 1e6, "peso_total_kg": float(peso.sum() / 1000),
        "carga_max_kg": CARGA_NOMINAL_G / 1000, "carga_limite_kg": CARGA_MAX_G / 1000, "bau_util_cm": [COMPR_U, LARG_U, ALT_U], "pesada_g": PESADA_G, "valor_total": float(preco.sum()), "sementes": args.sementes, "geracoes": args.geracoes,
        "populacao": args.populacao, "bau_cm": [COMPR, LARG, ALT],
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
