"""Advisor retrieval evaluation + inference latency of the pretrained models.

    python paper/retrieval_eval.py

Each knowledge-base entry gets two paraphrased test queries that avoid reusing the
stored question's wording. A query counts as a hit at k if its source entry is
among the top-k retrieved entries.
"""
from __future__ import annotations

import io
import json
import math
import os
import sys
import time
from collections import Counter

import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))
import common as C  # noqa: E402
from app.knowledge import KB  # noqa: E402

PARAPHRASES = [
    ("ideal nutrient strength and acidity for growing lettuce hydroponically", "what conductivity reading should my lettuce reservoir have"),
    ("edges of my lettuce leaves are turning brown and crispy", "how to stop tipburn in lettuce"),
    ("how warm should the room be for lettuce", "lettuce keeps bolting because it is too hot, what range is ideal"),
    ("role of nitrogen phosphorus and potassium in plant growth", "what does each number on an NPK fertiliser mean"),
    ("leaves are turning yellow, which nutrient is missing", "how to diagnose a nutrient shortage by looking at the foliage"),
    ("when should I replace the water in my hydroponic tank", "how frequently should the nutrient reservoir be refreshed"),
    ("what does electrical conductivity measure in a nutrient solution", "why should I monitor conductivity"),
    ("best acidity level for a hydroponic system", "ideal pH for soilless growing"),
    ("how do I bring down the pH of my reservoir", "safe way to increase pH in hydroponics"),
    ("how many hours of light does basil need", "photoperiod for growing basil indoors"),
    ("what is daily light integral", "how much total light per day do leafy greens need"),
    ("how high should I hang my LED lamps above the plants", "best height for grow lights"),
    ("small green insects on my indoor plants, how do I remove them", "natural aphid control indoors"),
    ("fine webbing and tiny dots on the leaves", "how to kill spider mites"),
    ("small flies buzzing around the growing medium", "getting rid of gnats in my grow room"),
    ("roots are brown and slimy in my hydro system", "how to stop pythium in the reservoir"),
    ("white powder on the leaves", "how to treat mildew on plants"),
    ("dark lesions spreading on my tomato leaves", "treatment for tomato blight"),
    ("when should I cut my microgreens", "how many days until microgreens can be harvested"),
    ("fuzzy white growth on my microgreen tray", "how to prevent mould on microgreens"),
    ("how to get more tomatoes from indoor plants", "increase fruit production of indoor tomatoes"),
    ("black rotten patch on the bottom of my tomatoes", "why do tomato fruits rot at the base"),
    ("hand pollination of tomatoes indoors", "my indoor tomatoes flower but do not set fruit"),
    ("compare nutrient film technique and deep water culture", "which is better, NFT or deep water culture"),
    ("easiest hydroponic setup for someone new", "what hydroponic system should a beginner start with"),
    ("rockwool vs coco coir vs perlite", "which substrate is best for hydroponics"),
    ("ideal relative humidity in a grow room", "how humid should my indoor farm be"),
    ("is CO2 enrichment worth it", "does adding carbon dioxide increase growth"),
    ("what is vapour pressure deficit", "how do temperature and humidity together affect transpiration"),
    ("is it okay to fill my reservoir with tap water", "do I need RO water for hydroponics"),
    ("best reservoir temperature for roots", "how cold should the nutrient solution be"),
    ("why is vertical farming good for the environment", "environmental benefits of indoor vertical farms"),
    ("do plants grow quicker in hydroponics than in soil", "how fast is hydroponic growth compared to soil"),
    ("best beginner crops for a vertical farm", "what should I grow first in an indoor farm"),
]
assert len(PARAPHRASES) == len(KB)


def ranks_from_scores(S, gold):
    order = np.argsort(-S, axis=1)
    return np.array([int(np.where(order[i] == g)[0][0]) + 1 for i, g in enumerate(gold)])


def summarise(ranks, lat):
    return {"r1": float((ranks <= 1).mean()), "r3": float((ranks <= 3).mean()), "r5": float((ranks <= 5).mean()),
            "mrr": float((1.0 / ranks).mean()), "latency_ms": lat, "n": int(len(ranks))}


def tokenize(s):
    return [w for w in "".join(ch.lower() if ch.isalnum() else " " for ch in s).split() if len(w) > 1]


def bm25_scores(docs, queries, k1=1.5, b=0.75):
    toks = [tokenize(d) for d in docs]
    N, avg = len(toks), np.mean([len(t) for t in toks])
    df = Counter(w for t in toks for w in set(t))
    idf = {w: math.log(1 + (N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
    S = np.zeros((len(queries), N))
    for qi, q in enumerate(queries):
        for di, t in enumerate(toks):
            tf = Counter(t)
            S[qi, di] = sum(idf.get(w, 0) * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(t) / avg)) for w in tokenize(q) if w in tf)
    return S


def main():
    queries = [q for pair in PARAPHRASES for q in pair]
    gold = [i for i in range(len(KB)) for _ in range(2)]
    docs_q = [k["q"] for k in KB]
    docs_qa = [k["q"] + " " + k["a"] for k in KB]
    out = {"retrieval": {}}

    for label, docs in [("TF-IDF (question)", docs_q), ("TF-IDF (question + answer)", docs_qa)]:
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(docs + queries)
        D = vec.transform(docs)
        S = (vec.transform(queries) @ D.T).toarray()
        lat = float(np.median([(lambda t0: (vec.transform([q]) @ D.T, (time.perf_counter() - t0) * 1000)[1])(time.perf_counter()) for q in queries]))
        out["retrieval"][label] = summarise(ranks_from_scores(S, gold), lat)
    for label, docs in [("BM25 (question)", docs_q), ("BM25 (question + answer)", docs_qa)]:
        out["retrieval"][label] = summarise(ranks_from_scores(bm25_scores(docs, queries), gold), None)

    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    Q = emb.encode(queries, normalize_embeddings=True)
    for label, docs in [("MiniLM dense (question) - ours", docs_q), ("MiniLM dense (question + answer)", docs_qa)]:
        D = emb.encode(docs, normalize_embeddings=True)
        S = Q @ D.T
        times = []
        for q in queries[:40]:
            t0 = time.perf_counter()
            qv = emb.encode(q, normalize_embeddings=True)
            _ = D @ qv
            times.append((time.perf_counter() - t0) * 1000)
        r = ranks_from_scores(S, gold)
        out["retrieval"][label] = summarise(r, float(np.median(times)))
        if "ours" in label:
            top = S.max(1)
            out["minilm_top_scores"] = {"hit": top[r == 1].tolist(), "miss": top[r != 1].tolist()}
            out["minilm_misses"] = [{"query": queries[i], "gold": docs_q[gold[i]], "got": docs_q[int(S[i].argmax())]} for i in np.where(r != 1)[0]]
    for k, v in out["retrieval"].items():
        print(f"  {k:36s} R@1={v['r1']:.3f} R@3={v['r3']:.3f} MRR={v['mrr']:.3f} lat={v['latency_ms']}")

    # latency + size of the pretrained models (CPU)
    from transformers import pipeline
    img = Image.fromarray((np.random.default_rng(0).random((256, 256, 3)) * 255).astype(np.uint8))
    lat = {}
    for key, mid in [("disease_vit", "wambugu71/crop_leaf_diseases_vit"), ("species_vit", "dima806/medicinal_plants_image_detection")]:
        pipe = pipeline("image-classification", model=mid)
        n_params = sum(p.numel() for p in pipe.model.parameters())
        for _ in range(3):
            pipe(img)
        t = []
        for _ in range(20):
            t0 = time.perf_counter()
            pipe(img, top_k=5)
            t.append((time.perf_counter() - t0) * 1000)
        lat[key] = {"params": n_params, "latency_ms": float(np.median(t)), "n_labels": len(pipe.model.config.id2label),
                    "labels": list(pipe.model.config.id2label.values())}
        print(f"  {key}: {n_params/1e6:.1f}M params, {np.median(t):.1f} ms, {lat[key]['n_labels']} labels")
    lat["minilm"] = {"params": sum(p.numel() for p in emb.parameters()), "latency_ms": out["retrieval"]["MiniLM dense (question) - ours"]["latency_ms"]}
    out["pretrained"] = lat
    with open(os.path.join(C.RESULTS_DIR, "retrieval.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("  -> results/retrieval.json")


if __name__ == "__main__":
    main()
