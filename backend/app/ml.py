"""Model loading + inference for VERDANT.

Three pretrained HuggingFace models (no training), exported to ONNX and int8-quantized
so the API runs without PyTorch in ~300 MB of RAM. Weights come from the ONNX_REPO model
repo on the Hub (see download_models.py). Everything is lazy-loaded and cached.
"""

from __future__ import annotations

import io
import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .knowledge import KB

DISEASE_MODEL = "wambugu71/crop_leaf_diseases_vit"          # Vision Transformer
SPECIES_MODEL = "dima806/medicinal_plants_image_detection"  # ViT — 52 plants/herbs (public)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"      # Sentence-Transformer

# int8 ONNX exports of the three models above, one subfolder each (disease/, species/, embed/)
ONNX_REPO = os.getenv("VERDANT_ONNX_REPO", "Ninisha/verdant-onnx")
ONNX_DIR = Path(os.getenv("VERDANT_ONNX_DIR", Path(__file__).parent.parent / "onnx_models"))

_cache: dict[str, Any] = {}
_locks: dict[str, threading.Lock] = {k: threading.Lock() for k in ("disease", "species", "embed", "kb")}


def _model_dir(key: str) -> Path:
    d = ONNX_DIR / key
    if not (d / "model.onnx").exists():
        from huggingface_hub import snapshot_download

        snapshot_download(ONNX_REPO, local_dir=ONNX_DIR, allow_patterns=[f"{key}/*"])
    return d


def _session(path: Path):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(os.getenv("VERDANT_THREADS", "1"))
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


class ImageClassifier:
    """Minimal stand-in for transformers' image-classification pipeline (ViTImageProcessor + softmax)."""

    def __init__(self, d: Path):
        cfg = json.loads((d / "config.json").read_text())
        pre = json.loads((d / "preprocessor_config.json").read_text())
        self.labels = {int(k): v for k, v in cfg["id2label"].items()}
        self.size = (pre["size"]["width"], pre["size"]["height"])
        self.resample = Image.Resampling(pre.get("resample", 2))
        self.scale = pre.get("rescale_factor", 1 / 255)
        self.mean = np.array(pre["image_mean"], dtype=np.float32)
        self.std = np.array(pre["image_std"], dtype=np.float32)
        self.sess = _session(d / "model.onnx")

    def __call__(self, img: Image.Image, top_k: int = 5) -> list[dict]:
        x = np.asarray(img.resize(self.size, self.resample), dtype=np.float32) * self.scale
        x = ((x - self.mean) / self.std).transpose(2, 0, 1)[None]
        logits = self.sess.run(None, {"pixel_values": x})[0][0]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        top = np.argsort(p)[::-1][:top_k]
        return [{"label": self.labels[int(i)], "score": float(p[i])} for i in top]


class Embedder:
    """all-MiniLM-L6-v2: tokenize, run ONNX (pooling baked in), L2-normalize."""

    def __init__(self, d: Path):
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(256)
        self.tok.enable_padding()
        self.sess = _session(d / "model.onnx")

    def encode(self, texts: list[str]) -> np.ndarray:
        enc = self.tok.encode_batch(texts)
        feeds = {
            "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
        }
        v = self.sess.run(["sentence_embedding"], feeds)[0]
        return v / np.linalg.norm(v, axis=1, keepdims=True)


# ----------------------------------------------------------------------------
# loaders (thread-safe, lazy)
# ----------------------------------------------------------------------------
def _image_pipe(key: str):
    if key not in _cache:
        with _locks[key]:
            if key not in _cache:
                _cache[key] = ImageClassifier(_model_dir(key))
    return _cache[key]


def get_disease():
    return _image_pipe("disease")


def get_species():
    return _image_pipe("species")


def get_embedder():
    if "embed" not in _cache:
        with _locks["embed"]:
            if "embed" not in _cache:
                _cache["embed"] = Embedder(_model_dir("embed"))
    return _cache["embed"]


def _kb_embeddings():
    if "kb_emb" not in _cache:
        with _locks["kb"]:
            if "kb_emb" not in _cache:
                _cache["kb_emb"] = get_embedder().encode([item["q"] for item in KB])
    return _cache["kb_emb"]


def loaded() -> dict[str, bool]:
    return {
        "disease": "disease" in _cache,
        "species": "species" in _cache,
        "advisor": "kb_emb" in _cache,
    }


# ----------------------------------------------------------------------------
# image inference
# ----------------------------------------------------------------------------
def classify(pipe, data: bytes, top_k: int = 5) -> list[dict]:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    out = pipe(img, top_k=top_k)
    return [{"label": str(o["label"]), "score": float(o["score"])} for o in out]


def prettify(label: str) -> str:
    s = label.replace("___", " · ").replace("__", " ").replace("_", " ").strip()
    parts = []
    for token in s.split(" "):
        if token in {"·", "-"}:
            parts.append(token)
        elif token.isupper() and len(token) <= 3:
            parts.append(token)  # keep acronyms
        else:
            parts.append(token.capitalize())
    return " ".join(parts)


_CROPS = {
    "corn": "Corn", "maize": "Corn", "potato": "Potato", "rice": "Rice",
    "wheat": "Wheat", "tomato": "Tomato", "pepper": "Pepper", "bell": "Pepper",
    "apple": "Apple", "grape": "Grape", "strawberry": "Strawberry",
    "soybean": "Soybean", "cotton": "Cotton", "sugarcane": "Sugarcane",
}


def parse_label(raw: str) -> tuple[str | None, str]:
    low = raw.lower()
    crop = next((v for k, v in _CROPS.items() if k in low), None)
    cond = raw.split("___", 1)[1] if "___" in raw else raw
    cond = cond.replace("_", " ").strip()
    # drop crop word from condition if it leads
    if crop and cond.lower().startswith(crop.lower()):
        cond = cond[len(crop):].strip()
    cond_pretty = " ".join(w.capitalize() for w in cond.split()) or "Detected"
    return crop, cond_pretty


# ordered (keywords, advice) — first match wins, so list specific first
_HEALTHY = {
    "severity": "none",
    "summary": "Good news — no disease detected on {crop}. The leaf looks healthy. Keep conditions stable and keep monitoring.",
    "treatment": [
        "No treatment needed — the plant is healthy.",
        "Maintain current light, water and nutrient routine.",
    ],
    "prevention": [
        "Scout leaves (especially undersides) twice a week.",
        "Keep airflow steady and humidity below 75%.",
        "Sanitise tools between zones to avoid spreading pathogens.",
    ],
}

_DEFAULT = {
    "severity": "moderate",
    "summary": "{condition} detected on {crop}. Treat it as an active foliar problem and isolate the plant while you act.",
    "treatment": [
        "Remove and bag affected leaves; do not compost them.",
        "Apply an appropriate fungicide or biocontrol on a weekly cycle.",
        "Improve airflow and reduce leaf wetness.",
    ],
    "prevention": [
        "Water at the root zone, never on the foliage.",
        "Space plants for airflow and prune dense growth.",
        "Rotate crops and disinfect surfaces between cycles.",
    ],
}

_ADVICE: list[tuple[tuple[str, ...], dict]] = [
    (("late blight", "late_blight"), {
        "severity": "high",
        "summary": "Late blight (Phytophthora) detected on {crop} — a fast, destructive water-mould that can collapse a crop in days under cool, humid conditions.",
        "treatment": [
            "Isolate and remove affected foliage immediately; bag and discard.",
            "Apply a copper-based or biofungicide every 5–7 days.",
            "Drop canopy humidity below 80% and boost airflow between racks.",
        ],
        "prevention": [
            "Water only at the roots — keep leaves dry.",
            "Increase spacing and prune lower leaves for airflow.",
            "Rotate solanaceous crops and sanitise tools.",
        ],
    }),
    (("early blight", "early_blight"), {
        "severity": "moderate",
        "summary": "Early blight (Alternaria) detected on {crop} — look for concentric 'target' rings on older leaves. It spreads in warm, humid spells.",
        "treatment": [
            "Prune and bag the lowest affected leaves first.",
            "Apply a chlorothalonil-free or copper fungicide weekly.",
            "Mulch the base to stop soil splash onto leaves.",
        ],
        "prevention": [
            "Avoid overhead watering; keep foliage dry.",
            "Feed steadily — stressed plants are more susceptible.",
            "Rotate beds and remove crop debris each cycle.",
        ],
    }),
    (("blight",), _DEFAULT),
    (("rust", "common rust", "yellow rust", "brown rust"), {
        "severity": "moderate",
        "summary": "Rust detected on {crop} — orange/brown powdery pustules on leaves, favoured by humidity and moderate temperatures.",
        "treatment": [
            "Remove the most heavily infected leaves.",
            "Apply a suitable fungicide (e.g. triazole) per label.",
            "Lower humidity and improve ventilation.",
        ],
        "prevention": [
            "Choose resistant varieties where possible.",
            "Avoid wetting leaves and overcrowding.",
            "Clear volunteer plants and debris that harbour spores.",
        ],
    }),
    (("blast", "leaf blast"), {
        "severity": "high",
        "summary": "Blast (Magnaporthe) detected on {crop} — diamond-shaped lesions that can spread rapidly and hit yields hard.",
        "treatment": [
            "Remove infected tissue and reduce leaf wetness.",
            "Apply a recommended systemic fungicide promptly.",
            "Avoid excess nitrogen, which worsens blast.",
        ],
        "prevention": [
            "Balance nitrogen; add silicon if available.",
            "Maintain steady moisture and good drainage.",
            "Use clean, treated seed and resistant cultivars.",
        ],
    }),
    (("powdery mildew",), {
        "severity": "moderate",
        "summary": "Powdery mildew detected on {crop} — a white dusty fungus driven by high humidity and stagnant air.",
        "treatment": [
            "Remove affected leaves and improve ventilation.",
            "Spray potassium bicarbonate or a biofungicide.",
            "Lower humidity and avoid wetting foliage.",
        ],
        "prevention": [
            "Keep strong airflow over the canopy.",
            "Avoid crowding; prune for light penetration.",
            "Hold humidity below 70% during growth.",
        ],
    }),
    (("downy mildew", "mildew"), {
        "severity": "moderate",
        "summary": "Mildew detected on {crop} — fungal growth that thrives in damp, still conditions.",
        "treatment": [
            "Strip affected leaves and increase airflow.",
            "Apply an appropriate fungicide or biocontrol.",
            "Reduce humidity and leaf wetness.",
        ],
        "prevention": [
            "Ventilate well and space plants.",
            "Water at the base, early in the day.",
            "Remove debris that harbours spores.",
        ],
    }),
    (("bacterial",), {
        "severity": "moderate",
        "summary": "A bacterial infection was detected on {crop} — typically water-soaked spots that may ooze and spread by splash.",
        "treatment": [
            "Remove infected tissue; sanitise tools after.",
            "Apply a copper-based bactericide per label.",
            "Stop overhead watering immediately.",
        ],
        "prevention": [
            "Use clean seed and disease-free transplants.",
            "Avoid working among wet plants.",
            "Disinfect surfaces and rotate crops.",
        ],
    }),
    (("mosaic", "virus"), {
        "severity": "high",
        "summary": "A viral infection (e.g. mosaic) was detected on {crop}. Viruses can't be cured — the priority is to stop spread.",
        "treatment": [
            "Remove and destroy infected plants promptly.",
            "Control insect vectors (aphids, thrips, whitefly).",
            "Sanitise hands and tools thoroughly.",
        ],
        "prevention": [
            "Use certified virus-free seed/stock.",
            "Manage insect vectors proactively.",
            "Disinfect tools; avoid tobacco near plants (TMV).",
        ],
    }),
    (("septoria", "leaf spot", "gray leaf spot", "brown spot", "spot"), {
        "severity": "moderate",
        "summary": "Leaf-spot disease detected on {crop} — small dark/grey lesions, often starting on lower leaves in humid conditions.",
        "treatment": [
            "Remove the lowest spotted leaves first.",
            "Apply a copper or biofungicide weekly.",
            "Mulch to prevent soil splash; improve airflow.",
        ],
        "prevention": [
            "Keep foliage dry; water at the roots.",
            "Space and prune for airflow.",
            "Clear infected debris each cycle.",
        ],
    }),
    (("scab",), _DEFAULT),
    (("rot",), {
        "severity": "high",
        "summary": "Rot detected on {crop} — tissue breakdown usually tied to excess moisture or low oxygen.",
        "treatment": [
            "Remove affected tissue and reduce watering.",
            "Improve drainage and root-zone oxygen.",
            "Apply a biocontrol (e.g. Trichoderma) if available.",
        ],
        "prevention": [
            "Avoid overwatering; keep reservoir below 21°C.",
            "Oxygenate the solution; keep light off it.",
            "Sterilise media between crops.",
        ],
    }),
]


def advice_for(raw: str) -> dict:
    low = raw.lower()
    if "healthy" in low:
        return _HEALTHY
    for keys, adv in _ADVICE:
        if any(k in low for k in keys):
            return adv
    return _DEFAULT


def build_diagnosis(preds_raw: list[dict]) -> dict:
    top = preds_raw[0]
    raw = top["label"]
    crop, condition = parse_label(raw)
    healthy = "healthy" in raw.lower()
    adv = advice_for(raw)
    summary = adv["summary"].format(condition=condition, crop=crop or "this crop")
    return {
        "ok": True,
        "offline": False,
        "model": DISEASE_MODEL,
        "top": {"label": prettify(raw), "score": top["score"]},
        "predictions": [{"label": prettify(p["label"]), "score": p["score"]} for p in preds_raw],
        "crop": crop,
        "condition": "Healthy" if healthy else condition,
        "healthy": healthy,
        "severity": adv["severity"],
        "summary": summary,
        "treatment": adv["treatment"],
        "prevention": adv["prevention"],
    }


# ----------------------------------------------------------------------------
# advisor (semantic search)
# ----------------------------------------------------------------------------
def answer(question: str) -> dict:
    kb = _kb_embeddings()
    qv = get_embedder().encode([question])[0]
    scores = kb @ qv
    idx = int(scores.argmax())
    score = float(scores[idx])
    item = KB[idx]
    related = [k["q"] for k in KB if k["topic"] == item["topic"] and k["q"] != item["q"]][:3]
    if not related:
        related = [k["q"] for k in KB if k["q"] != item["q"]][:3]

    ans = item["a"]
    if score < 0.35:
        ans = (
            "I'm not fully sure I have a precise answer for that, but here's the closest guidance "
            "from my knowledge base:\n\n" + ans
        )
    return {
        "ok": True,
        "offline": False,
        "answer": ans,
        "score": score,
        "topic": item["topic"],
        "related": related,
        "sources": ["VERDANT agronomy knowledge base"],
    }
