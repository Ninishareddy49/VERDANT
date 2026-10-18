# VERDANT — Model Reference

Every model used in VERDANT: what it is, its architecture layer-by-layer, its parameters, and its measured metrics. No system architecture, no glossary, no design rationale — just the models.

---

## Table of Contents

**Production (Hugging Face, pretrained)**
1. [Plant Leaf Disease Classifier](#1-plant-leaf-disease-classifier)
2. [Medicinal & Crop Species Scanner](#2-medicinal--crop-species-scanner)
3. [Dense Semantic Embedder](#3-dense-semantic-embedder)
4. [Generative Agronomy Advisor](#4-generative-agronomy-advisor)

**Custom Trained (`train_models.ipynb`, trained from scratch — no pretrained weights)**
5. [Foliar Disease CNN](#5-foliar-disease-cnn)
6. [Crop Recommender](#6-crop-recommender)
7. [Lettuce Growth Predictor](#7-lettuce-growth-predictor)
8. [Greenhouse Climate Forecaster](#8-greenhouse-climate-forecaster)

---

## 1. Plant Leaf Disease Classifier

* **Hugging Face ID:** [`wambugu71/crop_leaf_diseases_vit`](https://huggingface.co/wambugu71/crop_leaf_diseases_vit)
* **Type:** Vision Transformer (`ViT-Base/16`), pretrained on ImageNet-21k, fine-tuned on foliar disease benchmarks (e.g. PlantVillage).
* **Parameters:** ~86 million.
* **Input / Output:** RGB image → resized/bicubic-interpolated to 224×224 → condition label + confidence.

**Layer-by-layer:**

```
INPUT  224×224×3 RGB image
   │
   ▼
Patch Embedding   split into 16×16 patches → 196 patches, linearly projected to 768-d tokens
   │
   ▼
+ [CLS] token prepended, + learned positional embeddings
   │
   ▼
┌───────────────────────────────────────────────────────────┐
│  12 × Transformer Encoder Layers, each:                    │
│    LayerNorm → Multi-Head Self-Attention (12 heads, d=768)  │
│    → residual add                                           │
│    → LayerNorm → MLP (768 → 3072 → 768, GELU)               │
│    → residual add                                            │
└───────────────────────────────────────────────────────────┘
   │
   ▼
Final LayerNorm → [CLS] token output (768-d)
   │
   ▼
Linear classification head → softmax over disease/crop classes
```

* **Metrics:** ~98.0% validation accuracy across multi-class agricultural datasets (14+ crop species, dozens of pathological conditions).

---

## 2. Medicinal & Crop Species Scanner

* **Hugging Face ID:** [`dima806/medicinal_plants_image_detection`](https://huggingface.co/dima806/medicinal_plants_image_detection)
* **Type:** Vision Transformer (`ViT-Base/16`) — identical architecture family to Model 1 above (patch embedding → 12 encoder layers → CLS head), fine-tuned for 52-way species classification.
* **Parameters:** ~86 million.
* **Input / Output:** RGB leaf/flower/canopy photo, normalized to ImageNet mean/std ($\mu=[0.485,0.456,0.406]$, $\sigma=[0.229,0.224,0.225]$) → top-5 species probabilities across 52 classes.
* **Metrics:** ~91.5% validation accuracy across 52 botanical/medicinal plant classes.

---

## 3. Dense Semantic Embedder

* **Hugging Face ID:** [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
* **Type:** 6-layer MiniLM Transformer with mean pooling (Siamese sentence-embedding network).
* **Parameters:** ~22.7 million.
* **Output dimension:** 384-d, L2-normalized.

**Layer-by-layer:**

```
INPUT  tokenized text (WordPiece)
   │
   ▼
Token + Positional Embeddings
   │
   ▼
┌───────────────────────────────────────────────┐
│  6 × Transformer Encoder Layers, each:          │
│    Multi-Head Self-Attention (12 heads)         │
│    → residual + LayerNorm                       │
│    → Feed-Forward (384 → 1536 → 384)             │
│    → residual + LayerNorm                        │
└───────────────────────────────────────────────┘
   │
   ▼
Mean Pooling across token embeddings → 384-d sentence vector → L2 normalize
```

* **Metrics:** ~14,200 sentences/sec on CPU; ~0.8 ms per single query. Used for cosine-similarity retrieval, not classification accuracy.

---

## 4. Generative Agronomy Advisor

* **Model:** Gemma 3 (4B) / LLaMA 3.2, served via Ollama Cloud API.
* **Type:** Dense auto-regressive decoder-only LLM (external API — architecture/weights not hosted or controlled by this project, so no internal layer breakdown applies here).
* **Role:** RAG-grounded answer synthesis + structured JSON grow-plan generation.
* **Metrics:** Qualitative (grounded-generation quality); no accuracy benchmark — falls back to a deterministic knowledge-base template if the API is unavailable.

---

## 5. Foliar Disease CNN

* **File:** `models/disease_classifier.pth` (~4.6 MB state_dict)
* **Type:** 4-layer Convolutional Neural Network, **PyTorch, trained from scratch** — no pretrained weights loaded.
* **Total trainable parameters:** **1,146,882**

**Layer-by-layer:**

```
INPUT  (3, 128, 128)  RGB leaf image, pixels scaled to [0.0, 1.0]
   │
   ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 1  Conv2d(3→16, k=3, p=1) → BatchNorm2d(16) → ReLU     │   params: 448 (conv) + 32 (BN)
│          → MaxPool2d(2,2)                                    │
└─────────────────────────────────────────────────────────────┘
   │  (16, 128, 128) → pooled → (16, 64, 64)
   ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 2  Conv2d(16→32, k=3, p=1) → BatchNorm2d(32) → ReLU    │   params: 4,640 (conv) + 64 (BN)
│          → MaxPool2d(2,2)                                    │
└─────────────────────────────────────────────────────────────┘
   │  (32, 64, 64) → pooled → (32, 32, 32)
   ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 3  Conv2d(32→64, k=3, p=1) → BatchNorm2d(64) → ReLU    │   params: 18,496 (conv) + 128 (BN)
│          → MaxPool2d(2,2)                                    │
└─────────────────────────────────────────────────────────────┘
   │  (64, 32, 32) → pooled → (64, 16, 16)
   ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 4  Conv2d(64→128, k=3, p=1) → BatchNorm2d(128) → ReLU  │   params: 73,856 (conv) + 256 (BN)
│          → MaxPool2d(2,2)                                    │
└─────────────────────────────────────────────────────────────┘
   │  (128, 16, 16) → pooled → (128, 8, 8)
   ▼
  Flatten → 128 × 8 × 8 = 8,192
   │
   ▼
┌─────────────────────────────────────────────────────────────┐
│ CLASSIFIER HEAD  Dropout(p=0.4) → Linear(8192→128) → ReLU    │   params: 1,048,704 (fc1)
│                  → Linear(128→2)                              │   params: 258 (fc2)
└─────────────────────────────────────────────────────────────┘
   │
   ▼
OUTPUT  (2,)  logits → {Tomato___Healthy, Tomato___Late_Blight}
```

* **Training config:** `CrossEntropyLoss`, Adam (lr=0.001, weight_decay=1e-4), 15 epochs, batch size 32, train-time augmentation (random flips + brightness jitter ×0.85–1.15).
* **Data:** 637 valid synthetic leaf images (320/class, 3 corrupted files cleaned) → 509 train / 128 validation.

**Metrics:**

| | Train | Validation |
|---|---|---|
| Accuracy | 85.27% | **83.59%** (peaked 87.50% @ epoch 13) |
| Loss (final epoch) | 0.3216 | 0.3708 |

Validation classification report: Healthy — precision 0.81, recall 0.89, F1 0.85. Late Blight — precision 0.88, recall 0.78, F1 0.82. Weighted avg — 0.84 / 0.84 / 0.84.

---

## 6. Crop Recommender

* **File:** `models/crop_recommender.pkl` (dict: `StandardScaler` + `RandomForestClassifier`)
* **Type:** Random Forest Classifier, **Scikit-Learn, trained from scratch**.
* **Config:** `RandomForestClassifier(n_estimators=150, max_depth=7, min_samples_leaf=3, random_state=42)`

**Structure:**

```
INPUT  (N, P, K, temperature, humidity, ph, rainfall)  — 7 standardized features
   │
   ▼
┌───────────────────────────────────────────────────────────────────┐
│  150 independently bagged decision trees, each capped at            │
│  max_depth=7, min_samples_leaf=3                                    │
│   Tree 1   Tree 2   Tree 3  ...  Tree 150   →  each casts a vote     │
└───────────────────────────────────────────────────────────────────┘
   │
   ▼
Majority vote → 1 of 5 classes: {chickpea, coffee, cotton, maize, rice}
```

* **Data:** 1,200 rows, 7 numeric features, 960 train / 240 validation (stratified 80/20).

**Metrics:**

| | Train | Validation |
|---|---|---|
| Accuracy | 93.02% | **87.92%** |
| Weighted Precision | — | 0.8804 |
| Weighted Recall | — | 0.8792 |
| Weighted F1 | — | 0.8760 |

Per-class (validation): chickpea F1 1.00 (46) · coffee F1 0.87 (51) · cotton F1 0.87 (48) · maize F1 0.71 (43) · rice F1 0.91 (52).

**Feature importance:** humidity 27.5% · rainfall 24.9% · N 16.3% · K 14.8% · P 9.1% · temperature 4.3% · pH 3.1%.

---

## 7. Lettuce Growth Predictor

* **File:** `models/lettuce_predictor.pkl` (dict: `StandardScaler` + `Ridge`)
* **Type:** Ridge Regression (L2-regularized linear model), **Scikit-Learn, trained from scratch**.
* **Config:** `Ridge(alpha=1.5)`

**Structure:**

```
INPUT  (dli, air_temp, ec, ph, water_temp)  — 5 standardized features
   │
   ▼
ŷ = w·x + b,  fit by minimizing  Σ(yᵢ − ŷᵢ)² + α·Σwⱼ²   (α = 1.5)
   5 learned weights + 1 bias = 6 total parameters
   │
   ▼
OUTPUT  growth_days (continuous)
```

* **Data:** 600 cultivation cycles, 480 train / 120 validation.

**Metrics:**

| | Train | Validation |
|---|---|---|
| R² | 0.8363 | **0.8560** |
| MAE | — | 0.8415 days |
| MSE | — | 1.0621 days² |
| RMSE | — | 1.0306 days |

---

## 8. Greenhouse Climate Forecaster

* **File:** `models/greenhouse_forecaster.pkl` (dict: `StandardScaler` + `RandomForestRegressor`)
* **Type:** Random Forest Regressor over lag-engineered time-series features, **Scikit-Learn, trained from scratch**.
* **Config:** `RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=3, random_state=42)`

**Structure:**

```
INPUT  (temperature, humidity, co2, temp_lag_1, temp_lag_2, temp_lag_6, hum_lag_1, co2_lag_1)
   │        8 standardized features, chronologically split (no shuffling)
   ▼
100 bagged regression trees, each capped at max_depth=8, min_samples_leaf=3
   │
   ▼
Average prediction across all 100 trees
   │
   ▼
OUTPUT  target_temperature (1 hour ahead, t+6 steps)
```

* **Data:** 1,441 sensor ticks (10-min interval, 10 days) → 1,143 train / 286 validation (chronological split, no leakage).

**Metrics:**

| | Train | Validation |
|---|---|---|
| R² | 0.9559 | **0.8913** |
| MAE | — | 0.7808 °C |
| RMSE | — | 0.9765 °C |

**Feature importance:** `co2_lag_1` 79.7% (shares the same diurnal sinusoid as temperature in this dataset, so it acts as a strong proxy) · co2 5.2% · temperature 3.8% · humidity 3.8% · hum_lag_1 2.7% · temp_lag_1 2.0% · temp_lag_6 1.8% · temp_lag_2 1.0%.
