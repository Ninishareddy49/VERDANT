# VERDANT — Project Setup & Initialization Guide

This document describes how to initialize, configure, and launch the VERDANT Urban Farming and Intelligent Agronomy platform from scratch on a new machine.

---

## 1. System Prerequisites

Before initializing the project, ensure you have the following installed on your operating system:
* **Python 3.10+** (Ensure Python is added to your system `PATH`)
* **Node.js 18+** (Includes `npm`)
* **Internet Connection** (Required for the first-run download of pre-trained HuggingFace model weights and package installations)

---

## 2. Directory Layout & Structure

The repository contains two main directories:
* `/backend` — FastAPI server exposing local PyTorch image inference (Leaf Disease ViT and Species ViT), Sentence-Transformer RAG semantic search, and the Ollama Cloud Chat integration.
* `/frontend` — React 19 SPA built with TypeScript, Vite, Tailwind CSS v4, and Framer Motion.
* `train_models.ipynb` — Jupyter Notebook detailing the data cleaning, exploratory data analysis (EDA), scaling, splitting, training, and metrics visualizations for the ML pipelines.

---

## 3. Step-by-Step Initialization

Follow these commands to install dependencies and configure the environment:

### Step A: Backend Environment Setup
Navigate to the `backend` folder, create a virtual environment, and install dependencies:
```powershell
# Navigate into backend directory
cd backend

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.\.venv\Scripts\activate

# Upgrade pip and install all required modules (PyTorch, FastAPI, Pandas, Scikit-Learn, etc.)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step B: Configure Credentials & Environment Variables
In the `backend/` directory, create a configuration file named `.env` to store your local credentials and API keys:
```env
# backend/.env
OLLAMA_API_KEY=your_ollama_cloud_api_key_here
OLLAMA_MODEL=gemma3:4b
```
> **Note:** If `OLLAMA_API_KEY` is omitted, the AI Advisor module will automatically fall back to its internal, curated local Knowledge Base (grounding context) without crashing.

### Step C: Pre-downloading HuggingFace Weights (Optional)
To avoid delay on the first API request, pre-download the model weights using the download script:
```powershell
python download_models.py
```
This script will download and cache the following models in your home directory (`~/.cache/huggingface/`):
* `wambugu71/crop_leaf_diseases_vit` (~350 MB)
* `dima806/medicinal_plants_image_detection` (~350 MB)
* `sentence-transformers/all-MiniLM-L6-v2` (~90 MB)

### Step D: Frontend Package Installation
In a separate terminal window, navigate to the `frontend` folder and install the required Node modules:
```powershell
# Navigate into frontend directory
cd ../frontend

# Install node dependencies
npm install
```

---

## 4. Running the Application

### Method 1: Automatic Launch (Windows)
Run the one-shot PowerShell script at the root directory of the project:
```powershell
./start.ps1
```
This will open two terminal instances launching:
1. **FastAPI Backend Server** on `http://127.0.0.1:8000`
2. **Vite Frontend Server** on `http://localhost:5173`

### Method 2: Manual Launch
* **Backend:**
  ```powershell
  cd backend
  .\.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
  ```
* **Frontend:**
  ```powershell
  cd frontend
  npm run dev
  ```
Open `http://localhost:5173` in your browser.

---

## 5. Model Training Pipeline

If you need to show the machine learning training pipeline to a professor or reviewer:
1. Ensure your backend virtual environment is activated: `cd backend && .\.venv\Scripts\activate`
2. Open and run the Jupyter notebook `train_models.ipynb` located in the root folder.
3. This notebook executes cleaning, EDA, feature scaling, model training, metric evaluations, and generates the serialized model outputs in `/models`.
