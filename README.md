
# 🛸 Alien Signal Detective — ML Mystery Game # 

An AI-powered data detective game where you classify alien signals, cluster their
origins, decode hidden clues with a local LLM, and solve a deep-space mystery.

---

## Quick Start (macOS / MacBook Air)

### Step 1 — Install prerequisites (one-time)

**Python 3.10+** (check with `python3 --version`). If missing:
```bash
brew install python@3.12
```

**Ollama** (local LLM engine):
1. Download from https://ollama.com/download and install the Mac app
2. Open the Ollama app (it adds a menu bar icon)
3. Pull the model:
```bash
ollama pull llama3.2
```

### Step 2 — Run the setup script

```bash
cd alien-detective
bash setup.sh
```

This will:
- Create a `.venv` virtual environment
- Install all Python packages
- Generate the signal dataset (`data/signals_dataset.csv`)
- Check Ollama availability

### Step 3 — Launch the game

```bash
source .venv/bin/activate
streamlit run app.py
```

The game opens at **http://localhost:8501** in your browser.

---

## Launch from VS Code

1. Open the `alien-detective/` folder in VS Code (`File → Open Folder`)
2. VS Code will prompt you to install recommended extensions — click **Install All**
3. Select the Python interpreter: `Ctrl+Shift+P` → `Python: Select Interpreter` →
   choose `.venv/bin/python`
4. Press **F5** to launch (uses the `Run Alien Detective (Streamlit)` launch config)
5. The game opens at http://localhost:8501

---

## Game Flow

| Mission | What you do |
|---------|-------------|
| 🛸 Intro | Enter your analyst name, read the mission brief |
| 📡 Signal Lab | Explore 500 intercepted alien signals — maps, frequency charts, timeline |
| 🤖 Classifier | Train a RandomForest, evaluate accuracy, inspect anomaly probabilities |
| 🗺 Cluster Map | Run KMeans + DBSCAN, find the tight anomaly cluster in Sector 7-Delta |
| 🔍 Clue Decoder | Pick anomaly signals, let ARIA (local LLM) decode hidden clue fragments |
| 📋 Case Board | Review all evidence — radar chart, gauges, evidence log |
| 💡 Final Deduction | Submit your theory — ARIA evaluates and reveals the truth |

---

## Project Structure

```
alien-detective/
├── app.py                  ← Main Streamlit app (run this)
├── requirements.txt        ← Python dependencies
├── setup.sh                ← One-click Mac setup script
├── src/
│   ├── data_generator.py   ← Synthetic alien signal dataset
│   ├── ml_models.py        ← RandomForest + KMeans/DBSCAN
│   ├── llm_engine.py       ← Ollama local LLM integration
│   ├── game_state.py       ← Streamlit session state manager
│   └── visualizations.py  ← Plotly chart builders
├── data/                   ← Auto-generated CSV dataset
├── models/                 ← Saved trained model files
└── .vscode/
    ├── launch.json         ← F5 run config for VS Code
    ├── settings.json       ← Python interpreter + formatting
    └── extensions.json     ← Recommended extensions
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| UI | Streamlit 1.35 |
| Classification | scikit-learn RandomForest |
| Clustering | KMeans + DBSCAN + PCA |
| LLM / NLP | Ollama (llama3.2, local) |
| Visualisation | Plotly |
| Data | pandas + numpy |

---

## Offline Mode

ARIA's NLP features require Ollama running locally. If Ollama is not available,
the game still works — missions 1–3 (Signal Lab, Classifier, Clustering) are fully
functional. ARIA messages show an `[OFFLINE MODE]` prefix instead.

To start Ollama: open the Ollama app from your Applications folder, or run:
```bash
ollama serve
```

---

## Troubleshooting

**`ModuleNotFoundError`** — Make sure your virtual environment is activated:
```bash
source .venv/bin/activate
```

**`streamlit: command not found`** — Activate the venv first (see above).

**ARIA shows offline** — Open the Ollama app, then click "Check ARIA Status" in the sidebar.

**Wrong Python interpreter in VS Code** — Press `Ctrl+Shift+P` → `Python: Select Interpreter`
→ select the `.venv` option.
