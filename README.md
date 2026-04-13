# 🫁 PneumoScan AI — Chest X-Ray Diagnostic Assistant

> **Disclaimer:** This is an educational/portfolio project only. Not a substitute for clinical diagnosis. Always consult a licensed physician.

A full-stack medical AI web app that analyzes chest X-ray images and generates tailored recommendations for **children and pregnant women** using a CNN + RAG pipeline.

---

## How It Works

1. Upload a chest X-ray image
2. CNN model classifies it as **Normal** or **Pneumonia**
3. LangChain RAG retrieves relevant chunks from medical PDFs
4. LLM generates a structured report tailored to the patient (age + pregnancy status)

```
Browser → POST /analyze → Flask → CNN (.keras) + FAISS RAG + LLM → JSON Report → Browser
```

---

## Project Structure

```
your-project/
├── app.py                    # Flask backend (CNN + RAG + LLM pipeline)
├── index.html                # Frontend web interface
├── requirements.txt          # Python dependencies
├── my_model_2.keras          # Your trained CNN model
├── 1_2_3_4_5_merged.pdf      # Medical knowledge base PDF
└── faiss_chest_index/        # FAISS vector index (auto-generated on first run)
```

---

## Setup & Running

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure `app.py`
Edit the CONFIG section at the top:
```python
CNN_MODEL_PATH   = "my_model_2.keras"
IMAGE_SIZE       = (256, 256)
PDF_PATHS        = ["1_2_3_4_5_merged.pdf"]

os.environ["OPENAI_API_KEY"] = "sk-or-v1-your-openrouter-key"
OPENROUTER_MODEL = "nvidia/nemotron-super-49b-v1"   # no :free suffix
```

### 3. Add index route to `app.py`
Add these lines just before `if __name__ == "__main__":` so Flask serves the HTML:
```python
from flask import send_from_directory

@app.route("/")
def index():
    return send_from_directory(".", "index.html")
```

### 4. Start the server
```bash
python app.py
```

### 5. Open the app
Go to [http://localhost:5000](http://localhost:5000) in your browser.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML / CSS / JavaScript |
| Backend | Python Flask + Flask-CORS |
| CNN Model | TensorFlow / Keras (binary sigmoid) |
| Embeddings | HuggingFace `BAAI/bge-base-en-v1.5` |
| Vector DB | FAISS |
| RAG | LangChain |
| LLM | OpenRouter (Nemotron) |

---

## CNN Model Details

- Input: `(256, 256, 3)` RGB image normalized to `[0, 1]`
- Output: single sigmoid neuron → `> 0.5` = Pneumonia, `≤ 0.5` = Normal
- Confidence = raw score for Pneumonia, `1 - raw` for Normal

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Server error 500 | Check Flask terminal. Likely wrong OpenRouter model name (remove `:free`) or bad API key |
| 404 on localhost:5000 | Add the index route to `app.py` (Step 3 above) |
| Connection error in browser | Make sure `python app.py` is running on port 5000 |
| FAISS error | Delete `faiss_chest_index/` folder and restart — it rebuilds automatically |

---

## Author

**Neeraj** — ML Portfolio Project  
Focus: Computer Vision + NLP/LLM Applications
