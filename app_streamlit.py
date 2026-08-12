import os
import io
import json
import re
import shutil
import time
import requests
from PIL import Image
import streamlit as st
from langchain_core.embeddings import Embeddings
st.set_page_config(
    page_title="Chest X-Ray Analyzer",
    page_icon="🫁",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }
.stApp { background: #0d1117; color: #c9d1d9; }
.card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.pill {
    display: inline-block;
    padding: 2px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    letter-spacing: .04em;
}
.pill-normal { background:#0d3321; color:#3fb950; border:1px solid #238636; }
.pill-pneumonia{ background:#3d1c1c; color:#f85149; border:1px solid #da3633; }
.pill-warning { background:#2d2000; color:#e3b341; border:1px solid #9e6a03; }
.metric-box {
    background:#0d1117;
    border:1px solid #21262d;
    border-radius:6px;
    padding:.8rem 1rem;
    text-align:center;
}
.metric-label { font-size:.72rem; color:#8b949e; text-transform:uppercase; letter-spacing:.08em; }
.metric-value { font-size:1.6rem; font-family:'IBM Plex Mono',monospace; font-weight:600; color:#58a6ff; }
.med-chip {
    display:inline-block;
    background:#1c2d45;
    border:1px solid #1f6feb;
    color:#79c0ff;
    border-radius:4px;
    padding:3px 10px;
    margin:3px;
    font-size:.82rem;
    font-family:'IBM Plex Mono',monospace;
}
.section-title {
    font-size:.7rem;
    text-transform:uppercase;
    letter-spacing:.12em;
    color:#8b949e;
    margin-bottom:.4rem;
    font-family:'IBM Plex Mono',monospace;
}
.disclaimer-box {
    background:#1c1c1c;
    border-left:3px solid #e3b341;
    padding:.7rem 1rem;
    border-radius:0 6px 6px 0;
    color:#8b949e;
    font-size:.85rem;
}
div[data-testid="stFileUploader"] {
    border: 2px dashed #30363d;
    border-radius: 8px;
    padding: .5rem;
}
div[data-testid="stFileUploader"]:hover {
    border-color: #58a6ff;
}
</style>
""", unsafe_allow_html=True)

MODEL_API_URL = "https://pneumoscan-model-api.onrender.com/predict"
CLASSES = ["Normal", "Pneumonia"]
PDF_PATHS = ["final_used_pdf.pdf"]
FAISS_INDEX_DIR = "faiss_chest_index"

OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b:free"

PROMPT_TEMPLATE = """\
You are a cautious, evidence-based medical assistant AI.
You provide guidance on chest conditions detected from X-rays.

STRICT RULES:
- Base your response ONLY on the CONTEXT provided.
- If information is missing, say "Not enough information in the knowledge base."
- NEVER give a definitive prescription. Only suggest possible approaches.
- Always recommend consulting a qualified doctor.
- Pay special attention to safety considerations for children and pregnant women.
- If the condition is "Normal", reassure but still recommend a doctor visit.

━━━━━━━━━━━━━━━━━━━━━━
PATIENT PROFILE:
- Detected Condition : {condition}
- CNN Confidence      : {confidence}%
- Patient Age         : {age} years old
- Pregnant            : {pregnant}
━━━━━━━━━━━━━━━━━━━━━━
KNOWLEDGE BASE CONTEXT:
{context}
━━━━━━━━━━━━━━━━━━━━━━

Return a JSON object (no markdown, no backticks) with this exact structure:
{{
  "summary": "2-3 sentences explaining the detected condition",
  "severity": "1-2 sentences on severity based on confidence level",
  "medicines": ["medicine1", "medicine2"],
  "treatmentText": "2-3 sentences on treatment from context",
  "specialConsiderations": "2-3 sentences specific to age/pregnancy including what to avoid",
  "emergencySigns": "2-3 sentences on warning signs needing emergency care",
  "disclaimer": "1-2 sentence safety note"
}}
"""

def _require_secret(key: str) -> str:
    try:
        value = st.secrets[key]
    except (KeyError, FileNotFoundError):
        st.error(
            f"Missing secret `{key}`. Add it in Streamlit Cloud → "
            f"**Manage app → Settings → Secrets** (or `.streamlit/secrets.toml` locally), "
            f"e.g.\n\n```toml\n{key} = \"...\"\n```"
        )
        st.stop()
    if not value:
        st.error(f"Secret `{key}` is set but empty.")
        st.stop()
    return value

OPENAI_API_KEY = _require_secret("OPENAI_API_KEY")           
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

class OpenRouterEmbeddings(Embeddings):
    """Minimal LangChain-compatible Embeddings implementation
    (embed_documents / embed_query) backed by OpenRouter's /embeddings API."""

    def __init__(self, model: str, api_key: str,
                 base_url: str = OPENROUTER_BASE,
                 batch_size: int = 64, timeout: int = 60):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout

    def _embed_batch(self, texts):
        resp = requests.post(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "input": texts},
            timeout=self.timeout,
        )
        if not resp.ok:
            raise RuntimeError(
                f"OpenRouter embeddings request failed ({resp.status_code}): {resp.text[:300]}"
            )
        data = resp.json()
        return [item["embedding"] for item in data["data"]]

    def embed_documents(self, texts):
        texts = list(texts)
        out = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._embed_batch(texts[i:i + self.batch_size]))
        return out

    def embed_query(self, text):
        return self._embed_batch([text])[0]


@st.cache_resource(show_spinner="Loading embeddings & knowledge base…")
def load_rag():
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    emb = OpenRouterEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

    def _build_index():
        all_docs = []
        for p in PDF_PATHS:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Knowledge-base PDF not found: {p}. "
                    f"Make sure it's committed to the repo root."
                )
            all_docs.extend(PyPDFLoader(p).load())

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600, chunk_overlap=120,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(all_docs)
        vs = FAISS.from_documents(chunks, emb)
        vs.save_local(FAISS_INDEX_DIR)
        return vs

    if os.path.exists(FAISS_INDEX_DIR):
        try:
            vs = FAISS.load_local(
                FAISS_INDEX_DIR, emb, allow_dangerous_deserialization=True
            )
        except Exception:
            shutil.rmtree(FAISS_INDEX_DIR, ignore_errors=True)
            vs = _build_index()
    else:
        vs = _build_index()

    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    llm = ChatOpenAI(
        model=OPENROUTER_MODEL,
        base_url=OPENROUTER_BASE,
        api_key=OPENAI_API_KEY,
        temperature=0.4,
    )

    prompt = PromptTemplate(
        input_variables=["condition", "confidence", "context", "age", "pregnant"],
        template=PROMPT_TEMPLATE,
    )
    chain = prompt | llm | StrOutputParser()
    return retriever, chain

def classify_xray(image_bytes: bytes, filename: str = "xray.jpg"):
    """
    Calls the hosted FastAPI model at MODEL_API_URL/predict.
    Request:  multipart/form-data, field name "file" -> image bytes
    Response: JSON containing a class/label field and a confidence field,
              e.g. {"class": "PNEUMONIA", "confidence": 0.87}

    Render's free tier spins containers down when idle, so the first call
    after inactivity can take 30-60s to "cold start" — we retry a couple
    of times with backoff instead of failing immediately on a timeout.
    """
    files = {"file": (filename, image_bytes, "image/jpeg")}

    last_err = None
    for attempt in range(3):
        try:
            resp = requests.post(MODEL_API_URL, files=files, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.exceptions.RequestException as e:
            last_err = e
            time.sleep(5 * (attempt + 1))  
    else:
        raise RuntimeError(
            f"Could not reach the model API at {MODEL_API_URL} after 3 attempts "
            f"(it may be cold-starting on Render's free tier — try again in a moment). "
            f"Last error: {last_err}"
        )

    label_raw = (
        data.get("class")
        or data.get("label")
        or data.get("prediction")
        or data.get("result")
    )
    if label_raw is None:
        raise ValueError(f"Unexpected response shape from model API: {data}")

    label_str = str(label_raw).strip().lower()
    if label_str in ("1", "pneumonia", "true"):
        condition = "Pneumonia"
    elif label_str in ("0", "normal", "false"):
        condition = "Normal"
    else:
        condition = next(
            (c for c in CLASSES if c.lower() == label_str), str(label_raw)
        )

    conf_raw = data.get("confidence", data.get("probability", data.get("score")))
    if conf_raw is None:
        raise ValueError(f"Unexpected response shape from model API: {data}")
    confidence = float(conf_raw)
    if confidence > 1.0:  
        confidence /= 100.0

    pneu_conf = confidence if condition == "Pneumonia" else (1.0 - confidence)
    all_probs = {"Normal": round(1.0 - pneu_conf, 4), "Pneumonia": round(pneu_conf, 4)}
    return condition, round(confidence, 4), all_probs


def format_docs(docs) -> str:
    return "\n\n---\n\n".join(
        f"[Page {d.metadata.get('page', '?')}]\n{d.page_content}" for d in docs
    )


_FALLBACK_REPORT = {
    "summary": "The AI report could not be generated (see error above). "
                "Please retry, or consult a physician directly.",
    "severity": "Not available.",
    "medicines": [],
    "treatmentText": "Not available.",
    "specialConsiderations": "Not available.",
    "emergencySigns": "Seek emergency care for severe shortness of breath, "
                       "chest pain, bluish lips/face, confusion, or high fever "
                       "that doesn't respond to treatment.",
    "disclaimer": "Always consult a qualified doctor.",
}


def run_rag_llm(condition, confidence, age, pregnant, retriever, chain) -> dict:
    query = (
        f"{condition} treatment guidelines "
        + ("children pediatric" if age < 18 else "adult")
        + (" pregnancy pregnant women" if pregnant else "")
        + " medications precautions management"
    )
    docs = retriever.invoke(query)
    context = format_docs(docs)

    raw = chain.invoke({
        "condition": condition,
        "confidence": f"{confidence * 100:.1f}",
        "age": age,
        "pregnant": "Yes" if pregnant else "No",
        "context": context,
    })

    clean = raw.replace("```json", "").replace("```", "").strip()
    match = re.search(r'\{.*\}', clean, re.DOTALL)
    try:
        return json.loads(match.group() if match else clean)
    except (json.JSONDecodeError, AttributeError):
        st.warning(
            "The model didn't return valid JSON, so a fallback report is shown below. "
            "You can try clicking **Analyze X-Ray** again."
        )
        fallback = dict(_FALLBACK_REPORT)
        fallback["summary"] = f"Raw model output could not be parsed: {raw[:300]}"
        return fallback

st.markdown("# 🫁 Chest X-Ray Analyzer")
st.markdown(
    "<p style='color:#8b949e;font-size:.9rem;margin-top:-.5rem;'>"
    "CNN classification · RAG-powered medical report · Strictly informational</p>",
    unsafe_allow_html=True,
)
st.divider()

with st.sidebar:
    st.markdown("### Patient Profile")
    age = st.number_input("Age (years)", min_value=0, max_value=120, value=30, step=1)
    pregnant = st.checkbox("Pregnant", value=False)
    st.markdown("---")
    st.markdown(
        "<div class='disclaimer-box'>⚠️ This tool is for informational purposes only. "
        "Always consult a qualified physician before making any medical decisions.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        "<p style='font-size:.75rem;color:#484f58;'>Model: CNN via hosted API + RAG (LangChain + FAISS)<br>"
        f"Embedding: {EMBEDDING_MODEL}<br>LLM: {OPENROUTER_MODEL}</p>",
        unsafe_allow_html=True,
    )

col_upload, col_preview = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("#### Upload X-Ray Image")
    uploaded_file = st.file_uploader(
        "Drop a chest X-ray (JPG / PNG / DICOM-to-PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    analyze_btn = st.button(
        "🔬 Analyze X-Ray",
        type="primary",
        disabled=uploaded_file is None,
        use_container_width=True,
    )

with col_preview:
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded X-Ray", use_container_width=True)
    else:
        st.markdown(
            "<div style='height:220px;display:flex;align-items:center;justify-content:center;"
            "border:1px dashed #30363d;border-radius:8px;color:#484f58;font-size:.9rem;'>"
            "Preview will appear here</div>",
            unsafe_allow_html=True,
        )

if analyze_btn and uploaded_file:
    image_bytes = uploaded_file.getvalue()

    try:
        with st.spinner("Running CNN classifier (calling model API)…"):
            condition, confidence, all_probs = classify_xray(
                image_bytes, filename=uploaded_file.name
            )
    except Exception as e:
        st.error(f"CNN classification failed: {e}")
        st.stop()

    st.divider()
    st.markdown("### Classification Result")

    pill_class = "pill-normal" if condition == "Normal" else "pill-pneumonia"
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"<div class='metric-box'><div class='metric-label'>Detected Condition</div>"
            f"<div style='margin-top:.4rem'><span class='pill {pill_class}'>{condition}</span></div></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div class='metric-box'><div class='metric-label'>Confidence</div>"
            f"<div class='metric-value'>{confidence*100:.1f}%</div></div>",
            unsafe_allow_html=True,
        )
    with col3:
        norm_pct = all_probs['Normal'] * 100
        pneu_pct = all_probs['Pneumonia'] * 100
        st.markdown(
            f"<div class='metric-box'><div class='metric-label'>Class Probabilities</div>"
            f"<div style='font-size:.82rem;margin-top:.4rem;font-family:IBM Plex Mono,monospace;'>"
            f"<span style='color:#3fb950'>Normal: {norm_pct:.1f}%</span><br>"
            f"<span style='color:#f85149'>Pneumonia: {pneu_pct:.1f}%</span></div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    bar_color = "#3fb950" if condition == "Normal" else "#f85149"
    st.markdown(
        f"<div class='section-title'>Confidence bar</div>"
        f"<div style='background:#21262d;border-radius:4px;height:8px;overflow:hidden;'>"
        f"<div style='width:{confidence*100:.1f}%;height:100%;background:{bar_color};"
        f"border-radius:4px;transition:width .6s ease'></div></div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("### 📋 Medical Report")

    try:
        with st.spinner("Retrieving knowledge base & generating report…"):
            retriever, chain = load_rag()
            report = run_rag_llm(condition, confidence, age, pregnant, retriever, chain)
    except Exception as e:
        st.error(
            f"Report generation failed: {e}\n\n"
            f"If this mentions an embeddings format error, verify `OPENAI_API_KEY` "
            f"in secrets is a valid **OpenRouter** key (starts with `sk-or-v1-`) and that "
            f"`EMBEDDING_MODEL` ({EMBEDDING_MODEL}) is still listed as an active embeddings "
            f"model on OpenRouter — free models occasionally get rotated out."
        )
        report = dict(_FALLBACK_REPORT)

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown(
            f"<div class='card'><div class='section-title'>Summary</div>"
            f"<p style='margin:0;line-height:1.6'>{report.get('summary','—')}</p></div>",
            unsafe_allow_html=True,
        )
    with c_right:
        st.markdown(
            f"<div class='card'><div class='section-title'>Severity Assessment</div>"
            f"<p style='margin:0;line-height:1.6'>{report.get('severity','—')}</p></div>",
            unsafe_allow_html=True,
        )

    meds = report.get("medicines", [])
    if meds:
        chips = "".join(f"<span class='med-chip'>{m}</span>" for m in meds)
        st.markdown(
            f"<div class='card'><div class='section-title'>Suggested Medicine Approaches</div>"
            f"<div style='margin-top:.3rem'>{chips}</div></div>",
            unsafe_allow_html=True,
        )

    for label, key in [
        ("Treatment Guidance", "treatmentText"),
        ("Special Considerations", "specialConsiderations"),
        ("🚨 Emergency Warning Signs", "emergencySigns"),
    ]:
        value = report.get(key, "—")
        border = "#da3633" if "Emergency" in label else "#30363d"
        st.markdown(
            f"<div class='card' style='border-color:{border}'>"
            f"<div class='section-title'>{label}</div>"
            f"<p style='margin:0;line-height:1.6'>{value}</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<div class='disclaimer-box'>{report.get('disclaimer','Always consult a qualified doctor.')}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    full_result = {
        "condition": condition,
        "confidence": confidence,
        "normalProb": all_probs["Normal"],
        "pneumoniaProb": all_probs["Pneumonia"],
        **report,
    }
    st.download_button(
        label="⬇️ Download Report (JSON)",
        data=json.dumps(full_result, indent=2),
        file_name="xray_report.json",
        mime="application/json",
        use_container_width=True,
    )
