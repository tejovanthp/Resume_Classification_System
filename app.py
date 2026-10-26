import streamlit as st
import pandas as pd
import re
import nltk
import joblib
import fitz
import textwrap

from pathlib import Path
from datetime import datetime
from docx import Document
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ResumeAI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PATH CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "resume_model.pkl"
COMPARISON_PATH = BASE_DIR / "results" / "model_comparison.png"
CONFUSION_PATH = BASE_DIR / "results" / "confusion_matrix.png"


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       GLOBAL
       ===================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(37, 99, 235, 0.18),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(14, 165, 233, 0.12),
                transparent 30%
            ),
            #070b16;
        color: #f8fafc;
    }

    .main .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* =====================================================
       SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #0b1120 0%,
                #080d18 100%
            );
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }

    section[data-testid="stSidebar"] h1 {
        color: #60a5fa;
    }

    /* =====================================================
       TYPOGRAPHY
       ===================================================== */

    h1 {
        font-weight: 800 !important;
        letter-spacing: -1px;
    }

    h2 {
        font-weight: 700 !important;
    }

    h3 {
        font-weight: 650 !important;
    }

    p {
        color: #cbd5e1;
    }

    /* =====================================================
       HERO
       ===================================================== */

    .hero {
        padding: 45px 40px;
        border-radius: 25px;
        background:
            linear-gradient(
                135deg,
                rgba(30, 64, 175, 0.85),
                rgba(8, 47, 73, 0.85)
            );
        border: 1px solid rgba(96,165,250,0.25);
        box-shadow: 0 20px 60px rgba(0,0,0,0.35);
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 250px;
        height: 250px;
        right: -80px;
        top: -100px;
        border-radius: 50%;
        background: rgba(96,165,250,0.08);
        filter: blur(5px);
    }

    .hero-title {
        font-size: 46px;
        font-weight: 850;
        margin-bottom: 8px;
        position: relative;
        z-index: 1;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #cbd5e1;
        line-height: 1.6;
        max-width: 850px;
        position: relative;
        z-index: 1;
    }

    .hero-status {
        display: inline-flex;
        align-items: center;
        margin-top: 20px;
        padding: 7px 13px;
        border-radius: 999px;
        background: rgba(15,23,42,0.5);
        border: 1px solid rgba(96,165,250,0.22);
        color: #bfdbfe;
        font-size: 13px;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        background: #22c55e;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        box-shadow: 0 0 12px rgba(34,197,94,0.7);
    }

    /* =====================================================
       CARDS & CONTAINERS
       ===================================================== */

    .card {
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(148,163,184,0.13);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 10px 35px rgba(0,0,0,0.22);
        margin-bottom: 20px;
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    }

    .card:hover {
        transform: translateY(-3px);
        border-color: rgba(96,165,250,0.35);
        box-shadow: 0 15px 40px rgba(37,99,235,0.14);
    }

    .kpi-card {
        background: linear-gradient(145deg, rgba(15,23,42,0.95), rgba(17,24,39,0.82));
        border: 1px solid rgba(96,165,250,0.14);
        border-radius: 18px;
        padding: 22px;
        min-height: 125px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.20);
        transition: all 0.25s ease;
    }

    .kpi-card:hover {
        transform: translateY(-4px);
        border-color: rgba(96,165,250,0.40);
        box-shadow: 0 14px 35px rgba(37,99,235,0.16);
    }

    .kpi-label {
        color: #94a3b8;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1.1px;
        margin-bottom: 8px;
    }

    .kpi-value {
        color: #f8fafc;
        font-size: 25px;
        font-weight: 750;
    }

    .kpi-description {
        color: #64748b;
        font-size: 12px;
        margin-top: 5px;
    }

    .model-card {
        background: linear-gradient(145deg, rgba(15,23,42,0.95), rgba(17,24,39,0.8));
        border: 1px solid rgba(96,165,250,0.16);
        border-radius: 18px;
        padding: 20px;
        min-height: 135px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        transition: transform 0.25s ease, border-color 0.25s ease;
    }

    .model-card:hover {
        transform: translateY(-4px);
        border-color: rgba(96,165,250,0.38);
    }

    .model-name {
        color: #94a3b8;
        font-size: 13px;
        margin-bottom: 8px;
        line-height: 1.4;
    }

    .model-result {
        color: #60a5fa;
        font-size: 20px;
        font-weight: 750;
        line-height: 1.25;
    }

    .result-card {
        padding: 40px;
        border-radius: 25px;
        text-align: center;
        background: linear-gradient(145deg, rgba(30,41,59,0.95), rgba(15,23,42,0.95));
        border: 1px solid rgba(96,165,250,0.25);
        box-shadow: 0 15px 50px rgba(0,0,0,0.35);
        margin: 25px 0;
        position: relative;
        overflow: hidden;
    }

    .result-card::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #2563eb, #0284c7, #60a5fa);
    }

    .result-label {
        color: #94a3b8;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .result-value {
        font-size: 42px;
        font-weight: 850;
        margin-top: 10px;
        letter-spacing: -1px;
        color: #38bdf8;
    }

    .agreement-card {
        background: rgba(15,23,42,0.78);
        border: 1px solid rgba(96,165,250,0.14);
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 20px;
    }

    .agreement-title {
        color: #94a3b8;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1.3px;
    }

    .agreement-value {
        font-size: 28px;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 5px;
    }

    .agreement-bar {
        height: 8px;
        background: rgba(148,163,184,0.12);
        border-radius: 20px;
        margin-top: 13px;
        overflow: hidden;
    }

    .agreement-fill {
        height: 100%;
        border-radius: 20px;
        background: linear-gradient(90deg, #2563eb, #38bdf8);
    }

    /* =====================================================
       FILE CARD
       ===================================================== */

    .file-card {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(96, 165, 250, 0.2);
        border-radius: 16px;
        padding: 16px 20px;
        margin: 15px 0 25px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .file-name {
        font-size: 15px;
        font-weight: 650;
        color: #f8fafc;
    }

    .file-info {
        font-size: 12px;
        color: #94a3b8;
        margin-top: 4px;
    }

    .file-type {
        background: rgba(37, 99, 235, 0.2);
        color: #60a5fa;
        border: 1px solid rgba(96, 165, 250, 0.3);
        border-radius: 8px;
        padding: 6px 12px;
        font-size: 12px;
        font-weight: 700;
    }

    /* =====================================================
       PIPELINE
       ===================================================== */

    .pipeline {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
        overflow-x: auto;
        padding: 20px 5px;
    }

    .pipeline-step {
        min-width: 125px;
        padding: 17px;
        border-radius: 15px;
        text-align: center;
        background: rgba(30,41,59,0.8);
        border: 1px solid rgba(96,165,250,0.15);
        color: #cbd5e1;
        font-size: 13px;
        transition: all 0.25s ease;
    }

    .pipeline-step:hover {
        transform: translateY(-4px);
        border-color: rgba(96,165,250,0.4);
        background: rgba(30,64,175,0.16);
    }

    .pipeline-number {
        display: block;
        width: 28px;
        height: 28px;
        line-height: 28px;
        margin: 0 auto 8px;
        border-radius: 50%;
        background: rgba(37,99,235,0.18);
        color: #93c5fd;
        font-weight: 700;
    }

    /* =====================================================
       INPUTS & BUTTONS
       ===================================================== */

    .stButton > button {
        width: 100%;
        border-radius: 14px;
        border: 1px solid rgba(96,165,250,0.35);
        background: linear-gradient(135deg, #2563eb, #0284c7);
        color: white;
        font-size: 15px;
        font-weight: 700;
        padding: 12px 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        box-shadow: 0 8px 25px rgba(37,99,235,0.25);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(37,99,235,0.4);
        border-color: #60a5fa;
    }

    textarea {
        background: rgba(15,23,42,0.9) !important;
        color: #f8fafc !important;
        border: 1px solid rgba(148,163,184,0.2) !important;
        border-radius: 17px !important;
        font-size: 14px !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed rgba(96,165,250,0.35) !important;
        border-radius: 18px !important;
        background: rgba(15,23,42,0.7) !important;
    }

    div[data-testid="stMetric"] {
        background: rgba(15,23,42,0.72);
        border: 1px solid rgba(148,163,184,0.12);
        padding: 18px;
        border-radius: 18px;
    }

    .status-box {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 12px 15px;
        border-radius: 12px;
        background: rgba(15,23,42,0.72);
        border: 1px solid rgba(148,163,184,0.12);
        color: #cbd5e1;
        font-size: 13px;
    }

    .status-online {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #22c55e;
        box-shadow: 0 0 10px rgba(34,197,94,0.7);
    }

    .history-card {
        padding: 15px 18px;
        border-radius: 14px;
        background: rgba(15,23,42,0.65);
        border: 1px solid rgba(148,163,184,0.10);
        margin-bottom: 8px;
    }

    .history-text {
        color: #cbd5e1;
        font-size: 13px;
        line-height: 1.5;
    }

    .history-meta {
        color: #64748b;
        font-size: 11px;
        margin-top: 5px;
    }

    .footer {
        text-align: center;
        color: #64748b;
        padding: 30px 0;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# NLTK SETUP
# =========================================================

@st.cache_resource
def setup_nltk():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4")
    ]

    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


setup_nltk()


# =========================================================
# NLP & MODELS LOAD
# =========================================================

@st.cache_resource
def initialize_nlp():
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    return stop_words, lemmatizer


stop_words, lemmatizer = initialize_nlp()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model_package = load_model()
    tfidf = model_package.get("tfidf")
    voting_model = model_package.get("voting_ensemble")
    model_loaded = True if (tfidf and voting_model) else False
except Exception as e:
    model_package = None
    tfidf = None
    voting_model = None
    model_loaded = False


# =========================================================
# PREPROCESSING & EXTRACTION
# =========================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)


def extract_pdf_text(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    document = fitz.open(stream=file_bytes, filetype="pdf")
    pages = [page.get_text() for page in document if page.get_text()]
    page_count = len(document)
    document.close()
    return "\n".join(pages), page_count


def extract_docx_text(uploaded_file):
    document = Document(uploaded_file)
    content = []
    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            content.append(paragraph.text.strip())
    for table in document.tables:
        for row in table.rows:
            row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_text:
                content.append(" ".join(row_text))
    return "\n".join(content)


def extract_resume_text(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return extract_pdf_text(uploaded_file)
    elif name.endswith(".docx"):
        return extract_docx_text(uploaded_file), None
    return "", None


def format_category(category):
    return str(category).replace("_", " ").strip().title()


def get_predictions(package, vector):
    predictions = {}
    model_mapping = {
        "Naive Bayes": "naive_bayes",
        "Logistic Regression": "logistic_regression",
        "SVM": "svm",
        "Random Forest": "random_forest",
        "Voting Ensemble": "voting_ensemble"
    }

    for display_name, key in model_mapping.items():
        model = package.get(key)
        if model is not None:
            pred = model.predict(vector)[0]
            predictions[display_name] = format_category(pred)

    return predictions


# =========================================================
# SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None


def save_history(name, result):
    history_item = {
        "filename": name,
        "prediction": result["final_prediction"],
        "agreement": result["agreement"],
        "time": datetime.now().strftime("%H:%M:%S")
    }
    st.session_state.history.insert(0, history_item)
    st.session_state.history = st.session_state.history[:10]


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding:20px 5px;">
            <div style="font-size:40px; color:#60a5fa; font-weight:700;">◈</div>
            <h1>ResumeAI</h1>
            <p style="color:#94a3b8;">NLP Ensemble Classifier</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Resume Classifier",
            "Model Performance",
            "About"
        ]
    )

    st.divider()

    if model_loaded:
        st.markdown(
            """
            <div class="status-box">
                <span class="status-online"></span>
                Model system online
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.error("Model system unavailable")

    st.write("")
    st.caption("Python • PyMuPDF • NLTK • Scikit-learn • Streamlit")


# =========================================================
# PAGE 1: DASHBOARD
# =========================================================

if page == "Dashboard":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">◈ ResumeAI</div>
            <div class="hero-subtitle">
                Automated resume screening powered by Natural Language Processing and
                an ensemble of multi-class machine learning classifiers.
            </div>
            <div class="hero-status">
                <span class="status-dot"></span>
                Resume classification system ready
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">Classifiers</div>
                <div class="kpi-value">4 + Ensemble</div>
                <div class="kpi-description">ML Algorithms Evaluated</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">NLP Feature Extractor</div>
                <div class="kpi-value">TF-IDF</div>
                <div class="kpi-description">Tokenized Term Frequency</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">Ensemble Decision</div>
                <div class="kpi-value">Hard Voting</div>
                <div class="kpi-description">Consensus Prediction</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">Supported Formats</div>
                <div class="kpi-value">PDF / DOCX</div>
                <div class="kpi-description">Digital Documents</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    st.subheader("NLP & ML Pipeline")

    st.markdown(
        """
        <div class="pipeline">
            <div class="pipeline-step"><span class="pipeline-number">01</span>Document Parsing</div>
            <div class="pipeline-step"><span class="pipeline-number">02</span>Text Cleaning</div>
            <div class="pipeline-step"><span class="pipeline-number">03</span>Tokenization</div>
            <div class="pipeline-step"><span class="pipeline-number">04</span>Stopwords</div>
            <div class="pipeline-step"><span class="pipeline-number">05</span>Lemmatization</div>
            <div class="pipeline-step"><span class="pipeline-number">06</span>TF-IDF Matrix</div>
            <div class="pipeline-step"><span class="pipeline-number">07</span>Model Inference</div>
            <div class="pipeline-step"><span class="pipeline-number">08</span>Voting Result</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("Evaluated Classifiers")

    cols = st.columns(4)
    model_names = ["Naive Bayes", "Logistic Regression", "Support Vector Machine", "Random Forest"]

    for col, name in zip(cols, model_names):
        with col:
            st.markdown(
                textwrap.dedent(f"""
                <div class="model-card">
                    <div class="model-name">CLASSIFIER</div>
                    <div class="model-result" style="font-size: 18px;">{name}</div>
                    <div style="color:#64748b; font-size:12px; margin-top:12px;">
                        Ready for inference
                    </div>
                </div>
                """),
                unsafe_allow_html=True
            )

    st.divider()

    st.subheader("System Status")

    status1, status2, status3 = st.columns(3)

    with status1:
        st.markdown(
            """
            <div class="status-box">
                <span class="status-online"></span>
                Document parser ready (PDF/DOCX)
            </div>
            """,
            unsafe_allow_html=True
        )

    with status2:
        st.markdown(
            """
            <div class="status-box">
                <span class="status-online"></span>
                TF-IDF vectorizer ready
            </div>
            """,
            unsafe_allow_html=True
        )

    with status3:
        st.markdown(
            """
            <div class="status-box">
                <span class="status-online"></span>
                Voting ensemble loaded
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="footer">
            ResumeAI • Machine Learning & Natural Language Processing
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# PAGE 2: RESUME CLASSIFIER
# =========================================================

elif page == "Resume Classifier":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">Resume Classifier</div>
            <div class="hero-subtitle">
                Upload a candidate resume in PDF or DOCX format to parse,
                clean, and classify the candidate profile automatically.
            </div>
            <div class="hero-status">
                <span class="status-dot"></span>
                Automated document extraction
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload Resume",
        type=["pdf", "docx"],
        help="Upload candidate PDF or DOCX document"
    )

    if uploaded_file is not None:
        file_size_kb = len(uploaded_file.getvalue()) / 1024
        file_ext = uploaded_file.name.split(".")[-1].upper()

        st.markdown(
            textwrap.dedent(f"""
            <div class="file-card">
                <div>
                    <div class="file-name">{uploaded_file.name}</div>
                    <div class="file-info">{file_size_kb:.1f} KB</div>
                </div>
                <div class="file-type">{file_ext}</div>
            </div>
            """),
            unsafe_allow_html=True
        )

        with st.spinner("Extracting document contents..."):
            resume_text, page_count = extract_resume_text(uploaded_file)

        if not resume_text.strip():
            st.error("No readable text found. Please upload a digital, text-based PDF or DOCX resume.")
        else:
            word_count = len(resume_text.split())
            char_count = len(resume_text)
            page_display = str(page_count) if page_count is not None else "DOCX"

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Words", f"{word_count:,}")
            with col2:
                st.metric("Total Characters", f"{char_count:,}")
            with col3:
                st.metric("Pages / Document", page_display)

            st.write("")

            action1, action2 = st.columns([3, 1])
            with action1:
                classify_clicked = st.button("CLASSIFY RESUME", key="classify_btn")
            with action2:
                clear_clicked = st.button("CLEAR", key="clear_btn")

            if clear_clicked:
                st.session_state.analysis_result = None
                st.rerun()

            if classify_clicked:
                if not model_loaded:
                    st.error("Classification models not loaded. Check model files.")
                else:
                    with st.status("Analyzing resume...", expanded=True) as status:
                        st.write("Cleaning and lemmatizing text tokens...")
                        cleaned_text = clean_text(resume_text)

                        st.write("Generating TF-IDF feature matrix...")
                        vector = tfidf.transform([cleaned_text])

                        st.write("Generating predictions across classifiers...")
                        predictions = get_predictions(model_package, vector)

                        status.update(label="Classification complete!", state="complete", expanded=False)

                    final_pred = predictions.get("Voting Ensemble", "Unknown")

                    # Calculate agreement among base models (excluding the ensemble entry itself)
                    base_predictions = [v for k, v in predictions.items() if k != "Voting Ensemble"]
                    total_base = len(base_predictions) if base_predictions else 1
                    match_count = base_predictions.count(final_pred)
                    agreement = (match_count / total_base) * 100

                    st.session_state.analysis_result = {
                        "filename": uploaded_file.name,
                        "raw_text": resume_text,
                        "cleaned_text": cleaned_text,
                        "predictions": predictions,
                        "final_prediction": final_pred,
                        "agreement": agreement,
                        "base_predictions": base_predictions
                    }

                    save_history(uploaded_file.name, st.session_state.analysis_result)

    # Display Analysis Result
    res = st.session_state.analysis_result
    if res is not None:
        st.success("Resume classified successfully.")

        st.markdown(
            textwrap.dedent(f"""
            <div class="result-card">
                <div class="result-label">Predicted Professional Category</div>
                <div class="result-value">{res["final_prediction"]}</div>
                <div style="color:#94a3b8; margin-top:12px; font-size:14px;">
                    Consensus reached via Voting Ensemble Classifier
                </div>
            </div>
            """),
            unsafe_allow_html=True
        )

        tab1, tab2, tab3 = st.tabs(["Classifier Breakdown", "Summary Table", "Extracted & Processed Text"])

        with tab1:
            st.subheader("Model Predictions")
            m_cols = st.columns(len(res["predictions"]))
            for c, (m_name, m_pred) in zip(m_cols, res["predictions"].items()):
                with c:
                    st.markdown(
                        textwrap.dedent(f"""
                        <div class="model-card">
                            <div class="model-name">{m_name}</div>
                            <div class="model-result" style="font-size: 17px;">{m_pred}</div>
                            <div style="color:#64748b; font-size:11px; margin-top:12px;">
                                Individual result
                            </div>
                        </div>
                        """),
                        unsafe_allow_html=True
                    )

            st.markdown(
                textwrap.dedent(f"""
                <div class="agreement-card" style="margin-top: 20px;">
                    <div class="agreement-title">Model Agreement</div>
                    <div class="agreement-value">{res["agreement"]:.0f}%</div>
                    <div class="agreement-bar">
                        <div class="agreement-fill" style="width:{res["agreement"]:.0f}%"></div>
                    </div>
                    <div style="color:#64748b; font-size:12px; margin-top:9px;">
                        Percentage of base models agreeing with the final ensemble category
                    </div>
                </div>
                """),
                unsafe_allow_html=True
            )

        with tab2:
            st.subheader("Summary Table")
            summary_data = [{"Model": k, "Predicted Category": v} for k, v in res["predictions"].items()]
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

        with tab3:
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                st.markdown("**Original Text**")
                st.text_area("Original", res["raw_text"], height=280, label_visibility="collapsed")
            with t_col2:
                st.markdown("**Processed Text (Lemmatized)**")
                st.text_area("Processed", res["cleaned_text"], height=280, label_visibility="collapsed")

        # History Section
        st.divider()
        h_col1, h_col2 = st.columns([4, 1])
        with h_col1:
            st.subheader("Recent Evaluations")
        with h_col2:
            if st.session_state.history:
                if st.button("Clear History", key="clear_hist"):
                    st.session_state.history = []
                    st.rerun()

        for item in st.session_state.history:
            st.markdown(
                textwrap.dedent(f"""
                <div class="history-card">
                    <div class="history-text"><strong>{item["filename"]}</strong></div>
                    <div class="history-meta">
                        {item["time"]} &nbsp; • &nbsp; Category:
                        <span style="color:#38bdf8; font-weight:700;">{item["prediction"]}</span>
                        &nbsp; • &nbsp; Agreement: {item["agreement"]:.0f}%
                    </div>
                </div>
                """),
                unsafe_allow_html=True
            )


# =========================================================
# PAGE 3: MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">Model Performance</div>
            <div class="hero-subtitle">
                Visual evaluation metrics, accuracy benchmarks, and confusion matrices
                recorded during training.
            </div>
            <div class="hero-status">
                <span class="status-dot"></span>
                Validation benchmarks loaded
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Evaluation Metrics Overview")

    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">Trained Classifiers</div>
                <div class="kpi-value">5 Models</div>
                <div class="kpi-description">Including Voting Ensemble</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with p2:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">Vectorization</div>
                <div class="kpi-value">TF-IDF</div>
                <div class="kpi-description">N-gram vocabulary representation</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with p3:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">Ensemble Mechanism</div>
                <div class="kpi-value">Hard Voting</div>
                <div class="kpi-description">Majority consensus output</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    perf_tab1, perf_tab2 = st.tabs(["Model Accuracy Comparison", "Confusion Matrix"])

    with perf_tab1:
        st.subheader("Model Accuracy Comparison")
        if COMPARISON_PATH.exists():
            st.image(str(COMPARISON_PATH), use_container_width=True)
        else:
            st.warning("Model comparison image not found at results/model_comparison.png")

    with perf_tab2:
        st.subheader("Voting Ensemble Confusion Matrix")
        if CONFUSION_PATH.exists():
            st.image(str(CONFUSION_PATH), use_container_width=True)
        else:
            st.warning("Confusion matrix image not found at results/confusion_matrix.png")


# =========================================================
# PAGE 4: ABOUT
# =========================================================

elif page == "About":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">About ResumeAI</div>
            <div class="hero-subtitle">
                An enterprise-grade resume classification pipeline built to automate candidate
                categorization using modern NLP techniques and scikit-learn models.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="card">
                <h3>NLP Pipeline</h3>
                <p>Digital text extraction via PyMuPDF (PDF) & python-docx (DOCX)</p>
                <p>URL & email address regex stripping</p>
                <p>Alphanumeric character normalization</p>
                <p>Word tokenization via NLTK</p>
                <p>Stopword filtering</p>
                <p>WordNet lemmatization</p>
                <p>TF-IDF term weighting</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="card">
                <h3>Classification Architecture</h3>
                <p>Multinomial Naive Bayes</p>
                <p>Logistic Regression</p>
                <p>Support Vector Machine (Linear SVM)</p>
                <p>Random Forest Classifier</p>
                <p>Hard Voting Ensemble Classifier</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    st.subheader("Technology Stack")

    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">Language</div>
                <div class="kpi-value">Python</div>
                <div class="kpi-description">Core application runtime</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with t2:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">Document Extraction</div>
                <div class="kpi-value">PyMuPDF / docx</div>
                <div class="kpi-description">File parsers</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with t3:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">Machine Learning</div>
                <div class="kpi-value">Scikit-learn</div>
                <div class="kpi-description">Classifiers & TF-IDF</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with t4:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">Dashboard UI</div>
                <div class="kpi-value">Streamlit</div>
                <div class="kpi-description">Interactive dashboard</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="footer">
            ◈ ResumeAI<br>
            Automated Resume Classification System<br><br>
            Built using Python, PyMuPDF, NLTK, Scikit-learn, and Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )