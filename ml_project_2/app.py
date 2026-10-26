import re
import time
from datetime import datetime
from html import escape
from pathlib import Path

import altair as alt
import joblib
import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SentimentAI",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "sentiment_model.pkl"

MODEL_KEYS = {
    "Logistic Regression": "logistic_regression",
    "Naive Bayes": "naive_bayes",
    "Linear SVM": "linear_svm",
    "Random Forest": "random_forest",
    "Decision Tree": "decision_tree",
}

COLORS = {
    "positive": "#34d399",
    "negative": "#fb7185",
    "neutral": "#fbbf24",
}


# =========================================================
# MODERN DARK THEME
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        color-scheme: dark;
        --background: #0b0d14;
        --surface: #141722;
        --border: #282d40;
        --text: #f1f3fc;
        --muted: #a0a8c0;
        --accent: #a78bfa;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(
                ellipse at 80% 0%,
                rgba(124, 58, 237, 0.12),
                transparent 42%
            ),
            #0b0d14;
        color: var(--text);
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1360px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        color: var(--text) !important;
        letter-spacing: -0.04em;
    }

    p, label {
        color: #c0c6d9;
    }

    [data-testid="stCaptionContainer"] p {
        color: var(--muted);
    }

    .brand {
        font-size: 27px;
        font-weight: 800;
        letter-spacing: -1.3px;
        color: #f4f3ff;
    }

    .brand span {
        color: #a78bfa;
    }

    .brand-subtitle {
        color: #8892ad;
        font-size: 12px;
        margin-top: 3px;
    }

    .status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border: 1px solid #2b3449;
        background: #141a27;
        color: #c9d0e3;
        padding: 8px 13px;
        border-radius: 100px;
        font-size: 12px;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #34d399;
    }

    .status-dot.offline {
        background: #fbbf24;
    }

    .hero {
        margin: 24px 0 28px;
        padding: 36px;
        border: 1px solid #34314d;
        border-radius: 22px;
        background:
            radial-gradient(
                circle at 95% 20%,
                rgba(139, 92, 246, 0.20),
                transparent 45%
            ),
            linear-gradient(120deg, #181b2b, #121520);
    }

    .eyebrow {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
        color: #b39afa;
        text-transform: uppercase;
        margin-bottom: 14px;
    }

    .hero h1 {
        font-size: clamp(30px, 4vw, 48px);
        font-weight: 800;
        line-height: 1.12;
        margin: 0 0 14px;
        max-width: 850px;
    }

    .hero p {
        max-width: 700px;
        color: #aab3cc;
        line-height: 1.75;
        font-size: 15px;
        margin: 0;
    }

    .stat-card {
        background: linear-gradient(145deg, #181c2a, #131621);
        border: 1px solid #2a3043;
        padding: 23px;
        border-radius: 17px;
        min-height: 148px;
        margin-bottom: 18px;
        transition: border-color .2s, transform .2s;
    }

    .stat-card:hover {
        transform: translateY(-3px);
        border-color: #8066bf;
    }

    .stat-label {
        color: #a0a8c0;
        font-size: 12px;
        font-weight: 500;
    }

    .stat-value {
        color: #f0efff;
        font-size: 32px;
        font-weight: 750;
        letter-spacing: -1px;
        margin: 9px 0 4px;
    }

    .stat-note {
        color: #8792ae;
        font-size: 12px;
    }

    .empty-state {
        padding: 42px 25px;
        border: 1px dashed #3b4058;
        border-radius: 17px;
        background: #121621;
        text-align: center;
    }

    .empty-state h3 {
        font-size: 23px;
        margin-bottom: 12px;
    }

    .empty-state p {
        font-size: 14px;
        color: #98a4bf;
        line-height: 1.7;
    }

    .result-card {
        padding: 27px;
        border: 1px solid #3b3b59;
        background:
            radial-gradient(
                circle at top right,
                rgba(139, 92, 246, .14),
                transparent 65%
            ),
            #171b29;
        border-radius: 18px;
        margin-bottom: 18px;
    }

    .result-label {
        font-size: 11px;
        color: #b2bad2;
        letter-spacing: 1.8px;
        text-transform: uppercase;
    }

    .result-value {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: -1.4px;
        margin: 12px 0;
    }

    .result-note {
        font-size: 13px;
        color: #a4aec7;
    }

    .vote-row {
        display: flex;
        justify-content: space-between;
        gap: 15px;
        align-items: center;
        padding: 15px 0;
        border-bottom: 1px solid #292f42;
    }

    .vote-name {
        font-size: 13px;
        color: #c4cce0;
    }

    .sentiment-pill {
        padding: 5px 11px;
        border: 1px solid currentColor;
        border-radius: 100px;
        font-size: 11px;
        font-weight: 600;
        white-space: nowrap;
    }

    .model-row {
        display: flex;
        align-items: center;
        gap: 15px;
        padding: 16px 0;
        border-bottom: 1px solid #272c3d;
    }

    .model-number {
        background: #252239;
        color: #bca5ff;
        width: 35px;
        height: 35px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: 700;
        flex-shrink: 0;
    }

    .model-title {
        color: #e2e6f5;
        font-size: 14px;
        font-weight: 500;
    }

    .footer {
        margin-top: 36px;
        padding-top: 20px;
        border-top: 1px solid #252a3b;
        color: #7f8aa5;
        font-size: 12px;
    }

    [data-testid="stRadio"] > div {
        background: #141823;
        border: 1px solid #2a3042;
        border-radius: 12px;
        padding: 8px 15px;
        gap: 24px;
        width: fit-content;
        margin-top: 20px;
    }

    [data-testid="stRadio"] label p {
        color: #d2d7e9 !important;
        font-size: 13px;
        font-weight: 500;
    }

    [data-testid="stWidgetLabel"] p {
        color: #dce1f2 !important;
    }

    [data-baseweb="textarea"] {
        background: #10141f !important;
        border: 1px solid #343c53 !important;
        border-radius: 13px !important;
    }

    [data-baseweb="textarea"]:focus-within {
        border-color: #a78bfa !important;
        box-shadow: 0 0 0 3px rgba(167,139,250,.10);
    }

    textarea {
        background: #10141f !important;
        color: #edf0fc !important;
        caret-color: #c4b5fd !important;
        font-size: 15px !important;
        line-height: 1.75 !important;
    }

    textarea::placeholder {
        color: #76829e !important;
    }

    .stButton button,
    .stDownloadButton button,
    [data-testid="stFormSubmitButton"] button {
        min-height: 45px;
        border-radius: 11px;
        border: 1px solid #343c54;
        background: #1b2130;
        color: #e2e7f7;
        font-weight: 600;
        transition: all .2s;
    }

    .stButton button p,
    .stDownloadButton button p {
        color: inherit !important;
    }

    .stButton button:hover,
    .stDownloadButton button:hover {
        background: #272b40;
        color: #ffffff;
        border-color: #a78bfa;
    }

    .stButton button[kind="primary"] {
        background: linear-gradient(110deg, #7c3aed, #9255ec);
        border-color: #9466ee;
        color: white;
    }

    .stButton button[kind="primary"]:hover {
        background: linear-gradient(110deg, #8b46f5, #a16af4);
        box-shadow: 0 5px 22px rgba(124,58,237,.22);
    }

    .stButton button:disabled {
        opacity: .45;
    }

    [data-testid="stMetric"] {
        background: #151a28;
        border: 1px solid #2b3247;
        border-radius: 14px;
        padding: 19px;
    }

    [data-testid="stMetricValue"] {
        color: #eceaff;
    }

    button[data-baseweb="tab"] {
        color: #a8b2cc;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #c4b5fd;
    }

    [data-testid="stExpander"] {
        background: #131824;
        border: 1px solid #2a3247;
        border-radius: 12px;
    }

    [data-testid="stAlert"] p {
        color: inherit;
    }

    hr {
        border-color: #292f42;
    }

    @media (max-width: 700px) {
        .hero {
            padding: 25px;
        }

        .block-container {
            padding-top: 1rem;
        }

        .result-value {
            font-size: 34px;
        }

        [data-testid="stRadio"] > div {
            gap: 10px;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        * {
            transition: none !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# RESOURCE LOADING
# =========================================================

@st.cache_resource
def load_nlp():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
    ]

    for resource_path, name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.data.find(resource_path + ".zip")
            except LookupError:
                if not nltk.download(name, quiet=True):
                    raise RuntimeError(
                        f"Could not download NLTK resource: {name}"
                    )

    words = set(stopwords.words("english"))
    lemma = WordNetLemmatizer()

    word_tokenize("Resource check.")
    lemma.lemmatize("words")

    return words, lemma


@st.cache_resource
def load_models():
    package = joblib.load(MODEL_PATH)

    required = [
        "tfidf",
        "voting_ensemble",
        *MODEL_KEYS.values(),
    ]

    missing = [key for key in required if key not in package]

    if missing:
        raise ValueError(
            "Missing entries in model package: " + ", ".join(missing)
        )

    return package


package = None
stop_words = set()
lemmatizer = None
setup_errors = []

try:
    stop_words, lemmatizer = load_nlp()
except Exception as exc:
    setup_errors.append(f"NLP resources: {exc}")

try:
    package = load_models()
except Exception as exc:
    setup_errors.append(f"Model loading: {exc}")

ready = package is not None and lemmatizer is not None


# =========================================================
# STATE AND ACTIONS
# =========================================================

DEFAULTS = {
    "history": [],
    "result": None,
    "input_text": "",
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def set_example(text):
    st.session_state.input_text = text
    st.session_state.result = None


def clear_input():
    st.session_state.input_text = ""
    st.session_state.result = None


# =========================================================
# MODEL INFERENCE
# =========================================================

def clean_text(text):
    # Matches the preprocessing used in the supplied original app.
    text = re.sub(r"http\S+|www\S+|https\S+", "", str(text))
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)


def normalize_sentiment(value):
    # Numeric mapping follows your supplied code.
    # Change it if your notebook used a different label encoding.
    mapping = {
        "negative": "negative",
        "neg": "negative",
        "0": "negative",
        "-1": "negative",
        "positive": "positive",
        "pos": "positive",
        "1": "positive",
        "neutral": "neutral",
        "neu": "neutral",
        "2": "neutral",
    }

    label = str(value).strip().lower()

    if label not in mapping:
        raise ValueError(
            f"Unknown label '{label}'. Check the training label mapping."
        )

    return mapping[label]


def predict(text):
    start = time.perf_counter()
    cleaned = clean_text(text)

    if not cleaned:
        return None

    vector = package["tfidf"].transform([cleaned])

    predictions = {
        name: normalize_sentiment(
            package[key].predict(vector)[0]
        )
        for name, key in MODEL_KEYS.items()
    }

    final = normalize_sentiment(
        package["voting_ensemble"].predict(vector)[0]
    )

    votes = {
        label: list(predictions.values()).count(label)
        for label in COLORS
    }

    return {
        "text": text,
        "cleaned": cleaned,
        "sentiment": final,
        "predictions": predictions,
        "votes": votes,
        "agreement": votes[final] / len(predictions) * 100,
        "features": int(vector.nnz),
        "duration_ms": round(
            (time.perf_counter() - start) * 1000
        ),
        "time": datetime.now().strftime("%H:%M:%S"),
    }


# =========================================================
# DISPLAY HELPERS
# =========================================================

def html(content):
    st.markdown(content, unsafe_allow_html=True)


def hero(section, title, description):
    html(
        '<div class="hero">'
        f'<div class="eyebrow">{escape(section)}</div>'
        f"<h1>{escape(title)}</h1>"
        f"<p>{escape(description)}</p>"
        "</div>"
    )


def stat(label, value, note):
    html(
        '<div class="stat-card">'
        f'<div class="stat-label">{escape(label)}</div>'
        f'<div class="stat-value">{escape(str(value))}</div>'
        f'<div class="stat-note">{escape(note)}</div>'
        "</div>"
    )


def history_frame():
    return pd.DataFrame(
        [
            {
                "Text": item["text"],
                "Sentiment": item["sentiment"].title(),
                "Agreement (%)": item["agreement"],
                "Time": item["time"],
            }
            for item in st.session_state.history
        ]
    )


def vote_chart(votes):
    data = pd.DataFrame(
        {
            "Sentiment": [
                label.title() for label in votes
            ],
            "Votes": list(votes.values()),
        }
    )

    chart = (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=7, size=32)
        .encode(
            x=alt.X(
                "Votes:Q",
                scale=alt.Scale(domain=[0, 5]),
                axis=alt.Axis(
                    values=[0, 1, 2, 3, 4, 5],
                    title="Number of model votes",
                    gridColor="#282e42",
                    labelColor="#aeb8d0",
                    titleColor="#aeb8d0",
                ),
            ),
            y=alt.Y(
                "Sentiment:N",
                sort=["Positive", "Negative", "Neutral"],
                axis=alt.Axis(
                    title=None,
                    labelColor="#d5dcef",
                    labelPadding=12,
                ),
            ),
            color=alt.Color(
                "Sentiment:N",
                scale=alt.Scale(
                    domain=["Positive", "Negative", "Neutral"],
                    range=[
                        COLORS["positive"],
                        COLORS["negative"],
                        COLORS["neutral"],
                    ],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Sentiment:N"),
                alt.Tooltip("Votes:Q", format=".0f"),
            ],
        )
        .properties(height=210, background="#141722")
        .configure_view(strokeWidth=0)
    )

    st.altair_chart(
        chart,
        use_container_width=True,
        theme=None,
    )


# =========================================================
# HEADER
# =========================================================

brand, status = st.columns([5, 1.3])

with brand:
    html(
        '<div class="brand">Sentiment<span>AI</span></div>'
        '<div class="brand-subtitle">'
        "Your sentiment analysis workspace"
        "</div>"
    )

with status:
    dot_class = "" if ready else "offline"
    status_text = "Models connected" if ready else "Setup required"

    html(
        '<div class="status">'
        f'<span class="status-dot {dot_class}"></span>'
        f"{status_text}"
        "</div>"
    )

page = st.radio(
    "Navigation",
    ["Overview", "Analyze", "Performance"],
    horizontal=True,
    label_visibility="collapsed",
)

if setup_errors:
    st.warning(
        "Some resources could not be loaded. "
        "Analysis will be available after setup is complete."
    )

    with st.expander("View setup details"):
        st.code("\n\n".join(setup_errors))


# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":
    hero(
        "SENTIMENT INTELLIGENCE",
        "Understand the feeling behind every sentence.",
        "Analyze tweets, reviews, and comments. Compare five "
        "classifiers and explore the ensemble decision in one workspace.",
    )

    cols = st.columns(4)

    with cols[0]:
        stat("Individual classifiers", "05", "Five model perspectives")

    with cols[1]:
        stat("Sentiment categories", "03", "Positive · Negative · Neutral")

    with cols[2]:
        stat(
            "Recent analyses",
            len(st.session_state.history),
            "Latest 20 retained this session",
        )

    with cols[3]:
        stat("Ensemble method", "Voting", "Your saved ensemble model")

    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.subheader("The models behind your result")
        st.caption(
            "Each classifier makes its own prediction."
        )

        for index, name in enumerate(MODEL_KEYS, 1):
            html(
                '<div class="model-row">'
                f'<div class="model-number">{index:02d}</div>'
                f'<div class="model-title">{escape(name)}</div>'
                "</div>"
            )

    with right:
        st.subheader("Recent activity")

        if st.session_state.history:
            st.dataframe(
                history_frame().head(5),
                hide_index=True,
                use_container_width=True,
            )
        else:
            html(
                '<div class="empty-state">'
                "<h3>Your next insight starts here.</h3>"
                "<p>Open Analyze and enter your first text. "
                "Your recent results will appear here.</p>"
                "</div>"
            )

        st.write("")
        st.info(
            "Model agreement shows how many individual "
            "classifiers match the ensemble result. "
            "It does not measure accuracy or probability."
        )


# =========================================================
# ANALYZER
# =========================================================

elif page == "Analyze":
    hero(
        "ANALYSIS STUDIO",
        "From text to insight.",
        "Enter your text, run the ensemble, and explore "
        "the votes behind the prediction.",
    )

    editor, insight = st.columns([1.2, 1], gap="large")

    with editor:
        st.subheader("Your text")
        st.caption("A tweet, review, comment, or sentence.")

        st.text_area(
            "Text to analyze",
            key="input_text",
            height=235,
            max_chars=10000,
            placeholder=(
                "Example: The product arrived early and "
                "the quality exceeded my expectations."
            ),
            label_visibility="collapsed",
        )

        current = st.session_state.input_text

        st.caption(
            f"{len(current.split())} words"
            f"  ·  {len(current):,} / 10,000 characters"
        )

        action, reset = st.columns([3, 1])

        clicked = action.button(
            "Analyze sentiment",
            type="primary",
            use_container_width=True,
            disabled=not ready,
        )

        reset.button(
            "Clear",
            on_click=clear_input,
            use_container_width=True,
        )

        st.write("")
        st.caption("Need inspiration? Load an example.")

        examples = [
            (
                "Product review",
                "The product arrived early and the "
                "quality exceeded my expectations.",
            ),
            (
                "Customer feedback",
                "I waited for hours and support "
                "still did not resolve the issue.",
            ),
            (
                "Everyday update",
                "The meeting is scheduled for "
                "tomorrow afternoon.",
            ),
        ]

        example_cols = st.columns(3)

        for col, (label, text) in zip(example_cols, examples):
            col.button(
                label,
                on_click=set_example,
                args=(text,),
                use_container_width=True,
            )

        if clicked:
            st.session_state.result = None

            if not current.strip():
                st.warning("Enter some text before analyzing.")
            else:
                try:
                    with st.spinner(
                        "Processing text and comparing model votes..."
                    ):
                        result = predict(current)

                    if result is None:
                        st.warning(
                            "No usable words remained after "
                            "preprocessing. Try a longer sentence."
                        )
                    else:
                        st.session_state.result = result
                        st.session_state.history.insert(0, result)
                        st.session_state.history = (
                            st.session_state.history[:20]
                        )

                except Exception as exc:
                    st.error("The analysis could not be completed.")

                    with st.expander("Error details"):
                        st.code(str(exc))

    with insight:
        st.subheader("Ensemble result")
        result = st.session_state.result

        if result is None:
            html(
                '<div class="empty-state">'
                "<h3>Ready when you are.</h3>"
                "<p>Your sentiment result, model agreement, "
                "and individual predictions will appear here.</p>"
                "</div>"
            )
        else:
            label = result["sentiment"]
            color = COLORS[label]

            html(
                '<div class="result-card">'
                '<div class="result-label">Predicted sentiment</div>'
                f'<div class="result-value" style="color:{color}">'
                f"{label.title()}"
                "</div>"
                '<div class="result-note">'
                f'{result["votes"][label]} of 5 model predictions '
                "match the ensemble"
                "</div>"
                "</div>"
            )

            st.progress(
                result["agreement"] / 100,
                text=f'{result["agreement"]:.0f}% model agreement',
            )

            st.caption(
                "Agreement is a vote count, not a confidence score."
            )

            if current != result["text"]:
                st.info(
                    "The text has changed. Analyze again "
                    "to update this result."
                )

            for name, prediction in result["predictions"].items():
                html(
                    '<div class="vote-row">'
                    f'<span class="vote-name">{escape(name)}</span>'
                    '<span class="sentiment-pill" '
                    f'style="color:{COLORS[prediction]}">'
                    f"{prediction.title()}"
                    "</span>"
                    "</div>"
                )

    result = st.session_state.result

    if result:
        st.divider()

        votes_tab, processing_tab = st.tabs(
            ["Explore model votes", "Inspect processed text"]
        )

        with votes_tab:
            chart_col, metrics_col = st.columns(
                [2, 1],
                gap="large",
            )

            with chart_col:
                vote_chart(result["votes"])

            with metrics_col:
                st.metric(
                    "Analysis time",
                    f'{result["duration_ms"]:,} ms',
                )
                st.metric(
                    "Active TF-IDF features",
                    result["features"],
                )

        with processing_tab:
            original, cleaned = st.columns(2)

            original.text_area(
                "Analyzed input",
                value=result["text"],
                height=150,
                disabled=True,
            )

            cleaned.text_area(
                "Cleaned text",
                value=result["cleaned"],
                height=150,
                disabled=True,
            )

            st.caption(
                "URLs and mentions removed → lowercase → "
                "special characters removed → tokenization → "
                "stopword removal → lemmatization → TF-IDF."
            )

        export = {
            "text": result["text"],
            "sentiment": result["sentiment"],
            "agreement_percent": result["agreement"],
            **result["predictions"],
        }

        st.download_button(
            "Download analysis",
            data=pd.DataFrame([export]).to_csv(
                index=False
            ).encode("utf-8"),
            file_name="sentiment_analysis.csv",
            mime="text/csv",
        )

    if st.session_state.history:
        st.divider()
        st.subheader("Session history")

        sentiment_filter = st.radio(
            "Filter history",
            ["All", "Positive", "Negative", "Neutral"],
            horizontal=True,
            key="history_filter",
        )

        frame = history_frame()

        if sentiment_filter != "All":
            frame = frame[
                frame["Sentiment"] == sentiment_filter
            ]

        st.dataframe(
            frame,
            hide_index=True,
            use_container_width=True,
        )

        export_col, clear_col = st.columns([3, 1])

        export_col.download_button(
            "Export filtered history",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name="sentiment_history.csv",
            mime="text/csv",
        )

        if clear_col.button(
            "Clear history",
            use_container_width=True,
        ):
            st.session_state.history = []
            st.rerun()


# =========================================================
# PERFORMANCE
# =========================================================

elif page == "Performance":
    hero(
        "MODEL EVALUATION",
        "Explore the evidence.",
        "Inspect the saved evaluation charts from your "
        "training workflow.",
    )

    chart_type = st.radio(
        "Evaluation view",
        ["Model comparison", "Confusion matrix"],
        horizontal=True,
    )

    if chart_type == "Model comparison":
        image_path = BASE_DIR / "results" / "model_comparison.png"
        st.subheader("Classifier comparison")
    else:
        image_path = BASE_DIR / "results" / "confusion_matrix.png"
        st.subheader("Confusion matrix")

    if image_path.exists():
        st.image(
            str(image_path),
            use_container_width=True,
        )
    else:
        st.info(
            f"Place {image_path.name} inside the results folder "
            "to display this chart."
        )

    st.caption(
        "These images come from your saved evaluation results. "
        "The interface does not recalculate performance metrics."
    )


html(
    '<div class="footer">'
    "SentimentAI / Ensemble sentiment analysis"
    "</div>"
)