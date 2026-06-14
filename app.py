"""
AI-Powered Career Recommendation System
========================================
Matches a user's skills to the best-fit job roles using
Sentence-Transformers (all-MiniLM-L6-v2) + cosine similarity.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import io
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sentence_transformers import SentenceTransformer

# ── Optional CV parsers (gracefully degrade if missing) ──────────────────────
try:
    import pdfplumber
    PDF_OK = True
except ImportError:
    PDF_OK = False

try:
    from docx import Document as DocxDocument
    DOCX_OK = True
except ImportError:
    DOCX_OK = False

# ═══════════════════════════════════════════════════════════════════════════════
# 0.  PAGE CONFIG  (must be first Streamlit call)
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CareerMatch AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  GLOBAL STYLES
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
    /* ── Base & fonts ───────────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Main background ────────────────────────────────────────────────────── */
    .stApp {
        background: #0F1117;
    }

    /* ── Sidebar ────────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: #161B27 !important;
        border-right: 1px solid #2A2F3E;
    }
    [data-testid="stSidebar"] * {
        color: #C9D1E0 !important;
    }

    /* ── Header hero ────────────────────────────────────────────────────────── */
    .hero-block {
        padding: 2.4rem 2rem 1.8rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #1A1F2E 0%, #0F1117 100%);
        border: 1px solid #2A2F3E;
        margin-bottom: 1.6rem;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.5px;
        margin: 0 0 0.4rem;
    }
    .hero-pill {
        display: inline-block;
        background: #2563EB22;
        border: 1px solid #2563EB66;
        color: #60A5FA;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 3px 10px;
        border-radius: 999px;
        margin-bottom: 0.7rem;
    }
    .hero-sub {
        color: #8892A4;
        font-size: 0.97rem;
        line-height: 1.6;
        margin: 0;
        max-width: 560px;
    }

    /* ── Section labels ─────────────────────────────────────────────────────── */
    .section-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #5A6478;
        margin-bottom: 0.5rem;
    }

    /* ── Input card ─────────────────────────────────────────────────────────── */
    .input-card {
        background: #161B27;
        border: 1px solid #2A2F3E;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
    }

    /* ── Job result card ────────────────────────────────────────────────────── */
    .job-card {
        background: #161B27;
        border: 1px solid #2A2F3E;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.9rem;
        transition: border-color 0.2s;
    }
    .job-card:hover { border-color: #3B82F6; }
    .job-rank {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        color: #3B82F6;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    .job-title-text {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: #F0F4FF;
        margin-bottom: 0.3rem;
    }
    .job-category {
        font-size: 0.8rem;
        color: #8892A4;
        margin-bottom: 0.6rem;
    }
    .score-bar-bg {
        background: #1E2535;
        border-radius: 999px;
        height: 6px;
        overflow: hidden;
        margin-top: 0.5rem;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #2563EB, #60A5FA);
    }
    .score-text {
        font-size: 0.82rem;
        color: #60A5FA;
        font-weight: 600;
        margin-top: 0.35rem;
    }
    .skill-chip {
        display: inline-block;
        background: #1E2535;
        border: 1px solid #2A2F3E;
        color: #94A3B8;
        font-size: 0.72rem;
        padding: 3px 9px;
        border-radius: 999px;
        margin: 2px 3px 2px 0;
    }
    .skill-chip-match {
        background: #1D3461;
        border-color: #2563EB66;
        color: #93C5FD;
    }

    /* ── Stat box ───────────────────────────────────────────────────────────── */
    .stat-box {
        background: #161B27;
        border: 1px solid #2A2F3E;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        text-align: center;
    }
    .stat-num {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #F0F4FF;
    }
    .stat-label {
        font-size: 0.72rem;
        color: #5A6478;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* ── Streamlit widget overrides ─────────────────────────────────────────── */
    .stTextArea textarea {
        background: #0F1117 !important;
        border: 1px solid #2A2F3E !important;
        color: #E2E8F0 !important;
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
    }
    .stTextArea textarea::placeholder { color: #4A5568 !important; }
    .stTextArea textarea:focus { border-color: #3B82F6 !important; box-shadow: none !important; }

    .stButton button {
        background: linear-gradient(135deg, #2563EB, #1D4ED8) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        padding: 0.55rem 1.6rem !important;
        font-family: 'Inter', sans-serif !important;
        transition: opacity 0.2s !important;
    }
    .stButton button:hover { opacity: 0.88 !important; }

    .stFileUploader {
        background: #161B27 !important;
        border: 1px dashed #2A2F3E !important;
        border-radius: 10px !important;
        padding: 0.6rem !important;
    }
    .stFileUploader label { color: #8892A4 !important; }

    /* ── Slider ─────────────────────────────────────────────────────────────── */
    .stSlider [data-baseweb="slider"] { color: #3B82F6 !important; }

    /* ── Tabs ───────────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: #161B27 !important;
        border-bottom: 1px solid #2A2F3E !important;
        gap: 0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #5A6478 !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
    }
    .stTabs [aria-selected="true"] {
        color: #60A5FA !important;
        border-bottom-color: #3B82F6 !important;
    }

    /* ── Misc ───────────────────────────────────────────────────────────────── */
    hr { border-color: #2A2F3E !important; margin: 1rem 0; }
    .stMarkdown p, .stMarkdown li { color: #C9D1E0; }
    .stSuccess { background: #0F2A1A !important; color: #4ADE80 !important; border-color: #166534 !important; }
    .stWarning { background: #2A1F0A !important; color: #FCD34D !important; }
    .stError   { background: #2A0F0F !important; color: #F87171 !important; }
    div[data-testid="stExpander"] { background: #161B27 !important; border: 1px solid #2A2F3E !important; border-radius: 10px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  PATHS
# ═══════════════════════════════════════════════════════════════════════════════
DATA_DIR       = Path(__file__).parent / "data"
EMBEDDINGS_PATH = DATA_DIR / "job_embeddings.npy"
SLIM_CSV_PATH   = DATA_DIR / "job_titles_skills.csv"
MODEL_NAME      = "all-MiniLM-L6-v2"

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  CACHED RESOURCES
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


@st.cache_data(show_spinner=False)
def load_data() -> tuple[np.ndarray, pd.DataFrame]:
    if not EMBEDDINGS_PATH.exists():
        st.error(
            f"Embeddings not found at `{EMBEDDINGS_PATH}`. "
            "Run the Colab notebook first and place `job_embeddings.npy` in the `data/` folder."
        )
        st.stop()
    if not SLIM_CSV_PATH.exists():
        st.error(
            f"Slim CSV not found at `{SLIM_CSV_PATH}`. "
            "Run the Colab notebook first and place `job_titles_skills.csv` in the `data/` folder."
        )
        st.stop()

    embeddings = np.load(str(EMBEDDINGS_PATH)).astype(np.float32)
    df = pd.read_csv(SLIM_CSV_PATH)
    return embeddings, df


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def clean_skills(raw: str) -> str:
    """Normalise a comma-separated skill string (mirrors Colab logic)."""
    if not raw or not raw.strip():
        return ""
    raw = raw.lower()
    skills = re.split(r"[,;|/\n]+", raw)
    cleaned = []
    for skill in skills:
        skill = re.sub(r"[^a-z0-9 +#.\-]", " ", skill)
        skill = re.sub(r"\s+", " ", skill).strip()
        if skill:
            cleaned.append(skill)
    return ", ".join(cleaned)


def extract_text_from_file(uploaded_file) -> str:
    """Extract plain text from PDF, DOCX, or TXT upload."""
    fname = uploaded_file.name.lower()

    if fname.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    if fname.endswith(".pdf"):
        if not PDF_OK:
            st.warning("Install `pdfplumber` to parse PDFs: `pip install pdfplumber`")
            return ""
        with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
            return "\n".join(
                page.extract_text() or "" for page in pdf.pages
            )

    if fname.endswith(".docx"):
        if not DOCX_OK:
            st.warning("Install `python-docx` to parse DOCX files: `pip install python-docx`")
            return ""
        doc = DocxDocument(io.BytesIO(uploaded_file.read()))
        return "\n".join(p.text for p in doc.paragraphs)

    st.warning(f"Unsupported file type: {fname}")
    return ""


def extract_skills_from_cv_text(text: str) -> str:
    """
    Lightweight skill extractor from CV text.
    Strategy: look for a 'skills' section; fall back to full text.
    """
    lines = text.splitlines()
    skill_lines: list[str] = []
    in_skills = False

    for line in lines:
        lower = line.lower().strip()
        # Detect start of skills section
        if re.match(r"^(technical\s+)?skills?[\s:]*$", lower):
            in_skills = True
            continue
        # Detect start of a new section (stop collecting)
        if in_skills and re.match(r"^[A-Z][A-Za-z\s]{2,25}:?\s*$", line.strip()) and lower not in ("", " "):
            if not re.search(r"[,•\-|/]", line):
                break
        if in_skills:
            skill_lines.append(line)

    raw = " ".join(skill_lines) if skill_lines else text[:3000]
    # Extract comma/bullet/pipe-separated tokens
    tokens = re.split(r"[,•\-|/\n\t]+", raw)
    skills = [t.strip() for t in tokens if 2 < len(t.strip()) < 40]
    return ", ".join(dict.fromkeys(skills))  # deduplicate, preserve order


def rank_jobs(
    user_skills_clean: str,
    model: SentenceTransformer,
    job_embeddings: np.ndarray,
    df: pd.DataFrame,
    top_n: int = 5,
    category_filter: str = "All",
) -> pd.DataFrame:
    """Encode user skills and return top-N cosine-similar jobs."""
    user_emb = model.encode(
        [user_skills_clean],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)                      # (1, 384)

    scores = (job_embeddings @ user_emb.T).flatten()  # cosine sim (L2-normalised)

    df = df.copy()
    df["match_score"] = scores
    df["match_pct"]   = (scores * 100).round(1)

    if category_filter != "All" and "category" in df.columns:
        df = df[df["category"].str.lower() == category_filter.lower()]

    top = (
        df.nlargest(top_n, "match_score")
          .reset_index(drop=True)
    )
    return top


def build_bar_chart(results: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of match scores."""
    titles = results["job_title"].tolist()[::-1]
    scores = results["match_pct"].tolist()[::-1]

    colours = [
        "#3B82F6" if s == max(scores) else "#1D3461"
        for s in scores
    ]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=titles,
            orientation="h",
            marker_color=colours,
            text=[f"{s}%" for s in scores],
            textposition="outside",
            textfont=dict(color="#94A3B8", size=11),
            hovertemplate="<b>%{y}</b><br>Match: %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor="#0F1117",
        plot_bgcolor="#0F1117",
        margin=dict(l=10, r=60, t=20, b=20),
        height=max(220, len(titles) * 52),
        xaxis=dict(
            range=[0, 110],
            showgrid=True,
            gridcolor="#1E2535",
            tickfont=dict(color="#5A6478", size=10),
            ticksuffix="%",
        ),
        yaxis=dict(
            tickfont=dict(color="#C9D1E0", size=12),
            automargin=True,
        ),
        font=dict(family="Inter", color="#C9D1E0"),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  LOAD RESOURCES
# ═══════════════════════════════════════════════════════════════════════════════
with st.spinner("Loading AI model and job database…"):
    model          = load_model()
    job_embeddings, df_jobs = load_data()

categories = ["All"]
if "category" in df_jobs.columns:
    categories += sorted(df_jobs["category"].dropna().unique().tolist())

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        """
        <div style='padding: 0.4rem 0 1.2rem;'>
            <div style='font-family:Space Grotesk,sans-serif;font-size:1.1rem;
                        font-weight:700;color:#F0F4FF;'>🎯 CareerMatch AI</div>
            <div style='font-size:0.75rem;color:#5A6478;margin-top:2px;'>
                Powered by sentence-transformers
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("<div class='section-label'>Settings</div>", unsafe_allow_html=True)

    top_n = st.slider("Results to show", min_value=3, max_value=10, value=5, step=1)

    category_filter = st.selectbox("Filter by category", categories)

    show_desc = st.toggle("Show job description", value=False)
    show_req_skills = st.toggle("Show required skills", value=True)
    highlight_matches = st.toggle("Highlight matching skills", value=True)

    st.markdown("---")

    # Quick stats
    st.markdown("<div class='section-label'>Dataset</div>", unsafe_allow_html=True)
    n_jobs = len(df_jobs)
    n_cats = len(categories) - 1
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"<div class='stat-box'><div class='stat-num'>{n_jobs:,}</div>"
            "<div class='stat-label'>Jobs</div></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div class='stat-box'><div class='stat-num'>{n_cats}</div>"
            "<div class='stat-label'>Categories</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem;color:#3A4154;line-height:1.7;'>"
        "Model: <code style='color:#4A5568'>all-MiniLM-L6-v2</code><br>"
        "Similarity: cosine (L2-normalised dot product)<br>"
        "Embeddings: 384-dim"
        "</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class='hero-block'>
        <div class='hero-pill'>AI Career Intelligence</div>
        <div class='hero-title'>Find Your Best-Fit Role</div>
        <p class='hero-sub'>
            Enter your skills below or upload a CV. Our AI matches them against
            thousands of job profiles using semantic similarity — not just keywords.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Input tabs ────────────────────────────────────────────────────────────────
tab_manual, tab_cv = st.tabs(["  ✏️  Type Skills  ", "  📄  Upload CV  "])

user_raw_skills = ""

with tab_manual:
    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    user_raw_skills_manual = st.text_area(
        label="Your skills",
        placeholder=(
            "e.g.  Python, machine learning, SQL, data visualisation, "
            "pandas, scikit-learn, communication, project management"
        ),
        height=110,
        label_visibility="collapsed",
    )
    if user_raw_skills_manual:
        user_raw_skills = user_raw_skills_manual

with tab_cv:
    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    uploaded_cv = st.file_uploader(
        "Upload your CV",
        type=["pdf", "docx", "txt"],
        label_visibility="collapsed",
    )
    if uploaded_cv:
        with st.spinner("Extracting skills from CV…"):
            raw_text = extract_text_from_file(uploaded_cv)
            extracted = extract_skills_from_cv_text(raw_text)

        if extracted:
            st.success(f"✅ Extracted {len(extracted.split(','))} skills from your CV")
            editable = st.text_area(
                "Extracted skills (edit if needed)",
                value=extracted,
                height=90,
            )
            user_raw_skills = editable
        else:
            st.warning("Could not extract skills automatically. Try the manual tab.")

# ── Action row ────────────────────────────────────────────────────────────────
st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
col_btn, col_clr, col_spacer = st.columns([1.4, 1, 5])
with col_btn:
    run = st.button("🔍  Match My Skills", use_container_width=True)
with col_clr:
    if st.button("✕  Clear", use_container_width=True):
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# 8.  RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
if run:
    if not user_raw_skills.strip():
        st.warning("Please enter at least one skill before searching.")
        st.stop()

    cleaned = clean_skills(user_raw_skills)
    if not cleaned:
        st.warning("Could not parse any skills. Check your input format.")
        st.stop()

    user_skill_set = {s.strip() for s in cleaned.split(",") if s.strip()}

    with st.spinner("Finding your best matches…"):
        t0      = time.perf_counter()
        results = rank_jobs(
            cleaned,
            model,
            job_embeddings,
            df_jobs,
            top_n=top_n,
            category_filter=category_filter,
        )
        elapsed = time.perf_counter() - t0

    if results.empty:
        st.warning("No jobs found for the selected filters. Try 'All' categories.")
        st.stop()

    st.markdown("---")

    # ── Summary row ──────────────────────────────────────────────────────────
    best_score = results.iloc[0]["match_pct"]
    avg_score  = results["match_pct"].mean()
    n_skills   = len(user_skill_set)

    c1, c2, c3, c4 = st.columns(4)
    for col, num, label in [
        (c1, f"{len(results)}", "Matches Found"),
        (c2, f"{best_score:.0f}%", "Best Match"),
        (c3, f"{avg_score:.0f}%", "Avg Match"),
        (c4, f"{n_skills}", "Skills Detected"),
    ]:
        with col:
            st.markdown(
                f"<div class='stat-box'><div class='stat-num'>{num}</div>"
                f"<div class='stat-label'>{label}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        f"<div style='font-size:0.72rem;color:#3A4154;text-align:right;"
        f"margin-top:0.4rem;'>Matched in {elapsed*1000:.0f} ms</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Two-column layout: cards + chart ─────────────────────────────────────
    col_cards, col_chart = st.columns([1.1, 0.9], gap="large")

    with col_cards:
        st.markdown(
            "<div class='section-label'>Top Job Matches</div>",
            unsafe_allow_html=True,
        )

        rank_labels = ["🥇 Best Match", "🥈 2nd", "🥉 3rd", "4th", "5th",
                       "6th", "7th", "8th", "9th", "10th"]

        for i, row in results.iterrows():
            title     = row["job_title"]
            score_pct = row["match_pct"]
            category  = row.get("category", "")
            req_skills_raw = row.get("cleaned_skills", "")
            description    = row.get("job_description", "")

            req_skill_set = {s.strip() for s in req_skills_raw.split(",") if s.strip()}
            matched       = user_skill_set & req_skill_set

            bar_width = min(int(score_pct), 100)
            rank_lbl  = rank_labels[i] if i < len(rank_labels) else f"#{i+1}"

            # ── skill chips ──────────────────────────────────────────────────
            chips_html = ""
            if show_req_skills and req_skill_set:
                for sk in sorted(req_skill_set)[:18]:
                    is_match = highlight_matches and sk in matched
                    cls = "skill-chip skill-chip-match" if is_match else "skill-chip"
                    chips_html += f"<span class='{cls}'>{sk}</span>"

            # ── description snippet ──────────────────────────────────────────
            desc_html = ""
            if show_desc and description:
                snippet = str(description)[:220].rstrip()
                if len(str(description)) > 220:
                    snippet += "…"
                desc_html = (
                    f"<div style='font-size:0.8rem;color:#6B7A90;"
                    f"line-height:1.55;margin-top:0.5rem;'>{snippet}</div>"
                )

            match_info = (
                f"<span style='color:#4ADE80;font-size:0.78rem;'>✓ {len(matched)} skill"
                f"{'s' if len(matched)!=1 else ''} matched</span> &nbsp;"
                if matched else ""
            )

            st.markdown(
                f"""
                <div class='job-card'>
                    <div class='job-rank'>{rank_lbl}</div>
                    <div class='job-title-text'>{title}</div>
                    <div class='job-category'>{category}</div>
                    {match_info}
                    <div class='score-bar-bg'>
                        <div class='score-bar-fill' style='width:{bar_width}%'></div>
                    </div>
                    <div class='score-text'>{score_pct}% match</div>
                    {f"<div style='margin-top:0.7rem'>{chips_html}</div>" if chips_html else ""}
                    {desc_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_chart:
        st.markdown(
            "<div class='section-label'>Match Score Comparison</div>",
            unsafe_allow_html=True,
        )
        fig = build_bar_chart(results)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # ── Detected skills recap ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown(
            "<div class='section-label'>Your Skills Detected</div>",
            unsafe_allow_html=True,
        )
        chips = "".join(
            f"<span class='skill-chip' style='background:#12202E;border-color:#1E3A5F;"
            f"color:#60A5FA;'>{s}</span>"
            for s in sorted(user_skill_set)[:30]
        )
        st.markdown(f"<div style='line-height:2'>{chips}</div>", unsafe_allow_html=True)
        if len(user_skill_set) > 30:
            st.caption(f"…and {len(user_skill_set)-30} more")

    # ── Download results ──────────────────────────────────────────────────────
    st.markdown("---")
    csv_bytes = results[["job_title", "category", "match_pct", "cleaned_skills"]].rename(
        columns={"match_pct": "match_score_%", "cleaned_skills": "required_skills"}
    ).to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️  Download Results as CSV",
        data=csv_bytes,
        file_name="career_matches.csv",
        mime="text/csv",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# 9.  FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div style='text-align:center;padding:2.5rem 0 1rem;color:#3A4154;font-size:0.75rem;'>
        CareerMatch AI &nbsp;·&nbsp; Built with Sentence-Transformers &amp; Streamlit<br>
        Model: <code>all-MiniLM-L6-v2</code> &nbsp;|&nbsp; Similarity: Cosine
    </div>
    """,
    unsafe_allow_html=True,
)