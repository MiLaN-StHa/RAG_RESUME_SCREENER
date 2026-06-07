import os
import json
import tempfile
from datetime import datetime

import streamlit as st

from document_loader import DocumentLoader
from text_processor import TextProcessor
from embeddings import EmbeddingManager
from vector_store import VectorStoreManager
from screener import ResumeScreener
from ui_components import (
    render_sidebar_config,
    render_score_card,
    render_skills,
    render_strengths,
    render_interview_questions,
    render_summary,
    show_error,
    show_success,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Resume Screener",
    layout="wide"
)

# ── Minimal global CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    h1 { letter-spacing: -1px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in {
    "screen_result": None,
    "vector_store": None,
    "screening_history": [],
    "resume_text": "",
    "jd_text": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────
selected_provider = render_sidebar_config()

# ── API key status (live, shown in sidebar) ───────────────────────────────────
st.sidebar.divider()
st.sidebar.subheader("🔑 API Key Status")
_key_valid, _key_err = ResumeScreener.validate_api_key(selected_provider)
if _key_valid:
    st.sidebar.success(f"✅ {selected_provider.upper()} key is valid")
else:
    st.sidebar.error(f"❌ {selected_provider.upper()} key invalid")
    st.sidebar.caption(_key_err.replace("**", ""))

st.sidebar.divider()
st.sidebar.subheader("📜 Screening History")

search_term = st.sidebar.text_input("🔍 Filter history", placeholder="Search...")

history = st.session_state.screening_history
if search_term:
    history = [
        h for h in history
        if search_term.lower() in h.get("candidate", "").lower()
        or search_term.lower() in h.get("summary", "").lower()
    ]

col_clr, col_exp = st.sidebar.columns(2)
with col_clr:
    if st.button("🗑 Clear", use_container_width=True):
        st.session_state.screening_history = []
        st.rerun()
with col_exp:
    st.download_button(
        label="📥 Export",
        data=json.dumps(st.session_state.screening_history, indent=2),
        file_name=f"screening_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True
    )

if not history:
    st.sidebar.caption("No history yet")
else:
    for i, h in enumerate(reversed(history[-15:])):
        with st.sidebar.expander(
            f"🎯 {h.get('candidate','Resume')} — {h.get('score', '?')}%"
        ):
            st.markdown(f"**Provider:** `{h.get('provider','').upper()}`")
            st.markdown(f"**Recommendation:** {h.get('recommendation','N/A')}")
            st.caption(h.get("timestamp", ""))

st.sidebar.divider()
st.sidebar.caption(f"📚 Total screenings: {len(st.session_state.screening_history)}")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("RAG Resume Screener")
st.markdown(
    "Upload a **resume** and a **job description** — get an instant AI-powered match analysis."
)
st.divider()

# ── Upload section ────────────────────────────────────────────────────────────
col_resume, col_jd = st.columns(2)

with col_resume:
    st.subheader("📄 Resume")
    resume_file = st.file_uploader(
        "Upload resume",
        type=["pdf", "txt", "docx"],
        key="resume_uploader",
        label_visibility="collapsed"
    )

with col_jd:
    st.subheader("📋 Job Description")
    jd_mode = st.radio(
        "Provide JD as",
        ["Paste text", "Upload file"],
        horizontal=True,
        label_visibility="collapsed"
    )
    if jd_mode == "Paste text":
        jd_text_input = st.text_area(
            "Paste job description here",
            height=200,
            placeholder="e.g. Looking for an AI/ML intern with Python, TensorFlow, FastAPI...",
            label_visibility="collapsed"
        )
        jd_file = None
    else:
        jd_file = st.file_uploader(
            "Upload job description",
            type=["pdf", "txt", "docx"],
            key="jd_uploader",
            label_visibility="collapsed"
        )
        jd_text_input = ""

st.divider()

# ── Analyse button ────────────────────────────────────────────────────────────
analyse_btn = st.button(
    "🚀 Analyse Match",
    type="primary",
    use_container_width=True
)

if analyse_btn:
    # ── Validation ────────────────────────────────────────────────────────────
    if not resume_file:
        show_error("Please upload a resume.")
        st.stop()

    has_jd = (jd_mode == "Paste text" and jd_text_input.strip()) or \
             (jd_mode == "Upload file" and jd_file is not None)
    if not has_jd:
        show_error("Please provide a job description.")
        st.stop()

    # ── API key validation BEFORE any heavy processing ────────────────────────
    with st.spinner(f"Validating {selected_provider.upper()} API key..."):
        key_ok, key_err_msg = ResumeScreener.validate_api_key(selected_provider)
    if not key_ok:
        st.error(key_err_msg)
        st.stop()

    with st.spinner(f"Processing with {selected_provider.upper()}..."):

        # ── Load resume ───────────────────────────────────────────────────────
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(resume_file.name)[1]
        ) as tmp:
            tmp.write(resume_file.getbuffer())
            resume_path = tmp.name

        resume_docs = DocumentLoader.load_document(resume_path)
        resume_text = TextProcessor.extract_full_text(resume_docs)
        st.session_state.resume_text = resume_text

        # ── Load JD ───────────────────────────────────────────────────────────
        if jd_mode == "Paste text":
            jd_text = jd_text_input.strip()
        else:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=os.path.splitext(jd_file.name)[1]
            ) as tmp:
                tmp.write(jd_file.getbuffer())
                jd_path = tmp.name
            jd_docs = DocumentLoader.load_document(jd_path)
            jd_text = TextProcessor.extract_full_text(jd_docs)

        st.session_state.jd_text = jd_text

        # ── Build vector store for follow-up Q&A ─────────────────────────────
        resume_chunks = TextProcessor.split_documents(resume_docs)
        embeddings = EmbeddingManager.load_embeddings()
        vector_store = VectorStoreManager.create_vector_store(
            resume_chunks, embeddings
        )
        st.session_state.vector_store = vector_store

        # ── Run screening ─────────────────────────────────────────────────────
        result = ResumeScreener.screen(
            resume_text=resume_text,
            jd_text=jd_text,
            provider=selected_provider
        )
        st.session_state.screen_result = result

        # ── Save to history ───────────────────────────────────────────────────
        st.session_state.screening_history.append({
            "candidate": os.path.splitext(resume_file.name)[0],
            "score": result.get("match_score", 0),
            "recommendation": result.get("recommendation", "N/A"),
            "summary": result.get("summary", ""),
            "provider": selected_provider,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    show_success("Analysis complete!")

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.screen_result:
    r = st.session_state.screen_result
    st.divider()

    tabs = st.tabs([
        "📊 Score & Skills",
        "💬 Details",
        "🔍 Ask the Resume"
    ])

    # ── Tab 1: Score & Skills ─────────────────────────────────────────────────
    with tabs[0]:
        left, right = st.columns([1, 2])
        with left:
            render_score_card(r.get("match_score", 0))
            render_summary(
                r.get("summary", ""),
                r.get("recommendation", "N/A")
            )
        with right:
            render_skills(
                r.get("matched_skills", []),
                r.get("missing_skills", [])
            )

    # ── Tab 2: Strengths + Interview Questions ────────────────────────────────
    with tabs[1]:
        render_strengths(r.get("strengths", []))
        st.divider()
        render_interview_questions(r.get("interview_questions", []))

    # ── Tab 3: Follow-up Q&A against the resume ───────────────────────────────
    with tabs[2]:
        st.markdown("Ask anything about the candidate's resume.")
        followup = st.text_input(
            "Your question",
            placeholder="e.g. Does the candidate have FastAPI experience?",
            key="followup_input"
        )
        if followup and st.session_state.vector_store:
            with st.spinner("Searching resume..."):
                qa_result = ResumeScreener.ask(
                    question=followup,
                    vector_store=st.session_state.vector_store,
                    provider=selected_provider
                )
            st.markdown(f"**Answer:** {qa_result['result']}")

            with st.expander("📑 Source chunks"):
                for doc in qa_result.get("source_documents", []):
                    st.markdown(
                        f"> {doc.page_content[:400]}..."
                        if len(doc.page_content) > 400
                        else f"> {doc.page_content}"
                    )
        elif followup and not st.session_state.vector_store:
            show_error("Run a screening first to enable Q&A.")