import streamlit as st


# ── Recommendation badge colours ─────────────────────────────────────────────
RECOMMENDATION_STYLES = {
    "Strong Hire": ("🟢", "#16a34a", "#dcfce7"),
    "Hire":        ("🟡", "#ca8a04", "#fef9c3"),
    "Maybe":       ("🟠", "#ea580c", "#ffedd5"),
    "No Hire":     ("🔴", "#dc2626", "#fee2e2"),
}


def render_sidebar_config():
    """Render LLM selector in sidebar. Returns selected provider string."""
    st.sidebar.title("⚙️ Settings")
    st.sidebar.subheader("🤖 LLM Provider")
    provider = st.sidebar.selectbox(
        "Choose LLM",
        options=["groq", "openai", "gemini"],
        index=0
    )
    model_name = {
        "groq": "llama-3.1-8b-instant",
        "openai": "gpt-4o-mini",
        "gemini": "gemini-2.5-flash",
    }[provider]
    st.sidebar.info(f"Model: `{model_name}`")
    return provider


def render_score_card(score: int):
    """Large animated score dial using st.metric + progress bar."""
    if score >= 80:
        colour, label = "#16a34a", "Excellent Match"
    elif score >= 60:
        colour, label = "#ca8a04", "Good Match"
    elif score >= 40:
        colour, label = "#ea580c", "Partial Match"
    else:
        colour, label = "#dc2626", "Poor Match"

    st.markdown(
        f"""
        <div style="
            text-align:center;
            padding:2rem 1rem;
            border-radius:1rem;
            background:linear-gradient(135deg,#0f172a,#1e293b);
            margin-bottom:1.5rem;
        ">
            <div style="font-size:4.5rem;font-weight:900;color:{colour};
                        line-height:1;letter-spacing:-2px;">{score}%</div>
            <div style="font-size:1.1rem;color:#94a3b8;margin-top:.4rem;">
                Match Score
            </div>
            <div style="font-size:.95rem;font-weight:600;color:{colour};
                        margin-top:.3rem;">{label}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.progress(score / 100)


def render_skills(matched: list, missing: list):
    """Two-column skill chips — green matched, red missing."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ✅ Matched Skills")
        if matched:
            chips = " ".join(
                f'<span style="display:inline-block;background:#dcfce7;'
                f'color:#166534;padding:.25rem .75rem;border-radius:999px;'
                f'font-size:.85rem;margin:.2rem;">{s}</span>'
                for s in matched
            )
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.caption("None detected")

    with col2:
        st.markdown("#### ❌ Missing Skills")
        if missing:
            chips = " ".join(
                f'<span style="display:inline-block;background:#fee2e2;'
                f'color:#991b1b;padding:.25rem .75rem;border-radius:999px;'
                f'font-size:.85rem;margin:.2rem;">{s}</span>'
                for s in missing
            )
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.caption("No gaps found")


def render_strengths(strengths: list):
    if not strengths:
        return
    st.markdown("#### 💪 Candidate Strengths")
    for s in strengths:
        st.markdown(f"- {s}")


def render_interview_questions(questions: list):
    if not questions:
        return

    st.markdown("#### 🎤 Suggested Interview Questions")

    for i, q in enumerate(questions, 1):
        st.markdown(
            f"""
            <div style="
                background:#1e293b;
                color:#ffffff;
                border-left:4px solid #6366f1;
                padding:0.8rem 1rem;
                margin:0.5rem 0;
                border-radius:0.5rem;
                font-size:0.95rem;
            ">
                <b style="color:#a5b4fc;">Q{i}.</b> {q}
            </div>
            """,
            unsafe_allow_html=True
        )

def render_recommendation(rec: str):
    icon, fg, bg = RECOMMENDATION_STYLES.get(rec, ("⚪", "#475569", "#f1f5f9"))
    st.markdown(
        f"""
        <div style="display:inline-block;background:{bg};color:{fg};
                    padding:.5rem 1.5rem;border-radius:999px;font-weight:700;
                    font-size:1.05rem;margin-top:.5rem;">
            {icon} Recommendation: {rec}
        </div>
        """,
        unsafe_allow_html=True
    )


def render_summary(summary: str, recommendation: str):
    st.markdown("#### 📋 Summary")
    st.info(summary)
    render_recommendation(recommendation)


def show_success(msg):
    st.success(msg)


def show_error(msg):
    st.error(msg)
