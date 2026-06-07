import json
import re
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import Config


# Human-readable names for env var hints per provider
_ENV_VAR_HINTS = {
    "groq":   "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY or OPENAI_ADMIN_KEY",
    "gemini": "GOOGLE_API_KEY or GEMINI_API_KEY",
}


class ResumeScreener:

    @staticmethod
    def get_llm(provider: str = "groq"):
        """Build and return the LLM client for the given provider."""
        provider = provider.lower()
        if provider not in Config.MODELS:
            raise ValueError(f"Unsupported provider: '{provider}'.")

        model_info = Config.MODELS[provider]
        api_key = model_info.get("api_key")

        if not api_key:
            raise ValueError(
                f"No API key found for provider '{provider.upper()}'. "
                f"Add {_ENV_VAR_HINTS[provider]} to your .env file."
            )

        if provider == "groq":
            return ChatGroq(
                api_key=api_key,
                model=model_info["model"],
                temperature=0
            )
        elif provider == "openai":
            return ChatOpenAI(
                api_key=api_key,
                model=model_info["model"],
                temperature=0
            )
        elif provider == "gemini":
            return ChatGoogleGenerativeAI(
                api_key=api_key,
                model=model_info["model"],
                temperature=0
            )

    @staticmethod
    def validate_api_key(provider: str) -> tuple[bool, str]:
        """
        Make a minimal test call to the provider to confirm the key is valid.
        Returns (True, "") on success or (False, error_message) on failure.
        """
        provider = provider.lower()
        model_info = Config.MODELS.get(provider, {})
        api_key = model_info.get("api_key")

        # ── 1. Key missing from .env entirely ────────────────────────────────
        if not api_key:
            return False, (
                f"**No API key found for {provider.upper()}.** "
                f"Add `{_ENV_VAR_HINTS[provider]}` to your `.env` file and restart the app."
            )

        # ── 2. Structural sanity checks (catch obvious copy-paste mistakes) ──
        api_key_stripped = api_key.strip()

        if provider == "groq" and not api_key_stripped.startswith("gsk_"):
            return False, (
                "**Invalid Groq API key format.** "
                "Groq keys start with `gsk_`. Check your `.env` file."
            )
        if provider == "openai" and not api_key_stripped.startswith("sk-"):
            return False, (
                "**Invalid OpenAI API key format.** "
                "OpenAI keys start with `sk-`. Check your `.env` file."
            )

        # ── 3. Live test call ─────────────────────────────────────────────────
        try:
            llm = ResumeScreener.get_llm(provider)
            llm.invoke([HumanMessage(content="Hi")])
            return True, ""

        except Exception as e:
            err = str(e)

            # Translate common API error messages into plain English
            if "401" in err or "authentication" in err.lower() or "invalid api key" in err.lower():
                return False, (
                    f"**{provider.upper()} API key is invalid or revoked.** "
                    f"Check `{_ENV_VAR_HINTS[provider]}` in your `.env` file."
                )
            if "403" in err or "permission" in err.lower():
                return False, (
                    f"**{provider.upper()} API key lacks permission.** "
                    "Ensure the key has the required scopes/access."
                )
            if "429" in err or "rate limit" in err.lower() or "quota" in err.lower():
                return False, (
                    f"**{provider.upper()} rate limit or quota exceeded.** "
                    "Wait a moment or check your billing/plan."
                )
            if "connect" in err.lower() or "network" in err.lower() or "timeout" in err.lower():
                return False, (
                    f"**Could not reach {provider.upper()} servers.** "
                    "Check your internet connection and try again."
                )

            # Fallback: show the raw error
            return False, f"**{provider.upper()} error:** {err}"

    @staticmethod
    def screen(resume_text: str, jd_text: str, provider: str = "groq") -> dict:
        """Send resume + JD to LLM and return structured screening result."""
        llm = ResumeScreener.get_llm(provider)

        prompt = Config.SCREENING_PROMPT.format(
            resume_text=resume_text[:6000],
            jd_text=jd_text[:3000]
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        # Strip markdown fences if model wraps output in ```json ... ```
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {
                "match_score": 0,
                "matched_skills": [],
                "missing_skills": [],
                "strengths": [],
                "interview_questions": [],
                "summary": "Could not parse LLM response. Please try again.",
                "recommendation": "N/A",
                "_raw": raw
            }

        return result

    @staticmethod
    def ask(question: str, vector_store, provider: str = "groq") -> dict:
        """
        Follow-up Q&A against the resume vector store.
        Retrieves top-k chunks then prompts the LLM directly.
        """
        llm = ResumeScreener.get_llm(provider)

        docs = vector_store.similarity_search(question, k=4)
        context = "\n\n".join(doc.page_content for doc in docs)

        messages = [
            SystemMessage(content=(
                "You are an expert recruiter assistant. "
                "Answer questions about a candidate based ONLY on the resume context provided. "
                "Be concise and specific."
            )),
            HumanMessage(content=(
                f"Resume context:\n{context}\n\n"
                f"Question: {question}"
            ))
        ]

        response = llm.invoke(messages)
        return {
            "result": response.content,
            "source_documents": docs
        }