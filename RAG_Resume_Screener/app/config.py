from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    # Chunking
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100

    # Embedding
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_ADMIN_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    # LLM Provider models
    MODELS = {
        "groq": {
            "model": "llama-3.1-8b-instant",
            "api_key": GROQ_API_KEY
        },
        "openai": {
            "model": "gpt-4o-mini",
            "api_key": OPENAI_API_KEY
        },
        "gemini": {
            "model": "gemini-2.5-flash",
            "api_key": GOOGLE_API_KEY
        }
    }

    # Screening prompt template
    SCREENING_PROMPT = """You are an expert technical recruiter and HR specialist.

You are given:
1. RESUME CONTENT (extracted from the candidate's resume)
2. JOB DESCRIPTION (the role being hired for)

Your task is to perform a detailed screening analysis and return a JSON object ONLY — no explanation, no markdown, no extra text.

Resume Content:
{resume_text}

Job Description:
{jd_text}

Return exactly this JSON structure:
{{
  "match_score": <integer 0-100>,
  "matched_skills": [<list of skill strings found in both resume and JD>],
  "missing_skills": [<list of skill strings in JD but not in resume>],
  "strengths": [<list of 3-5 candidate strength strings>],
  "interview_questions": [<list of 4-5 tailored interview question strings>],
  "summary": "<2-3 sentence overall assessment>",
  "recommendation": "<one of: Strong Hire, Hire, Maybe, No Hire>"
}}
"""
