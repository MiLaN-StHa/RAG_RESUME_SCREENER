from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import Config


class TextProcessor:
    @staticmethod
    def split_documents(documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(documents)
        return chunks

    @staticmethod
    def extract_full_text(documents) -> str:
        """Join all document pages into a single string for LLM prompting."""
        return "\n\n".join(doc.page_content for doc in documents)
