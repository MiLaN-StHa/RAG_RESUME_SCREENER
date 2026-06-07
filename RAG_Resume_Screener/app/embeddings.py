from langchain_huggingface import HuggingFaceEmbeddings
from config import Config


class EmbeddingManager:
    @staticmethod
    def load_embeddings():
        embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL
        )
        return embeddings
