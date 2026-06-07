from langchain_community.vectorstores import FAISS


class VectorStoreManager:
    @staticmethod
    def create_vector_store(chunks, embeddings):
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        return vector_store
