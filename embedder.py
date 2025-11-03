# embedder.py
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initializes the SentenceTransformer model for generating embeddings.
        model_name: The name of the Sentence-Transformer model to use.
        """
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            logging.info(f"Embedder initialized with model: {model_name}, dimension: {self.embedding_dimension}")
        except Exception as e:
            logging.error(f"Failed to load SentenceTransformer model {model_name}: {e}")
            raise

    def get_embedding(self, text):
        """
        Generates an embedding for a single text string.
        """
        if not text:
            return None
        try:
            return self.model.encode(text, convert_to_numpy=True).tolist() # Return as list for Chroma
        except Exception as e:
            logging.error(f"Error generating embedding for text: {text[:50]}... Error: {e}")
            return None

    def get_embeddings_batch(self, texts):
        """
        Generates embeddings for a list of text strings in a batch.
        """
        if not texts:
            return []
        try:
            return self.model.encode(texts, convert_to_numpy=True).tolist() # Return as list for Chroma
        except Exception as e:
            logging.error(f"Error generating embeddings for batch. Error: {e}")
            return [None] * len(texts)

