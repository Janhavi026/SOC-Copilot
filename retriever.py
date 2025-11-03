import chromadb
import uuid
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Retriever:
    def __init__(self, chroma_data_path="chroma_data", collection_name="rag_documents"):
        """
        Initializes the Retriever with a ChromaDB client and collection.
        """
        self.chroma_data_path = chroma_data_path
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialize_chroma()
        logging.info(f"Retriever initialized with ChromaDB at: {chroma_data_path}, Collection: {collection_name}")

    def _initialize_chroma(self):
        """Initializes ChromaDB client and gets/creates the collection."""
        try:
            os.makedirs(self.chroma_data_path, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.chroma_data_path)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            logging.info(f"ChromaDB collection '{self.collection_name}' ready. Total documents: {self.collection.count()}")
        except Exception as e:
            logging.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def add_documents(self, documents, embedder):
        """
        Adds documents to the ChromaDB collection in manageable batches to avoid size limits.
        """
        if not documents:
            logging.info("No documents to add.")
            return

        texts = [doc["text"] for doc in documents]
        embeddings = embedder.get_embeddings_batch(texts)

        if not embeddings or all(e is None for e in embeddings):
            logging.error("Failed to generate embeddings for any documents.")
            return

        # --- FIX: Process documents in batches to stay within ChromaDB limits ---
        batch_size = 4000  # A safe batch size, well below the 5461 limit shown in the error
        total_added = 0

        for i in range(0, len(documents), batch_size):
            batch_documents = documents[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]

            chroma_documents = []
            chroma_metadatas = []
            chroma_ids = []
            chroma_embeddings = []

            for j, emb in enumerate(batch_embeddings):
                if emb is not None:
                    doc_data = batch_documents[j]
                    metadata = {
                        "file_path": os.path.basename(doc_data["file_path"]),
                        "source_type": doc_data.get("source_type", "generic") # Ensure metadata is included
                    }
                    chroma_documents.append(doc_data["text"])
                    chroma_metadatas.append(metadata)
                    chroma_ids.append(str(uuid.uuid4()))
                    chroma_embeddings.append(emb)
                else:
                    logging.warning(f"Skipping document due to failed embedding: {batch_documents[j].get('file_path', 'unknown')}")

            if chroma_documents:
                try:
                    self.collection.add(
                        embeddings=chroma_embeddings,
                        documents=chroma_documents,
                        metadatas=chroma_metadatas,
                        ids=chroma_ids
                    )
                    total_added += len(chroma_documents)
                    logging.info(f"Added batch of {len(chroma_documents)} documents to ChromaDB. Total so far: {total_added}")
                except Exception as e:
                    logging.error(f"Error adding batch to ChromaDB: {e}")
                    # Decide if you want to stop or continue on batch failure
                    continue
        
        logging.info(f"Finished adding documents. Total added: {total_added}. Collection size: {self.collection.count()}")

    def search(self, query_text, embedder, k=5, metadata_filter: dict = None):
        """
        Searches the ChromaDB collection for the top-k most similar documents,
        with an optional metadata filter.
        """
        query_embedding = embedder.get_embedding(query_text)
        if query_embedding is None:
            logging.error("Failed to generate embedding for query.")
            return []

        try:
            # Add the 'where' clause for metadata filtering if provided
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": k,
                "include": ['documents', 'distances', 'metadatas']
            }
            if metadata_filter:
                query_params["where"] = metadata_filter

            results = self.collection.query(**query_params)

            retrieved_docs = []
            if results and results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    retrieved_docs.append({
                        "text": results['documents'][0][i],
                        "file_path": results['metadatas'][0][i].get("file_path", "unknown_file"),
                        "score": results['distances'][0][i]
                    })
            logging.info(f"Retrieved {len(retrieved_docs)} documents from ChromaDB.")
            return retrieved_docs
        except Exception as e:
            logging.error(f"Error searching ChromaDB: {e}")
            return []

    def clear_index(self):
        """Clears the ChromaDB collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            logging.info(f"ChromaDB collection '{self.collection_name}' cleared.")
        except Exception as e:
            logging.error(f"Error clearing ChromaDB collection: {e}")

