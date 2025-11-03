# question_validator.py
import logging
import re
import os
from datetime import datetime

# Assuming these are available from your project structure
# If Embedder and Retriever are not directly importable like this,
# you might need to adjust paths or pass their instances differently.
# For this example, we assume they are in the same logical "level"
# as this validator and can be imported.
from embedder import Embedder
from retriever import Retriever

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QuestionValidator:
    def __init__(self, embedder: Embedder, retriever: Retriever, knowledge_threshold: float = 0.6, max_query_length: int = 200):
        self.embedder = embedder
        self.retriever = retriever
        self.knowledge_threshold = knowledge_threshold # Similarity threshold for "in-knowledge" check
        self.max_query_length = max_query_length # Max characters for a valid query

        # --- CUSTOMIZE THESE KEYWORDS FOR VELOX SOLUTION'S KNOWLEDGE DOMAIN ---
        # These keywords help quickly identify if a question is relevant to your business.
        self.domain_keywords = [
            "planetguard", "velox solution", "endpoint security", "cybersecurity", "edr", "threat detection",
            "unified security", "data protection", "firewall", "antivirus", "malware", "ransomware",
            "security management", "compliance", "incident response", "vulnerability", "patch management",
            "network security", "device management", "security policy", "threat intelligence",
            "security monitoring", "forensics", "backup", "encryption", "access control",
            "pricing", "subscription", "personal package", "sme package", "enterprise package",
            "features", "benefits", "installation", "support", "integration", "api",
            "dashboard", "reporting", "alerts", "notifications", "quarantine", "sandbox"
        ]
        
        # Define keywords for common "out-of-scope" queries (general knowledge, personal, etc.)
        self.off_topic_keywords = [
            "weather", "news", "jokes", "recipes", "sports", "politics", "celebrity",
            "current events", "personal advice", "gossip", "tell me about yourself",
            "what is your name", "who created you", "how are you", "what time is it",
            "what is the meaning of life", "stock market", "financial advice", "health advice"
        ]
        
        # Regex for identifying spam/malicious patterns (e.g., very long strings of special chars)
        self.spam_pattern = re.compile(r'([^a-zA-Z0-9\s]){10,}|.{' + str(max_query_length * 2) + ',}', re.DOTALL)
        logging.info("QuestionValidator initialized.")

    def is_within_knowledge_boundary(self, query: str) -> bool:
        """
        Checks if the query is likely within the application's knowledge boundary.
        This uses a combination of keyword matching and a quick vector search.
        """
        query_lower = query.lower()

        # 1. Fast Keyword-based check: If it contains core domain keywords, it's likely relevant.
        if any(keyword in query_lower for keyword in self.domain_keywords):
            logging.debug(f"Query '{query}' matched domain keywords.")
            return True
        
        # 2. Semantic similarity check using retriever: More robust, but involves an embedding call.
        # This checks if the query is semantically similar to *any* document in the knowledge base.
        try:
            # The retriever's search function returns a list of dictionaries.
            # We need to ensure it returns a score to use the knowledge_threshold.
            # If your retriever.search doesn't return scores, this part needs adjustment
            # or you can simply check if relevant_docs is not empty.
            
            # Assuming retriever.search returns [{"text": ..., "file_path": ..., "score": ...}]
            # If not, you might need to modify retriever.py or use a simpler check here.
            
            # For now, let's assume `retriever.search` returns documents, and we'll
            # rely on the *presence* of a document as an indicator, or modify retriever.py
            # to expose scores. For a quick check, we can just see if anything is returned.
            
            # To properly use knowledge_threshold, retriever.search needs to return scores.
            # Let's assume for this example, if relevant_docs is not empty, it's "relevant enough".
            # For production, modify retriever.py to return scores and check against threshold.
            
            relevant_docs = self.retriever.search(query, self.embedder, k=1) # Search for just 1 top document
            if relevant_docs: # If at least one document is found
                # If retriever.search returns scores, you'd do:
                # if relevant_docs[0].get("score", 0) > self.knowledge_threshold:
                #     logging.debug(f"Query '{query}' found relevant doc with score above threshold.")
                #     return True
                logging.debug(f"Query '{query}' found at least one relevant document in KB.")
                return True

        except Exception as e:
            logging.error(f"Error during knowledge boundary retrieval check for query '{query}': {e}")
            # If retrieval fails, better to let it pass to LLM with context or a general error,
            # rather than rejecting it outright if it might be valid.
            return True # Assume valid if check fails, to avoid false negatives

        logging.info(f"Query '{query}' is likely outside knowledge boundary (no keyword match, no relevant doc found).")
        return False

    def is_valid_question(self, query: str) -> bool:
        """
        Performs basic validation on the query: length, spam, obvious off-topic keywords.
        """
        # 1. Check for empty or too short query
        if not query or len(query.strip()) < 5: # Minimum length of 5 characters
            logging.info(f"Validation failed: Query '{query}' too short or empty.")
            return False

        # 2. Check for excessively long queries (potential spam/injection or just bad user input)
        if len(query) > self.max_query_length:
            logging.info(f"Validation failed: Query '{query}' too long (>{self.max_query_length} chars).")
            return False

        # 3. Check for spam/malicious patterns (e.g., repetitive special characters)
        if self.spam_pattern.search(query):
            logging.info(f"Validation failed: Query '{query}' matched spam pattern.")
            return False

        # 4. Check for obvious off-topic keywords (e.g., general chit-chat not related to business)
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in self.off_topic_keywords):
            logging.info(f"Validation failed: Query '{query}' matched off-topic keywords.")
            return False
            
        logging.debug(f"Query '{query}' passed basic validity checks.")
        return True

    def validate_and_filter(self, query: str) -> str:
        """
        Combines validation steps. Returns a rejection message if invalid/out-of-scope,
        otherwise returns an empty string (indicating it's valid to proceed to RAG).
        """
        # First, perform cheap and fast validity checks
        if not self.is_valid_question(query):
            return "That doesn't seem like a valid question or it's too general. Please ask a more specific question relevant to Velox Solution."
        
        # If basic validity passes, then perform the more resource-intensive knowledge boundary check
        if not self.is_within_knowledge_boundary(query):
            return "I apologize, but your question appears to be outside the scope of my knowledge base about Velox Solution. I can only answer questions related to our products, services, and technical information."
        
        # If both checks pass, the query is considered valid and within scope
        logging.info(f"Query '{query}' passed all validation checks. Proceeding to RAG.")
        return "" # Valid to proceed