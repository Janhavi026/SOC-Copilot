# text_splitter.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

def get_text_splitter():
    """
    Returns a configured RecursiveCharacterTextSplitter.
    Adjust chunk_size and chunk_overlap based on your data and LLM context window.
    """
    # These values are crucial for RAG performance.
    # A common starting point for LLMs with decent context windows.
    # Adjust based on your average document size and the detail required in answers.
    # chunk_size refers to the maximum number of characters in a chunk.
    # chunk_overlap refers to the number of characters that overlap between consecutive chunks.
    chunk_size = 1000
    chunk_overlap = 200

    logging.info(f"Initializing RecursiveCharacterTextSplitter with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len, # Use character length for splitting
        add_start_index=True # Optional: adds metadata about chunk's start position in the original document
    )

def split_document_into_chunks(text_content):
    """
    Splits a single document's text into chunks using the configured splitter.
    Returns a list of dictionaries, where each dictionary has 'text' (the chunk content)
    and 'metadata' (including any original document metadata and the chunk's start index).
    """
    if not text_content or not text_content.strip():
        logging.warning("Received empty or whitespace-only text_content for splitting. Returning empty list.")
        return []

    splitter = get_text_splitter()
    
    # create_documents expects a list of strings
    chunks = splitter.create_documents([text_content])
    
    # The output of create_documents is a list of Document objects.
    # We need to extract the page_content (the text) and metadata from each Document.
    formatted_chunks = []
    for chunk in chunks:
        formatted_chunks.append({
            "text": chunk.page_content,
            "metadata": chunk.metadata
        })
    
    return formatted_chunks