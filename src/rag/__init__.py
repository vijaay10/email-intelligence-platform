"""
RAG layer (Sentence-Transformers + ChromaDB)
============================================
Embedding generation and the ChromaDB vector store that powers semantic search
and Retrieval-Augmented Generation for the "Chat with Emails" assistant.

Like the LLM layer, every component degrades gracefully when its optional
dependencies (``sentence-transformers``, ``chromadb``) are not installed.

Populated in Phase 1.
"""
