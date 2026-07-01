"""
Service layer (orchestration / dependency injection)
====================================================
Thin services that wire the ML, LLM, RAG, and database layers together via
dependency injection — so new capabilities slot in *alongside* the existing
``EmailSentimentAnalyzer`` without rewriting it.

Populated from Phase 1 onward.
"""
