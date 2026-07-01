"""
LLM layer (local-only, via Ollama)
==================================
Houses the Ollama client wrapper and the prompt-driven analysers that add an
intelligence layer *on top of* the existing TF-IDF + Naive Bayes pipeline.

Nothing here is required for the original ML features to work — every component
is designed to degrade gracefully when Ollama is not installed or a model is
unavailable (see Phase 2).

Populated in Phase 2.
"""
