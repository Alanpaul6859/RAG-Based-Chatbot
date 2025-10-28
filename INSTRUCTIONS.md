# Instructions & Notes

## Overview
This scaffold provides a minimal RAG flow:
1. Add `.txt` documents to `data/`.
2. Run `scripts/generate_embeddings.py` to create a FAISS index (uses sentence-transformers).
3. Run `app.py` to start a Flask web UI for chat. The RAG system will retrieve relevant docs and
   return them as a prompt if no LLM is configured.

## Choosing an LLM
- For local inference, you can use smaller models from Hugging Face (e.g., `gpt2`, `distilgpt2`) but
  quality will vary. To enable, pass `llm_model` when creating `RAGSystem`.
- Alternatively, use a hosted LLM API (OpenAI, Anthropic, etc.) by replacing the generate section in
  `scripts/rag_inference.py` with an API call.

## Prompt Engineering Tips
- Provide a clear context block separated from the user query.
- Limit context length to fit model prompt size.
- Use system instructions to set tone, length, and format.

## Security
- Do NOT commit API keys. Use environment variables or secrets manager.
