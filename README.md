# RAG-Based Chatbot

This repository contains a scaffold for a Retrieval-Augmented Generation (RAG) chatbot using Python, TensorFlow, LLMs, and prompt engineering.

**What's included**
- Scripts to preprocess documents, generate embeddings, and build simple retriever index
- Example Flask app (`app.py`) to run a RAG-style chat service
- Prompts and engineering guidelines in `docs/`
- Notebook template to demo RAG flow
- CI workflow (linting)

**Important**
- This is a scaffold — it does not ship with large LLM weights or commercial API keys.
- You can use open-source LLMs (locally) or call hosted LLM APIs. See `docs/INSTRUCTIONS.md` for options.

## Quickstart (developer)
1. Create a Python virtual environment and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS / Linux
   venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```
2. Prepare your documents in `data/` (one `.txt` per document) or use the example script to add sample docs.
3. Generate embeddings & index:
   ```bash
   python scripts/generate_embeddings.py --data-dir data --index-dir models/index
   ```
4. Run the Flask app:
   ```bash
   python app.py
   ```
5. Open http://127.0.0.1:5000 and chat.

## License
MIT
