"""Simple RAG system that retrieves nearest docs and uses an LLM (transformers) to generate answer.
This is a lightweight scaffold — adapt model calls to suit your compute or hosted APIs.
"""
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

class RAGSystem:
    def __init__(self, index_dir='models/index', embed_model='all-MiniLM-L6-v2', llm_model=None):
        self.index_dir = index_dir
        self.embed_model = SentenceTransformer(embed_model)
        self.index = None
        meta_path = os.path.join(index_dir, 'metas.npy')
        if os.path.exists(meta_path):
            self.metas = np.load(meta_path, allow_pickle=True)
        else:
            self.metas = []
        faiss_index_path = os.path.join(index_dir, 'faiss.index')
        if os.path.exists(faiss_index_path):
            self.index = faiss.read_index(faiss_index_path)
        # LLM - optional: user can provide a local or hosted model
        if llm_model:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
            self.llm = AutoModelForCausalLM.from_pretrained(llm_model)
        else:
            self.tokenizer = None
            self.llm = None

    def retrieve(self, query, k=3):
        q_emb = self.embed_model.encode([query], convert_to_numpy=True)
        if self.index is None:
            return []
        D, I = self.index.search(q_emb, k)
        results = []
        for idx in I[0]:
            if idx < len(self.metas):
                results.append(self.metas[idx].item()['source'])
        return results

    def answer(self, query):
        docs = self.retrieve(query)
        # Load doc texts
        texts = []
        for fname in docs:
            try:
                with open(os.path.join('data', fname), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
            except Exception:
                pass
        context = '\n\n'.join(texts)
        prompt = f"Context:\n{context}\n\nUser: {query}\nAssistant:" 
        # If no LLM available, return retrieved docs and prompt template
        if self.llm is None:
            return f"(No LLM configured) Retrieved documents: {docs}\n---\nPrompt you can use with an LLM:\n{prompt}"
        # Otherwise run generation (careful: large models may be heavy)
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.llm.generate(**inputs, max_new_tokens=256)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
