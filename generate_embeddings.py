"""Generate embeddings for text files using sentence-transformers and save an index.
This script creates a simple FAISS index from the embeddings.
"""
import os
import argparse
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm

def load_texts(data_dir):
    texts = []
    metas = []
    for fname in os.listdir(data_dir):
        if fname.endswith('.txt'):
            path = os.path.join(data_dir, fname)
            with open(path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
                metas.append({'source': fname})
    return texts, metas

def main(data_dir, index_dir, model_name='all-MiniLM-L6-v2'):
    os.makedirs(index_dir, exist_ok=True)
    model = SentenceTransformer(model_name)
    texts, metas = load_texts(data_dir)
    if not texts:
        print('No text files found in', data_dir)
        return
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(index_dir, 'faiss.index'))
    np.save(os.path.join(index_dir, 'metas.npy'), np.array(metas, dtype=object))
    print('Index saved to', index_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data', help='Directory with .txt documents')
    parser.add_argument('--index-dir', default='models/index', help='Directory to store index')
    parser.add_argument('--model-name', default='all-MiniLM-L6-v2', help='SentenceTransformer model')
    args = parser.parse_args()
    main(args.data_dir, args.index_dir, args.model_name)
