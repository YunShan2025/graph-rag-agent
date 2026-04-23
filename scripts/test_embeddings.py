import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from sentence_transformers import SentenceTransformer
import torch
import faiss

from graphrag_agent.config.settings import CACHE_SENTENCE_TRANSFORMER_MODEL

def main():
    model_name = CACHE_SENTENCE_TRANSFORMER_MODEL or 'all-MiniLM-L6-v2'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)

    print('Loading model:', model_name)
    model = SentenceTransformer(model_name, device=device)

    texts = [
        '华东理工大学的国家奖学金评定办法有哪些要求？',
        '学生申诉管理规定的流程是什么？',
        '如何申请上海市奖学金？',
    ]

    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    print('Embeddings shape:', embeddings.shape)

    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    print('Index ntotal:', index.ntotal)

    query = '奖学金 申请 条件'
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    k = 3
    scores, indices = index.search(q_emb, k)
    print('Query:', query)
    print('Results:')
    for score, idx in zip(scores[0], indices[0]):
        print(f'  idx={idx} score={score:.4f} text="{texts[idx]}"')

if __name__ == '__main__':
    main()
