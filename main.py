from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from rapidfuzz import process, fuzz
import ollama 


def dedupe_pairs(
    raw_pairs: list[tuple[str,str]],
    threshold: int = 85
) -> list[tuple[str,str]]:

    unique_pairs = []
    for reason, action in raw_pairs:
        combo = f"{reason} || {action}"
        if unique_pairs:
            # build list of existing combos
            existing = [f"{r} || {a}" for r,a in unique_pairs]
            match, score, _ = process.extractOne(combo, existing, scorer=fuzz.token_sort_ratio)
            if score >= threshold:
                continue
        unique_pairs.append((reason, action))
    return unique_pairs


def build_pair_index(
    pairs: list[tuple[str,str]],
    embed_model_name: str = 'all-MiniLM-L6-v2'
):

    texts = [f"{r} || {a}" for r,a in pairs]
    embedder = SentenceTransformer(embed_model_name)
    embs = embedder.encode(texts, convert_to_numpy=True)
    faiss.normalize_L2(embs)
    dim = embs.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(embs)
    return embedder, idx


def retrieve_pairs(
    category: str,
    details: str,
    unique_pairs: list[tuple[str,str]],
    embedder: SentenceTransformer,
    index: faiss.IndexFlatIP,
    top_k: int = 5
) -> list[tuple[str,str]]:

    query = f"CATEGORY: {category} || DETAILS: {details}"
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    _, I = index.search(q_emb, top_k)
    return [unique_pairs[i] for i in I[0]]


def init_ollama_client() -> ollama.Client:
    return ollama.Client()


def suggest_pairs(
    category: str,
    details: str,
    unique_pairs: list[tuple[str,str]],
    embedder: SentenceTransformer,
    index: faiss.IndexFlatIP,
    client: ollama.Client,
    top_k: int = 5,
    n_new: int = 5,
    temperature: float = 0.2,
    max_tokens: int = 256
) -> str:

    hist = retrieve_pairs(category, details, unique_pairs, embedder, index, top_k)
    hist_block = "\n".join(f"- Reason: {r}  →  Action: {a}" for r,a in hist)

    prompt = f"""
You are a safety‑alert assistant. Given the alert below:

Alert category: {category}
Details: {details}

1) First list which of the following historical (reason, action) pairs apply:
{hist_block}

2) Then generate {n_new} NEW concise (reason, action) pairs, formatted the same way.
"""

    resp = client.generate(
        model='llama3.2',
        prompt=prompt,
        options={'max_tokens': max_tokens, 'temperature': temperature}
    )
    return resp.get('text') or resp.get('response')


def main():
    raw_pairs = [
        ("check", "verify sensor reading"),
        ("chekk", "verify sensor reading"),
        ("sensor misaligned", "realign sensor"),
    ]

    unique_pairs = dedupe_pairs(raw_pairs)

    embedder, pair_index = build_pair_index(unique_pairs)

    client = init_ollama_client()

    category = "Overheat Alert"
    details = "Temp sensor above threshold for 3 min."
    suggestions = suggest_pairs(
        category, details,
        unique_pairs, embedder, pair_index, client,
        top_k=5, n_new=3
    )

    print(suggestions)


if __name__ == "__main__":
    main()
