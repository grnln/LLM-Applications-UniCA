"""Read the text corpus, embed each paragraph, upsert into Weaviate."""
from __future__ import annotations

import os
import re
from pathlib import Path

import weaviate
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from weaviate.classes.config import Configure, DataType, Property

load_dotenv()

TEXT_FILE = Path(os.environ.get("OSCAR_TEXT_FILE", "ontology/oscar.txt"))
COLLECTION = os.environ.get("WEAVIATE_COLLECTION", "OscarFilms")
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


def read_chunks(path: Path, target_chars: int = 800) -> list[str]:
    """Sentence-aware chunking.

    Reconstructs hard-wrapped text into a single stream, splits on sentence
    boundaries, and greedily packs sentences into chunks of ~target_chars.
    Works for both prose documents and one-paragraph-per-line files.
    """
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    text = " ".join(lines)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    buf = ""
    for s in sentences:
        if buf and len(buf) + 1 + len(s) > target_chars:
            chunks.append(buf.strip())
            buf = s
        else:
            buf = f"{buf} {s}".strip()
    if buf:
        chunks.append(buf.strip())
    return chunks


def embed_all(chunks: list[str]) -> list[list[float]]:
    """Turn every chunk into a normalised embedding vector."""
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model.encode(chunks, normalize_embeddings=True).tolist()


def reset_collection(client: weaviate.WeaviateClient) -> None:
    """Drop the collection if it exists and recreate it empty (we supply our own vectors)."""
    if client.collections.exists(COLLECTION):
        client.collections.delete(COLLECTION)
    client.collections.create(
        name=COLLECTION,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="text", data_type=DataType.TEXT),
            Property(name="chunk_id", data_type=DataType.INT),
            Property(name="source", data_type=DataType.TEXT),
        ],
    )


def upsert(client: weaviate.WeaviateClient, chunks: list[str], vectors: list[list[float]]) -> None:
    """Batch-insert every chunk with its vector into the Weaviate collection."""
    collection = client.collections.get(COLLECTION)
    with collection.batch.dynamic() as batch:
        for i, (text, vec) in enumerate(zip(chunks, vectors)):
            batch.add_object(
                properties={"text": text, "chunk_id": i, "source": TEXT_FILE.name},
                vector=vec,
            )


def main() -> None:
    """End-to-end: read → embed → wipe collection → upsert."""
    chunks = read_chunks(TEXT_FILE)
    vectors = embed_all(chunks)
    client = weaviate.connect_to_local()
    try:
        reset_collection(client)
        upsert(client, chunks, vectors)
        print(f"Ingested {len(chunks)} chunks into {COLLECTION}.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
