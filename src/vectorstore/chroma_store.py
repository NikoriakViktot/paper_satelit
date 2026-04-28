"""
Vector store layer — ChromaDB (persistent, local).

Uses ChromaDB ≥ 0.4 API (PersistentClient).
Embeddings are supplied externally so the collection uses no_embedding function.
"""
import logging
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings

from src.config import CHROMA_DIR, COLLECTION_NAME
from src.processing.chunker import TextChunk

logger = logging.getLogger(__name__)

# ChromaDB stores metadata values as str | int | float | bool only
_META_FIELDS = ("filename", "page_start", "page_end", "chunk_index")


class VectorStore:
    """Thin wrapper around a ChromaDB persistent collection."""

    def __init__(
        self,
        persist_dir: Path = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorStore ready: collection=%s  existing_docs=%d",
            collection_name, self._collection.count(),
        )

    # ── write ────────────────────────────────────────────────────────────────

    def upsert(
        self,
        chunks: list[TextChunk],
        embeddings: np.ndarray,
        batch_size: int = 512,
    ) -> None:
        """
        Upsert chunks + embeddings.
        Idempotent: existing chunk_ids are overwritten.
        """
        n = len(chunks)
        assert embeddings.shape[0] == n, "chunks / embeddings length mismatch"

        logger.info("Upserting %d chunks into ChromaDB …", n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_chunks = chunks[start:end]
            batch_vecs   = embeddings[start:end].tolist()

            self._collection.upsert(
                ids        = [c.chunk_id for c in batch_chunks],
                embeddings = batch_vecs,
                documents  = [c.text for c in batch_chunks],
                metadatas  = [_meta(c) for c in batch_chunks],
            )
            logger.debug("  Upserted batch [%d:%d]", start, end)

        logger.info("Upsert complete. Total in collection: %d", self._collection.count())

    def clear(self) -> None:
        """Delete all documents from the collection."""
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("Collection cleared.")

    # ── read ─────────────────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Return top-k nearest chunks as list of dicts:
            {text, chunk_id, filename, page_start, page_end, distance}
        """
        kwargs: dict = dict(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        hits = []
        for doc, meta, dist, cid in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0],
        ):
            hits.append({
                "text":       doc,
                "chunk_id":   cid,
                "filename":   meta.get("filename", ""),
                "page_start": meta.get("page_start", 0),
                "page_end":   meta.get("page_end",   0),
                "distance":   dist,
            })
        return hits

    def count(self) -> int:
        return self._collection.count()

    def get_by_filename(self, filename: str) -> list[dict]:
        """
        Return ALL stored chunks for *filename*, sorted by page_start.
        Used by SectionExtractor to reconstruct full document text.
        """
        results = self._collection.get(
            where={"filename": filename},
            include=["documents", "metadatas"],
        )
        hits = []
        for doc, meta, cid in zip(
            results.get("documents") or [],
            results.get("metadatas") or [],
            results.get("ids")       or [],
        ):
            hits.append({
                "text":       doc,
                "chunk_id":   cid,
                "filename":   meta.get("filename", ""),
                "page_start": meta.get("page_start", 0),
                "page_end":   meta.get("page_end",   0),
                "distance":   0.0,
            })
        hits.sort(key=lambda h: (h["page_start"], h["chunk_id"]))
        return hits

    def list_filenames(self) -> list[str]:
        """Return sorted list of unique PDF filenames in the collection."""
        results = self._collection.get(include=["metadatas"])
        seen: set[str] = set()
        names: list[str] = []
        for meta in results.get("metadatas") or []:
            fn = meta.get("filename", "")
            if fn and fn not in seen:
                seen.add(fn)
                names.append(fn)
        return sorted(names)


# ── helpers ───────────────────────────────────────────────────────────────────

def _meta(chunk: TextChunk) -> dict:
    return {
        "filename":    chunk.filename,
        "page_start":  chunk.page_start,
        "page_end":    chunk.page_end,
        "chunk_index": chunk.chunk_index,
    }
