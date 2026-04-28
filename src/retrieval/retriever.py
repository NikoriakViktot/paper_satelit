"""
Retrieval layer.
Executes multiple semantic queries and merges results,
deduplicating by chunk_id and grouping by source file.
"""
import logging
from collections import defaultdict

from src.config import RETRIEVAL_QUERIES, RETRIEVAL_TOP_K
from src.embedding.embedder import Embedder
from src.vectorstore.chroma_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self._store   = store
        self._embedder = embedder

    def retrieve_for_queries(
        self,
        queries: list[str] = RETRIEVAL_QUERIES,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> dict[str, list[dict]]:
        """
        Run every query and return a mapping:
            filename → [deduplicated hit dicts]
        Hits are sorted by ascending distance (best first).
        """
        seen_ids: set[str] = set()
        by_file: dict[str, list[dict]] = defaultdict(list)

        for query in queries:
            logger.info("Querying: '%s'", query[:60])
            vec  = self._embedder.embed_query(query)
            hits = self._store.query(vec, top_k=top_k)

            for hit in hits:
                if hit["chunk_id"] in seen_ids:
                    continue
                seen_ids.add(hit["chunk_id"])
                by_file[hit["filename"]].append(hit)

        # sort each file's hits by distance (closest first)
        for fname in by_file:
            by_file[fname].sort(key=lambda h: h["distance"])

        total = sum(len(v) for v in by_file.values())
        logger.info(
            "Retrieved %d unique chunks across %d files",
            total, len(by_file),
        )
        return dict(by_file)

    def fetch_full_context(self) -> dict[str, list[dict]]:
        """
        Return ALL stored chunks for every file, each list sorted by page order.
        Used by SectionExtractor to reconstruct full document text per paper.
        """
        filenames = self._store.list_filenames()
        logger.info("Fetching full context for %d files …", len(filenames))
        by_file: dict[str, list[dict]] = {}
        for fn in filenames:
            chunks = self._store.get_by_filename(fn)
            if chunks:
                by_file[fn] = chunks
        total = sum(len(v) for v in by_file.values())
        logger.info("Full context: %d chunks across %d files", total, len(by_file))
        return by_file

    def retrieve_for_file(
        self,
        filename: str,
        queries: list[str] = RETRIEVAL_QUERIES,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> list[dict]:
        """Retrieve chunks restricted to a single source file."""
        seen_ids: set[str] = set()
        hits: list[dict] = []

        for query in queries:
            vec  = self._embedder.embed_query(query)
            raw  = self._store.query(
                vec, top_k=top_k,
                where={"filename": filename},
            )
            for h in raw:
                if h["chunk_id"] not in seen_ids:
                    seen_ids.add(h["chunk_id"])
                    hits.append(h)

        hits.sort(key=lambda h: h["distance"])
        return hits
