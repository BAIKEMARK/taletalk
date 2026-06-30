from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config
from .memory import SceneMemory
from .memory_pack import scene_memory_text
from .retrieval import tokenize


@dataclass
class SemanticSearchResult:
    scene: SceneMemory
    score: float


class SemanticMemoryIndex:
    def __init__(self, scenes: list[SceneMemory], vectors: list[list[float]], config: Config):
        self.scenes = scenes
        self.vectors = [_normalize(vector) for vector in vectors]
        self.config = config

    @classmethod
    def from_artifacts(cls, scenes: list[SceneMemory], embedding_path: Path, config: Config) -> "SemanticMemoryIndex":
        vectors = read_npy_float32(embedding_path)
        if len(vectors) != len(scenes):
            raise ValueError(f"embedding rows({len(vectors)}) != scenes({len(scenes)})")
        return cls(scenes=scenes, vectors=vectors, config=config)

    def search(self, query: str, top_k: int, exclude_narrator_only: bool = True) -> list[SemanticSearchResult]:
        if not self.vectors:
            return []
        query_vector = embed_query(self.config, query, dimensions=len(self.vectors[0]))
        return self.search_with_vector(query_vector, top_k=top_k, exclude_narrator_only=exclude_narrator_only)

    def search_with_vector(
        self,
        query_vector: list[float],
        top_k: int,
        exclude_narrator_only: bool = True,
    ) -> list[SemanticSearchResult]:
        if not self.vectors:
            return []
        if len(query_vector) != len(self.vectors[0]):
            return []
        query_vector = _normalize(query_vector)
        results: list[SemanticSearchResult] = []
        for scene, vector in zip(self.scenes, self.vectors):
            if exclude_narrator_only and not scene.target_role_knows:
                continue
            score = _dot(query_vector, vector)
            if score > 0:
                results.append(SemanticSearchResult(scene=scene, score=score))
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]


def write_embedding_artifacts(config: Config, scenes: list[SceneMemory]) -> str:
    texts = [scene_memory_text(scene, max_chars=max(config.max_one_scene_chars, 600)) for scene in scenes]
    backend = _embedding_backend(config)
    if backend == "cloud":
        vectors = embed_texts(config, texts)
        provider = "cloud"
    else:
        vectors = [_hashed_embedding(text, config.embedding_dimensions) for text in texts]
        provider = "local_hash"
    _write_npy_float32(vectors, config.embedding_npy)
    _write_embedding_meta(config, scenes, provider=provider, dimensions=len(vectors[0]) if vectors else 0)
    return provider


def embed_query(config: Config, query: str, dimensions: int) -> list[float]:
    backend = _embedding_backend(config)
    if backend == "cloud":
        return embed_texts(config, [query])[0]
    return _hashed_embedding(query, dimensions)


def embed_texts(config: Config, texts: list[str]) -> list[list[float]]:
    base_url = _embedding_base_url(config)
    api_key = _embedding_api_key(config)
    model = _embedding_model(config)
    if not base_url or not api_key or not model:
        if config.embedding_backend == "cloud":
            raise ValueError("embedding_backend=cloud 需要 embedding_base_url、embedding_api_key、embedding_model")
        return [_hashed_embedding(text, config.embedding_dimensions) for text in texts]

    vectors: list[list[float]] = []
    batch_size = max(1, config.embedding_batch_size)
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(_call_embedding_api(base_url, api_key, model, batch))
    return vectors


def rerank_items(config: Config, query: str, items: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    if not items or not config.use_reranker:
        return items[:top_k]
    backend = _reranker_backend(config)
    if backend != "cloud":
        return items[:top_k]

    base_url = _reranker_base_url(config)
    api_key = _reranker_api_key(config)
    model = _reranker_model(config)
    if not base_url or not api_key or not model:
        if config.reranker_backend == "cloud":
            raise ValueError("reranker_backend=cloud 需要 reranker_base_url、reranker_api_key、reranker_model")
        return items[:top_k]

    documents = [str(item.get("text", "")) for item in items]
    ranked = _call_rerank_api(base_url, api_key, model, query, documents, top_k, _reranker_provider(config))
    return [items[index] for index, _score in ranked if 0 <= index < len(items)]


def read_npy_float32(path: Path) -> list[list[float]]:
    with path.open("rb") as f:
        magic = f.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"not a npy file: {path}")
        major, _minor = f.read(2)
        if major != 1:
            raise ValueError("only npy v1 is supported")
        header_len = struct.unpack("<H", f.read(2))[0]
        header = f.read(header_len).decode("latin1")
        shape_text = header.split("'shape':", 1)[1].split(")", 1)[0] + ")"
        rows, cols = _parse_shape(shape_text)
        data = f.read()
    values = struct.unpack("<" + "f" * (rows * cols), data)
    return [list(values[row * cols : (row + 1) * cols]) for row in range(rows)]


def _call_embedding_api(base_url: str, api_key: str, model: str, texts: list[str]) -> list[list[float]]:
    url = _embedding_url(base_url)
    payload = {"model": model, "input": texts}
    result = _post_json(url, api_key, payload)
    if isinstance(result.get("data"), list):
        vectors = [item.get("embedding") for item in result["data"]]
    else:
        vectors = result.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise ValueError("embedding API response missing vectors")
    return [[float(value) for value in vector] for vector in vectors]


def _call_rerank_api(
    base_url: str,
    api_key: str,
    model: str,
    query: str,
    documents: list[str],
    top_k: int,
    provider: str,
) -> list[tuple[int, float]]:
    url = _rerank_url(base_url, provider)
    payload = {"model": model, "query": query, "documents": documents, "top_n": top_k}
    result = _post_json(url, api_key, payload)
    rows = result.get("results") or result.get("data") or []
    ranked: list[tuple[int, float]] = []
    for row in rows:
        index = row.get("index")
        score = row.get("relevance_score", row.get("score", 0.0))
        if index is None and isinstance(row.get("document"), dict):
            index = row["document"].get("index")
        if index is not None:
            ranked.append((int(index), float(score)))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:top_k]


def _post_json(url: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{url} failed: HTTP {exc.code} {detail}") from exc


def _embedding_url(base_url: str) -> str:
    url = _normalize_base_url(base_url)
    if url.endswith("/embeddings"):
        return url
    return f"{url}/embeddings"


def _rerank_url(base_url: str, provider: str) -> str:
    url = _normalize_base_url(base_url)
    if url.endswith("/rerank"):
        return url
    if provider == "parallel_cloud":
        return f"{url}/p002/rerank"
    return f"{url}/rerank"


def _normalize_base_url(base_url: str) -> str:
    url = base_url.strip().rstrip("/")
    if url and "://" not in url:
        url = f"https://{url}"
    return url


def _embedding_backend(config: Config) -> str:
    backend = config.embedding_backend
    if backend == "auto":
        return "cloud" if _embedding_base_url(config) and _embedding_api_key(config) else "local"
    return backend


def _reranker_backend(config: Config) -> str:
    backend = config.reranker_backend
    if backend == "auto":
        return "cloud" if _reranker_base_url(config) and _reranker_api_key(config) else "local"
    return backend


def _embedding_base_url(config: Config) -> str:
    return config.embedding_base_url or os.getenv("EMBEDDING_BASE_URL", "")


def _embedding_api_key(config: Config) -> str:
    return config.embedding_api_key or os.getenv("EMBEDDING_API_KEY", "")


def _embedding_model(config: Config) -> str:
    return config.embedding_model or os.getenv("EMBEDDING_MODEL", "") or os.getenv("EMBEDDING_MODEL_NAME", "")


def _embedding_provider(config: Config) -> str:
    return config.embedding_provider or os.getenv("EMBEDDING_PROVIDER", "")


def _reranker_base_url(config: Config) -> str:
    return config.reranker_base_url or os.getenv("RERANKER_BASE_URL", "")


def _reranker_api_key(config: Config) -> str:
    return config.reranker_api_key or os.getenv("RERANKER_API_KEY", "")


def _reranker_model(config: Config) -> str:
    return config.reranker_model or os.getenv("RERANKER_MODEL", "") or os.getenv("RERANKER_MODEL_NAME", "")


def _reranker_provider(config: Config) -> str:
    return config.reranker_provider or os.getenv("RERANKER_PROVIDER", "")


def _write_embedding_meta(config: Config, scenes: list[SceneMemory], provider: str, dimensions: int) -> None:
    config.embedding_meta_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with config.embedding_meta_jsonl.open("w", encoding="utf-8") as f:
        for row_index, scene in enumerate(scenes):
            json.dump(
                {
                    "row": row_index,
                    "scene_id": scene.scene_id,
                    "knowledge_level": scene.knowledge_level,
                    "source_start": scene.source_start,
                    "source_end": scene.source_end,
                    "embedding_provider": provider,
                    "embedding_model": config.embedding_model,
                    "embedding_dimensions": dimensions,
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")


def _hashed_embedding(text: str, dimensions: int) -> list[float]:
    vector = [0.0] * dimensions
    for token in tokenize(text):
        index = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % dimensions
        vector[index] += 1.0
    return _normalize(vector)


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _write_npy_float32(vectors: list[list[float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = len(vectors)
    cols = len(vectors[0]) if vectors else 0
    header = f"{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {cols}), }}"
    padding = 16 - ((10 + len(header) + 1) % 16)
    header_bytes = (header + " " * padding + "\n").encode("latin1")
    with path.open("wb") as f:
        f.write(b"\x93NUMPY")
        f.write(bytes([1, 0]))
        f.write(struct.pack("<H", len(header_bytes)))
        f.write(header_bytes)
        for vector in vectors:
            f.write(struct.pack("<" + "f" * cols, *vector))


def _parse_shape(shape_text: str) -> tuple[int, int]:
    numbers = [int(part.strip()) for part in shape_text.strip("() ").split(",") if part.strip()]
    if len(numbers) != 2:
        raise ValueError(f"unsupported npy shape: {shape_text}")
    return numbers[0], numbers[1]
