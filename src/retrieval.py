from __future__ import annotations

import json
import math
import re
import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .memory import SceneMemory


def tokenize(text: str) -> list[str]:
    text = text.lower()
    ascii_words = re.findall(r"[a-z0-9_]+", text)
    cjk = re.findall(r"[\u4e00-\u9fff]", text)
    cjk_bigrams = ["".join(pair) for pair in zip(cjk, cjk[1:])]
    return ascii_words + cjk + cjk_bigrams


@dataclass
class SearchResult:
    scene: SceneMemory
    score: float


class BM25MemoryIndex:
    def __init__(
        self,
        scenes: list[SceneMemory],
        documents: list[list[str]],
        term_frequencies: list[dict[str, int]],
        document_lengths: list[int],
        idf: dict[str, float],
        avgdl: float,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.scenes = scenes
        self.documents = documents
        self.term_frequencies = term_frequencies
        self.document_lengths = document_lengths
        self.idf = idf
        self.avgdl = avgdl or 1.0
        self.k1 = k1
        self.b = b

    @classmethod
    def from_scenes(cls, scenes: list[SceneMemory]) -> "BM25MemoryIndex":
        documents = [tokenize(_scene_document(scene)) for scene in scenes]
        term_frequencies = [_count_terms(doc) for doc in documents]
        document_lengths = [len(doc) for doc in documents]
        doc_count = len(documents)
        df: dict[str, int] = {}
        for doc in documents:
            for token in set(doc):
                df[token] = df.get(token, 0) + 1
        idf = {
            token: math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))
            for token, freq in df.items()
        }
        avgdl = sum(len(doc) for doc in documents) / doc_count if doc_count else 1.0
        return cls(
            scenes=scenes,
            documents=documents,
            term_frequencies=term_frequencies,
            document_lengths=document_lengths,
            idf=idf,
            avgdl=avgdl,
        )

    def search(self, query: str, top_k: int = 3, exclude_narrator_only: bool = True) -> list[SearchResult]:
        query_tokens = tokenize(query)
        results: list[SearchResult] = []
        for scene, frequencies, doc_len in zip(self.scenes, self.term_frequencies, self.document_lengths):
            if exclude_narrator_only and not scene.target_role_knows:
                continue
            score = self._score(query_tokens, frequencies, doc_len)
            if score > 0:
                results.append(SearchResult(scene=scene, score=score))
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]

    def _score(self, query_tokens: list[str], frequencies: dict[str, int], doc_len: int) -> float:
        if not frequencies:
            return 0.0
        score = 0.0
        for token in query_tokens:
            freq = frequencies.get(token, 0)
            if freq == 0:
                continue
            idf = self.idf.get(token, 0.0)
            denom = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * freq * (self.k1 + 1) / denom
        return score

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenes": [scene.__dict__ for scene in self.scenes],
            "documents": self.documents,
            "term_frequencies": self.term_frequencies,
            "document_lengths": self.document_lengths,
            "idf": self.idf,
            "avgdl": self.avgdl,
            "k1": self.k1,
            "b": self.b,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BM25MemoryIndex":
        documents = data["documents"]
        term_frequencies = data.get("term_frequencies") or [_count_terms(doc) for doc in documents]
        document_lengths = data.get("document_lengths") or [len(doc) for doc in documents]
        return cls(
            scenes=[SceneMemory(**scene) for scene in data["scenes"]],
            documents=documents,
            term_frequencies=term_frequencies,
            document_lengths=document_lengths,
            idf={str(k): float(v) for k, v in data["idf"].items()},
            avgdl=float(data["avgdl"]),
            k1=float(data.get("k1", 1.5)),
            b=float(data.get("b", 0.75)),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BM25MemoryIndex":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _scene_document(scene: SceneMemory) -> str:
    parts = [
        scene.summary,
        scene.text,
        " ".join(scene.characters),
        " ".join(scene.events),
        " ".join(scene.quotes),
    ]
    return "\n".join(part for part in parts if part)


def _count_terms(doc: list[str]) -> dict[str, int]:
    frequencies: dict[str, int] = {}
    for token in doc:
        frequencies[token] = frequencies.get(token, 0) + 1
    return frequencies


def write_hashed_embedding_artifacts(
    scenes: list[SceneMemory],
    embedding_path: Path,
    meta_path: Path,
    dimensions: int = 64,
) -> None:
    vectors = [_hashed_embedding(_scene_document(scene), dimensions) for scene in scenes]
    _write_npy_float32(vectors, embedding_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        for row_index, scene in enumerate(scenes):
            json.dump(
                {
                    "row": row_index,
                    "scene_id": scene.scene_id,
                    "knowledge_level": scene.knowledge_level,
                    "source_start": scene.source_start,
                    "source_end": scene.source_end,
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
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


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
