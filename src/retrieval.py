from __future__ import annotations

import json
import math
import re
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
        idf: dict[str, float],
        avgdl: float,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.scenes = scenes
        self.documents = documents
        self.idf = idf
        self.avgdl = avgdl or 1.0
        self.k1 = k1
        self.b = b

    @classmethod
    def from_scenes(cls, scenes: list[SceneMemory]) -> "BM25MemoryIndex":
        documents = [tokenize(_scene_document(scene)) for scene in scenes]
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
        return cls(scenes=scenes, documents=documents, idf=idf, avgdl=avgdl)

    def search(self, query: str, top_k: int = 3, exclude_narrator_only: bool = True) -> list[SearchResult]:
        query_tokens = tokenize(query)
        results: list[SearchResult] = []
        for scene, doc in zip(self.scenes, self.documents):
            if exclude_narrator_only and not scene.target_role_knows:
                continue
            score = self._score(query_tokens, doc)
            if score > 0:
                results.append(SearchResult(scene=scene, score=score))
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]

    def _score(self, query_tokens: list[str], doc: list[str]) -> float:
        if not doc:
            return 0.0
        frequencies: dict[str, int] = {}
        for token in doc:
            frequencies[token] = frequencies.get(token, 0) + 1
        score = 0.0
        doc_len = len(doc)
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
            "idf": self.idf,
            "avgdl": self.avgdl,
            "k1": self.k1,
            "b": self.b,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BM25MemoryIndex":
        return cls(
            scenes=[SceneMemory(**scene) for scene in data["scenes"]],
            documents=data["documents"],
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
