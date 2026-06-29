from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class SceneSkeleton:
    scene_id: str
    chapter: str
    scene_index: int
    source_start: int
    source_end: int
    raw_text: str
    dialogues: list[dict[str, str]]
    dialogue_alignment: str
    alignment_score: float
    source_refs: list[dict[str, Any]]
    coverage: str = "full"


def build_scene_skeletons(
    novel_txt: Path,
    raw_dialogue_jsonl: Path | None,
    max_chars: int,
    overlap_chars: int,
) -> tuple[list[SceneSkeleton], dict[str, Any]]:
    text = novel_txt.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    scenes = _split_text_to_scenes(text, max_chars=max_chars, overlap_chars=overlap_chars)
    raw_dialogues = _load_raw_dialogues(raw_dialogue_jsonl) if raw_dialogue_jsonl and raw_dialogue_jsonl.exists() else []
    dialogue_matcher = DialogueMatcher(raw_dialogues)

    aligned_dialogue_count = 0
    missing_dialogue_count = 0
    for scene in scenes:
        matched = dialogue_matcher.match(scene.raw_text)
        scene.dialogues = [{"role": row["role"], "dialogue": row["dialogue"]} for row in matched]
        scene.source_refs = [
            {"raw_dialogue_id": row.get("id") or f"chunk_{row.get('chunk_id', '')}_{row.get('dialogue_index', '')}", "match_type": "text"}
            for row in matched
        ]
        if matched:
            scene.dialogue_alignment = "matched"
            scene.alignment_score = min(1.0, len(matched) / 5)
            aligned_dialogue_count += len(matched)
        else:
            scene.dialogue_alignment = "missing"
            scene.alignment_score = 0.0
            missing_dialogue_count += 1

    avg_chars = round(sum(len(scene.raw_text) for scene in scenes) / len(scenes), 2) if scenes else 0
    report = {
        "scene_count": len(scenes),
        "avg_scene_chars": avg_chars,
        "aligned_dialogue_count": aligned_dialogue_count,
        "missing_dialogue_count": missing_dialogue_count,
        "alignment_success_rate": round(
            (len(scenes) - missing_dialogue_count) / len(scenes),
            4,
        )
        if scenes
        else 0.0,
        "alignment_strategy": "keyed_exact_dialogue_match",
        "source_file": str(novel_txt),
        "scene_max_chars": max_chars,
        "scene_overlap_chars": overlap_chars,
    }
    return scenes, report


def write_scene_skeletons(scenes: list[SceneSkeleton], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for scene in scenes:
            json.dump(asdict(scene), f, ensure_ascii=False)
            f.write("\n")


def read_scene_skeletons(path: Path) -> list[SceneSkeleton]:
    scenes: list[SceneSkeleton] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                scenes.append(SceneSkeleton(**json.loads(line)))
    return scenes


def write_scene_build_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _split_text_to_scenes(text: str, max_chars: int, overlap_chars: int) -> list[SceneSkeleton]:
    paragraphs = _paragraph_spans(text)
    scenes: list[SceneSkeleton] = []
    current: list[tuple[int, int, str]] = []
    current_len = 0
    chapter = ""

    for start, end, para in paragraphs:
        maybe_chapter = _chapter_title(para)
        if maybe_chapter:
            chapter = maybe_chapter
        if current and current_len + len(para) > max_chars:
            scenes.append(_make_scene(text, current, len(scenes), chapter))
            current = _overlap_tail(current, overlap_chars)
            current_len = sum(len(item[2]) for item in current)
        current.append((start, end, para))
        current_len += len(para)

    if current:
        scenes.append(_make_scene(text, current, len(scenes), chapter))
    return scenes


def _paragraph_spans(text: str) -> list[tuple[int, int, str]]:
    matches = list(re.finditer(r"\S(?:.*?)(?=\n\s*\n|\Z)", text, flags=re.S))
    spans: list[tuple[int, int, str]] = []
    for match in matches:
        para = match.group(0).strip()
        if para:
            spans.append((match.start(), match.end(), para))
    if spans:
        return spans
    stripped = text.strip()
    return [(0, len(text), stripped)] if stripped else []


def _chapter_title(paragraph: str) -> str | None:
    first_line = paragraph.splitlines()[0].strip()
    if re.match(r"^第[一二三四五六七八九十百千万零〇\d]+[章节回卷部].{0,40}$", first_line):
        return first_line
    return None


def _make_scene(text: str, paragraphs: list[tuple[int, int, str]], scene_index: int, chapter: str) -> SceneSkeleton:
    source_start = paragraphs[0][0]
    source_end = paragraphs[-1][1]
    return SceneSkeleton(
        scene_id=f"scene_{scene_index + 1:06d}",
        chapter=chapter,
        scene_index=scene_index + 1,
        source_start=source_start,
        source_end=source_end,
        raw_text=text[source_start:source_end].strip(),
        dialogues=[],
        dialogue_alignment="missing",
        alignment_score=0.0,
        source_refs=[],
    )


def _overlap_tail(paragraphs: list[tuple[int, int, str]], overlap_chars: int) -> list[tuple[int, int, str]]:
    if overlap_chars <= 0:
        return []
    total = 0
    tail: list[tuple[int, int, str]] = []
    for item in reversed(paragraphs):
        tail.append(item)
        total += len(item[2])
        if total >= overlap_chars:
            break
    return list(reversed(tail))


def _load_raw_dialogues(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                role = str(row.get("role", "")).strip()
                dialogue = str(row.get("dialogue", "")).strip()
                if role and dialogue:
                    rows.append(row)
    return rows


class DialogueMatcher:
    def __init__(self, raw_dialogues: list[dict[str, Any]], key_size: int = 6):
        self.key_size = key_size
        self.by_key: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for row in raw_dialogues:
            dialogue = _normalize_match_text(str(row.get("dialogue", "")))
            if len(dialogue) < key_size:
                continue
            key = dialogue[:key_size]
            self.by_key.setdefault(key, []).append((dialogue, row))

    def match(self, raw_text: str, limit: int = 20) -> list[dict[str, Any]]:
        text = _normalize_match_text(raw_text)
        if not text or not self.by_key:
            return []
        matched: list[dict[str, Any]] = []
        seen_rows: set[int] = set()
        for key in _candidate_keys(text, self.key_size):
            for dialogue, row in self.by_key.get(key, []):
                row_id = id(row)
                if row_id in seen_rows:
                    continue
                if dialogue in text:
                    matched.append(row)
                    seen_rows.add(row_id)
                    if len(matched) >= limit:
                        return matched
        return matched


def _normalize_match_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _candidate_keys(text: str, key_size: int) -> set[str]:
    if len(text) < key_size:
        return set()
    return {text[index : index + key_size] for index in range(0, len(text) - key_size + 1)}
