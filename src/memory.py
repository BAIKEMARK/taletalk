from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class CharacterProfile:
    role: str
    aliases: list[str]
    novel_title: str
    identity: str
    core_goals: list[str]
    personality: list[str]
    speech_style: list[str]
    relationships: list[dict[str, Any]]
    knowledge_boundary: str
    answer_rules: list[str]


@dataclass
class SceneMemory:
    scene_id: str
    chunk_id: int
    chapter: str
    text: str
    summary: str
    characters: list[str]
    target_role_present: bool
    target_role_knows: bool
    events: list[str]
    relations: list[dict[str, Any]]
    quotes: list[str]
    source: dict[str, Any]


def build_default_profile(role: str, aliases: list[str], novel_title: str) -> CharacterProfile:
    return CharacterProfile(
        role=role,
        aliases=aliases,
        novel_title=novel_title,
        identity=f"《{novel_title}》中的{role}。",
        core_goals=[],
        personality=[],
        speech_style=[],
        relationships=[],
        knowledge_boundary="Only answer from first-hand experience, heard information, or reasonable inference from retrieved memories.",
        answer_rules=[
            "Use memory for facts.",
            "Use the character voice for expression.",
            "If memory is insufficient, do not invent specific novel facts.",
            "Stay in first person unless the role naturally would not.",
        ],
    )


def read_dialogue_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _contains_alias(text: str, aliases: Iterable[str]) -> bool:
    return any(alias and alias in text for alias in aliases)


def _scene_text(rows: list[dict[str, Any]]) -> str:
    for row in rows:
        chunk_text = str(row.get("chunk_text") or "").strip()
        if chunk_text:
            return chunk_text
    return "\n".join(f"{row.get('role', '')}: {row.get('dialogue', '')}".strip() for row in rows).strip()


def build_scene_memories(
    raw_jsonl: Path,
    canonical_role: str,
    aliases: list[str],
    novel_title: str,
) -> list[SceneMemory]:
    rows = read_dialogue_jsonl(raw_jsonl)
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if "chunk_id" not in row:
            continue
        buckets[int(row["chunk_id"])].append(row)

    scenes: list[SceneMemory] = []
    for chunk_id in sorted(buckets):
        chunk_rows = sorted(buckets[chunk_id], key=lambda row: int(row.get("dialogue_index", 0)))
        text = _scene_text(chunk_rows)
        roles = [str(row.get("role", "")).strip() for row in chunk_rows if str(row.get("role", "")).strip()]
        target_role_present = any(role in aliases for role in roles) or _contains_alias(text, aliases)
        target_quotes = [
            str(row.get("dialogue", "")).strip()
            for row in chunk_rows
            if str(row.get("role", "")).strip() in aliases and str(row.get("dialogue", "")).strip()
        ]
        summary = text[:120].replace("\n", " ").strip()
        scenes.append(
            SceneMemory(
                scene_id=f"chunk_{chunk_id:06d}",
                chunk_id=chunk_id,
                chapter="",
                text=text,
                summary=summary,
                characters=sorted(set(roles)),
                target_role_present=target_role_present,
                target_role_knows=target_role_present,
                events=[],
                relations=[],
                quotes=target_quotes[:5],
                source={"novel_title": novel_title, "chunk_id": chunk_id},
            )
        )
    return scenes


def write_scene_memories(scenes: list[SceneMemory], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for scene in scenes:
            json.dump(asdict(scene), f, ensure_ascii=False)
            f.write("\n")


def read_scene_memories(path: Path) -> list[SceneMemory]:
    scenes: list[SceneMemory] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                scenes.append(SceneMemory(**json.loads(line)))
    return scenes


def write_profile(profile: CharacterProfile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(profile), ensure_ascii=False, indent=2), encoding="utf-8")


def read_profile(path: Path) -> CharacterProfile:
    return CharacterProfile(**json.loads(path.read_text(encoding="utf-8")))
