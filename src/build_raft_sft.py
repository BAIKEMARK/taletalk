from __future__ import annotations

from collections import defaultdict
from typing import Any

from .memory import CharacterProfile, SceneMemory
from .prompting import build_roleplay_system_prompt


def build_raft_sharegpt(
    raw_rows: list[dict[str, Any]],
    scenes: list[SceneMemory],
    profile: CharacterProfile,
    target_roles: set[str],
    max_memory_chars: int = 1800,
    max_one_scene_chars: int = 600,
) -> list[dict[str, Any]]:
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        if "chunk_id" in row:
            buckets[int(row["chunk_id"])].append(row)
    scenes_by_chunk = {scene.chunk_id: scene for scene in scenes}

    samples: list[dict[str, Any]] = []
    for chunk_id in sorted(buckets):
        chunk_rows = sorted(buckets[chunk_id], key=lambda row: int(row.get("dialogue_index", 0)))
        scene = scenes_by_chunk.get(chunk_id)
        if scene is None:
            continue
        conversations = _build_chunk_conversation(chunk_rows, target_roles)
        if conversations is None:
            continue
        system_prompt = build_roleplay_system_prompt(
            profile,
            [scene],
            max_memory_chars=max_memory_chars,
            max_one_scene_chars=max_one_scene_chars,
        )
        samples.append(
            {
                "id": f"{profile.role}_{chunk_id:06d}_raft",
                "system": system_prompt,
                "conversations": conversations,
                "metadata": {
                    "oracle_scene_ids": [scene.scene_id],
                    "distractor_scene_ids": [],
                    "sample_type": "grounded_dialogue",
                },
            }
        )
    return samples


def _build_chunk_conversation(chunk_rows: list[dict[str, Any]], target_roles: set[str]) -> list[dict[str, str]] | None:
    if not any(str(row.get("role", "")).strip() in target_roles for row in chunk_rows):
        return None
    if str(chunk_rows[0].get("role", "")).strip() in target_roles:
        return None

    conversations: list[dict[str, str]] = []
    side: str | None = None
    parts: list[str] = []

    def flush() -> None:
        nonlocal side, parts
        if side and parts:
            text = "\n".join(parts).strip()
            if text:
                conversations.append({"from": side, "value": text})
        side = None
        parts = []

    for row in chunk_rows:
        role = str(row.get("role", "")).strip()
        dialogue = str(row.get("dialogue", "")).strip()
        if not role or not dialogue:
            continue
        next_side = "gpt" if role in target_roles else "human"
        piece = dialogue if next_side == "gpt" else f"{role}：{dialogue}"
        if side == next_side:
            parts.append(piece)
        else:
            flush()
            side = next_side
            parts = [piece]
    flush()

    while conversations and conversations[-1]["from"] != "gpt":
        conversations.pop()
    if len(conversations) < 2 or not any(item["from"] == "gpt" for item in conversations):
        return None
    return conversations
