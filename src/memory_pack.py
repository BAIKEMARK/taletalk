from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .memory import SceneMemory
from .retrieval import BM25MemoryIndex


def scene_memory_text(scene: SceneMemory, max_chars: int = 600) -> str:
    parts = [scene.summary]
    if scene.events:
        parts.append("事件：" + "；".join(scene.events[:3]))
    if scene.relations:
        relation_texts = [str(item.get("text", item)) for item in scene.relations[:2]]
        parts.append("关系：" + "；".join(relation_texts))
    if scene.quotes:
        parts.append("原话：" + " / ".join(scene.quotes[:3]))
    body = "\n".join(part for part in parts if part).strip() or scene.raw_text or scene.text
    return body[:max_chars].rstrip() + ("…" if len(body) > max_chars else "")


def build_memory_packs(
    candidates: list[dict[str, Any]],
    scenes: list[SceneMemory],
    index: BM25MemoryIndex,
    *,
    max_one_scene_chars: int,
    include_distractors: bool,
) -> list[dict[str, Any]]:
    by_id = {scene.scene_id: scene for scene in scenes}
    packs: list[dict[str, Any]] = []
    rng = random.Random(42)

    for candidate in candidates:
        question = str(candidate.get("question", ""))
        refs = [ref for ref in candidate.get("source_scene_ids", []) if ref in by_id]
        items: list[dict[str, Any]] = []
        for ref in refs[:2]:
            items.append(_pack_item(by_id[ref], "gold_evidence", max_one_scene_chars))

        if refs:
            for result in index.search(question, top_k=5, exclude_narrator_only=True):
                if result.scene.scene_id not in refs:
                    items.append(_pack_item(result.scene, "supporting_context", max_one_scene_chars))
                    break
        if include_distractors and refs:
            candidates_for_distractor = [scene for scene in scenes if scene.scene_id not in refs and scene.target_role_knows]
            if candidates_for_distractor:
                items.append(_pack_item(rng.choice(candidates_for_distractor), "hard_distractor", max_one_scene_chars))

        packs.append(
            {
                "question_id": candidate["id"],
                "items": items,
                "metadata": {
                    "sample_type": candidate.get("sample_type"),
                    "source_scene_ids": refs,
                    "answer_policy": candidate.get("answer_policy"),
                },
            }
        )
    return packs


def write_memory_packs(packs: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for pack in packs:
            json.dump(pack, f, ensure_ascii=False)
            f.write("\n")


def read_memory_packs(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def render_memory_pack(items: list[dict[str, Any]], max_chars: int = 1800) -> str:
    if not items:
        return "【记忆片段 1｜记忆不足】\n没有可靠记忆片段能直接回答这个问题。"

    blocks: list[str] = []
    total = 0
    for index, item in enumerate(items, start=1):
        label = _knowledge_label(str(item.get("knowledge_level", "")))
        text = str(item.get("text", "")).strip()
        block = f"【记忆片段 {index}｜{label}】\n{text}"
        if blocks and total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)


def _pack_item(scene: SceneMemory, hidden_role: str, max_chars: int) -> dict[str, Any]:
    return {
        "scene_id": scene.scene_id,
        "hidden_role": hidden_role,
        "knowledge_level": scene.knowledge_level,
        "text": scene_memory_text(scene, max_chars=max_chars),
    }


def _knowledge_label(level: str) -> str:
    if level == "first_hand":
        return "齐夏亲历"
    if level == "heard_or_inferred":
        return "他人告知或可推断"
    if level == "narrator_only":
        return "读者视角，不可直接当作齐夏记忆"
    return "证据不确定"
