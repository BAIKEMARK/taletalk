#!/usr/bin/env python3
"""Convert per-chunk dialogue jsonl into multi-turn ShareGPT SFT data.

Input format (one JSON object per line, output of extract/dialogue_extractor.py):
    {"chunk_id": 3, "dialogue_index": 0, "role": "角色名", "dialogue": "..."}

Output format (LLaMA Factory ShareGPT):
    [
        {
            "id": "<run_name>-<chunk_id>",
            "system": "你正在扮演...",
            "conversations": [
                {"from": "human", "value": "..."},
                {"from": "gpt",   "value": "..."},
                ...
            ]
        },
        ...
    ]

Slicing rule:
    - Group dialogues by chunk_id.
    - Drop chunks that contain zero lines from TARGET_ROLE.
    - Within a chunk, walk dialogues in dialogue_index order. Consecutive
      lines from the same "side" (target role vs others) are merged into one
      turn. Target role becomes "gpt", everyone else becomes "human".
    - Drop chunks that start with the target role (no leading human turn).
    - Drop chunks where the resulting conversation has < 2 turns or no "gpt".
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def load_jsonl(path: Path) -> list[dict]:
    lines = []
    with path.open(encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            lines.append(json.loads(raw))
    return lines


def group_by_chunk(lines: Iterable[dict]) -> dict[int, list[dict]]:
    buckets: dict[int, list[dict]] = defaultdict(list)
    for item in lines:
        buckets[int(item["chunk_id"])].append(item)
    for cid in buckets:
        buckets[cid].sort(key=lambda x: int(x.get("dialogue_index", 0)))
    return buckets


def build_conversation(
    chunk_lines: list[dict],
    target_role: str,
) -> list[dict] | None:
    """Return ShareGPT conversations list, or None to drop this chunk."""
    if not any(line["role"] == target_role for line in chunk_lines):
        return None
    # Skip chunks that lead with the target role (no human context).
    if chunk_lines[0]["role"] == target_role:
        return None

    conversations: list[dict] = []
    buf_side: str | None = None  # "human" or "gpt"
    buf_parts: list[str] = []

    def flush() -> None:
        if buf_side and buf_parts:
            text = "\n".join(buf_parts).strip()
            if text:
                conversations.append({"from": buf_side, "value": text})

    for line in chunk_lines:
        side = "gpt" if line["role"] == target_role else "human"
        # Format other roles as "角色名：内容" so the model sees who is speaking.
        if side == "human":
            piece = f"{line['role']}：{line['dialogue'].strip()}"
        else:
            piece = line["dialogue"].strip()
        if not piece:
            continue
        if buf_side == side:
            buf_parts.append(piece)
        else:
            flush()
            buf_side = side
            buf_parts = [piece]
    flush()

    # Need at least one human + one gpt turn.
    has_gpt = any(c["from"] == "gpt" for c in conversations)
    if len(conversations) < 2 or not has_gpt:
        return None
    # Must end with a gpt turn (last training target).
    if conversations[-1]["from"] != "gpt":
        # Trim trailing human turns so the last is gpt.
        while conversations and conversations[-1]["from"] != "gpt":
            conversations.pop()
        if len(conversations) < 2:
            return None
    return conversations


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="dialogues jsonl from extract module")
    ap.add_argument("--target-role", required=True, help="character name, e.g. 齐夏")
    ap.add_argument("--run-name", required=True, help="dataset prefix, e.g. shiri_qixia")
    ap.add_argument("--novel-title", default="", help="novel title used in system prompt")
    ap.add_argument("--out-dir", required=True, help="output directory")
    ap.add_argument("--valid-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-conversations", type=int, default=0, help="0 = no cap")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = load_jsonl(in_path)
    print(f"loaded {len(lines)} dialogue lines from {in_path}")
    buckets = group_by_chunk(lines)
    print(f"grouped into {len(buckets)} chunks")

    novel_phrase = f"《{args.novel_title}》中的" if args.novel_title else ""
    system_prompt = (
        f"你正在扮演{novel_phrase}{args.target_role}。"
        f"严格保持{args.target_role}的语气、性格、说话习惯和价值观，"
        f"根据对话上下文自然回应，不要跳出角色，不要续写其他角色的发言。"
    )

    samples: list[dict] = []
    dropped_no_target = 0
    dropped_lead = 0
    dropped_short = 0
    for cid in sorted(buckets):
        chunk_lines = buckets[cid]
        if not any(l["role"] == args.target_role for l in chunk_lines):
            dropped_no_target += 1
            continue
        if chunk_lines[0]["role"] == args.target_role:
            dropped_lead += 1
            continue
        conv = build_conversation(chunk_lines, args.target_role)
        if conv is None:
            dropped_short += 1
            continue
        samples.append({
            "id": f"{args.run_name}-{cid:05d}",
            "system": system_prompt,
            "conversations": conv,
        })

    print(f"kept {len(samples)} multi-turn samples")
    print(f"dropped: no_target={dropped_no_target} leads_with_target={dropped_lead} too_short={dropped_short}")

    if args.max_conversations and len(samples) > args.max_conversations:
        random.Random(args.seed).shuffle(samples)
        samples = samples[: args.max_conversations]
        print(f"capped to {len(samples)}")

    rng = random.Random(args.seed)
    rng.shuffle(samples)
    n_valid = max(1, int(len(samples) * args.valid_ratio))
    valid = samples[:n_valid]
    train = samples[n_valid:]
    print(f"split: train={len(train)} valid={len(valid)}")

    train_path = out_dir / f"{args.run_name}_chat_train.json"
    valid_path = out_dir / f"{args.run_name}_chat_valid.json"
    train_path.write_text(json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8")
    valid_path.write_text(json.dumps(valid, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {train_path}")
    print(f"wrote {valid_path}")

    # Stats
    turn_counts = [len(s["conversations"]) for s in samples]
    if turn_counts:
        avg_turns = sum(turn_counts) / len(turn_counts)
        print(f"avg turns/sample: {avg_turns:.2f}  max: {max(turn_counts)}  min: {min(turn_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
