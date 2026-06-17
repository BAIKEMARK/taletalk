#!/usr/bin/env python3
"""Build role-play SFT data from a Chinese novel.

This follows the same broad shape as huanhuan-chat/generation_dataset:
extract role/dialogue records first, then create training pairs where the
previous dialogue/context is the user message and the target character's line is
the assistant response.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


SPEECH_VERBS = (
    "说道",
    "问道",
    "答道",
    "回答",
    "开口道",
    "低声道",
    "冷声道",
    "沉声道",
    "喃喃",
    "喊道",
    "叫道",
    "笑道",
    "叹道",
    "说",
    "问",
    "答",
    "开口",
    "低声",
    "冷声",
    "沉声",
    "喊",
    "叫",
    "笑",
    "叹",
)


DEFAULT_EXCLUDE_NAMES = (
    "山羊头",
    "乔家劲",
    "李警官",
    "甜甜",
    "赵医生",
    "赵海博",
    "林檎",
    "楚天秋",
    "陈俊南",
    "韩一墨",
    "章晨泽",
    "云瑶",
    "余念安",
    "许流年",
    "地虎",
    "青龙",
    "白虎",
    "玄武",
    "朱雀",
)


@dataclass
class Dialogue:
    chapter: str
    role: str
    dialogue: str
    context_before: str
    context_after: str
    confidence: float
    method: str
    offset: int


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def current_chapter(text: str, pos: int, chapter_matches: list[re.Match[str]]) -> str:
    chapter = ""
    for match in chapter_matches:
        if match.start() <= pos:
            chapter = match.group(0).strip()
        else:
            break
    return chapter


def clean_quote(quote: str) -> str:
    quote = quote.strip()
    quote = quote.strip("“”\"'")
    quote = re.sub(r"\s+", " ", quote)
    return quote.strip()


def speech_verb_pattern() -> str:
    # Single-char verbs such as "说" must not match words like "说谎者".
    single = "说问答喊叫笑叹"
    strict_long = (
        "说道",
        "问道",
        "答道",
        "回答",
        "开口道",
        "低声道",
        "冷声道",
        "沉声道",
        "喊道",
        "叫道",
        "笑道",
        "叹道",
    )
    long_part = "|".join(re.escape(v) for v in strict_long)
    soft_part = r"(?:开口|低声|冷声|沉声)(?=[道说问：:，,。\s]|$)"
    return rf"(?:{long_part}|喃喃(?:自语)?(?:地)?说|[{single}](?=[道着了：:，,。\s]|$)|{soft_part})"


def has_speech_verb(s: str) -> bool:
    return re.search(speech_verb_pattern(), s) is not None


def has_excluded_name(segment: str, exclude_names: list[str]) -> bool:
    return any(name and name in segment for name in exclude_names)


def has_negative_speech_marker(segment: str) -> bool:
    markers = (
        "没有回答",
        "没有直接回答",
        "没有开口",
        "并没有开口",
        "并未开口",
        "没有说话",
        "并没有说话",
        "没有理会",
        "没有回应",
        "没有答话",
        "没有言语",
        "没有阻拦",
    )
    return any(m in segment for m in markers)


def detect_speaker(
    before: str,
    after: str,
    aliases: list[str],
    pronouns: list[str],
    exclude_names: list[str],
) -> tuple[str, float, str]:
    names = "|".join(re.escape(a) for a in aliases)
    pronoun_re = "|".join(re.escape(p) for p in pronouns)
    verbs = speech_verb_pattern()

    before_tail = before[-120:]
    last_alias_match = None
    for m in re.finditer(names, before_tail):
        last_alias_match = m

    def alias_before_is_clean() -> bool:
        if not last_alias_match:
            return False
        segment = before_tail[last_alias_match.end() :]
        # If another prominent character appears between the target name and the
        # speech verb, the quote probably belongs to that character.
        return not has_excluded_name(segment, exclude_names)

    # 齐夏说道：“……”
    if alias_before_is_clean() and re.search(rf"({names})[^。！？“”\n]{{0,50}}({verbs})[：:，,、\s]*$", before_tail):
        return aliases[0], 0.98, "name_before_quote"

    after_head = after[:90]

    # “……”齐夏说道
    if not has_negative_speech_marker(after_head) and re.match(rf"^[，,。\s]*({names})[^。！？“”\n]{{0,50}}({verbs})", after):
        return aliases[0], 0.98, "name_after_quote"

    # 齐夏看向众人：“……”
    if alias_before_is_clean() and re.search(rf"({names})[^。！？“”\n]{{0,50}}[：:]\s*$", before_tail):
        return aliases[0], 0.94, "name_colon_before_quote"

    # “……”齐夏看向众人
    if (
        not has_negative_speech_marker(after_head)
        and re.match(rf"^[，,。\s]*({names})[^。！？“”\n]{{0,35}}", after)
        and has_speech_verb(after[:80])
    ):
        return aliases[0], 0.86, "name_near_after_quote"

    # Pronoun fallback only when the target name is very close before the quote.
    # Example: 齐夏摇了摇头，他说道：“……”
    if pronouns and alias_before_is_clean() and re.search(rf"({names})[^。！？“”\n]{{0,80}}({pronoun_re})[^。！？“”\n]{{0,35}}({verbs})[：:，,、\s]*$", before_tail):
        return aliases[0], 0.78, "near_name_pronoun_before_quote"

    if pronouns and not has_negative_speech_marker(after_head) and re.match(rf"^[，,。\s]*({pronoun_re})[^。！？“”\n]{{0,35}}({verbs})", after):
        if re.search(rf"({names})[^。！？“”\n]{{0,120}}$", before):
            return aliases[0], 0.70, "near_name_pronoun_after_quote"

    return "", 0.0, "unknown"


def iter_dialogues(text: str, aliases: list[str], pronouns: list[str], exclude_names: list[str]) -> Iterable[Dialogue]:
    chapter_matches = list(re.finditer(r"(?m)^第[0-9零〇一二三四五六七八九十百千万两]+[章节卷][^\n]*", text))
    quote_pattern = re.compile(r"[“\"]([^“”\"]{1,260})[”\"]")

    for match in quote_pattern.finditer(text):
        quote = clean_quote(match.group(1))
        if not quote:
            continue

        before = text[max(0, match.start() - 220) : match.start()]
        after = text[match.end() : min(len(text), match.end() + 220)]
        role, confidence, method = detect_speaker(before, after, aliases, pronouns, exclude_names)
        if not role:
            continue

        yield Dialogue(
            chapter=current_chapter(text, match.start(), chapter_matches),
            role=role,
            dialogue=quote,
            context_before=before.strip()[-180:],
            context_after=after.strip()[:180],
            confidence=confidence,
            method=method,
            offset=match.start(),
        )


def filter_dialogue(text: str, min_len: int, max_len: int) -> bool:
    if len(text) < min_len or len(text) > max_len:
        return False
    if text in {"嗯", "好", "是", "不", "走", "等等", "什么"}:
        return False
    if re.fullmatch(r"[啊嗯哦呃哈…？！。，、\s]+", text):
        return False
    return True


def make_user_message(prev: Dialogue | None, cur: Dialogue, character: str) -> str:
    if prev and prev.dialogue:
        return (
            f"上一句对话：{prev.dialogue}\n"
            f"请以《十日终焉》中{character}的口吻自然回应。"
        )

    context = cur.context_before[-140:]
    return (
        f"场景：{context}\n"
        f"请以《十日终焉》中{character}的口吻自然回应。"
    )


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--novel", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--character", default="齐夏")
    parser.add_argument("--aliases", default="齐夏")
    parser.add_argument("--pronouns", default="他")
    parser.add_argument("--exclude-names", default=",".join(DEFAULT_EXCLUDE_NAMES))
    parser.add_argument("--min-len", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=180)
    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--write-weclone-dir", type=Path, default=None)
    args = parser.parse_args()

    aliases = [x.strip() for x in args.aliases.split(",") if x.strip()]
    pronouns = [x.strip() for x in args.pronouns.split(",") if x.strip()]
    exclude_names = [x.strip() for x in args.exclude_names.split(",") if x.strip()]
    exclude_names = [x for x in exclude_names if x not in aliases]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    text = normalize_text(read_text(args.novel))

    raw_dialogues = list(iter_dialogues(text, aliases, pronouns, exclude_names))
    dialogues = [
        d
        for d in raw_dialogues
        if d.confidence >= args.min_confidence and filter_dialogue(d.dialogue, args.min_len, args.max_len)
    ]
    dialogues.sort(key=lambda d: d.offset)

    raw_jsonl = args.out_dir / f"{args.character}_dialogue_candidates.jsonl"
    write_jsonl(raw_jsonl, (asdict(d) for d in dialogues))

    preview_csv = args.out_dir / f"{args.character}_dialogue_preview.csv"
    with preview_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "chapter",
                "role",
                "dialogue",
                "confidence",
                "method",
                "context_before",
                "context_after",
            ],
        )
        writer.writeheader()
        for d in dialogues:
            row = asdict(d)
            row.pop("offset", None)
            writer.writerow(row)

    system_prompt = args.system_prompt or (
        f"你正在扮演《十日终焉》中的{args.character}。"
        "你冷静、理性、善于观察和推演，不轻易暴露情绪。"
        "回答要简洁自然，不要自称 AI，不要解释你在扮演角色。"
    )

    sft_rows = []
    prev: Dialogue | None = None
    for d in dialogues:
        user = make_user_message(prev, d, args.character)
        sft_rows.append(
            {
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": d.dialogue},
                ],
            }
        )
        prev = d

    sft_path = args.out_dir / "sft-my.json"
    sft_path.write_text(json.dumps(sft_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    dataset_info = {
        "chat-sft": {
            "file_name": "./sft-my.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "system": "system"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }
    }
    (args.out_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.write_weclone_dir:
        target = args.write_weclone_dir / "dataset" / "res_csv" / "sft"
        target.mkdir(parents=True, exist_ok=True)
        (target / "sft-my.json").write_text(sft_path.read_text(encoding="utf-8"), encoding="utf-8")
        (target / "dataset_info.json").write_text(
            (args.out_dir / "dataset_info.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    stats = {
        "raw_candidates": len(raw_dialogues),
        "filtered_dialogues": len(dialogues),
        "sft_examples": len(sft_rows),
        "raw_jsonl": str(raw_jsonl),
        "preview_csv": str(preview_csv),
        "sft_path": str(sft_path),
    }
    (args.out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
