from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .config import Config
from .memory_pack import read_memory_packs
from .utils import check_step_done, init_logger, mark_step_done


def run_eval(config: Config) -> None:
    step_name = "eval"
    logger = init_logger(step_name, config.logs_dir)
    if check_step_done(step_name, config.status_dir):
        logger.info("评测报告已生成，跳过")
        return

    train_rows = _load_json(config.train_json)
    valid_rows = _load_json(config.valid_json) if config.valid_json.exists() else []
    candidates = _load_jsonl(config.raft_candidates_raw_jsonl)
    packs = read_memory_packs(config.raft_memory_packs_jsonl) if config.raft_memory_packs_jsonl.exists() else []

    eval_questions = _build_eval_questions(candidates, config.eval_question_count)
    _write_jsonl(eval_questions, config.eval_questions_jsonl)

    pack_by_id = {pack["question_id"]: pack for pack in packs}
    missing_gold = []
    for candidate in candidates:
        refs = set(candidate.get("source_scene_ids", []))
        if not refs:
            continue
        pack = pack_by_id.get(candidate.get("id"), {"items": []})
        gold = {item.get("scene_id") for item in pack.get("items", []) if item.get("hidden_role") == "gold_evidence"}
        if not refs.intersection(gold):
            missing_gold.append(candidate.get("id"))

    format_errors = [
        row.get("id", f"row_{index}")
        for index, row in enumerate(train_rows + valid_rows)
        if _has_format_error(row)
    ]
    report = {
        "run_name": config.run_name,
        "train_count": len(train_rows),
        "valid_count": len(valid_rows),
        "candidate_count": len(candidates),
        "memory_pack_count": len(packs),
        "eval_question_count": len(eval_questions),
        "retrieval_recall_at_5": round(1 - len(missing_gold) / len(candidates), 4) if candidates else 0,
        "format_error_count": len(format_errors),
        "missing_gold_examples": missing_gold[:10],
        "format_error_examples": format_errors[:10],
    }
    config.eval_report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    config.eval_report_md.write_text(_render_eval_report(report), encoding="utf-8")
    logger.info(f"写入评测问题: {config.eval_questions_jsonl}")
    logger.info(f"写入评测报告: {config.eval_report_md}")
    mark_step_done(step_name, config.status_dir)


def run_export_role(config: Config) -> None:
    step_name = "export_role"
    logger = init_logger(step_name, config.logs_dir)
    if check_step_done(step_name, config.status_dir):
        logger.info("角色包已导出，跳过")
        return

    root = config.role_package_dir
    for subdir in ["memory", "prompts", "datasets", "eval"]:
        (root / subdir).mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema": "taletalk-role-package-v1",
        "run_name": config.run_name,
        "novel_title": config.novel_title,
        "target_role": config.canonical_role,
        "aliases": config.target_role_aliases,
        "export_mode": config.export_mode,
        "generation_mode": config.generation_mode,
        "ai_passes": config.ai_passes,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "README.md").write_text(
        f"# {config.run_name}\n\nTaleTalk 角色包，角色：{config.canonical_role}，小说：《{config.novel_title}》。\n",
        encoding="utf-8",
    )

    _copy_if_exists(config.profile_json, root / "profile.json")
    _copy_if_exists(config.scene_memory_jsonl, root / "memory" / "scenes.jsonl")
    _copy_if_exists(config.memory_index_json, root / "memory" / "bm25.json")
    _copy_if_exists(config.embedding_npy, root / "memory" / "embeddings.npy")
    _copy_if_exists(config.embedding_meta_jsonl, root / "memory" / "embedding_meta.jsonl")
    _copy_if_exists(config.train_json, root / "datasets" / "chat_train.json")
    _copy_if_exists(config.valid_json, root / "datasets" / "chat_valid.json")
    _copy_if_exists(config.eval_questions_jsonl, root / "eval" / "eval_questions.jsonl")
    _copy_if_exists(config.eval_report_md, root / "eval" / "eval_report.md")

    runtime_prompt = (
        "你正在扮演目标角色。运行时必须先检索角色记忆，使用 TaleTalk memory pack 协议渲染到系统提示词，"
        "再回答用户问题；不要把记忆作为普通 user message。"
    )
    (root / "prompts" / "runtime_system.txt").write_text(runtime_prompt, encoding="utf-8")
    logger.info(f"角色包导出: {root}")
    mark_step_done(step_name, config.status_dir)


def _load_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


def _build_eval_questions(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for candidate in candidates[:limit]:
        questions.append(
            {
                "id": f"eval_{candidate.get('id')}",
                "question": candidate.get("question"),
                "expected_source_scene_ids": candidate.get("source_scene_ids", []),
                "category": candidate.get("sample_type", "grounded_fact"),
            }
        )
    return questions


def _has_format_error(row: dict[str, Any]) -> bool:
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        return True
    if conversations[0].get("from") != "human" or conversations[-1].get("from") != "gpt":
        return True
    return any("user" in str(message.get("value", "")).lower() for message in conversations if message.get("from") == "gpt")


def _render_eval_report(report: dict[str, Any]) -> str:
    lines = [
        f"# TaleTalk 阶段一评测报告：{report['run_name']}",
        "",
        f"- 训练样本：{report['train_count']}",
        f"- 验证样本：{report['valid_count']}",
        f"- 候选样本：{report['candidate_count']}",
        f"- Memory pack：{report['memory_pack_count']}",
        f"- 评测问题：{report['eval_question_count']}",
        f"- retrieval_recall_at_5：{report['retrieval_recall_at_5']}",
        f"- format_error_count：{report['format_error_count']}",
        "",
        "## 失败样例",
        "",
        f"- missing_gold_examples：{report['missing_gold_examples']}",
        f"- format_error_examples：{report['format_error_examples']}",
    ]
    return "\n".join(lines) + "\n"


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
