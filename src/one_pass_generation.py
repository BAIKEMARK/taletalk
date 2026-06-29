from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import Config
from .memory import SceneMemory
from .preprocess import SceneSkeleton

PROMPT_VERSION = "taletalk-one-pass-v1"


class OnePassGenerationError(RuntimeError):
    pass


def run_one_pass_generation(
    config: Config,
    scenes: list[SceneSkeleton],
) -> tuple[list[SceneMemory], list[dict[str, Any]], list[dict[str, Any]]]:
    completed = _load_completed(config.generation_checkpoint_dir)
    scene_memories: list[SceneMemory] = []
    profile_observations: list[dict[str, Any]] = []
    candidate_samples: list[dict[str, Any]] = []

    for batch in _batches(scenes, max(1, config.teacher_batch_size)):
        batch_key = "_".join(scene.scene_id for scene in batch)
        if batch_key in completed:
            result = completed[batch_key]
        else:
            result = _generate_batch(config, batch)
            _write_checkpoint(config.generation_checkpoint_dir, batch_key, result)
        scene_memories.extend(_scene_memories_from_result(config, batch, result))
        profile_observations.extend(_normalize_profile_observations(result.get("profile_observations", []), batch))
        candidate_samples.extend(_normalize_candidates(config, result.get("candidate_samples", []), batch))

    return scene_memories, profile_observations, candidate_samples


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


def _generate_batch(config: Config, scenes: list[SceneSkeleton]) -> dict[str, Any]:
    api_key = _api_key(config)
    base_url = _base_url(config)
    model = _model_name(config)
    payload = _batch_payload(config, scenes)

    if api_key and base_url and model and config.teacher_backend not in {"mock", "rule", "heuristic"}:
        try:
            text = _call_chat_completion(base_url, api_key, model, _load_one_pass_prompt(config.repo_dir), payload)
            return _validate_or_repair(config, scenes, text)
        except Exception as exc:
            _append_failed(config.raft_failed_jsonl, scenes, str(exc))
    return _heuristic_result(config, scenes)


def _call_chat_completion(base_url: str, api_key: str, model: str, prompt: str, payload: dict[str, Any]) -> str:
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = f"{url}/chat/completions"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "temperature": 0.2,
        "max_tokens": 4096,
    }
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            raise

    raise OnePassGenerationError("chat completion failed after retries")


def _api_key(config: Config) -> str:
    return (
        config.custom_api_key
        or os.getenv("CUSTOM_API_KEY", "")
        or os.getenv("STEPFUN_API_KEY", "")
        or os.getenv("OPENAI_API_KEY", "")
    )


def _base_url(config: Config) -> str:
    return (
        config.custom_base_url
        or os.getenv("CUSTOM_BASE_URL", "")
        or os.getenv("STEPFUN_BASE_URL", "")
        or os.getenv("OPENAI_BASE_URL", "")
    )


def _model_name(config: Config) -> str:
    return (
        config.teacher_model
        or config.custom_model_name
        or os.getenv("CUSTOM_MODEL_NAME", "")
        or os.getenv("STEPFUN_MODEL", "")
        or os.getenv("OPENAI_MODEL", "")
    )


def _validate_or_repair(config: Config, scenes: list[SceneSkeleton], text: str) -> dict[str, Any]:
    try:
        return _validate_result(json.loads(_strip_code_fence(text)), scenes)
    except Exception as first_error:
        api_key = _api_key(config)
        base_url = _base_url(config)
        model = _model_name(config)
        if not api_key or not base_url or not model:
            raise first_error
        repair_payload = {
            "original_input": _batch_payload(config, scenes),
            "failed_output": text,
            "error": str(first_error),
        }
        repaired = _call_chat_completion(base_url, api_key, model, _load_repair_prompt(config.repo_dir), repair_payload)
        return _validate_result(json.loads(_strip_code_fence(repaired)), scenes)


def _validate_result(data: dict[str, Any], scenes: list[SceneSkeleton]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("one-pass output must be object")
    scene_ids = {scene.scene_id for scene in scenes}
    result = {
        "version": data.get("version") or PROMPT_VERSION,
        "scene_memories": data.get("scene_memories") or [],
        "profile_observations": data.get("profile_observations") or [],
        "candidate_samples": data.get("candidate_samples") or [],
        "batch_warnings": data.get("batch_warnings") or [],
    }
    if result["version"] != PROMPT_VERSION:
        result["version"] = PROMPT_VERSION
    for memory in result["scene_memories"]:
        if memory.get("scene_id") not in scene_ids:
            raise ValueError(f"unknown scene_id in scene_memories: {memory.get('scene_id')}")
    for sample in result["candidate_samples"]:
        refs = sample.get("source_scene_ids") or []
        if any(ref not in scene_ids for ref in refs):
            raise ValueError(f"unknown source_scene_ids: {refs}")
    if len(result["candidate_samples"]) > len(scenes):
        result["candidate_samples"] = result["candidate_samples"][: len(scenes)]
    return result


def _scene_memories_from_result(
    config: Config,
    batch: list[SceneSkeleton],
    result: dict[str, Any],
) -> list[SceneMemory]:
    by_id = {str(item.get("scene_id")): item for item in result.get("scene_memories", [])}
    memories: list[SceneMemory] = []
    for scene in batch:
        item = by_id.get(scene.scene_id, {})
        characters = _string_list(item.get("characters")) or _dialogue_roles(scene)
        quotes = _quote_texts(item.get("quotes")) or [d["dialogue"] for d in scene.dialogues if d.get("role") in config.target_role_aliases][:2]
        knowledge_level = str(item.get("knowledge_level") or _infer_knowledge_level(scene, config.target_role_aliases))
        target_present = bool(item.get("target_role_present", _contains_any(scene.raw_text, config.target_role_aliases)))
        target_knows = bool(item.get("target_role_knows", target_present and knowledge_level != "narrator_only"))
        raw_text = scene.raw_text
        summary = str(item.get("summary") or _shorten(raw_text, 100))
        memories.append(
            SceneMemory(
                scene_id=scene.scene_id,
                chunk_id=scene.scene_index,
                chapter=scene.chapter,
                text=raw_text,
                summary=summary,
                characters=characters,
                target_role_present=target_present,
                target_role_knows=target_knows,
                events=_string_list(item.get("events"))[:3],
                relations=[{"text": value} for value in _string_list(item.get("relations"))[:2]],
                quotes=quotes[:3],
                source={
                    "novel_title": config.novel_title,
                    "source_start": scene.source_start,
                    "source_end": scene.source_end,
                },
                raw_text=raw_text,
                coverage=str(item.get("coverage") or scene.coverage),
                knowledge_level=knowledge_level,
                source_start=scene.source_start,
                source_end=scene.source_end,
                source_risks=_string_list(item.get("source_risks")),
            )
        )
    return memories


def _normalize_profile_observations(rows: list[dict[str, Any]], batch: list[SceneSkeleton]) -> list[dict[str, Any]]:
    scene_ids = {scene.scene_id for scene in batch}
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        refs = [ref for ref in row.get("source_scene_ids", []) if ref in scene_ids]
        value = str(row.get("value", "")).strip()
        if not refs or not value:
            continue
        normalized.append(
            {
                "id": f"profile_obs_{refs[0]}_{index:02d}",
                "aspect": str(row.get("aspect") or "personality"),
                "value": value,
                "source_scene_ids": refs,
                "confidence": float(row.get("confidence", 0.6) or 0.6),
            }
        )
    return normalized


def _normalize_candidates(config: Config, rows: list[dict[str, Any]], batch: list[SceneSkeleton]) -> list[dict[str, Any]]:
    scene_ids = {scene.scene_id for scene in batch}
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        refs = [ref for ref in row.get("source_scene_ids", []) if ref in scene_ids]
        policy = str(row.get("answer_policy") or "answer_from_memory")
        if not question or not answer:
            continue
        if policy != "insufficient_memory" and not refs:
            continue
        normalized.append(
            {
                "id": f"{config.run_name}_candidate_{batch[0].scene_index:06d}_{index:02d}",
                "sample_type": str(row.get("sample_type") or "grounded_fact"),
                "question": question,
                "answer": answer,
                "source_scene_ids": refs,
                "knowledge_level": str(row.get("knowledge_level") or "first_hand"),
                "answer_policy": policy,
                "must_not_claim": _string_list(row.get("must_not_claim")),
                "risk_tags": _string_list(row.get("risk_tags")),
                "generation_mode": config.generation_mode,
            }
        )
    return normalized


def _heuristic_result(config: Config, scenes: list[SceneSkeleton]) -> dict[str, Any]:
    scene_memories: list[dict[str, Any]] = []
    profile_observations: list[dict[str, Any]] = []
    candidate_samples: list[dict[str, Any]] = []
    for scene in scenes:
        present = _contains_any(scene.raw_text, config.target_role_aliases)
        roles = _dialogue_roles(scene)
        if present and config.canonical_role not in roles:
            roles.insert(0, config.canonical_role)
        summary = _shorten(scene.raw_text, 110)
        quotes = [
            {"role": item["role"], "text": item["dialogue"]}
            for item in scene.dialogues
            if item.get("role") in config.target_role_aliases
        ][:2]
        knowledge_level = "first_hand" if present else "narrator_only"
        scene_memories.append(
            {
                "scene_id": scene.scene_id,
                "coverage": "full",
                "summary": summary,
                "characters": roles,
                "target_role_present": present,
                "target_role_knows": present,
                "knowledge_level": knowledge_level,
                "events": [_shorten(scene.raw_text, 24)] if scene.raw_text else [],
                "relations": [],
                "quotes": quotes,
                "source_risks": ["heuristic_fallback"],
            }
        )
        if present:
            answer = "先别急着下结论。把眼前能确认的事列出来，再判断哪一条真正能救命。"
            if quotes:
                answer = _shorten(quotes[0]["text"], 100)
            candidate_samples.append(
                {
                    "sample_type": "grounded_fact",
                    "question": f"这一段里你经历了什么？",
                    "answer": answer,
                    "source_scene_ids": [scene.scene_id],
                    "knowledge_level": knowledge_level,
                    "answer_policy": "answer_from_memory",
                    "must_not_claim": [],
                    "risk_tags": ["heuristic_fallback"],
                }
            )
            profile_observations.append(
                {
                    "aspect": "speech_style",
                    "value": f"{config.canonical_role}回答时应冷静、克制，优先判断风险和证据。",
                    "source_scene_ids": [scene.scene_id],
                    "confidence": 0.5,
                }
            )
    return {
        "version": PROMPT_VERSION,
        "scene_memories": scene_memories,
        "profile_observations": profile_observations[:2],
        "candidate_samples": candidate_samples[: len(scenes)],
        "batch_warnings": ["heuristic fallback used"],
    }


def _batch_payload(config: Config, scenes: list[SceneSkeleton]) -> dict[str, Any]:
    return {
        "target_role": config.canonical_role,
        "target_role_aliases": config.target_role_aliases,
        "novel_title": config.novel_title,
        "scenes": [
            {
                "scene_id": scene.scene_id,
                "chapter": scene.chapter,
                "coverage": scene.coverage,
                "source_start": scene.source_start,
                "source_end": scene.source_end,
                "raw_text": scene.raw_text,
                "dialogues": scene.dialogues[:20],
            }
            for scene in scenes
        ],
    }


def _load_one_pass_prompt(repo_dir: Path) -> str:
    return _extract_prompt(repo_dir / "prompts/taletalk/02_one_pass_scene_generation.md")


def _load_repair_prompt(repo_dir: Path) -> str:
    return _extract_prompt(repo_dir / "prompts/taletalk/08_json_repair.md")


def _extract_prompt(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"<!-- PROMPT_START -->(.*?)<!-- PROMPT_END -->", text, flags=re.S)
    return match.group(1).strip() if match else text.strip()


def _load_completed(checkpoint_dir: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not checkpoint_dir.exists():
        return completed
    for path in checkpoint_dir.glob("batch_*.json"):
        completed[path.stem.removeprefix("batch_")] = json.loads(path.read_text(encoding="utf-8"))
    return completed


def _write_checkpoint(checkpoint_dir: Path, batch_key: str, result: dict[str, Any]) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / f"batch_{batch_key}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _append_failed(path: Path, scenes: list[SceneSkeleton], error: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"scene_ids": [scene.scene_id for scene in scenes], "error": error}
    with path.open("a", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False)
        f.write("\n")


def _batches(items: list[SceneSkeleton], size: int) -> list[list[SceneSkeleton]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _quote_texts(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    quotes: list[str] = []
    for item in value:
        if isinstance(item, dict):
            text = str(item.get("text", "")).strip()
        else:
            text = str(item).strip()
        if text:
            quotes.append(text)
    return quotes


def _dialogue_roles(scene: SceneSkeleton) -> list[str]:
    return sorted({item["role"] for item in scene.dialogues if item.get("role")})


def _contains_any(text: str, needles: list[str]) -> bool:
    return any(needle and needle in text for needle in needles)


def _infer_knowledge_level(scene: SceneSkeleton, aliases: list[str]) -> str:
    return "first_hand" if _contains_any(scene.raw_text, aliases) else "narrator_only"


def _shorten(text: str, limit: int) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    return clean if len(clean) <= limit else clean[:limit].rstrip() + "…"
