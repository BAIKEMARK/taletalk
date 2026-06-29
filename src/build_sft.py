from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import List
from .build_raft_sft import build_raft_sharegpt
from .config import Config
from .memory import read_dialogue_jsonl, read_profile, read_scene_memories
from .utils import init_logger, check_step_done, mark_step_done, run_with_progress

def _load_jsonl(path: Path) -> List[dict]:
    lines = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines.append(json.loads(line))
    return lines

def _group_by_chunk(lines: List[dict]) -> dict[int, List[dict]]:
    buckets = {}
    for item in lines:
        chunk_id = int(item["chunk_id"])
        if chunk_id not in buckets:
            buckets[chunk_id] = []
        buckets[chunk_id].append(item)
    
    for cid in buckets:
        buckets[cid].sort(key=lambda x: int(x.get("dialogue_index", 0)))
    
    return buckets

def _build_conversation(chunk_lines: List[dict], target_roles: set[str]) -> List[dict] | None:
    if not any(line["role"] in target_roles for line in chunk_lines):
        return None
    
    if chunk_lines[0]["role"] in target_roles:
        return None
    
    conversations = []
    buf_side = None
    buf_parts = []
    
    def flush():
        nonlocal buf_side, buf_parts
        if buf_side and buf_parts:
            text = "\n".join(buf_parts).strip()
            if text:
                conversations.append({"from": buf_side, "value": text})
        buf_side = None
        buf_parts = []
    
    for line in chunk_lines:
        side = "gpt" if line["role"] in target_roles else "human"
        if side == "human":
            piece = f"{line['role']}: {line['dialogue'].strip()}"
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
    
    has_gpt = any(c["from"] == "gpt" for c in conversations)
    if len(conversations) < 2 or not has_gpt:
        return None
    
    # 确保最后一条是gpt
    if conversations[-1]["from"] != "gpt":
        while conversations and conversations[-1]["from"] != "gpt":
            conversations.pop()
        if len(conversations) < 2:
            return None
    
    return conversations

def _split_samples(samples: List[dict], config: Config) -> tuple[List[dict], List[dict]]:
    random.Random(config.seed).shuffle(samples)
    if not samples:
        return [], []
    if len(samples) == 1:
        return samples, []
    n_valid = max(1, int(len(samples) * config.valid_ratio))
    valid_samples = samples[:n_valid]
    train_samples = samples[n_valid:]
    return train_samples, valid_samples

def _write_json(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

def _select_ratio(rows: List[dict], ratio: float, seed: int) -> List[dict]:
    if ratio >= 1:
        return list(rows)
    if ratio <= 0 or not rows:
        return []
    selected = list(rows)
    random.Random(seed).shuffle(selected)
    keep = max(1, int(len(selected) * ratio))
    return selected[:keep]

def run_build_sft(config: Config) -> None:
    """构建多轮SFT训练数据集"""
    step_name = "build_sft"
    logger = init_logger(step_name, config.logs_dir)
    
    if check_step_done(step_name, config.status_dir):
        logger.info("SFT数据集已构建完成，跳过")
        return
    
    logger.info("===== 开始构建SFT数据集 =====")
    logger.info(f"输入对话: {config.raw_jsonl}")
    logger.info(f"训练集输出: {config.train_json}")
    logger.info(f"验证集输出: {config.valid_json}")
    logger.info(f"目标角色别名: {', '.join(config.target_role_aliases)}")
    
    # 加载原始对话
    lines = _load_jsonl(config.raw_jsonl)
    logger.info(f"加载原始对话: {len(lines)} 条")
    
    # 统计别名命中情况
    target_roles = set(config.target_role_aliases)
    alias_counter = Counter(l["role"] for l in lines if l["role"] in target_roles)
    logger.info("别名命中统计:")
    for role, cnt in alias_counter.most_common():
        logger.info(f"  {role}: {cnt}次")
    
    if not alias_counter:
        logger.error("未找到任何目标角色的对话，请检查角色别名配置")
        raise RuntimeError("未找到目标角色对话")
    
    # 按chunk分组
    buckets = _group_by_chunk(lines)
    logger.info(f"共 {len(buckets)} 个对话块")
    
    # 构建多轮对话
    samples = []
    dropped_no_target = 0
    dropped_lead = 0
    dropped_short = 0
    
    for cid in run_with_progress(sorted(buckets), desc="构建对话", unit="chunk"):
        chunk_lines = buckets[cid]
        if not any(l["role"] in target_roles for l in chunk_lines):
            dropped_no_target += 1
            continue
        if chunk_lines[0]["role"] in target_roles:
            dropped_lead += 1
            continue
        conv = _build_conversation(chunk_lines, target_roles)
        if conv is None:
            dropped_short += 1
            continue
        samples.append({
            "id": f"{config.run_name}_{cid:05d}",
            "system": f"你正在扮演《{config.novel_title}》中的{config.canonical_role}。严格保持{config.canonical_role}的语气、性格、说话习惯和价值观，根据对话上下文自然回应，不要跳出角色，不要续写其他角色的发言。",
            "conversations": conv,
        })
    
    logger.info(f"共获得 {len(samples)} 条有效多轮对话样本")
    logger.info(f"丢弃统计:")
    logger.info(f"  无目标角色: {dropped_no_target}")
    logger.info(f"  目标角色开头: {dropped_lead}")
    logger.info(f"  对话过短: {dropped_short}")
    
    if len(samples) < 10:
        logger.warning(f"样本数过少({len(samples)}条)，训练效果可能不好")
    
    style_samples = samples
    raft_samples: List[dict] = []
    if config.enable_memory and config.sft_mode in {"raft", "mixed"}:
        logger.info(f"启用 memory-aware SFT: mode={config.sft_mode}")
        if not config.profile_json.exists():
            raise FileNotFoundError(f"缺 profile: {config.profile_json}，先运行 build_memory")
        if not config.scene_memory_jsonl.exists():
            raise FileNotFoundError(f"缺 scene memory: {config.scene_memory_jsonl}，先运行 build_memory")
        profile = read_profile(config.profile_json)
        scenes = read_scene_memories(config.scene_memory_jsonl)
        raw_rows = read_dialogue_jsonl(config.raw_jsonl)
        raft_samples = build_raft_sharegpt(
            raw_rows,
            scenes,
            profile,
            target_roles=target_roles,
            max_memory_chars=config.max_memory_chars,
            max_one_scene_chars=config.max_one_scene_chars,
        )
        logger.info(f"生成 RAFT 样本: {len(raft_samples)} 条")
        raft_train, raft_valid = _split_samples(list(raft_samples), config)
        _write_json(config.raft_train_json, raft_train)
        _write_json(config.raft_valid_json, raft_valid)
        logger.info(f"RAFT数据集: 训练集 {len(raft_train)} 条，验证集 {len(raft_valid)} 条")

        if config.sft_mode == "raft":
            samples = raft_samples
        else:
            style_part = _select_ratio(style_samples, config.style_data_ratio, config.seed)
            raft_part = _select_ratio(raft_samples, config.raft_data_ratio, config.seed + 1)
            samples = style_part + raft_part
            logger.info(f"mixed数据集: style {len(style_part)} 条 + raft {len(raft_part)} 条")
    elif config.sft_mode not in {"style", "raft", "mixed"}:
        raise ValueError(f"未知 sft_mode: {config.sft_mode}")

    # 限制最大样本数
    if config.max_conversations > 0 and len(samples) > config.max_conversations:
        logger.info(f"限制最大样本数为 {config.max_conversations}")
        random.Random(config.seed).shuffle(samples)
        samples = samples[:config.max_conversations]
    
    # 划分训练集验证集
    train_samples, valid_samples = _split_samples(samples, config)
    
    logger.info(f"数据集划分: 训练集 {len(train_samples)} 条，验证集 {len(valid_samples)} 条")
    
    # 保存数据集
    _write_json(config.train_json, train_samples)
    _write_json(config.valid_json, valid_samples)
    
    # 统计对话轮数
    turn_counts = [len(s["conversations"]) for s in samples]
    if turn_counts:
        avg_turns = sum(turn_counts) / len(turn_counts)
        logger.info(f"对话轮数统计: 平均 {avg_turns:.2f} 轮，最长 {max(turn_counts)} 轮，最短 {min(turn_counts)} 轮")
    
    mark_step_done(step_name, config.status_dir)
    logger.info("===== SFT数据集构建完成 =====")
