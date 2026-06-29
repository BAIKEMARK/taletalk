from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class Config:
    # 基础输入
    novel_txt: Path
    target_role: str
    target_role_aliases: List[str]
    canonical_role: str
    novel_title: str
    run_name: str
    
    # 模型配置
    model_choice: str
    model_ids: Dict[str, str]
    model_id: str
    
    # 抽取后端
    extraction_backend: str
    llm_platform: str
    custom_base_url: str
    custom_api_key: str
    custom_model_name: str
    local_model_id_override: str
    local_model_id: str
    local_model_port: int
    vllm_gpu_util: float
    max_workers: int
    chunk_size_tokens: int
    
    # SFT配置
    valid_ratio: float
    max_conversations: int
    seed: int

    # 记忆 / RAFT配置
    enable_memory: bool
    memory_backend: str
    top_k_memory: int
    max_memory_chars: int
    max_one_scene_chars: int
    prefer_target_present: bool
    exclude_narrator_only: bool
    sft_mode: str
    style_data_ratio: float
    raft_data_ratio: float
    raft_include_distractors: bool
    raft_no_answer_ratio: float
    roleplay_mode: str
    
    # 训练配置
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_train_epochs: float
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    cutoff_len: int
    warmup_ratio: float
    lr_scheduler_type: str
    logging_steps: int
    save_steps: int
    eval_steps: int
    gradient_checkpointing: bool
    model_cache_dir: Path
    output_dir: Path
    
    # 推理配置
    gradio_port: int
    share: bool
    stream_output: bool
    adapter_dir: Optional[Path]
    
    # 路径
    repo_dir: Path
    raw_dir: Path
    sft_dir: Path
    cache_dir: Path
    logs_dir: Path
    status_dir: Path
    raw_jsonl: Path
    train_json: Path
    valid_json: Path
    profile_json: Path
    scene_memory_jsonl: Path
    memory_index_json: Path
    raft_train_json: Path
    raft_valid_json: Path

def load_config(config_path: str = "config.toml") -> Config:
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    
    # 角色别名处理
    role_aliases = [r.strip() for r in cfg["target_role"].split(",") if r.strip()]
    assert role_aliases, "target_role 不能为空"
    canonical_role = role_aliases[0]
    
    # 模型ID
    model_id = cfg["model_ids"][cfg["model_choice"]]
    local_model_id = cfg["local_model_id_override"].strip() or model_id
    
    # 路径处理
    repo_dir = Path.cwd().resolve()
    while repo_dir != repo_dir.parent and not (repo_dir / "extract").is_dir() and not (repo_dir / "src").is_dir():
        repo_dir = repo_dir.parent
    
    novel_txt = Path(cfg["novel_txt"])
    if not novel_txt.is_absolute():
        novel_txt = (repo_dir / novel_txt).resolve()
    assert novel_txt.exists(), f"小说文件不存在: {novel_txt}"
    
    raw_dir = repo_dir / "data" / "raw"
    sft_dir = repo_dir / "data"
    profiles_dir = repo_dir / "data" / "profiles"
    memory_dir = repo_dir / "data" / "memory"
    cache_dir = repo_dir / "cache"
    logs_dir = repo_dir / "logs"
    status_dir = repo_dir / "status"
    adapter_dir_raw = cfg.get("adapter_dir", "").strip()
    adapter_dir = Path(adapter_dir_raw) if adapter_dir_raw else None
    if adapter_dir and not adapter_dir.is_absolute():
        adapter_dir = (repo_dir / adapter_dir).resolve()

    for d in [raw_dir, sft_dir, profiles_dir, memory_dir, cache_dir, logs_dir, status_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    run_name = cfg["run_name"]
    raw_jsonl = raw_dir / f"{run_name}_dialogues.jsonl"
    train_json = sft_dir / f"{run_name}_chat_train.json"
    valid_json = sft_dir / f"{run_name}_chat_valid.json"
    profile_json = profiles_dir / f"{run_name}_profile.json"
    scene_memory_jsonl = memory_dir / f"{run_name}_scenes.jsonl"
    memory_index_json = memory_dir / f"{run_name}_bm25.json"
    raft_train_json = sft_dir / f"{run_name}_raft_train.json"
    raft_valid_json = sft_dir / f"{run_name}_raft_valid.json"
    
    return Config(
        # 基础输入
        novel_txt=novel_txt,
        target_role=cfg["target_role"],
        target_role_aliases=role_aliases,
        canonical_role=canonical_role,
        novel_title=cfg["novel_title"],
        run_name=run_name,
        
        # 模型
        model_choice=cfg["model_choice"],
        model_ids=cfg["model_ids"],
        model_id=model_id,
        
        # 抽取
        extraction_backend=cfg["extraction_backend"],
        llm_platform=cfg["llm_platform"],
        custom_base_url=cfg["custom_base_url"],
        custom_api_key=cfg["custom_api_key"],
        custom_model_name=cfg["custom_model_name"],
        local_model_id_override=cfg["local_model_id_override"],
        local_model_id=local_model_id,
        local_model_port=cfg["local_model_port"],
        vllm_gpu_util=cfg["vllm_gpu_util"],
        max_workers=cfg["max_workers"],
        chunk_size_tokens=cfg["chunk_size_tokens"],
        
        # SFT
        valid_ratio=cfg["valid_ratio"],
        max_conversations=cfg["max_conversations"],
        seed=cfg["seed"],

        # 记忆 / RAFT
        enable_memory=cfg.get("enable_memory", False),
        memory_backend=cfg.get("memory_backend", "bm25"),
        top_k_memory=cfg.get("top_k_memory", 3),
        max_memory_chars=cfg.get("max_memory_chars", 1800),
        max_one_scene_chars=cfg.get("max_one_scene_chars", 600),
        prefer_target_present=cfg.get("prefer_target_present", True),
        exclude_narrator_only=cfg.get("exclude_narrator_only", True),
        sft_mode=cfg.get("sft_mode", "style"),
        style_data_ratio=cfg.get("style_data_ratio", 1.0),
        raft_data_ratio=cfg.get("raft_data_ratio", 0.0),
        raft_include_distractors=cfg.get("raft_include_distractors", True),
        raft_no_answer_ratio=cfg.get("raft_no_answer_ratio", 0.1),
        roleplay_mode=cfg.get("roleplay_mode", "in_character"),
        
        # 训练
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        lora_rank=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        cutoff_len=cfg["cutoff_len"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        eval_steps=cfg["eval_steps"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        model_cache_dir=Path(cfg["model_cache_dir"]),
        output_dir=Path(cfg["output_dir"]),
        
        # 推理
        gradio_port=cfg["gradio_port"],
        share=cfg["share"],
        stream_output=cfg["stream_output"],
        adapter_dir=adapter_dir,
        
        # 路径
        repo_dir=repo_dir,
        raw_dir=raw_dir,
        sft_dir=sft_dir,
        cache_dir=cache_dir,
        logs_dir=logs_dir,
        status_dir=status_dir,
        raw_jsonl=raw_jsonl,
        train_json=train_json,
        valid_json=valid_json,
        profile_json=profile_json,
        scene_memory_jsonl=scene_memory_jsonl,
        memory_index_json=memory_index_json,
        raft_train_json=raft_train_json,
        raft_valid_json=raft_valid_json,
    )
