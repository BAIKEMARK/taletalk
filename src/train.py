from __future__ import annotations

import os
import json
import time
import subprocess
from pathlib import Path
from .config import Config
from .utils import init_logger, check_step_done, mark_step_done


def _modelscope_cache_dir(cache_dir: Path, model_id: str) -> Path:
    """Return ModelScope's default snapshot dir for a model id."""
    return cache_dir / model_id.replace(".", "___")


def _template_for_model(model_id: str) -> str:
    model_id_lower = model_id.lower()
    if "qwen3.5" in model_id_lower or "qwen3___5" in model_id_lower:
        return "qwen3_5"
    if "qwen3.6" in model_id_lower or "qwen3___6" in model_id_lower:
        return "qwen3_6"
    if "qwen3" in model_id_lower:
        return "qwen3"
    return "qwen"

def run_train(config: Config) -> None:
    """训练角色LoRA"""
    step_name = "train"
    logger = init_logger(step_name, config.logs_dir)
    
    if check_step_done(step_name, config.status_dir):
        logger.info("LoRA训练已完成，跳过")
        return
    
    logger.info("===== 开始训练LoRA =====")
    logger.info(f"模型: {config.model_id}")
    logger.info(f"训练集: {config.train_json}")
    logger.info(f"验证集: {config.valid_json}")
    logger.info(f"输出目录: {config.output_dir}/{config.run_name}")
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("缺少 PyYAML，请先运行: pip install pyyaml") from exc
    
    # 生成LLaMA Factory训练配置
    cached_model_dir = _modelscope_cache_dir(config.model_cache_dir, config.model_id)
    model_name_or_path = str(cached_model_dir) if (cached_model_dir / "config.json").exists() else config.model_id
    cfg = {
        "model_name_or_path": model_name_or_path,
        "template": _template_for_model(model_name_or_path),
        "dataset": "taletalk_custom",
        "dataset_dir": str(config.sft_dir),
        "output_dir": str(config.output_dir / config.run_name),
        "finetuning_type": "lora",
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "lora_target": "all",
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "warmup_ratio": config.warmup_ratio,
        "lr_scheduler_type": config.lr_scheduler_type,
        "cutoff_len": config.cutoff_len,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "eval_steps": config.eval_steps,
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "load_best_model_at_end": True,
        "gradient_checkpointing": config.gradient_checkpointing,
        "bf16": True,
        "optim": "adamw_torch",
        "report_to": "none",
    }
    
    # 保存训练配置
    runtime_config = config.cache_dir / f"{config.run_name}_train_config.yaml"
    with open(runtime_config, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"训练配置已生成: {runtime_config}")
    
    # 生成dataset_info.json
    dataset_info = {
        "taletalk_custom": {
            "file_name": config.run_name + "_chat_train.json",
            "file_sha1": "",
            "columns": {
                "messages": "conversations",
                "system": "system",
            }
        },
        "taletalk_custom_valid": {
            "file_name": config.run_name + "_chat_valid.json",
            "file_sha1": "",
            "columns": {
                "messages": "conversations",
                "system": "system",
            }
        }
    }
    
    with open(config.sft_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    # 启动训练
    cmd = [
        "llamafactory-cli", "train", str(runtime_config)
    ]
    
    logger.info(f"训练命令: {' '.join(cmd)}")
    log_path = config.logs_dir / f"train_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logger.info(f"训练日志: {log_path}")
    
    env = os.environ.copy()
    env["DISABLE_VERSION_CHECK"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["HIP_VISIBLE_DEVICES"] = "0"
    
    with open(log_path, "wb") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
        )
        proc.wait()
    
    if proc.returncode != 0:
        logger.error(f"训练失败，退出码: {proc.returncode}，请查看日志: {log_path}")
        raise RuntimeError("训练失败")
    
    logger.info("训练完成")
    mark_step_done(step_name, config.status_dir)
    logger.info("===== LoRA训练完成 =====")
