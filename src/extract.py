from __future__ import annotations

import os
import signal
import sys
import time
import json
import socket
import subprocess
import urllib.request
from pathlib import Path
from typing import List
from .config import Config
from .utils import init_logger, check_step_done, mark_step_done, run_with_progress

# 复用之前的对话抽取器
def _load_extractor(logger, config):
    # 把extract模块加入路径
    sys.path.insert(0, str(config.repo_dir))
    from extract.dialogue_extractor import DialogueExtractor, Config as ExtractConfig
    
    # 初始化抽取配置
    extract_cfg = ExtractConfig(
        target_role=config.canonical_role,
        llm_platform=config.llm_platform,
        custom_api_key=config.custom_api_key,
        custom_base_url=config.custom_base_url,
        custom_model_name=config.custom_model_name,
        max_workers=config.max_workers,
        chunk_size_tokens=config.chunk_size_tokens,
        output_path=str(config.raw_jsonl),
        save_chunk_text=False,
    )
    extractor = DialogueExtractor(extract_cfg)
    return extractor

def _start_vllm_server(logger, config):
    """启动vLLM服务，返回进程对象"""
    port = config.local_model_port
    # 检查端口是否已经被占用
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        if s.connect_ex(('127.0.0.1', port)) == 0:
            logger.info(f"端口 {port} 已被占用，复用现有vLLM服务")
            return None
    
    # 启动vLLM服务
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["HIP_VISIBLE_DEVICES"] = "0"
    env.setdefault("VLLM_USE_TRITON_FLASH_ATTN", "0")
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    
    log_path = config.cache_dir / "vllm_serve.log"
    log_f = open(log_path, "wb")
    
    cmd = [
        "vllm", "serve", config.local_model_id,
        "--port", str(port),
        "--host", "127.0.0.1",
        "--served-model-name", config.local_model_id,
        "--trust-remote-code",
        "--dtype", "bfloat16",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", str(config.vllm_gpu_util),
        "--enforce-eager",
        "--limit-mm-per-prompt.image", "0",
        "--limit-mm-per-prompt.video", "0",
    ]
    
    logger.info(f"启动vLLM服务: {' '.join(cmd)}")
    logger.info(f"vLLM日志: {log_path}")
    
    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
    )
    
    # 等待服务启动
    max_wait = 900
    start_time = time.time()
    ready = False
    
    while time.time() - start_time < max_wait:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as r:
                if r.status == 200:
                    ready = True
                    break
        except Exception:
            pass
        if proc.poll() is not None:
            logger.warning(
                f"vLLM wrapper进程已退出(rc={proc.returncode})，继续等待EngineCore子进程开放端口"
            )
        
        elapsed = int(time.time() - start_time)
        if elapsed > 0 and elapsed % 20 == 0:
            logger.info(f"等待vLLM服务启动... {elapsed}s / {max_wait}s")
        
        time.sleep(2)
    
    if not ready:
        logger.error(f"vLLM服务 {max_wait}s 内未启动成功")
        raise RuntimeError("vLLM服务启动超时")
    
    logger.info("vLLM服务启动成功")
    return proc

def run_extract(config: Config) -> None:
    """抽取小说中的角色对话"""
    step_name = "extract"
    logger = init_logger(step_name, config.logs_dir)
    
    if check_step_done(step_name, config.status_dir):
        logger.info("对话抽取已完成，跳过")
        return
    
    logger.info("===== 开始抽取对话 =====")
    logger.info(f"小说: {config.novel_txt}")
    logger.info(f"目标角色: {config.canonical_role} (别名: {', '.join(config.target_role_aliases[1:])})")
    logger.info(f"输出: {config.raw_jsonl}")
    
    proc = None
    try:
        if config.extraction_backend == "local_model":
            proc = _start_vllm_server(logger, config)
            # 配置环境变量让抽取器用本地服务
            os.environ["LLM_PLATFORM"] = "custom"
            os.environ["CUSTOM_API_KEY"] = "EMPTY"
            os.environ["CUSTOM_BASE_URL"] = f"http://127.0.0.1:{config.local_model_port}/v1"
            os.environ["CUSTOM_MODEL_NAME"] = config.local_model_id
        
        # 加载抽取器
        import sys
        sys.path.insert(0, str(config.repo_dir))
        from extract.dialogue_extractor import DialogueExtractor, Config as ExtractConfig
        
        extract_cfg = ExtractConfig(
            target_role=config.canonical_role,
            llm_platform=os.environ.get("LLM_PLATFORM", config.llm_platform),
            custom_api_key=os.environ.get("CUSTOM_API_KEY", config.custom_api_key),
            custom_base_url=os.environ.get("CUSTOM_BASE_URL", config.custom_base_url),
            custom_model_name=os.environ.get("CUSTOM_MODEL_NAME", config.custom_model_name),
            max_workers=config.max_workers,
            chunk_size_tokens=config.chunk_size_tokens,
            output_path=str(config.raw_jsonl),
            save_chunk_text=False,
        )
        extractor = DialogueExtractor(extract_cfg)
        
        # 读取小说内容
        with open(config.novel_txt, encoding='utf-8') as f:
            content = f.read()
        
        # 抽取对话
        logger.info("开始抽取对话...")
        chunks = extractor.chunk_text(content)
        logger.info(f"文本分块完成，共 {len(chunks)} 块")
        
        dialogues = extractor.extract(chunks)
        logger.info(f"抽取完成，共获得 {len(dialogues)} 条对话")
        
        # 统计角色出现次数
        from collections import Counter
        role_counter = Counter(d["role"] for d in dialogues)
        logger.info("角色出现频次统计:")
        for role, cnt in role_counter.most_common(10):
            logger.info(f"  {role}: {cnt}次")
        
        mark_step_done(step_name, config.status_dir)
        logger.info("===== 对话抽取完成 =====")
        
    finally:
        # 关闭vLLM服务
        if proc is not None:
            logger.info("关闭vLLM服务...")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception as e:
                logger.warning(f"关闭vLLM服务失败: {e}")
