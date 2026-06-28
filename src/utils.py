from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional
from tqdm import tqdm

def init_logger(step_name: str, logs_dir: Path) -> logging.Logger:
    """初始化日志，同时输出到文件和控制台"""
    logger = logging.getLogger(step_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(console_handler)
    
    # 文件输出
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f"{step_name}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(lineno)d | %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"日志文件: {log_file}")
    return logger

def check_step_done(step_name: str, status_dir: Path) -> bool:
    """检查步骤是否已经完成"""
    status_file = status_dir / f"{step_name}.done"
    return status_file.exists()

def mark_step_done(step_name: str, status_dir: Path) -> None:
    """标记步骤完成"""
    status_file = status_dir / f"{step_name}.done"
    status_file.touch()
    with open(status_file, "w") as f:
        f.write(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def clear_step_status(step_name: str, status_dir: Path) -> None:
    """清除步骤完成标记，用于强制重跑"""
    status_file = status_dir / f"{step_name}.done"
    if status_file.exists():
        status_file.unlink()

def run_with_progress(iterable, desc: str, total: Optional[int] = None, unit: str = "it") -> tqdm:
    """带进度条的迭代器封装"""
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
