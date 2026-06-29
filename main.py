#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import sys
from src.config import load_config
from src.utils import clear_step_status, init_logger


def load_step_fn(step_name: str):
    """Lazy-load step modules so optional train/infer deps don't block startup."""
    if step_name == "extract":
        from src.extract import run_extract
        return run_extract
    if step_name == "build_memory":
        from src.build_memory import run_build_memory
        return run_build_memory
    if step_name == "build_sft":
        from src.build_sft import run_build_sft
        return run_build_sft
    if step_name == "eval":
        from src.eval_export import run_eval
        return run_eval
    if step_name == "export_role":
        from src.eval_export import run_export_role
        return run_export_role
    if step_name == "train":
        from src.train import run_train
        return run_train
    if step_name == "infer":
        from src.infer import run_infer
        return run_infer
    raise ValueError(f"未知步骤: {step_name}")

def main():
    parser = argparse.ArgumentParser(description="TaleTalk - 让小说角色活起来")
    parser.add_argument("--config", "-c", default="config.toml", help="配置文件路径，默认config.toml")
    step_choices = ["extract", "build_memory", "build_sft", "eval", "export_role", "train", "infer"]
    parser.add_argument("--rerun", "-r", nargs="+", choices=step_choices, help="强制重跑指定步骤")
    parser.add_argument("--only", "-o", nargs="+", choices=step_choices, help="只执行指定步骤，不执行后续步骤")
    args = parser.parse_args()
    
    # 加载配置
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"配置加载失败: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 初始化全局日志
    main_logger = init_logger("main", config.logs_dir)
    main_logger.info("===== TaleTalk 启动 =====")
    main_logger.info(f"运行标识: {config.run_name}")
    main_logger.info(f"目标角色: {config.canonical_role}")
    main_logger.info(f"小说: {config.novel_title}")
    
    # 处理强制重跑
    if args.rerun:
        for step in args.rerun:
            main_logger.info(f"强制重跑步骤: {step}")
            clear_step_status(step, config.status_dir)
    
    steps = [
        ("extract", "抽取对话"),
        ("build_memory", "构建角色记忆"),
        ("build_sft", "构建SFT数据集"),
        ("eval", "生成评测报告"),
        ("export_role", "导出角色包"),
        ("train", "训练LoRA"),
        ("infer", "启动推理服务"),
    ]
    only_steps = set(args.only or [])
    default_steps = {"build_memory", "build_sft", "eval", "export_role"}
    
    for step_name, step_desc in steps:
        if only_steps and step_name not in only_steps:
            continue
        if not only_steps and step_name not in default_steps:
            continue
        
        try:
            step_fn = load_step_fn(step_name)
            main_logger.info(f">>>>> 开始{step_desc} <<<<<")
            step_fn(config)
            main_logger.info(f">>>>> {step_desc}完成 <<<<<")
        except Exception as e:
            main_logger.error(f"{step_desc}失败: {e}", exc_info=True)
            main_logger.info("查看日志文件获取详细错误信息")
            sys.exit(1)
        
    if only_steps:
        main_logger.info("指定步骤执行完成，退出")
        sys.exit(0)
    
    main_logger.info("===== 所有步骤执行完成 =====")
    main_logger.info(f"训练数据集: {config.train_json}")
    main_logger.info(f"评测报告: {config.eval_report_md}")
    main_logger.info(f"角色包: {config.role_package_dir}")

if __name__ == "__main__":
    main()
