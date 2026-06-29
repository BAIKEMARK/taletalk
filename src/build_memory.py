from __future__ import annotations

from .config import Config
from .memory import build_default_profile, build_scene_memories, write_profile, write_scene_memories
from .retrieval import BM25MemoryIndex
from .utils import check_step_done, init_logger, mark_step_done


def run_build_memory(config: Config) -> None:
    """Build role profile, scene memory, and retrieval index."""
    step_name = "build_memory"
    logger = init_logger(step_name, config.logs_dir)

    if not config.enable_memory:
        logger.info("enable_memory=false，跳过记忆构建")
        return

    if check_step_done(step_name, config.status_dir):
        logger.info("角色记忆已构建完成，跳过")
        return

    if not config.raw_jsonl.exists():
        raise FileNotFoundError(f"缺少原始对话文件: {config.raw_jsonl}")

    logger.info("===== 开始构建角色记忆 =====")
    logger.info(f"输入对话: {config.raw_jsonl}")
    logger.info(f"profile: {config.profile_json}")
    logger.info(f"scenes: {config.scene_memory_jsonl}")
    logger.info(f"index: {config.memory_index_json}")

    if not config.profile_json.exists():
        profile = build_default_profile(config.canonical_role, config.target_role_aliases, config.novel_title)
        write_profile(profile, config.profile_json)
        logger.info("已生成默认角色 profile，可手工编辑后重跑 build_memory/build_sft")
    else:
        logger.info("profile 已存在，保留用户编辑版本")

    scenes = build_scene_memories(
        config.raw_jsonl,
        canonical_role=config.canonical_role,
        aliases=config.target_role_aliases,
        novel_title=config.novel_title,
    )
    write_scene_memories(scenes, config.scene_memory_jsonl)
    logger.info(f"写入 scene memory: {len(scenes)} 条")

    index = BM25MemoryIndex.from_scenes(scenes)
    index.save(config.memory_index_json)
    logger.info("BM25 记忆索引写入完成")

    mark_step_done(step_name, config.status_dir)
    logger.info("===== 角色记忆构建完成 =====")
