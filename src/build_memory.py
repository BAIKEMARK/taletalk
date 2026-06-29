from __future__ import annotations

from .config import Config
from .memory import build_default_profile, write_profile, write_scene_memories
from .one_pass_generation import run_one_pass_generation, write_jsonl
from .preprocess import build_scene_skeletons, write_scene_build_report, write_scene_skeletons
from .retrieval import BM25MemoryIndex, write_hashed_embedding_artifacts
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

    logger.info("===== 开始构建角色记忆 =====")
    logger.info(f"输入小说: {config.novel_txt}")
    logger.info(f"可选 raw dialogue: {config.raw_jsonl}")
    logger.info(f"scene skeleton: {config.raw_scene_jsonl}")
    logger.info(f"profile: {config.profile_json}")
    logger.info(f"scenes: {config.scene_memory_jsonl}")
    logger.info(f"index: {config.memory_index_json}")

    if not config.profile_json.exists():
        profile = build_default_profile(config.canonical_role, config.target_role_aliases, config.novel_title)
        write_profile(profile, config.profile_json)
        logger.info("已生成默认角色 profile，可手工编辑后重跑 build_memory/build_sft")
    else:
        logger.info("profile 已存在，保留用户编辑版本")

    scene_skeletons, report = build_scene_skeletons(
        config.novel_txt,
        config.raw_jsonl if config.raw_jsonl.exists() else None,
        max_chars=config.scene_max_chars,
        overlap_chars=config.scene_overlap_chars,
    )
    write_scene_skeletons(scene_skeletons, config.raw_scene_jsonl)
    write_scene_build_report(report, config.scene_build_report_json)
    logger.info(f"写入 scene skeleton: {len(scene_skeletons)} 条")

    scenes, observations, candidates = run_one_pass_generation(config, scene_skeletons)
    write_scene_memories(scenes, config.scene_memory_jsonl)
    logger.info(f"写入 scene memory: {len(scenes)} 条")
    write_jsonl(observations, config.profile_observations_jsonl)
    write_jsonl(candidates, config.raft_candidates_raw_jsonl)
    logger.info(f"写入 profile observations: {len(observations)} 条")
    logger.info(f"写入 candidate samples: {len(candidates)} 条")

    index = BM25MemoryIndex.from_scenes(scenes)
    index.save(config.memory_index_json)
    logger.info("BM25 记忆索引写入完成")
    write_hashed_embedding_artifacts(scenes, config.embedding_npy, config.embedding_meta_jsonl)
    logger.info("embedding 记忆索引占位产物写入完成")

    mark_step_done(step_name, config.status_dir)
    logger.info("===== 角色记忆构建完成 =====")
