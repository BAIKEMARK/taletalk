from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .memory import CharacterProfile, read_profile
from .prompting import build_roleplay_system_prompt
from .retrieval import BM25MemoryIndex


@dataclass
class RuntimeMemory:
    profile: CharacterProfile
    index: BM25MemoryIndex


def load_runtime_memory(config: Config) -> RuntimeMemory | None:
    if not config.enable_memory:
        return None
    if not config.profile_json.exists() or not config.memory_index_json.exists():
        return None
    return RuntimeMemory(
        profile=read_profile(config.profile_json),
        index=BM25MemoryIndex.load(config.memory_index_json),
    )


def build_runtime_system_prompt(config: Config, memory: RuntimeMemory | None, user_message: str) -> str:
    if memory is None:
        return (
            f"你正在扮演《{config.novel_title}》中的{config.canonical_role}。"
            f"严格保持{config.canonical_role}的语气、性格、说话习惯和价值观，"
            "根据对话上下文自然回应，不要跳出角色，不要续写其他角色的发言。"
        )
    results = memory.index.search(
        user_message,
        top_k=config.top_k_memory,
        exclude_narrator_only=config.exclude_narrator_only,
    )
    scenes = [result.scene for result in results]
    return build_roleplay_system_prompt(
        memory.profile,
        scenes,
        max_memory_chars=config.max_memory_chars,
        max_one_scene_chars=config.max_one_scene_chars,
    )
