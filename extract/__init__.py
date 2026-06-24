"""TaleTalk dialogue extraction module.

Vendored and adapted from https://github.com/KMnO4-zx/extract-dialogue
(part of the huanhuan-chat project family). Local modifications focus on
OpenAI-compatible endpoints (cloud APIs + local vLLM / LLaMA Factory servers)
and on returning per-chunk dialogue tuples that downstream multi-turn
ShareGPT conversion can group by chunk_id.
"""

from .dialogue_extractor import (
    DialogueExtractor,
    DialogueItem,
    ChunkDialogueItem,
)
from .config import Config, ModelPlatform

__all__ = [
    "DialogueExtractor",
    "DialogueItem",
    "ChunkDialogueItem",
    "Config",
    "ModelPlatform",
]
