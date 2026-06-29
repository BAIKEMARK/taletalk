from __future__ import annotations

from collections.abc import Sequence

from .memory import CharacterProfile, SceneMemory


def build_roleplay_system_prompt(
    profile: CharacterProfile,
    scenes: Sequence[SceneMemory],
    max_memory_chars: int = 1800,
    max_one_scene_chars: int = 600,
) -> str:
    memory_snippets = _format_memory_snippets(scenes, max_memory_chars, max_one_scene_chars)
    profile_text = _format_profile(profile)
    return f"""你正在扮演《{profile.novel_title}》中的{profile.role}。

你必须遵守：
1. 如果记忆片段包含答案，优先依据记忆回答。
2. 不要逐字复述记忆片段，要用{profile.role}自己的口吻回答。
3. 如果记忆片段没有答案，不要编造具体小说事实。
4. 始终保持第一人称，除非这个角色在该场景下不会这么说。
5. 不要续写 user/assistant，不要展开新对话。

【角色设定】
{profile_text}

【记忆片段】
{memory_snippets}
""".strip()


def _format_profile(profile: CharacterProfile) -> str:
    lines = [
        f"角色：{profile.role}",
        f"别名：{', '.join(profile.aliases)}",
        f"身份：{profile.identity}",
    ]
    if profile.core_goals:
        lines.append(f"核心目标：{'；'.join(profile.core_goals)}")
    if profile.personality:
        lines.append(f"性格：{'；'.join(profile.personality)}")
    if profile.speech_style:
        lines.append(f"说话风格：{'；'.join(profile.speech_style)}")
    if profile.knowledge_boundary:
        lines.append(f"认知边界：{profile.knowledge_boundary}")
    if profile.answer_rules:
        lines.append(f"回答规则：{'；'.join(profile.answer_rules)}")
    return "\n".join(lines)


def _format_memory_snippets(
    scenes: Sequence[SceneMemory],
    max_memory_chars: int,
    max_one_scene_chars: int,
) -> str:
    if not scenes:
        return "（没有检索到可靠记忆片段。）"
    snippets: list[str] = []
    total = 0
    for idx, scene in enumerate(scenes, start=1):
        body = (scene.summary or scene.text).strip()
        if scene.quotes:
            body += "\n原话：" + " / ".join(scene.quotes[:3])
        if len(body) > max_one_scene_chars:
            body = body[:max_one_scene_chars].rstrip() + "..."
        snippet = f"[{idx}] {scene.scene_id}\n{body}"
        if total + len(snippet) > max_memory_chars and snippets:
            break
        snippets.append(snippet)
        total += len(snippet)
    return "\n\n".join(snippets)
