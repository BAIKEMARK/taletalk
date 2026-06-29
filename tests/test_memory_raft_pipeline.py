import json

from src.build_raft_sft import build_raft_sharegpt
from src.memory import (
    CharacterProfile,
    SceneMemory,
    build_default_profile,
    build_scene_memories,
    read_profile,
    write_profile,
)
from src.prompting import build_roleplay_system_prompt
from src.retrieval import BM25MemoryIndex


def test_build_scene_memories_from_dialogue_rows(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "chunk_id": 1,
                        "dialogue_index": 0,
                        "role": "余念安",
                        "dialogue": "你为什么来？",
                        "chunk_text": "齐夏看见余念安。她问他为何而来。",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "chunk_id": 1,
                        "dialogue_index": 1,
                        "role": "齐夏",
                        "dialogue": "我只是确认规则。",
                        "chunk_text": "齐夏看见余念安。她问他为何而来。",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {"chunk_id": 2, "dialogue_index": 0, "role": "其他人", "dialogue": "这里没有齐夏。"},
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    scenes = build_scene_memories(raw, canonical_role="齐夏", aliases=["齐夏", "阿夏"], novel_title="十日终焉")

    assert len(scenes) == 2
    assert scenes[0].scene_id == "chunk_000001"
    assert scenes[0].target_role_present is True
    assert scenes[0].target_role_knows is True
    assert "余念安" in scenes[0].text
    assert scenes[1].target_role_present is True


def test_profile_roundtrip(tmp_path):
    profile = build_default_profile("齐夏", ["齐夏", "阿夏"], "十日终焉")
    path = tmp_path / "profile.json"

    write_profile(profile, path)
    loaded = read_profile(path)

    assert loaded.role == "齐夏"
    assert loaded.aliases == ["齐夏", "阿夏"]
    assert "Use memory for facts." in loaded.answer_rules


def test_bm25_retrieves_relevant_scene_and_filters_unknown():
    scenes = [
        _scene("known", "齐夏知道余念安的线索。", knows=True),
        _scene("unknown", "旁白透露余念安的秘密。", knows=False),
        _scene("other", "孙悟空大闹天宫。", knows=True),
    ]
    index = BM25MemoryIndex.from_scenes(scenes)

    results = index.search("余念安秘密", top_k=3, exclude_narrator_only=True)

    assert [result.scene.scene_id for result in results] == ["known"]


def test_prompt_contains_profile_and_memory_rules():
    profile = _profile()
    scene = _scene("s1", "齐夏确认了规则。", knows=True)
    scene.summary = "齐夏确认规则。"
    scene.quotes = ["我只是确认规则。"]

    prompt = build_roleplay_system_prompt(profile, [scene])

    assert "你正在扮演《十日终焉》中的齐夏" in prompt
    assert "如果记忆片段包含答案" in prompt
    assert "齐夏确认规则" in prompt
    assert "不要续写 user/assistant" in prompt


def test_build_raft_sharegpt_outputs_system_human_gpt():
    profile = _profile()
    scenes = [_scene("chunk_000001", "余念安询问齐夏。", knows=True, chunk_id=1)]
    raw_rows = [
        {"chunk_id": 1, "dialogue_index": 0, "role": "余念安", "dialogue": "你为什么来？"},
        {"chunk_id": 1, "dialogue_index": 1, "role": "齐夏", "dialogue": "我只是确认规则。"},
    ]

    samples = build_raft_sharegpt(raw_rows, scenes, profile, target_roles={"齐夏"}, max_memory_chars=1000)

    assert samples[0]["system"].startswith("你正在扮演")
    assert samples[0]["conversations"][0] == {"from": "human", "value": "余念安：你为什么来？"}
    assert samples[0]["conversations"][1] == {"from": "gpt", "value": "我只是确认规则。"}
    assert samples[0]["metadata"]["oracle_scene_ids"] == ["chunk_000001"]


def _profile():
    return CharacterProfile(
        role="齐夏",
        aliases=["齐夏"],
        novel_title="十日终焉",
        identity="终焉之地中的参与者。",
        core_goals=[],
        personality=["冷静"],
        speech_style=["克制"],
        relationships=[],
        knowledge_boundary="只回答自己知道的事。",
        answer_rules=["Use memory for facts."],
    )


def _scene(scene_id, text, knows=True, chunk_id=1):
    return SceneMemory(
        scene_id=scene_id,
        chunk_id=chunk_id,
        chapter="",
        text=text,
        summary=text,
        characters=[],
        target_role_present=knows,
        target_role_knows=knows,
        events=[],
        relations=[],
        quotes=[],
        source={},
    )
