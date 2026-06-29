# Memory RAFT Qixia Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first memory-first TaleTalk path for Qixia without running ROCm training: build character memory, retrieve it, generate RAFT/mixed SFT data, and smoke-test the local data path.

**Architecture:** Preserve the user-facing lifecycle but replace dialogue-only internals with a memory-first contract. Implement focused modules for profile loading, scene memory construction, BM25 retrieval, shared prompt assembly, and RAFT SFT generation; then wire them into the CLI as independently runnable steps.

**Tech Stack:** Python 3.11+/stdlib, existing TOML config loader, existing dialogue JSONL output, pytest. Use a lightweight in-repo BM25 implementation first to avoid adding a dependency.

---

## File Structure

- Create `src/memory.py`: scene memory dataclasses, profile dataclass, scene builder from raw dialogue JSONL and optional chunk text, JSONL read/write helpers.
- Create `src/retrieval.py`: tokenizer, in-repo BM25 index, save/load index, retrieval API.
- Create `src/prompting.py`: shared roleplay system prompt builder used by RAFT SFT and inference.
- Create `src/build_raft_sft.py`: build style/RAFT/mixed ShareGPT datasets from raw dialogues plus memory/profile.
- Modify `src/config.py`: add memory and RAFT config fields and output paths.
- Modify `config.example.toml`: document new memory/RAFT options.
- Modify `main.py` and step orchestration files if needed: add `build_memory` and memory-aware `build_sft`.
- Modify `src/infer.py`: optional memory retrieval before generation.
- Add tests under `tests/` for memory building, retrieval, prompt parity, and RAFT SFT output.

## Task 1: Config Contract

**Files:**
- Modify: `src/config.py`
- Modify: `config.example.toml`
- Test: `tests/test_config_memory.py`

- [ ] **Step 1: Write failing config test**

```python
from pathlib import Path

from src.config import load_config


def test_memory_config_defaults(tmp_path, monkeypatch):
    repo = tmp_path
    (repo / "extract").mkdir()
    (repo / "src").mkdir()
    (repo / "novels").mkdir()
    (repo / "novels" / "novel.txt").write_text("hello", encoding="utf-8")
    (repo / "config.toml").write_text(
        '''
novel_txt = "novels/novel.txt"
target_role = "齐夏,阿夏"
novel_title = "十日终焉"
run_name = "shiri_qixia"
model_choice = "qwen3_5_9b"
model_ids = { "qwen3_5_9b" = "Qwen/Qwen3.5-9B" }
extraction_backend = "cloud_api"
llm_platform = "custom"
custom_base_url = ""
custom_api_key = ""
custom_model_name = ""
local_model_id_override = ""
local_model_port = 8000
vllm_gpu_util = 0.85
max_workers = 2
chunk_size_tokens = 1000
valid_ratio = 0.05
max_conversations = 0
seed = 42
per_device_train_batch_size = 1
gradient_accumulation_steps = 1
learning_rate = 1e-4
num_train_epochs = 1.0
lora_rank = 8
lora_alpha = 16
lora_dropout = 0.05
cutoff_len = 2048
warmup_ratio = 0.05
lr_scheduler_type = "cosine"
logging_steps = 5
save_steps = 100
eval_steps = 100
gradient_checkpointing = false
model_cache_dir = "models"
output_dir = "outputs"
gradio_port = 7860
share = false
stream_output = true
adapter_dir = ""
enable_memory = true
memory_backend = "bm25"
top_k_memory = 3
max_memory_chars = 1800
max_one_scene_chars = 600
prefer_target_present = true
exclude_narrator_only = true
sft_mode = "mixed"
style_data_ratio = 0.35
raft_data_ratio = 0.65
raft_include_distractors = true
raft_no_answer_ratio = 0.1
roleplay_mode = "in_character"
''',
        encoding="utf-8",
    )
    monkeypatch.chdir(repo)

    cfg = load_config("config.toml")

    assert cfg.enable_memory is True
    assert cfg.memory_backend == "bm25"
    assert cfg.profile_json.name == "shiri_qixia_profile.json"
    assert cfg.scene_memory_jsonl.name == "shiri_qixia_scenes.jsonl"
    assert cfg.raft_train_json.name == "shiri_qixia_raft_train.json"
```

- [ ] **Step 2: Run failing test**

Run: `pytest tests/test_config_memory.py -q`

Expected: fail because memory config fields do not exist.

- [ ] **Step 3: Implement config fields**

Add dataclass fields and derived paths:

```python
enable_memory: bool
memory_backend: str
top_k_memory: int
max_memory_chars: int
max_one_scene_chars: int
prefer_target_present: bool
exclude_narrator_only: bool
sft_mode: str
style_data_ratio: float
raft_data_ratio: float
raft_include_distractors: bool
raft_no_answer_ratio: float
roleplay_mode: str
profile_json: Path
scene_memory_jsonl: Path
memory_index_json: Path
raft_train_json: Path
raft_valid_json: Path
```

Derived paths:

```python
profile_json = repo_dir / "data" / "profiles" / f"{run_name}_profile.json"
scene_memory_jsonl = repo_dir / "data" / "memory" / f"{run_name}_scenes.jsonl"
memory_index_json = repo_dir / "data" / "memory" / f"{run_name}_bm25.json"
raft_train_json = sft_dir / f"{run_name}_raft_train.json"
raft_valid_json = sft_dir / f"{run_name}_raft_valid.json"
```

- [ ] **Step 4: Update example config**

Add the memory/RAFT settings from the architecture spec to `config.example.toml`.

- [ ] **Step 5: Run test**

Run: `pytest tests/test_config_memory.py -q`

Expected: pass.

## Task 2: Memory Builder

**Files:**
- Create: `src/memory.py`
- Test: `tests/test_memory_builder.py`

- [ ] **Step 1: Write memory builder tests**

```python
import json

from src.memory import build_default_profile, build_scene_memories, write_profile, read_profile


def test_build_scene_memories_from_dialogue_rows(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        "\n".join(
            [
                json.dumps({"chunk_id": 1, "dialogue_index": 0, "role": "旁白", "dialogue": "齐夏看见余念安。", "chunk_text": "齐夏看见余念安。她问他为何而来。"}),
                json.dumps({"chunk_id": 1, "dialogue_index": 1, "role": "齐夏", "dialogue": "我只是确认规则。", "chunk_text": "齐夏看见余念安。她问他为何而来。"}),
                json.dumps({"chunk_id": 2, "dialogue_index": 0, "role": "其他人", "dialogue": "这里没有齐夏。"}),
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
    assert scenes[1].target_role_present is False
    assert scenes[1].target_role_knows is False


def test_profile_roundtrip(tmp_path):
    profile = build_default_profile("齐夏", ["齐夏", "阿夏"], "十日终焉")
    path = tmp_path / "profile.json"

    write_profile(profile, path)
    loaded = read_profile(path)

    assert loaded.role == "齐夏"
    assert loaded.aliases == ["齐夏", "阿夏"]
    assert "Use memory for facts." in loaded.answer_rules
```

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/test_memory_builder.py -q`

Expected: fail because `src.memory` does not exist.

- [ ] **Step 3: Implement `src/memory.py`**

Implement:

```python
@dataclass
class CharacterProfile: ...

@dataclass
class SceneMemory: ...

def build_default_profile(role: str, aliases: list[str], novel_title: str) -> CharacterProfile: ...
def read_dialogue_jsonl(path: Path) -> list[dict]: ...
def build_scene_memories(raw_jsonl: Path, canonical_role: str, aliases: list[str], novel_title: str) -> list[SceneMemory]: ...
def write_scene_memories(scenes: list[SceneMemory], path: Path) -> None: ...
def read_scene_memories(path: Path) -> list[SceneMemory]: ...
def write_profile(profile: CharacterProfile, path: Path) -> None: ...
def read_profile(path: Path) -> CharacterProfile: ...
```

Initial heuristic:

- group raw dialogue rows by `chunk_id`
- scene text is `chunk_text` if present, otherwise joined dialogues
- `target_role_present` is true if any row role or text contains any alias
- `target_role_knows` equals `target_role_present` in v1
- summary can be a short deterministic string from the first 120 chars

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_memory_builder.py -q`

Expected: pass.

## Task 3: BM25 Retrieval

**Files:**
- Create: `src/retrieval.py`
- Test: `tests/test_retrieval.py`

- [ ] **Step 1: Write retrieval tests**

```python
from src.memory import SceneMemory
from src.retrieval import BM25MemoryIndex


def scene(scene_id, text, knows=True):
    return SceneMemory(
        scene_id=scene_id,
        chunk_id=1,
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


def test_bm25_retrieves_relevant_scene():
    index = BM25MemoryIndex.from_scenes(
        [
            scene("a", "齐夏和余念安讨论规则。"),
            scene("b", "孙悟空大闹天宫。"),
        ]
    )

    results = index.search("余念安是谁", top_k=1)

    assert results[0].scene.scene_id == "a"


def test_retrieval_can_exclude_unknown_scenes():
    index = BM25MemoryIndex.from_scenes(
        [
            scene("known", "齐夏知道余念安的线索。", knows=True),
            scene("unknown", "旁白透露余念安的秘密。", knows=False),
        ]
    )

    results = index.search("余念安秘密", top_k=2, exclude_narrator_only=True)

    assert [r.scene.scene_id for r in results] == ["known"]
```

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/test_retrieval.py -q`

Expected: fail because `src.retrieval` does not exist.

- [ ] **Step 3: Implement retrieval**

Implement a small BM25 index using stdlib only:

- tokenize Chinese by character bigrams plus ASCII words
- compute IDF
- score scenes
- support `exclude_narrator_only`
- save/load JSON index if needed

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_retrieval.py -q`

Expected: pass.

## Task 4: Shared Prompt Builder

**Files:**
- Create: `src/prompting.py`
- Test: `tests/test_prompting.py`

- [ ] **Step 1: Write prompt tests**

```python
from src.memory import CharacterProfile, SceneMemory
from src.prompting import build_roleplay_system_prompt


def test_prompt_contains_profile_and_memory_rules():
    profile = CharacterProfile(
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
    scene = SceneMemory(
        scene_id="s1",
        chunk_id=1,
        chapter="",
        text="齐夏确认了规则。",
        summary="齐夏确认规则。",
        characters=["齐夏"],
        target_role_present=True,
        target_role_knows=True,
        events=[],
        relations=[],
        quotes=["我只是确认规则。"],
        source={},
    )

    prompt = build_roleplay_system_prompt(profile, [scene])

    assert "你正在扮演《十日终焉》中的齐夏" in prompt
    assert "如果记忆片段包含答案" in prompt
    assert "齐夏确认规则" in prompt
    assert "不要续写 user/assistant" in prompt
```

- [ ] **Step 2: Run failing test**

Run: `pytest tests/test_prompting.py -q`

Expected: fail because `src.prompting` does not exist.

- [ ] **Step 3: Implement prompt builder**

Implement:

```python
def build_roleplay_system_prompt(profile: CharacterProfile, scenes: Sequence[SceneMemory], max_memory_chars: int = 1800, max_one_scene_chars: int = 600) -> str:
    ...
```

Use the exact memory protocol from the architecture spec.

- [ ] **Step 4: Run test**

Run: `pytest tests/test_prompting.py -q`

Expected: pass.

## Task 5: RAFT SFT Builder

**Files:**
- Create: `src/build_raft_sft.py`
- Test: `tests/test_build_raft_sft.py`

- [ ] **Step 1: Write RAFT SFT test**

```python
import json

from src.build_raft_sft import build_raft_sharegpt
from src.memory import CharacterProfile, SceneMemory


def test_build_raft_sharegpt_outputs_system_human_gpt():
    profile = CharacterProfile(
        role="齐夏",
        aliases=["齐夏"],
        novel_title="十日终焉",
        identity="",
        core_goals=[],
        personality=[],
        speech_style=[],
        relationships=[],
        knowledge_boundary="",
        answer_rules=[],
    )
    scenes = [
        SceneMemory(
            scene_id="chunk_000001",
            chunk_id=1,
            chapter="",
            text="余念安询问齐夏。",
            summary="余念安询问齐夏。",
            characters=["齐夏", "余念安"],
            target_role_present=True,
            target_role_knows=True,
            events=[],
            relations=[],
            quotes=[],
            source={},
        )
    ]
    raw_rows = [
        {"chunk_id": 1, "dialogue_index": 0, "role": "余念安", "dialogue": "你为什么来？"},
        {"chunk_id": 1, "dialogue_index": 1, "role": "齐夏", "dialogue": "我只是确认规则。"},
    ]

    samples = build_raft_sharegpt(raw_rows, scenes, profile, target_roles={"齐夏"}, max_memory_chars=1000)

    assert samples[0]["conversations"][0]["from"] == "system"
    assert samples[0]["conversations"][1] == {"from": "human", "value": "余念安：你为什么来？"}
    assert samples[0]["conversations"][2] == {"from": "gpt", "value": "我只是确认规则。"}
    assert samples[0]["metadata"]["oracle_scene_ids"] == ["chunk_000001"]
```

- [ ] **Step 2: Run failing test**

Run: `pytest tests/test_build_raft_sft.py -q`

Expected: fail because builder does not exist.

- [ ] **Step 3: Implement builder**

Implement:

```python
def build_raft_sharegpt(raw_rows: list[dict], scenes: list[SceneMemory], profile: CharacterProfile, target_roles: set[str], max_memory_chars: int) -> list[dict]:
    ...
```

V1 behavior:

- group rows by `chunk_id`
- skip chunks with no target role response
- create one sample per target response block
- leading non-target rows become human context
- target rows become gpt response
- system prompt uses the chunk's oracle scene

- [ ] **Step 4: Run test**

Run: `pytest tests/test_build_raft_sft.py -q`

Expected: pass.

## Task 6: CLI Integration and Smoke Commands

**Files:**
- Modify: `main.py`
- Modify: `src/build_sft.py`
- Modify or create: `src/build_memory.py`
- Test: `tests/test_cli_memory_steps.py`

- [ ] **Step 1: Add minimal CLI smoke test**

Test should run a tiny fake config through `build_memory` and `build_sft` functions directly, not full subprocess training.

- [ ] **Step 2: Implement `build_memory` step**

The step should:

- read `cfg.raw_jsonl`
- create default profile if missing
- create `cfg.scene_memory_jsonl`
- create `cfg.memory_index_json`
- write status marker

- [ ] **Step 3: Implement memory-aware build_sft**

When `cfg.sft_mode` is `raft` or `mixed`, write `cfg.raft_train_json` and `cfg.raft_valid_json`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_config_memory.py tests/test_memory_builder.py tests/test_retrieval.py tests/test_prompting.py tests/test_build_raft_sft.py tests/test_cli_memory_steps.py -q`

Expected: all pass.

## Task 7: Qixia Local Data Smoke

**Files:**
- No required code files if previous tasks are complete.

- [ ] **Step 1: Find existing Qixia raw dialogues**

Run:

```bash
find /Users/zdl/project/taletalk/taletalk-repo/data -iname '*qixia*' -o -iname '*shiri*'
```

Expected: identify existing raw dialogue JSONL or confirm missing.

- [ ] **Step 2: If raw dialogue exists, run memory build**

Run:

```bash
python main.py -c configs/shiri_qixia.toml -o build_memory build_sft
```

Expected: profile, scenes, memory index, and RAFT/mixed dataset are generated. No training runs.

- [ ] **Step 3: If raw dialogue is missing, run extraction with StepFun**

Before running, confirm `.env.stepfun` exists and contains API config:

```bash
ls -l .env.stepfun
```

Then run only extract/build_memory/build_sft:

```bash
set -a && source .env.stepfun && set +a
python main.py -c configs/shiri_qixia.toml -o extract build_memory build_sft
```

Expected: extraction uses cloud API, then memory and SFT artifacts are generated. No training runs.

- [ ] **Step 4: Validate generated datasets**

Run:

```bash
python scripts/validate_dataset.py data/shiri_qixia_raft_train.json
```

Expected: validation passes or reports only known unsupported `metadata` sidecar issue.

## Task 8: Commit and Report

**Files:**
- All implementation and tests from tasks above.

- [ ] **Step 1: Run focused tests**

Run:

```bash
pytest tests/test_config_memory.py tests/test_memory_builder.py tests/test_retrieval.py tests/test_prompting.py tests/test_build_raft_sft.py tests/test_cli_memory_steps.py -q
```

Expected: pass.

- [ ] **Step 2: Run no-training smoke**

Run either the existing-data or StepFun path from Task 7.

Expected: no training starts; artifacts are generated.

- [ ] **Step 3: Commit scoped changes**

Run:

```bash
git add docs/superpowers/plans/2026-06-29-memory-raft-qixia.md src tests config.example.toml main.py
git commit -m "feat: add memory-first RAFT data pipeline"
```

Do not stage unrelated user changes unless they are directly required by this feature.

## Self-Review

- Spec coverage: Phase 1 runtime memory and Phase 2 RAFT SFT generation are covered. Phase 3 cognitive boundary is represented by `target_role_knows` heuristics, not full LLM classification. Phase 4 richer retrieval is intentionally excluded.
- Placeholder scan: no TBD/TODO placeholders are used.
- Scope check: this plan intentionally does not run ROCm training. It produces testable software and data artifacts that can be trained later on cloud hardware.
