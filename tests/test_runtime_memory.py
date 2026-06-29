import json

from src.build_memory import run_build_memory
from src.config import load_config
from src.runtime_memory import build_runtime_system_prompt, load_runtime_memory


def test_runtime_system_prompt_uses_retrieved_memory(tmp_path, monkeypatch):
    _write_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    cfg = load_config("config.toml")
    run_build_memory(cfg)

    memory = load_runtime_memory(cfg)
    prompt = build_runtime_system_prompt(cfg, memory, "余念安是谁")

    assert "【记忆片段】" in prompt
    assert "余念安" in prompt
    assert "如果记忆片段包含答案" in prompt


def _write_project(root):
    (root / "src").mkdir()
    (root / "extract").mkdir()
    (root / "novels").mkdir()
    (root / "data" / "raw").mkdir(parents=True)
    (root / "novels" / "novel.txt").write_text("齐夏看见余念安。", encoding="utf-8")
    (root / "data" / "raw" / "shiri_qixia_dialogues.jsonl").write_text(
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
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "config.toml").write_text(
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
valid_ratio = 0.5
max_conversations = 0
seed = 42
enable_memory = true
memory_backend = "bm25"
top_k_memory = 3
max_memory_chars = 1800
max_one_scene_chars = 600
prefer_target_present = true
exclude_narrator_only = true
sft_mode = "mixed"
style_data_ratio = 1.0
raft_data_ratio = 1.0
raft_include_distractors = true
raft_no_answer_ratio = 0.1
roleplay_mode = "in_character"
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
''',
        encoding="utf-8",
    )
