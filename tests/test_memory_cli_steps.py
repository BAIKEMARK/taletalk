import json

from src.build_memory import run_build_memory
from src.build_sft import run_build_sft
from src.config import load_config


def test_config_memory_fields_and_steps_smoke(tmp_path, monkeypatch):
    _write_minimal_project(tmp_path)
    monkeypatch.chdir(tmp_path)

    cfg = load_config("config.toml")

    assert cfg.enable_memory is True
    assert cfg.sft_mode == "mixed"
    assert cfg.profile_json.name == "shiri_qixia_profile.json"
    assert cfg.scene_memory_jsonl.name == "shiri_qixia_scenes.jsonl"
    assert cfg.raft_train_json.name == "shiri_qixia_raft_train.json"

    run_build_memory(cfg)
    run_build_sft(cfg)

    assert cfg.profile_json.exists()
    assert cfg.scene_memory_jsonl.exists()
    assert cfg.memory_index_json.exists()
    assert cfg.train_json.exists()
    assert cfg.valid_json.exists()
    assert cfg.raft_train_json.exists()

    train_rows = json.loads(cfg.train_json.read_text(encoding="utf-8"))
    valid_rows = json.loads(cfg.valid_json.read_text(encoding="utf-8"))
    assert train_rows
    assert any("【记忆片段】" in row.get("system", "") for row in train_rows + valid_rows)


def _write_minimal_project(root):
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
                json.dumps(
                    {"chunk_id": 2, "dialogue_index": 0, "role": "路人", "dialogue": "没有目标角色。"},
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
