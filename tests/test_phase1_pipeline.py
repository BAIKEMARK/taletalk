import json

from src.config import load_config
from src.memory_pack import render_memory_pack
from src.preprocess import build_scene_skeletons


def test_preprocess_scenes_keep_offsets(tmp_path):
    novel = tmp_path / "novel.txt"
    novel.write_text("第一章\n\n齐夏看着门。\n\n余念安问他怎么办。\n\n齐夏说先确认规则。", encoding="utf-8")

    scenes, report = build_scene_skeletons(novel, None, max_chars=12, overlap_chars=0)

    assert len(scenes) >= 2
    assert report["scene_count"] == len(scenes)
    source = novel.read_text(encoding="utf-8")
    for scene in scenes:
        assert source[scene.source_start : scene.source_end].strip() == scene.raw_text


def test_memory_pack_render_protocol():
    rendered = render_memory_pack(
        [
            {
                "scene_id": "scene_000001",
                "knowledge_level": "first_hand",
                "text": "齐夏确认规则。",
            }
        ]
    )

    assert "【记忆片段 1｜齐夏亲历】" in rendered
    assert "齐夏确认规则" in rendered


def test_phase1_cli_steps_smoke(tmp_path, monkeypatch):
    _write_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    cfg = load_config("config.toml")

    from src.build_memory import run_build_memory
    from src.build_sft import run_build_sft
    from src.eval_export import run_eval, run_export_role

    run_build_memory(cfg)
    run_build_sft(cfg)
    run_eval(cfg)
    run_export_role(cfg)

    assert cfg.raw_scene_jsonl.exists()
    assert cfg.scene_build_report_json.exists()
    assert cfg.scene_memory_jsonl.exists()
    assert cfg.profile_observations_jsonl.exists()
    assert cfg.embedding_npy.exists()
    assert cfg.embedding_meta_jsonl.exists()
    assert cfg.raft_candidates_raw_jsonl.exists()
    assert cfg.raft_memory_packs_jsonl.exists()
    assert cfg.train_json.exists()
    assert cfg.eval_report_md.exists()
    assert (cfg.role_package_dir / "manifest.json").exists()

    first_scene = json.loads(cfg.scene_memory_jsonl.read_text(encoding="utf-8").splitlines()[0])
    assert first_scene["raw_text"]
    assert first_scene["characters"]
    assert "quotes" in first_scene
    assert first_scene["knowledge_level"] in {"first_hand", "heard_or_inferred", "narrator_only", "uncertain"}

    train_rows = json.loads(cfg.train_json.read_text(encoding="utf-8"))
    assert train_rows
    assert "【记忆片段" in train_rows[0]["system"]


def _write_project(root):
    (root / "src").mkdir()
    (root / "extract").mkdir()
    (root / "novels").mkdir()
    (root / "novels" / "novel.txt").write_text(
        "第一章\n\n齐夏看见余念安。\n\n余念安问：你为什么来？\n\n齐夏说：我只是确认规则。\n\n乔家劲问他怎么办。\n\n齐夏说：别把恐惧当判断。",
        encoding="utf-8",
    )
    (root / "config.toml").write_text(
        '''
novel_txt = "novels/novel.txt"
target_role = "齐夏"
novel_title = "测试小说"
run_name = "phase1_test"
model_choice = "qwen3_5_9b"
model_ids = { "qwen3_5_9b" = "Qwen/Qwen3.5-9B" }
extraction_backend = "cloud_api"
llm_platform = "custom"
custom_base_url = ""
custom_api_key = ""
custom_model_name = "mock"
local_model_id_override = ""
local_model_port = 8000
vllm_gpu_util = 0.85
max_workers = 2
chunk_size_tokens = 1000
valid_ratio = 0.5
max_conversations = 0
seed = 42
enable_memory = true
generation_mode = "one_pass"
ai_passes = 1
ai_audit_mode = "rules"
scene_max_chars = 40
scene_overlap_chars = 0
teacher_backend = "mock"
teacher_model = "mock"
teacher_batch_size = 2
teacher_concurrency = 1
memory_backend = "bm25"
retrieval_mode = "bm25"
embedding_model = "BAAI/bge-m3"
reranker_model = "BAAI/bge-reranker-v2-m3"
use_reranker = false
bm25_top_k = 20
embedding_top_k = 20
rerank_top_k = 5
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
target_train_samples = 100
eval_question_count = 20
export_mode = "private_full"
per_device_train_batch_size = 1
gradient_accumulation_steps = 1
learning_rate = 1e-4
num_train_epochs = 1.0
lora_rank = 8
lora_alpha = 16
lora_dropout = 0.05
cutoff_len = 1024
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
