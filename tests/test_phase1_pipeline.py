import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from src.config import load_config
from src.build_sft import _build_phase_one_sft
from src.eval_export import _build_eval_questions
from src.train import _build_dataset_info
from src.memory import CharacterProfile, SceneMemory, read_scene_memories, write_profile, write_scene_memories
from src.memory_pack import render_memory_pack
from src.one_pass_generation import _load_completed
from src.preprocess import build_scene_skeletons
from src.semantic_retrieval import embed_texts, rerank_items, write_embedding_artifacts


def test_preprocess_scenes_keep_offsets(tmp_path):
    novel = tmp_path / "novel.txt"
    novel.write_text("第一章\n\n齐夏看着门。\n\n余念安问他怎么办。\n\n齐夏说先确认规则。", encoding="utf-8")

    scenes, report = build_scene_skeletons(novel, None, max_chars=12, overlap_chars=0)

    assert len(scenes) >= 2
    assert report["scene_count"] == len(scenes)
    source = novel.read_text(encoding="utf-8")
    for scene in scenes:
        assert source[scene.source_start : scene.source_end].strip() == scene.raw_text


def test_preprocess_aligns_large_raw_dialogue_without_skip(tmp_path):
    novel = tmp_path / "novel.txt"
    raw = tmp_path / "raw.jsonl"
    novel.write_text("第一章\n\n齐夏说：先确认规则，再判断出口。", encoding="utf-8")
    rows = [
        {"chunk_id": index, "dialogue_index": 0, "role": "路人", "dialogue": f"无关台词{index:05d}"}
        for index in range(6000)
    ]
    rows.append({"chunk_id": 9999, "dialogue_index": 0, "role": "齐夏", "dialogue": "先确认规则，再判断出口。"})
    raw.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

    scenes, report = build_scene_skeletons(novel, raw, max_chars=100, overlap_chars=0)

    assert report["alignment_strategy"] == "keyed_exact_dialogue_match"
    assert scenes[0].dialogue_alignment == "matched"
    assert scenes[0].dialogues == [{"role": "齐夏", "dialogue": "先确认规则，再判断出口。"}]


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


def test_cloud_embedding_and_reranker_clients(tmp_path, monkeypatch):
    _write_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    cfg = load_config("config.toml")

    server, thread = _start_fake_retrieval_server()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        cfg.embedding_backend = "cloud"
        cfg.embedding_base_url = base_url
        cfg.embedding_api_key = "test-key"
        cfg.embedding_model = "fake-embedding"
        cfg.reranker_backend = "cloud"
        cfg.reranker_base_url = base_url
        cfg.reranker_api_key = "test-key"
        cfg.reranker_model = "fake-reranker"
        cfg.use_reranker = True

        vectors = embed_texts(cfg, ["余念安", "规则"])
        ranked = rerank_items(
            cfg,
            "规则",
            [
                {"scene_id": "a", "text": "余念安"},
                {"scene_id": "b", "text": "规则"},
            ],
            top_k=2,
        )

        assert vectors == [[1.0, 0.0], [0.0, 1.0]]
        assert [item["scene_id"] for item in ranked] == ["b", "a"]
        assert "/embeddings" in server.paths
        assert "/rerank" in server.paths

        server.paths.clear()
        cfg.reranker_provider = "parallel_cloud"
        ranked = rerank_items(
            cfg,
            "规则",
            [
                {"scene_id": "a", "text": "余念安"},
                {"scene_id": "b", "text": "规则"},
            ],
            top_k=2,
        )
        assert [item["scene_id"] for item in ranked] == ["b", "a"]
        assert "/p002/rerank" in server.paths
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_real_ai_generation_ignores_heuristic_checkpoints(tmp_path, monkeypatch):
    _write_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    cfg = load_config("config.toml")
    cfg.teacher_backend = "stepfun"
    cfg.generation_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = cfg.generation_checkpoint_dir / "batch_scene_000001.json"
    checkpoint.write_text(
        json.dumps(
            {
                "version": "taletalk-one-pass-v1",
                "scene_memories": [
                    {
                        "scene_id": "scene_000001",
                        "source_risks": ["heuristic_fallback"],
                    }
                ],
                "candidate_samples": [
                    {
                        "sample_type": "grounded_fact",
                        "question": "这一段里你经历了什么？",
                        "answer": "先别急着下结论。",
                        "source_scene_ids": ["scene_000001"],
                        "risk_tags": ["heuristic_fallback"],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    completed = _load_completed(cfg.generation_checkpoint_dir, allow_heuristic=False)

    assert "scene_000001" not in completed


def test_memory_aware_mixed_sft_includes_style_samples(tmp_path, monkeypatch):
    _write_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    cfg = load_config("config.toml")
    cfg.sft_mode = "mixed"
    cfg.style_data_ratio = 1.0
    cfg.raft_data_ratio = 1.0

    write_profile(
        CharacterProfile(
            role="齐夏",
            aliases=["齐夏"],
            novel_title="测试小说",
            identity="测试身份",
            core_goals=[],
            personality=[],
            speech_style=[],
            relationships=[],
            knowledge_boundary="只回答自己知道的事。",
            answer_rules=[],
        ),
        cfg.profile_json,
    )
    write_scene_memories(
        [
            SceneMemory(
                scene_id="scene_000001",
                chunk_id=1,
                chapter="第一章",
                text="齐夏确认规则。",
                summary="齐夏确认规则。",
                characters=["齐夏", "乔家劲"],
                target_role_present=True,
                target_role_knows=True,
                events=["齐夏确认规则"],
                relations=[],
                quotes=["先确认规则。"],
                source={"novel_title": "测试小说"},
                raw_text="齐夏确认规则。",
                knowledge_level="first_hand",
            )
        ],
        cfg.scene_memory_jsonl,
    )
    cfg.raw_jsonl.parent.mkdir(parents=True, exist_ok=True)
    cfg.raw_jsonl.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in [
                {"chunk_id": 1, "dialogue_index": 0, "role": "乔家劲", "dialogue": "怎么办？"},
                {"chunk_id": 1, "dialogue_index": 1, "role": "齐夏", "dialogue": "先确认规则。"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg.raft_candidates_raw_jsonl.parent.mkdir(parents=True, exist_ok=True)
    cfg.raft_candidates_raw_jsonl.write_text(
        json.dumps(
            {
                "id": "cand_1",
                "sample_type": "grounded_fact",
                "question": "出口在哪里？",
                "answer": "我先确认规则。",
                "source_scene_ids": ["scene_000001"],
                "knowledge_level": "first_hand",
                "answer_policy": "answer_from_memory",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    rows = _build_phase_one_sft(cfg)
    sample_types = {row["metadata"]["sample_type"] for row in rows}

    assert {"grounded_fact", "style_dialogue"}.issubset(sample_types)


def test_phase_two_sft_includes_required_training_types(tmp_path, monkeypatch):
    _write_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    cfg = load_config("config.toml")
    cfg.sft_mode = "mixed"
    cfg.style_data_ratio = 1.0
    cfg.raft_data_ratio = 1.0
    _write_minimal_memory_sft_inputs(cfg)

    rows = _build_phase_one_sft(cfg)
    sample_types = {row["metadata"]["sample_type"] for row in rows}

    assert {
        "style_dialogue",
        "grounded_fact",
        "relationship",
        "motivation",
        "false_premise",
        "boundary_unknown",
    }.issubset(sample_types)


def test_eval_questions_cover_phase_two_categories():
    candidates = [
        {"id": f"{category}_{index}", "question": f"{category}?", "sample_type": category, "source_scene_ids": []}
        for category in ["grounded_fact", "relationship", "motivation", "false_premise", "boundary_unknown"]
        for index in range(2)
    ]

    questions = _build_eval_questions(candidates, limit=5)
    categories = {row["category"] for row in questions}

    assert categories == {"grounded_fact", "relationship", "motivation", "false_premise", "boundary_unknown"}


def test_train_dataset_info_uses_sharegpt_format(tmp_path, monkeypatch):
    _write_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    cfg = load_config("config.toml")

    info = _build_dataset_info(cfg)

    assert info["taletalk_custom"]["formatting"] == "sharegpt"
    assert info["taletalk_custom"]["tags"]["role_tag"] == "from"
    assert info["taletalk_custom"]["tags"]["user_tag"] == "human"
    assert info["taletalk_custom_valid"]["file_name"] == "phase1_test_chat_valid.json"


def test_training_memory_pack_can_skip_reranker(tmp_path, monkeypatch):
    _write_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    cfg = load_config("config.toml")
    cfg.sft_mode = "mixed"
    cfg.retrieval_mode = "hybrid"
    cfg.training_retrieval_mode = "bm25"
    cfg.embedding_backend = "cloud"
    cfg.embedding_base_url = "https://example.invalid/v1"
    cfg.embedding_api_key = "test-key"
    cfg.embedding_model = "test-embedding"
    cfg.use_reranker = True
    cfg.reranker_backend = "cloud"
    cfg.reranker_base_url = "https://example.invalid/v1"
    cfg.reranker_api_key = "test-key"
    cfg.reranker_model = "test-reranker"
    cfg.training_use_reranker = False
    _write_minimal_memory_sft_inputs(cfg)

    cfg.embedding_backend = "local"
    write_embedding_artifacts(cfg, read_scene_memories(cfg.scene_memory_jsonl))
    cfg.embedding_backend = "cloud"

    def fail_embedding(*args, **kwargs):
        raise AssertionError("training build should not call cloud embedding per query")

    def fail_rerank(*args, **kwargs):
        raise AssertionError("training build should not call reranker")

    monkeypatch.setattr("src.build_sft.embed_texts", fail_embedding)
    monkeypatch.setattr("src.build_sft.rerank_items", fail_rerank)

    rows = _build_phase_one_sft(cfg)

    assert rows


def test_training_semantic_query_embeddings_are_batched(tmp_path, monkeypatch):
    _write_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    cfg = load_config("config.toml")
    cfg.sft_mode = "mixed"
    cfg.training_retrieval_mode = "hybrid"
    cfg.embedding_backend = "cloud"
    cfg.embedding_base_url = "https://example.invalid/v1"
    cfg.embedding_api_key = "test-key"
    cfg.embedding_model = "test-embedding"
    cfg.use_reranker = False
    cfg.training_use_reranker = False
    _write_minimal_memory_sft_inputs(cfg)
    _write_candidates(
        cfg,
        [
            {
                "id": "cand_1",
                "sample_type": "grounded_fact",
                "question": "出口在哪里？",
                "answer": "我先确认规则。",
                "source_scene_ids": ["scene_000001"],
                "knowledge_level": "first_hand",
                "answer_policy": "answer_from_memory",
            },
            {
                "id": "cand_2",
                "sample_type": "grounded_fact",
                "question": "规则怎么判断？",
                "answer": "先看它是不是在替你做选择。",
                "source_scene_ids": ["scene_000002"],
                "knowledge_level": "first_hand",
                "answer_policy": "answer_from_memory",
            },
        ],
    )

    cfg.embedding_backend = "local"
    write_embedding_artifacts(cfg, read_scene_memories(cfg.scene_memory_jsonl))
    cfg.embedding_backend = "cloud"
    calls = []

    def fake_embed_texts(_config, texts):
        calls.append(list(texts))
        return [[1.0, 0.0] if "出口" in text else [0.0, 1.0] for text in texts]

    monkeypatch.setattr("src.build_sft.embed_texts", fake_embed_texts)

    rows = _build_phase_one_sft(cfg)

    assert rows
    assert len(calls) == 1
    assert calls[0][:2] == ["出口在哪里？", "规则怎么判断？"]


def _write_candidates(cfg, candidates):
    cfg.raft_candidates_raw_jsonl.parent.mkdir(parents=True, exist_ok=True)
    cfg.raft_candidates_raw_jsonl.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in candidates) + "\n",
        encoding="utf-8",
    )


def _write_minimal_memory_sft_inputs(cfg):
    write_profile(
        CharacterProfile(
            role="齐夏",
            aliases=["齐夏"],
            novel_title="测试小说",
            identity="测试身份",
            core_goals=[],
            personality=[],
            speech_style=[],
            relationships=[],
            knowledge_boundary="只回答自己知道的事。",
            answer_rules=[],
        ),
        cfg.profile_json,
    )
    write_scene_memories(
        [
            SceneMemory(
                scene_id="scene_000001",
                chunk_id=1,
                chapter="第一章",
                text="齐夏确认规则。",
                summary="齐夏确认规则。",
                characters=["齐夏", "乔家劲"],
                target_role_present=True,
                target_role_knows=True,
                events=["齐夏确认规则"],
                relations=[],
                quotes=["先确认规则。"],
                source={"novel_title": "测试小说"},
                raw_text="齐夏确认规则。",
                knowledge_level="first_hand",
            ),
            SceneMemory(
                scene_id="scene_000002",
                chunk_id=2,
                chapter="第一章",
                text="齐夏观察出口。",
                summary="齐夏观察出口。",
                characters=["齐夏"],
                target_role_present=True,
                target_role_knows=True,
                events=["齐夏观察出口"],
                relations=[],
                quotes=[],
                source={"novel_title": "测试小说"},
                raw_text="齐夏观察出口。",
                knowledge_level="first_hand",
            ),
        ],
        cfg.scene_memory_jsonl,
    )
    cfg.raw_jsonl.parent.mkdir(parents=True, exist_ok=True)
    cfg.raw_jsonl.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in [
                {"chunk_id": 1, "dialogue_index": 0, "role": "乔家劲", "dialogue": "怎么办？"},
                {"chunk_id": 1, "dialogue_index": 1, "role": "齐夏", "dialogue": "先确认规则。"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_candidates(
        cfg,
        [
            {
                "id": "cand_1",
                "sample_type": "grounded_fact",
                "question": "出口在哪里？",
                "answer": "我先确认规则。",
                "source_scene_ids": ["scene_000001"],
                "knowledge_level": "first_hand",
                "answer_policy": "answer_from_memory",
            }
        ],
    )


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
training_retrieval_mode = "bm25"
embedding_backend = "local"
embedding_provider = "openai_compatible"
embedding_base_url = ""
embedding_api_key = ""
embedding_model = "BAAI/bge-m3"
embedding_batch_size = 64
embedding_dimensions = 64
reranker_backend = "local"
reranker_provider = "openai_compatible"
reranker_base_url = ""
reranker_api_key = ""
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


def _start_fake_retrieval_server():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.server.paths.append(self.path)
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if self.path == "/embeddings":
                vectors = []
                for text in payload["input"]:
                    vectors.append([1.0, 0.0] if "余念安" in text else [0.0, 1.0])
                body = {"data": [{"embedding": vector} for vector in vectors]}
            elif self.path in {"/rerank", "/p002/rerank"}:
                rows = []
                query = payload["query"]
                for index, document in enumerate(payload["documents"]):
                    rows.append({"index": index, "relevance_score": 10.0 if query in document else 1.0})
                body = {"results": rows}
            else:
                body = {"error": "not found"}
            data = json.dumps(body).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    server.paths = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread
