# TaleTalk 齐夏低成本记忆 + RAFT 总计划

> **给后续执行者的要求：** 本文档是主入口。先按这里的开发顺序实现，再到模块文档看细节。默认路径只允许 1 轮 AI；最多 2 轮 AI，第二轮只属于增强模式。不要把本文档重新扩写成全量多轮 AI 流程。

**目标：** 把一本小说 TXT 和可选 raw dialogue 转成可训练、可推理、可评测、可导出的角色包。事实留在记忆库，LoRA 学角色口吻和使用记忆的协议。

**主入口文件：** `docs/superpowers/plans/2026-06-29-memory-raft-qixia.md`

**模块文档：**

- `docs/architecture/modules/01-data-preprocess.md`
- `docs/architecture/modules/02-one-pass-ai-generation.md`
- `docs/architecture/modules/03-memory-retrieval-pack.md`
- `docs/architecture/modules/04-sft-runtime-eval-export.md`

---

## 1. 第一性原理

角色系统要解决四件事：

- **像本人：** 语气、判断方式、价值观、回应节奏像目标角色。
- **有记忆：** 能基于小说证据回答经历、关系、动机。
- **有边界：** 不使用角色本人不该知道的旁白或读者视角知识。
- **能对话：** 能处理新问题、追问、错误前提和资料不足。

因此职责必须分开：

- **程序前处理：** 章节识别、场景切分、offset、raw dialogue 对齐。
- **AI 一轮生成：** 从 scene batch 生成结构化记忆、候选问答、profile observations。
- **检索与组包：** 建索引、召回、认知边界过滤、memory pack 渲染。
- **LoRA：** 学口吻和记忆使用协议，不承担整本小说记忆。
- **评测与导出：** 用固定评测集和角色包保证结果可复现、可迁移。

核心原则：

```text
AI 不做机械处理。
AI 只做必要的语义理解、角色化表达、少量高风险质量判断。
默认用户路径必须低成本可跑通。
```

## 2. 默认总流程

默认流程只跑 1 轮 AI：

```text
小说 TXT + 可选 raw dialogue
-> 程序切章节/场景/offset/对齐
-> 一轮 AI 批处理生成 scene memory + 候选问答 + profile observations
-> 程序建 BM25/embedding 索引
-> 程序构造训练 memory pack
-> 规则过滤候选样本
-> ShareGPT mixed SFT 数据
-> runtime agent / eval report / role package
```

增强流程最多增加第 2 轮 AI：

```text
默认流程产物
-> 程序找出重要关系线、动机线、跨场景事件、高风险样本
-> AI 第二轮只处理这些子集
-> 生成 aggregate memory 或修复关键样本
```

禁止默认扩展成：

- 全量 scene 抽取后再全量 aggregate。
- 全量 question generation 后再全量 answer generation。
- 对所有样本调用 AI verifier。
- 每个模块都单独调用 AI。

## 3. 配置默认值

```toml
# AI 生成策略
generation_mode = "one_pass"  # one_pass / enhanced
ai_passes = 1                 # 默认 1，最大 2
ai_audit_mode = "rules"       # rules / sampled / risk_only
enhanced_topics = ["relationship", "motivation", "cross_scene_event"]

# 场景切分
scene_max_chars = 1800
scene_overlap_chars = 250
scene_min_dialogues = 0

# 检索
retrieval_mode = "hybrid"  # bm25 / embedding / hybrid
embedding_model = "BAAI/bge-m3"
reranker_model = "BAAI/bge-reranker-v2-m3"
use_reranker = true
bm25_top_k = 20
embedding_top_k = 20
rerank_top_k = 5

# 一轮 AI 批处理
teacher_backend = "stepfun"
teacher_model = "step-3.7-flash"
teacher_batch_size = 5
teacher_concurrency = 6
generation_checkpoint_dir = "cache/raft_generation"

# 数据集混合比例
style_ratio = 0.25
grounded_ratio = 0.35
relationship_ratio = 0.20
multi_turn_ratio = 0.10
false_premise_ratio = 0.05
boundary_ratio = 0.05
target_train_samples = 3000

# 评测与导出
eval_question_count = 120
eval_report_dir = "reports"
role_package_dir = "dist/roles"
export_mode = "private_full"  # private_full / public_redacted
```

执行者必须把 `ai_passes > 2` 视为配置错误。

## 4. 模块开发顺序

### 模块 1：数据前处理

细节见 `docs/architecture/modules/01-data-preprocess.md`。

目标：

- 从小说 TXT 程序化生成 scene skeleton。
- 保留 `source_start/source_end`，能回到原文切片。
- 可选对齐 `data/raw/shiri_qixia_dialogues.jsonl`，辅助识别说话人和真实台词。

输出：

```text
data/memory/shiri_qixia_scenes.raw.jsonl
data/memory/shiri_qixia_scene_build_report.json
```

验收：

- 不调用 AI。
- 对齐失败不丢场景。
- 报告包含场景数量、平均长度、对齐成功率、失败数量。

### 模块 2：一轮 AI 数据生成

细节见 `docs/architecture/modules/02-one-pass-ai-generation.md`。

目标：

- 一次 AI 批处理每个 scene 或 scene batch。
- 同时产出 scene memory、候选训练问答、profile observations。
- 支持断点续跑和失败样本单独记录。

输出：

```text
data/memory/shiri_qixia_scenes.jsonl
data/profiles/shiri_qixia_profile.observations.jsonl
data/raft/shiri_qixia_candidates.raw.jsonl
cache/raft_generation/failed.jsonl
```

验收：

- 默认只需要这一轮 AI。
- 每条候选样本带 `source_scene_ids`。
- 不把问题生成、回答生成、审核器写成默认三段 AI 流程。

### 模块 3：记忆检索与 memory pack

细节见 `docs/architecture/modules/03-memory-retrieval-pack.md`。

目标：

- 程序建 BM25/embedding 索引。
- 训练时 memory pack 用 gold evidence + supporting context + hard distractor 模拟运行时噪声。
- 运行时 memory pack 实时检索、过滤、重排、压缩。
- 训练和运行时使用同一个 `render_memory_pack` 协议。

输出：

```text
data/memory/shiri_qixia_bm25.json
data/memory/shiri_qixia_embeddings.npy
data/memory/shiri_qixia_embedding_meta.jsonl
data/raft/shiri_qixia_memory_packs.jsonl
```

验收：

- 有证据样本至少包含 1 个 gold evidence。
- 无证据/认知边界样本不能包含直接答案。
- 运行时不把记忆混进普通用户消息。

### 模块 4：SFT、运行时、评测、角色包

细节见 `docs/architecture/modules/04-sft-runtime-eval-export.md`。

目标：

- 把通过过滤的候选样本转成 ShareGPT mixed SFT。
- 运行时 agent 使用同一套 profile + memory pack 协议。
- 生成评测报告。
- 导出 TaleTalk 自己的角色包。

输出：

```text
data/shiri_qixia_chat_train.json
data/shiri_qixia_chat_valid.json
data/eval/shiri_qixia_eval_questions.jsonl
reports/shiri_qixia_eval_report.md
dist/roles/shiri_qixia/manifest.json
```

验收：

- SFT 系统提示词和 runtime prompt 使用同一构造函数。
- 报告包含失败样例，不只给平均分。
- `public_redacted` 不默认保留整段小说原文。

## 5. 训练数据设计

mixed SFT 默认比例：

```text
风格模仿：20%-30%
有证据的事实/事件：30%-35%
人物关系/动机：20%-25%
多轮追问：10%-15%
错误前提纠正：8%-10%
无证据/认知边界：7%-10%
```

每条候选样本至少包含：

```json
{
  "id": "shiri_qixia_candidate_000001",
  "sample_type": "motivation",
  "question": "你当时为什么判断那条规则是陷阱？",
  "answer": "因为它太像答案了。真正想让人活下去的规则，不会急着替你做选择。",
  "source_scene_ids": ["scene_000032"],
  "knowledge_level": "first_hand",
  "generation_mode": "one_pass"
}
```

最终训练样本使用 `system + conversations`；`metadata` 只用于审计和评测。

## 6. 过滤策略

默认过滤是程序规则，不是逐样本 AI 审核。

必须规则过滤：

- 续写 `user/assistant`。
- 出现“作为 AI”“根据资料”等非角色表达。
- 回答为空或明显截断。
- 大段复制原文。
- 有证据样本缺少 `source_scene_ids`。
- 无证据样本却带直接答案场景。

可选 AI 审核只在以下情况触发：

- `generation_mode = "enhanced"`。
- `ai_audit_mode = "sampled"` 时抽样质检。
- `ai_audit_mode = "risk_only"` 时只审高风险样本。

AI 审核不得扩展成默认第三轮。

## 7. 增强模式边界

增强模式只解决默认一轮难处理的问题：

- 跨很多场景的人物关系。
- 长期动机和价值判断。
- 关键事件线。
- 高风险样本修复。

增强模式输出可选 aggregate memory：

```json
{
  "memory_id": "rel_qixia_yuniannian_001",
  "memory_type": "relationship_arc",
  "summary": "余念安是齐夏重要的情感牵引之一，影响他的生存目标和选择。",
  "supporting_scene_ids": ["scene_000088", "scene_000241"],
  "knowledge_level": "first_hand"
}
```

默认流程不得依赖 aggregate memory 才能跑通。

## 8. 命令路径

本地只验证数据链路，不运行 ROCm 训练：

```bash
python3 main.py -c configs/shiri_qixia.toml -r build_memory build_sft -o build_memory build_sft
```

如果需要重新用 StepFun 生成：

```bash
set -a
source .env.stepfun
set +a

python3 main.py -c configs/shiri_qixia.toml -r extract build_memory build_sft -o extract build_memory build_sft
```

验证：

```bash
python3 -m pytest -q
python3 scripts/validate_dataset.py data/shiri_qixia_chat_train.json
python3 main.py -c configs/shiri_qixia.toml -r eval export_role -o eval export_role
```

不运行：

```text
train
infer with loaded model
ROCm training
```

## 9. 阶段验收标准

第一阶段完成标准：

- 默认一轮 AI 能生成 scene memory、候选问答、profile observations。
- 场景记忆来自小说原文，不只是台词。
- 每条场景能看到 `raw_text`、`characters`、`quotes`、`knowledge_level`。
- 至少生成 100 条通过规则过滤的训练样本作为冒烟验证。
- 训练 memory pack 和 runtime memory pack 使用同一个渲染协议。
- 至少生成一份检索/数据质量评测报告。
- 至少能导出一个 `private_full` 角色包。
- 本地测试通过。
- 不运行训练。

第二阶段完成标准：

- 生成 1000-3000 条齐夏 mixed SFT 样本。
- 增强模式只处理重要关系/动机/高风险样本。
- 有固定评测集，覆盖事实、关系、动机、错误前提、边界、不知道。
- 有训练前后对比评测报告。
- 角色包能在云端直接加载进行推理或继续训练。
- 云端可直接训练 mixed 主训练集。

## 10. 明确不做

- 不把整本小说事实强行塞进 LoRA。
- 不把全量多轮 AI 流程设为默认路径。
- 不默认逐样本 AI 审核。
- 不让问题生成、回答生成、审核器变成默认三轮 API。
- 不为了迁就外部格式牺牲 TaleTalk 自己的运行协议。
- 不在本地跑 ROCm 训练。

## 11. 文档自检命令

```bash
rg -n "^## " docs/superpowers/plans/2026-06-29-memory-raft-qixia.md
git diff --check -- docs/superpowers/plans/2026-06-29-memory-raft-qixia.md docs/architecture/modules
```

另做一次方向残留检查：搜索旧版外部定位词、多轮默认化词、逐样本审核默认化词，预期无输出。不要把这些词写进正文，避免自匹配。
