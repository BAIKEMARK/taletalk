# 齐夏完整场景记忆 + RAFT 数据生成实施计划

> **给后续执行者的要求：** 按任务顺序实现。每个任务都要先写可验证测试或冒烟命令，再写实现。不要运行 ROCm 训练。本计划取代旧的“台词记忆 v1”计划。

**目标：** 为《十日终焉》齐夏构建一条更符合第一性原理的训练数据管线：从原始小说文本生成完整场景记忆，用混合检索组织证据，用强模型生成并审核 RAFT 数据，让后续 LoRA 学会“像齐夏一样使用记忆回答”。

**核心判断：** 旧的 `原始对话 -> 台词记忆 -> BM25 -> RAFT` 只能验证管线，不是最终方向。真正的知识主体应来自原始小说全文；原始对话 JSONL 是辅助对齐和提取齐夏原话的结构化标注。

**技术栈：** Python 3.11+、现有 TOML 配置、StepFun/OpenAI 兼容 API、pytest、numpy、sentence-transformers 或可替换语义向量后端、可选重排模型。训练仍由云端 LLaMA Factory 执行，本计划不跑训练。

---

## 0. 第一性原理

我们要训练的不是“会复述台词的模型”，而是“一个能扮演小说角色的系统”。

角色系统需要同时满足：

- **像本人：** 语气、判断方式、价值观、回应节奏像齐夏。
- **有记忆：** 能基于小说证据回答自己的经历、关系、动机。
- **有边界：** 不使用齐夏本人不该知道的旁白或读者视角知识。
- **能对话：** 能处理用户新问题、追问、错误前提和资料不足场景。

因此职责必须分开：

- **记忆库：** 保存小说事实和证据。
- **角色设定卡：** 保存角色身份、性格、说话规则和认知边界。
- **检索器：** 在用户问题下找出可支撑回答的记忆。
- **LoRA：** 学习齐夏的表达、判断方式，以及如何按协议使用记忆。
- **审核器：** 在数据生成阶段过滤无证据、跑偏、复述、越界样本。

## 1. 最终目标架构

```text
小说原文
+ 原始对话 jsonl
-> 完整场景记忆
-> 角色设定卡 / 关系状态 / 认知边界标签
-> BM25 精确召回 + 语义向量召回 + 重排
-> 教师模型生成 RAFT 样本
-> 审核器过滤后的数据集
-> 后续在 ROCm 上训练 LoRA
-> 运行时智能体使用同一套记忆协议
```

运行时智能体：

```text
用户问题
-> 查询分析器提取实体/别名/事件/追问上下文
-> BM25 召回专名、原话、规则名
-> 语义向量召回动机、因果、情绪等语义相关片段
-> 合并去重
-> 认知边界过滤器去掉齐夏不该知道的证据
-> 重排模型选 top 3-5
-> 提示词组装器拼角色设定 + 记忆片段
-> LoRA 生成器生成齐夏口吻回答
-> 可选审核器检查回答是否编造
```

## 2. 训练数据总体设计

训练集不是单一 RAFT 数据，而是混合数据。

推荐比例：

```text
风格模仿：20%-30%
有证据的事实/事件 RAFT：30%-35%
人物关系/动机 RAFT：20%-25%
多轮追问：10%-15%
错误前提纠正：8%-10%
无证据/认知边界：7%-10%
```

各类数据作用：

- **风格模仿：** 学齐夏真实台词风格、节奏、推理语气。
- **有证据 RAFT：** 学会基于记忆片段回答事实和事件问题。
- **人物关系/动机：** 学人物关系、动机、长期目标和价值判断。
- **多轮追问：** 学会连续追问里保持上下文和角色状态。
- **错误前提纠正：** 学会纠正用户错误前提，不顺着编。
- **无证据/边界：** 学会资料不足时克制，不编造小说事实。

每条高质量 RAFT 样本应该包含：

```json
{
  "id": "shiri_qixia_raft_000001",
  "system": "角色协议 + 角色设定 + 记忆片段",
  "conversations": [
    {"from": "human", "value": "你为什么判断那条规则是陷阱？"},
    {"from": "gpt", "value": "因为它太像答案了。真正想让人活下去的规则，不会急着替你做选择。"}
  ],
  "metadata": {
    "sample_type": "motivation",
    "oracle_scene_ids": ["scene_000032"],
    "retrieved_scene_ids": ["scene_000032", "scene_000041"],
    "distractor_scene_ids": ["scene_000877"],
    "knowledge_level": "first_hand",
    "teacher_model": "step-xxx",
    "verified": true
  }
}
```

训练时喂给模型的是 `system + conversations`；`metadata` 用于审计和评估。

## 3. 文件结构目标

新增或重写：

- `src/scene_memory.py`：从小说原文和原始对话构建完整场景记忆。
- `src/profile_builder.py`：用强模型生成/更新角色设定卡。
- `src/retrieval_hybrid.py`：BM25、语义向量、重排模型、候选合并、边界过滤。
- `src/teacher_client.py`：StepFun/OpenAI 兼容 API 封装，支持批处理、重试、断点。
- `src/raft_generation.py`：问题生成、记忆包构造、回答生成、样本落盘。
- `src/raft_verifier.py`：LLM 审核与规则审核。
- `src/runtime_agent.py`：运行时检索 + 提示词组装，不依赖 Gradio。
- `tests/test_scene_memory.py`
- `tests/test_hybrid_retrieval.py`
- `tests/test_raft_generation_schema.py`
- `tests/test_raft_verifier.py`
- `tests/test_runtime_agent_prompt.py`

保留但可重写内部语义：

- `build_memory`：应从“原始对话聚合”升级为“小说原文 + 原始对话 -> 完整场景记忆”。
- `build_sft`：应从“原始对话 -> ShareGPT”升级为“场景记忆 + 教师样本 -> 风格/RAFT/混合数据集”。
- `infer`：应使用运行时智能体构造提示词，而不是在 Gradio 函数里拼字符串。

## 4. 数据产物

目标产物：

```text
data/profiles/shiri_qixia_profile.json
data/memory/shiri_qixia_scenes.jsonl
data/memory/shiri_qixia_bm25.json
data/memory/shiri_qixia_embeddings.npy
data/memory/shiri_qixia_embedding_meta.jsonl
data/memory/shiri_qixia_retrieval_eval.jsonl
data/raft/shiri_qixia_questions.jsonl
data/raft/shiri_qixia_answers.raw.jsonl
data/raft/shiri_qixia_verified.jsonl
data/shiri_qixia_chat_train.json
data/shiri_qixia_chat_valid.json
data/shiri_qixia_raft_train.json
data/shiri_qixia_raft_valid.json
```

大文件和版权敏感数据默认不提交 Git。

## 5. 配置目标

新增配置项：

```toml
# 记忆来源
memory_source = "full_scene"  # full_scene / dialogue_only
scene_max_chars = 1800
scene_overlap_chars = 250
scene_min_dialogues = 0

# 语义向量 / 重排
retrieval_mode = "hybrid"  # bm25 / embedding / hybrid
embedding_model = "BAAI/bge-m3"
reranker_model = "BAAI/bge-reranker-v2-m3"
use_reranker = true
bm25_top_k = 20
embedding_top_k = 20
rerank_top_k = 5

# 教师模型生成
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
```

## 6. 任务 1：完整场景记忆构建

**目标：** 让记忆的知识主体来自原始小说，而不是只来自抽取台词。

**输入：**

- `novels/《十日终焉》（校对全本）.txt`
- `data/raw/shiri_qixia_dialogues.jsonl`

**输出：**

- `data/memory/shiri_qixia_scenes.jsonl`

**场景结构：**

```json
{
  "scene_id": "scene_000032",
  "source_start": 123456,
  "source_end": 125100,
  "raw_text": "小说原文片段，包括旁白、动作、心理、台词",
  "dialogues": [
    {"role": "乔家劲", "dialogue": "喂，骗子，快过来啊！"},
    {"role": "齐夏", "dialogue": "我不可能死在这里的..."}
  ],
  "summary": "",
  "characters": ["齐夏", "乔家劲"],
  "target_role_present": true,
  "target_role_knows": true,
  "knowledge_level": "first_hand",
  "events": [],
  "relations": [],
  "quotes": ["我不可能死在这里的，我有不得不出去的理由。"]
}
```

**实现策略：**

1. 先按原文长度切场景，使用重叠窗口保留上下文。
2. 用原始对话的 `chunk_id/dialogue/role` 辅助对齐到场景。
3. 对齐失败时保留原始场景，但标记 `dialogue_alignment = "missing"`。
4. 初始 `target_role_knows` 规则：
   - 齐夏在场景中出现或说话：`first_hand`
   - 场景由其他人对齐夏讲述：`heard_or_inferred`
   - 只有旁白或远处事件：`narrator_only`

**测试：**

- 构造小小说文本，包含旁白 + 台词。
- 构造原始对话。
- 验证场景的 `raw_text` 包含旁白，`dialogues` 包含台词，`target_role_present` 正确。

## 7. 任务 2：大模型场景增强

**目标：** 用强模型给场景补摘要、事件、人物关系、认知边界。

**输入：**

- `shiri_qixia_scenes.jsonl`

**输出：**

- 同结构的增强后场景数据。

**教师模型提示词要求：**

```text
你是小说角色记忆标注器。
给定一段小说原文和目标角色“齐夏”。
请抽取：
1. 这一段发生了什么
2. 出现了哪些角色
3. 齐夏是否在场
4. 齐夏是否应该知道这段信息
5. 事件列表
6. 人物关系变化
7. 齐夏原话

只输出 JSON。
不要补充原文没有的信息。
```

**实现要求：**

- 支持批处理。
- 支持断点续跑。
- 每个场景失败单独记录，不阻断整体。
- 保存原始教师模型响应，便于排查。

**测试：**

- 用假的教师模型返回固定 JSON。
- 验证场景增强结果能合并回场景。
- 验证失败样本进入 `failed.jsonl`。

## 8. 任务 3：角色设定卡生成

**目标：** 生成齐夏角色设定卡，但允许人工编辑。

**输入：**

- 高频齐夏场景
- 关系/动机相关场景
- 重要原话

**输出：**

- `data/profiles/shiri_qixia_profile.json`

**角色设定卡结构：**

```json
{
  "role": "齐夏",
  "aliases": ["齐夏", "老齐", "齐哥", "小齐"],
  "identity": "",
  "core_goals": [],
  "personality": [],
  "speech_style": [],
  "relationships": [],
  "knowledge_boundary": "",
  "answer_rules": []
}
```

**注意：**

- 自动生成不等于最终真理。
- 生成后必须保留为可编辑 JSON。
- 后续 `build_sft` 和 `infer` 只读取角色设定卡文件，不硬编码。

## 9. 任务 4：混合检索

**目标：** 检索不是只靠 BM25，也不是只靠语义向量，而是精确召回 + 语义召回 + 重排。

**召回层：**

```text
BM25 top 20：人名、别名、专有名词、原话、规则名
语义向量 top 20：动机、情绪、因果、语义相似
```

**合并层：**

```text
合并 BM25 和语义向量候选
按 scene_id 去重
保留来源分数
```

**过滤层：**

```text
默认排除 narrator_only
优先 first_hand
允许 heard_or_inferred
```

**排序层：**

```text
重排模型(question, scene.raw_text + summary + quotes)
输出 top 3-5
```

**测试：**

- 精确名字问题应由 BM25 召回。
- 语义动机问题应由语义向量召回。
- 只有旁白信息的场景默认被过滤。
- 重排模型的模拟分数能决定最终顺序。

## 10. 任务 5：问题生成

**目标：** 不只生成“这段发生了什么”，要覆盖角色用户真实会问的问题。

**每个场景或场景簇生成问题类型：**

```text
grounded_fact
relationship
motivation
false_premise
no_evidence
multi_turn_followup
casual_roleplay
```

**输出：**

```json
{
  "question_id": "q_000001",
  "sample_type": "motivation",
  "question": "你当时为什么判断那条规则是陷阱？",
  "oracle_scene_ids": ["scene_000032"],
  "requires_memory": true,
  "knowledge_level": "first_hand"
}
```

**生成策略：**

- 每个重要场景生成 2-5 个问题。
- 关系/动机问题可以基于场景簇，而不是单个场景。
- 错误前提问题要故意包含错误前提。
- 无证据问题要确保标准证据场景为空或无关。

## 11. 任务 6：记忆包构造

**目标：** 训练时的记忆包必须模拟运行时检索结果。

每个问题构造：

```text
标准证据场景：真正能回答问题
相关场景：补充背景
干扰场景：看似相关但不能回答
```

规则：

- 有证据样本至少 1 个标准证据场景。
- 人物关系/动机样本可有 2-4 个相关场景。
- 无证据样本不能包含能直接回答问题的证据。
- 干扰场景不要太离谱，要“看似相关”。

## 12. 任务 7：教师模型回答生成

**目标：** 用强模型生成齐夏口吻回答，但严格基于记忆包。

**教师模型提示词必须包含：**

```text
你是训练数据生成器。
目标角色：齐夏。
你只能使用给定记忆片段中的事实。
回答必须像齐夏本人。
不要逐字复述原文。
不要添加记忆外具体事实。
资料不足时，用齐夏口吻说明无法确认。
只输出 assistant 的回答文本。
```

**输出要求：**

- 不出现“根据资料/根据片段/作为AI”。
- 不续写 user/assistant。
- 不直接复制大段原文。
- 可以复用少量经典原话，但必须自然。

## 13. 任务 8：审核器审核

**目标：** API 不怕花，就用审核器保质量，少要垃圾数据。

审核维度：

```text
grounded：回答是否被记忆支撑
in_character：是否像齐夏
no_hallucination：是否编造了具体事实
boundary_ok：是否使用了角色不该知道的信息
not_copying：是否过度复述原文
format_ok：是否没有续写 user/assistant
```

审核器输出：

```json
{
  "accepted": true,
  "scores": {
    "grounded": 5,
    "in_character": 4,
    "no_hallucination": 5,
    "boundary_ok": 5,
    "not_copying": 4,
    "format_ok": 5
  },
  "reason": ""
}
```

过滤规则：

- 任一关键项低于 4 分，丢弃或重写。
- `no_hallucination < 5` 的样本优先重写。
- `boundary_ok < 5` 的样本不得进入训练。

## 14. 任务 9：最终 ShareGPT 数据构造

**目标：** 训练格式和运行时格式一致。

统一系统提示词：

```text
你正在扮演《十日终焉》中的齐夏。

你必须遵守：
1. 如果记忆片段包含答案，优先依据记忆回答。
2. 不要逐字复述记忆片段，要用齐夏自己的口吻回答。
3. 如果记忆片段没有答案，不要编造具体小说事实。
4. 始终保持第一人称。
5. 不要续写 user/assistant。

【角色设定】
...

【记忆片段】
...
```

最终输出：

```text
data/shiri_qixia_chat_train.json      # mixed 主训练集
data/shiri_qixia_chat_valid.json
data/shiri_qixia_raft_train.json      # 纯 RAFT 备查/对照
data/shiri_qixia_raft_valid.json
```

## 15. 任务 10：运行时智能体对齐

**目标：** 推理时用同一套协议，不要训练和推理两张皮。

实现：

- `runtime_agent.build_prompt(user_message, history)`
- 使用混合检索器。
- 使用同一 `build_roleplay_system_prompt`。
- 不把记忆混进普通用户消息。
- 对话历史和检索到的记忆分开。

测试：

- 给定问题“余念安是谁”，提示词中应出现相关记忆。
- 给定无关问题，提示词可出现“没有可靠记忆片段”。
- 不加载模型也能测试提示词。

## 16. 任务 11：齐夏数据生成冒烟验证

**目标：** 本地不训练，只验证数据生成链路。

已有输入：

```text
novels/《十日终焉》（校对全本）.txt
data/raw/shiri_qixia_dialogues.jsonl
```

命令：

```bash
python3 main.py -c configs/shiri_qixia.toml -r build_memory build_sft -o build_memory build_sft
```

如果要重新用 StepFun 抽取：

```bash
set -a
source .env.stepfun
set +a

# config 中必须设置 extraction_backend = "cloud_api"
python3 main.py -c configs/shiri_qixia.toml -r extract build_memory build_sft -o extract build_memory build_sft
```

验证：

```bash
python3 -m pytest -q
python3 scripts/validate_dataset.py data/shiri_qixia_chat_train.json
python3 scripts/validate_dataset.py data/shiri_qixia_raft_train.json
```

不运行：

```text
train
infer with loaded model
ROCm training
```

## 17. 阶段验收标准

第一阶段完成标准：

- 场景记忆来自原始小说文本，不只是台词。
- 每条场景能看到 `raw_text`、`dialogues`、`characters`、`quotes`、`knowledge_level`。
- 至少生成 100 条教师模型回答 + 审核通过样本作为冒烟验证。
- 训练数据系统提示词和运行时提示词使用同一个构造函数。
- 本地测试通过。
- 不运行训练。

第二阶段完成标准：

- 生成 1000-3000 条高质量齐夏训练样本。
- 审核器通过率和拒绝原因有统计。
- 有固定评测集，覆盖事实、关系、动机、错误前提、边界、不知道。
- 云端可直接训练 mixed 主训练集。

## 18. 明确不做

本计划当前不做：

- 不把整本小说事实强行塞进 LoRA。
- 不只基于原始对话生成最终记忆。
- 不用单纯 BM25 作为最终检索。
- 不在本地跑 ROCm 训练。
- 不为了兼容旧实现保留错误的数据边界。

## 19. 自检

- 架构已从台词记忆 v1 修正为完整场景记忆。
- 数据生成已从“规则拼接”修正为教师模型生成 + 审核器过滤。
- 检索已从只用 BM25 修正为 BM25 + 语义向量 + 重排模型 + 认知边界过滤。
- 训练目标已明确：LoRA 学口吻和记忆使用协议，事实留在记忆库。
- 计划全文使用中文。
