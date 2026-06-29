# 模块 4：SFT、运行时、评测、角色包

## 目标

把通过过滤的候选样本转成训练数据，并保证训练、运行时、评测、导出使用同一套角色记忆协议。

## 输入

```text
data/raft/shiri_qixia_candidates.raw.jsonl
data/raft/shiri_qixia_memory_packs.jsonl
data/memory/shiri_qixia_scenes.jsonl
data/profiles/shiri_qixia_profile.json
```

## 输出

```text
data/shiri_qixia_chat_train.json
data/shiri_qixia_chat_valid.json
data/eval/shiri_qixia_eval_questions.jsonl
reports/shiri_qixia_eval_report.json
reports/shiri_qixia_eval_report.md
dist/roles/shiri_qixia/manifest.json
```

## ShareGPT 构造

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

## 规则过滤

默认过滤不调用 AI：

- 去掉续写 `user/assistant` 的样本。
- 去掉出现“作为 AI”“根据资料”的样本。
- 去掉空回答、截断回答、大段复制原文。
- 去掉有证据但缺少 `source_scene_ids` 的样本。
- 去掉无证据却包含直接答案场景的样本。

AI 审核只允许在 `sampled` 或 `risk_only` 模式启用。

## Runtime Agent

运行时必须：

- 使用同一 profile。
- 使用同一 `render_memory_pack`。
- 区分对话历史和检索记忆。
- 不把记忆作为普通 user message。
- 对无可靠记忆的问题允许克制回答。

## 评测报告

评测类别：

```text
style_imitation
grounded_fact
relationship
motivation
false_premise
boundary_unknown
multi_turn_consistency
retrieval_quality
```

报告指标：

```text
retrieval_recall_at_5
grounded_score
character_score
boundary_score
hallucination_count
copy_rate
format_error_count
```

Markdown 报告必须包含失败样例。

## 角色包导出

导出目录：

```text
dist/roles/shiri_qixia/
```

目录结构：

```text
manifest.json
README.md
profile.json
memory/scenes.jsonl
memory/bm25.json
memory/embeddings.npy
memory/embedding_meta.jsonl
prompts/runtime_system.txt
datasets/chat_train.json
datasets/chat_valid.json
eval/eval_questions.jsonl
eval/eval_report.md
```

导出模式：

```text
private_full：本机/私有服务器使用，保留 raw_text，效果最好。
public_redacted：公开分享使用，移除 raw_text，只保留 summary、events、relations、quotes 的可配置子集。
```

## 测试

- 验证 ShareGPT 数据能通过 dataset validator。
- 验证 runtime prompt 包含相关记忆，且记忆不在 user message 中。
- 验证评测 JSON 和 Markdown 都生成。
- 验证 `public_redacted` 不包含 `raw_text`。
- 验证角色包能被 runtime agent 从目录加载。
