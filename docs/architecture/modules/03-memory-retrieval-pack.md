# 模块 3：记忆检索与 Memory Pack

## 目标

把 scene memory 变成可检索记忆库，并用同一协议构造训练时和运行时的 memory pack。

## 输入

```text
data/memory/shiri_qixia_scenes.jsonl
data/raft/shiri_qixia_candidates.raw.jsonl
```

## 输出

```text
data/memory/shiri_qixia_bm25.json
data/memory/shiri_qixia_embeddings.npy
data/memory/shiri_qixia_embedding_meta.jsonl
data/raft/shiri_qixia_memory_packs.jsonl
```

## 索引构建

- BM25 字段：`characters + aliases + quotes + summary + events + relations + raw_text`。
- embedding 字段：`summary + events + relations + quotes + raw_text`。
- meta 必须保留 `scene_id`、`knowledge_level`、`source_start/source_end`。
- 同一份 scenes 输入必须生成稳定 meta 顺序。

## 训练时 Memory Pack

训练时 memory pack 是程序构造的受控检索上下文：

```json
{
  "question_id": "q_000001",
  "items": [
    {
      "scene_id": "scene_000032",
      "hidden_role": "gold_evidence",
      "knowledge_level": "first_hand",
      "text": "场景摘要 + 关键事件 + 齐夏原话"
    },
    {
      "scene_id": "scene_000041",
      "hidden_role": "supporting_context",
      "knowledge_level": "first_hand",
      "text": "补充背景"
    },
    {
      "scene_id": "scene_000877",
      "hidden_role": "hard_distractor",
      "knowledge_level": "first_hand",
      "text": "看似相关但不能直接回答"
    }
  ]
}
```

`hidden_role` 只用于审计，不暴露给模型。

## 运行时 Memory Pack

运行时 memory pack 实时构造：

```text
用户问题 + 对话历史
-> query 分析
-> BM25 召回
-> embedding 召回
-> 认知边界过滤
-> reranker 重排
-> token budget 压缩
-> render_memory_pack
```

默认召回 scene memory；增强模式可以额外召回 aggregate memory。

## 渲染协议

训练和运行时都使用同一种格式：

```text
【记忆片段 1｜齐夏亲历】
...

【记忆片段 2｜他人告知或可推断】
...

【记忆片段 3｜记忆不足】
没有可靠记忆片段能直接回答这个问题。
```

## 构造规则

- 有证据样本至少 1 个 gold evidence。
- 关系/动机样本允许 2-4 个 supporting context。
- 无证据/认知边界样本不能包含直接答案。
- hard distractor 必须看似相关，不能太离谱。
- 默认排除 `narrator_only`，除非样本就是训练“不知道”。

## 测试

- 验证 BM25、embedding、meta 三类产物生成。
- 验证 meta 顺序和 embedding 行号一一对应。
- 验证有证据样本包含 gold evidence。
- 验证无证据样本不包含直接答案。
- 验证 runtime prompt 不把记忆混进普通用户消息。
