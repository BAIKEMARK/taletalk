# 模块 2：一轮 AI 数据生成

## 目标

默认只用一轮 AI 批处理，把 scene skeleton 转成 scene memory、候选训练问答和 profile observations。

## 输入

```text
data/memory/shiri_qixia_scenes.raw.jsonl
```

## 输出

```text
data/memory/shiri_qixia_scenes.jsonl
data/profiles/shiri_qixia_profile.observations.jsonl
data/raft/shiri_qixia_candidates.raw.jsonl
cache/raft_generation/failed.jsonl
```

## 一轮 AI 输入

每次请求处理一个 scene 或 scene batch：

```json
{
  "target_role": "齐夏",
  "novel_title": "十日终焉",
  "scenes": [
    {
      "scene_id": "scene_000032",
      "chapter": "第十二章",
      "raw_text": "...",
      "dialogues": []
    }
  ]
}
```

## 一轮 AI 输出

AI 只输出 JSON：

```json
{
  "scene_memories": [
    {
      "scene_id": "scene_000032",
      "summary": "这一段发生了什么",
      "characters": ["齐夏", "乔家劲"],
      "target_role_present": true,
      "target_role_knows": true,
      "knowledge_level": "first_hand",
      "events": ["齐夏判断规则存在陷阱"],
      "relations": [],
      "quotes": ["齐夏原话"]
    }
  ],
  "profile_observations": [
    {
      "aspect": "speech_style",
      "value": "冷静、克制、倾向先判断风险",
      "source_scene_ids": ["scene_000032"]
    }
  ],
  "candidate_samples": [
    {
      "sample_type": "motivation",
      "question": "你当时为什么判断那条规则是陷阱？",
      "answer": "因为它太像答案了。",
      "source_scene_ids": ["scene_000032"],
      "knowledge_level": "first_hand"
    }
  ]
}
```

## 生成要求

- 不要补充原文没有的信息。
- 不要续写 `user/assistant`。
- 不要输出 Markdown。
- 不要把候选回答写成“根据资料”。
- 每个候选样本必须有 `source_scene_ids`。
- 无证据样本必须明确 `source_scene_ids = []` 或 `answer_policy = "insufficient_memory"`。

## 断点续跑

- 每个 scene 或 scene batch 的结果按 id 落盘。
- 已完成 id 默认跳过。
- 解析失败写入 `failed.jsonl`，不阻断整体。
- 原始 AI 响应保留在 cache，便于排查。

## 测试

- 用 fake AI client 返回固定 JSON。
- 验证 scene memory 合并回原 scene。
- 验证候选样本包含 `source_scene_ids`。
- 验证失败样本进入 `failed.jsonl`。
- 验证默认流程不需要第二轮 AI。
