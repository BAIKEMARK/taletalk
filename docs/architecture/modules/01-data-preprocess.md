# 模块 1：数据前处理

## 目标

把小说 TXT 和可选 raw dialogue 转成可追溯的 scene skeleton。这个模块不调用 AI。

## 输入

```text
novels/《十日终焉》（校对全本）.txt
data/raw/shiri_qixia_dialogues.jsonl  # 可选
```

## 输出

```text
data/memory/shiri_qixia_scenes.raw.jsonl
data/memory/shiri_qixia_scene_build_report.json
```

## Scene Skeleton

```json
{
  "scene_id": "scene_000032",
  "chapter": "第十二章",
  "scene_index": 32,
  "source_start": 123456,
  "source_end": 125100,
  "raw_text": "小说原文片段，包括旁白、动作、心理、台词",
  "dialogues": [
    {"role": "齐夏", "dialogue": "我不可能死在这里的..."}
  ],
  "dialogue_alignment": "matched",
  "alignment_score": 0.97,
  "source_refs": [
    {"raw_dialogue_id": "chunk_000032_0004", "match_type": "exact"}
  ]
}
```

## 实现规则

- 原文归一化只统一换行和空白，不删除正文。
- `source_start/source_end` 必须能回到原文切片。
- 优先按章节标题切分；没有章节时按段落密度和长度切分。
- 场景切分以段落为基本单位，受 `scene_max_chars` 和 `scene_overlap_chars` 控制。
- raw dialogue 对齐先精确匹配，再做模糊匹配。
- 对齐失败时保留场景，标记 `dialogue_alignment = "missing"`。

## 构建报告

报告至少包含：

```json
{
  "scene_count": 319,
  "avg_scene_chars": 1420,
  "aligned_dialogue_count": 880,
  "missing_dialogue_count": 37,
  "alignment_success_rate": 0.96
}
```

## 测试

- 构造小小说文本，包含章节、旁白、台词。
- 验证 `source_start/source_end` 能回到原文切片。
- 验证对齐失败不会丢场景。
- 验证报告记录成功和失败数量。
