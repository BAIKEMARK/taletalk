#!/usr/bin/env python3
"""把『上下文 → 齐夏下一句』补全数据改造成『用户问 → 齐夏答』ShareGPT 聊天数据。

测试版：默认只跑前 100 条，用阶跃 API (StepFun)。
自动从 extract-dialogue/.env.stepfun 读取 CUSTOM_API_KEY/CUSTOM_BASE_URL/CUSTOM_MODEL_NAME。
也可以用环境变量 STEPFUN_API_KEY / STEPFUN_BASE_URL / STEPFUN_MODEL 覆盖。
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("请先安装：pip install openai", file=sys.stderr)
    sys.exit(1)

REPO_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_DIR / "data" / "qixia_train.json"
OUTPUT_PATH = REPO_DIR / "data" / "qixia_chat_test_100.json"
CHECKPOINT_PATH = REPO_DIR / "data" / "_chat_conversion_test_progress.json"
ENV_PATH = REPO_DIR / "extract-dialogue" / ".env.stepfun"

SAMPLE_COUNT = int(os.environ.get("SAMPLE_COUNT", "100"))


def load_env_file(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


env = load_env_file(ENV_PATH)

API_KEY = os.environ.get("STEPFUN_API_KEY") or env.get("CUSTOM_API_KEY") or env.get("STEPFUN_API_KEY")
BASE_URL = os.environ.get("STEPFUN_BASE_URL") or env.get("CUSTOM_BASE_URL") or "https://api.stepfun.com/v1"
MODEL = os.environ.get("STEPFUN_MODEL") or env.get("CUSTOM_MODEL_NAME") or "step-1.5-flash"

if not API_KEY:
    print(f"ERROR: 没找到 API key。检查 {ENV_PATH} 或设置 STEPFUN_API_KEY。", file=sys.stderr)
    sys.exit(1)

print(f"API base: {BASE_URL}")
print(f"Model:    {MODEL}")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

SYSTEM_PROMPT = (
    "你是一个 SFT 数据标注员。任务：把小说对话改造成一条高质量聊天训练数据。\n"
    "严格只输出一个 JSON 对象，不要 markdown 代码块，不要任何前后说明文字。\n"
    "JSON 格式：{\"user\": \"...\", \"qixia\": \"...\"}\n"
    "其中 user 是一个自然人会问齐夏的问题，结合上下文，不要书面，不要提小说片段。\n"
    "qixia 是齐夏的回答，保持原著语气：冷静、克制、善于观察、逻辑严密。基于原台词适当扩展成 30~200 字完整回答，不加原著没有的核心设定。"
)

USER_PROMPT_TPL = """
上下文：
{context}

齐夏原台词：
{line}
""".strip()


def clean_context(context_raw: str) -> str:
    # 去掉提示词，只留对话
    parts = context_raw.split("齐夏：")
    if len(parts) > 1:
        # 最后一个 "齐夏：" 是要补全的位置，前面是上下文
        body = "齐夏：".join(parts[:-1]).strip()
    else:
        body = context_raw
    if "下面是《十日终焉》" in body:
        body = body.split("\n\n", 1)[-1].strip()
    return body


def main() -> None:
    done = set()
    if CHECKPOINT_PATH.exists():
        done = set(json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8")))

    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))[:SAMPLE_COUNT]
    result = []

    if OUTPUT_PATH.exists():
        try:
            existing = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                result.extend(existing)
        except Exception:
            pass

    for i, item in enumerate(data):
        conv = item["conversations"]
        line = conv[1]["value"]
        context = clean_context(conv[0]["value"])

        key = f"{context[:80]}___{line[:40]}"
        if key in done:
            continue

        print(f"\n[{i+1}/{len(data)}] 生成中... 原台词: {line[:40]}")

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT_TPL.format(context=context, line=line)},
                ],
                temperature=0.6,
                stream=False,
                timeout=60,
            )
            text = (response.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"  ❌ API 失败: {e}")
            continue

        try:
            m = re.search(r"\{[\s\S]+\}", text)
            if not m:
                raise ValueError("no json")
            parsed = json.loads(m.group(0))
            if not parsed.get("user") or not parsed.get("qixia"):
                raise ValueError("missing user/qixia fields")
        except Exception as e:
            print(f"  ❌ 解析失败: {e}\n  原始: {text[:200]}")
            continue

        chat_item = {
            "id": f"qixia-chat-{i:04d}",
            "system": "你正在扮演《十日终焉》中的齐夏。保持冷静、克制、善于观察和推理，根据对话上下文自然回应。",
            "conversations": [
                {"from": "human", "value": parsed["user"].strip()},
                {"from": "gpt", "value": parsed["qixia"].strip()},
            ],
        }
        result.append(chat_item)
        done.add(key)

        if (i + 1) % 10 == 0:
            OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            CHECKPOINT_PATH.write_text(json.dumps(list(done), ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"  ✅ 已保存 {len(result)} 条")

    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    CHECKPOINT_PATH.write_text(json.dumps(list(done), ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"完成！共生成 {len(result)} 条")
    print(f"输出: {OUTPUT_PATH}")
    print("=" * 60)

    print("\n📋 预览前 5 条：\n")
    for j, x in enumerate(result[:5]):
        print(f"--- 第 {j+1} 条 ---")
        print(f"用户: {x['conversations'][0]['value']}")
        print(f"齐夏: {x['conversations'][1]['value']}")
        print()


if __name__ == "__main__":
    main()
