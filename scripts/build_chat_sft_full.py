#!/usr/bin/env python3
"""完整版：把 data/qixia_train.json 全部转成『用户问 → 齐夏答』ShareGPT 聊天数据。

- 并发调用阶跃 API 加速。
- 断点续传：中途 Ctrl+C 再跑会跳过已完成的。
- 输出：data/qixia_chat_train_full.json
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("请先安装：pip install openai", file=sys.stderr)
    sys.exit(1)

REPO_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_DIR / "data" / "qixia_train.json"
OUTPUT_PATH = REPO_DIR / "data" / "qixia_chat_train_full.json"
CHECKPOINT_PATH = REPO_DIR / "data" / "_chat_conversion_full_progress.json"
ENV_PATH = REPO_DIR / "extract-dialogue" / ".env.stepfun"

# 并发数。阶跃 flash 模型 QPS 一般够 8-16，先稳一点。
CONCURRENCY = int(os.environ.get("CONCURRENCY", "8"))
# 0 = 跑完所有
SAMPLE_COUNT = int(os.environ.get("SAMPLE_COUNT", "0"))


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
API_KEY = os.environ.get("STEPFUN_API_KEY") or env.get("CUSTOM_API_KEY")
BASE_URL = os.environ.get("STEPFUN_BASE_URL") or env.get("CUSTOM_BASE_URL") or "https://api.stepfun.com/v1"
MODEL = os.environ.get("STEPFUN_MODEL") or env.get("CUSTOM_MODEL_NAME") or "step-1.5-flash"

if not API_KEY:
    print(f"ERROR: 没找到 API key。检查 {ENV_PATH}", file=sys.stderr)
    sys.exit(1)

print(f"API base:   {BASE_URL}")
print(f"Model:      {MODEL}")
print(f"Concurrency: {CONCURRENCY}")

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
    parts = context_raw.split("齐夏：")
    body = "齐夏：".join(parts[:-1]).strip() if len(parts) > 1 else context_raw
    if "下面是《十日终焉》" in body:
        body = body.split("\n\n", 1)[-1].strip()
    return body


def make_key(context: str, line: str) -> str:
    return f"{context[:80]}___{line[:40]}"


def convert_one(item: dict, idx: int) -> tuple[str, dict | None]:
    conv = item["conversations"]
    line = conv[1]["value"]
    context = clean_context(conv[0]["value"])
    key = make_key(context, line)

    for attempt in range(3):
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
            m = re.search(r"\{[\s\S]+\}", text)
            if not m:
                raise ValueError("no json")
            parsed = json.loads(m.group(0))
            if not parsed.get("user") or not parsed.get("qixia"):
                raise ValueError("missing fields")
            chat_item = {
                "id": f"qixia-chat-{idx:05d}",
                "system": "你正在扮演《十日终焉》中的齐夏。保持冷静、克制、善于观察和推理，根据对话上下文自然回应。",
                "conversations": [
                    {"from": "human", "value": parsed["user"].strip()},
                    {"from": "gpt", "value": parsed["qixia"].strip()},
                ],
            }
            return key, chat_item
        except Exception as e:
            if attempt == 2:
                print(f"  ❌ #{idx} 失败: {e}")
                return key, None
            time.sleep(1 + attempt * 2)
    return key, None


def main() -> None:
    done = {}
    if OUTPUT_PATH.exists():
        try:
            for x in json.loads(OUTPUT_PATH.read_text(encoding="utf-8")):
                # 用 id 兜底，因为旧文件不一定有 key 记录
                done[x["id"]] = x
        except Exception:
            pass
    if CHECKPOINT_PATH.exists():
        try:
            for k in json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8")):
                done.setdefault(k, True)
        except Exception:
            pass

    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    if SAMPLE_COUNT > 0:
        data = data[:SAMPLE_COUNT]
    total = len(data)
    print(f"\n共 {total} 条，开始并发转换 (并发={CONCURRENCY})\n")

    # 已经做过的样本：通过 id 跳过
    pending = []
    results: list[dict] = []
    done_keys: set = set()
    for idx, item in enumerate(data):
        expected_id = f"qixia-chat-{idx:05d}"
        if expected_id in done and isinstance(done[expected_id], dict):
            results.append(done[expected_id])
            done_keys.add(expected_id)
        else:
            pending.append((idx, item))
    print(f"已完成 {len(results)} 条，本次需处理 {len(pending)} 条")

    lock = threading.Lock()
    saved_count = [len(results)]
    last_save_time = [time.time()]

    def save_progress():
        OUTPUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        CHECKPOINT_PATH.write_text(json.dumps(list(done_keys), ensure_ascii=False) + "\n", encoding="utf-8")

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futures = {ex.submit(convert_one, item, idx): idx for idx, item in pending}
        for fut in as_completed(futures):
            idx = futures[fut]
            key, chat_item = fut.result()
            with lock:
                if chat_item:
                    results.append(chat_item)
                    done_keys.add(chat_item["id"])
                    saved_count[0] += 1
                    if saved_count[0] % 50 == 0 or time.time() - last_save_time[0] > 30:
                        # 按 idx 排序，保持稳定顺序
                        results.sort(key=lambda r: r["id"])
                        save_progress()
                        last_save_time[0] = time.time()
                        print(f"  ✅ 已保存 {saved_count[0]}/{total}")

    results.sort(key=lambda r: r["id"])
    save_progress()

    print("\n" + "=" * 60)
    print(f"完成！共生成 {len(results)}/{total} 条")
    print(f"输出: {OUTPUT_PATH}")
    print("=" * 60)

    print("\n📋 预览前 5 条：\n")
    for j, x in enumerate(results[:5]):
        print(f"--- 第 {j+1} 条 ---")
        print(f"用户: {x['conversations'][0]['value']}")
        print(f"齐夏: {x['conversations'][1]['value']}\n")


if __name__ == "__main__":
    main()
