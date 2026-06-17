# Qixia ROCm LoRA

ROCm 单卡 LoRA 微调脚手架，用于把《十日终焉》齐夏角色语料整理成 WeClone / LLaMA Factory 可读的 ShareGPT 数据，并在 AMD GPU 云平台上跑训练。

## What Is Included

- `rocm_amd_gpu_smoke_test.ipynb`: AMD ROCm 环境自检，先确认 GPU、HIP、PyTorch 都可用。
- `weclone_rocm_single_gpu_lora.ipynb`: 单卡 LoRA 训练 notebook，默认先跑样例数据，确认后再换完整语料。
- `scripts/prepare_weclone_dataset.py`: 把 `extract-dialogue` 生成的 ShareGPT JSON 转成 WeClone 需要的 `sft-my.json` / `dataset_info.json`。
- `scripts/build_role_dataset_from_novel.py`: 备用的本地正则抽取脚本，不作为主训练数据来源。
- `role_data/qixia_sample/`: 6 条可公开的原创格式样例，只用于验证格式和冒烟测试。

## What Is Not Committed

完整小说、完整对白语料、`.env`、API key、训练输出和 LoRA checkpoint 都不会提交到公开仓库。完整数据在本地生成后放在 `role_data/qixia/`，该目录已被 `.gitignore` 忽略。

## Prepare Data Locally

如果你已经有 `extract-dialogue/outputs/sft/qixia_sharegpt_train.json`，运行：

```bash
python3 scripts/prepare_weclone_dataset.py \
  --source extract-dialogue/outputs/sft/qixia_sharegpt_train.json \
  --out-dir role_data/qixia \
  --dataset-name chat-sft
```

生成结果：

```text
role_data/qixia/sft-my.json
role_data/qixia/dataset_info.json
role_data/qixia/stats.json
```

如果只想生成本地抽样样例，可用同一脚本：

```bash
python3 scripts/prepare_weclone_dataset.py \
  --source extract-dialogue/outputs/sft/qixia_sharegpt_train.json \
  --out-dir role_data/qixia_sample \
  --dataset-name chat-sft \
  --limit 20
```

注意：如果样例来自原书文本，只用于本地检查格式，不要提交到公开仓库。当前仓库里的 `role_data/qixia_sample/` 是原创格式样例，不含原文摘录。

## Run On ROCm Cloud

1. 先打开 `rocm_amd_gpu_smoke_test.ipynb`，确认 `torch.version.hip` 有值，`torch.cuda.is_available()` 为 `True`。
2. 上传或挂载完整 `role_data/qixia/` 到 notebook 所在仓库目录。
3. 打开 `weclone_rocm_single_gpu_lora.ipynb`，从上到下运行到训练前检查。
4. 确认模型、数据、显存都正常后，把参数区的 `RUN_TRAIN = True` 再运行训练单元。

默认存储策略：

- 持久化目录：`/network-workspace/weclone-rocm`
- 临时模型缓存：`/workspace/model-cache/weclone-rocm`
- LoRA 输出：`/network-workspace/weclone-rocm/model_output_rocm_lora_qwen25_7b`

## Validate

本地脚本测试：

```bash
python3 -m unittest tests/test_prepare_weclone_dataset.py
```

数据格式快速检查：

```bash
python3 - <<'PY'
import json
from pathlib import Path
data = json.loads(Path("role_data/qixia_sample/sft-my.json").read_text(encoding="utf-8"))
assert data and data[0]["messages"][0]["role"] == "user"
assert data[0]["messages"][-1]["role"] == "assistant"
print("sample examples:", len(data))
PY
```

## Notes

第一轮建议只跑普通 LoRA，不先做 4bit QLoRA。48GB AMD 显存可以从 Qwen2.5-7B-Instruct、`cutoff_len=1024`、`batch_size=1`、`gradient_accumulation_steps=16` 开始，跑通后再调大上下文或训练轮数。
