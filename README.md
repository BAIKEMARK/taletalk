# TaleTalk

让小说角色"活过来"对话的端到端工具链。

输入一本小说 `.txt` + 一个角色名，输出一个能扮演这个角色和你多轮聊天的 LoRA + Gradio 对话界面。

## 流程

整个流水线就 3 个 notebook，按编号跑：

| 文件 | 作用 | 跑在哪 |
|---|---|---|
| `notebooks/01_extract.ipynb` | 小说 .txt → 角色多轮对话 jsonl → 多轮 ShareGPT SFT 数据 | 本地或云端（只调 LLM API） |
| `notebooks/02_train.ipynb` | 多轮 SFT 数据 → LoRA（ROCm + LLaMA Factory） | ROCm GPU 云端（如 Radeon Cloud） |
| `notebooks/03_infer.ipynb` | base + LoRA → Gradio 多轮聊天 + Cloudflare 隧道 | 训练同环境 |

三个 notebook 顶部都有 `RUN_NAME`/`MODEL_CHOICE`/`TARGET_ROLE` 参数区，保持一致就能串起来。换小说、换角色、换模型只改参数区。

## 抽取阶段支持的 LLM

`01_extract.ipynb` 不挑后端，只要 OpenAI 兼容协议都能用。在仓库根目录复制 `.env.example` 为 `.env`，按需填一个平台即可：

- **云端 API**：DeepSeek / OpenAI / SiliconFlow / Moonshot / 阶跃 / 任意 OpenAI 兼容服务
- **本地 vLLM**：`vllm serve <model> --port 8000`，填 `CUSTOM_BASE_URL=http://localhost:8000/v1`
- **本地 LLaMA Factory api server**：同样填 `CUSTOM_*`

支持**断点续跑**——每个 chunk 抽完写本地 cache，重跑不会重复花钱。

## 训练环境

`02_train.ipynb` 默认假设：

- 镜像：`crpi-t0r1tierahelwr8r.cn-beijing.personal.cr.aliyuncs.com/zhangnju/llamafactory_rocm:20260608`（LLaMA Factory + ROCm PyTorch 预装）
- 持久化目录：`/network-workspace`
- GPU：AMD W7900D 48GB 或同档

非 ROCm 环境也能用，把 `requirements-rocm.txt` 换成 CUDA 版 torch，把 patch 脚本去掉就行。

## 仓库布局

```
taletalk/
├── notebooks/
│   ├── 01_extract.ipynb
│   ├── 02_train.ipynb
│   └── 03_infer.ipynb
├── extract/                   # 多轮对话抽取模块（vendor + 改造自 KMnO4-zx/extract-dialogue）
│   ├── dialogue_extractor.py
│   ├── config.py
│   └── __init__.py
├── scripts/
│   ├── build_chat_sft_multiturn.py   # jsonl → 多轮 ShareGPT
│   ├── build_chat_sft_batch.py       # 单轮压扁版（备用，不推荐）
│   ├── train_lora.py                 # LLaMA Factory 训练入口
│   ├── validate_dataset.py
│   ├── patch_llamafactory_qwen35_text.py
│   ├── merge_lora.py
│   └── quick_infer.py
├── configs/                   # LoRA yaml 模板
│   ├── qwen3_5_9b_lora.yaml
│   └── qwen3_6_35b_a3b_lora.yaml
├── data/                      # SFT 数据（jsonl raw 在 data/raw/，gitignore）
├── requirements-rocm.txt      # 训练依赖（不含 torch，避免覆盖 ROCm 版）
├── requirements-extract.txt   # 抽取依赖
└── .env.example
```

## 致谢

- 对话抽取模块基于 [KMnO4-zx/extract-dialogue](https://github.com/KMnO4-zx/extract-dialogue)（huanhuan-chat 项目的一部分）改造而来。
- 训练阶段使用 [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory)。
