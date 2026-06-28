from __future__ import annotations

import os
import gradio as gr
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from .config import Config
from .utils import init_logger, check_step_done, mark_step_done

def run_infer(config: Config) -> None:
    """启动推理服务"""
    step_name = "infer"
    logger = init_logger(step_name, config.logs_dir)
    
    logger.info("===== 启动推理服务 =====")
    logger.info(f"模型: {config.model_id}")
    logger.info(f"LoRA路径: {config.output_dir}/{config.run_name}")
    
    # 加载模型
    logger.info("加载模型和LoRA权重...")
    model_path = config.model_cache_dir / config.model_id.replace('/', '_').replace('.', '_') if (config.model_cache_dir / config.model_id.replace('/', '_').replace('.', '_')).exists() else config.model_id
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    lora_path = config.output_dir / config.run_name
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()
    logger.info("模型加载完成")
    
    # 聊天函数
    def clean_reply(text: str) -> str:
        stops = [
            "\n用户", "\nuser", "\nUser", "\nassistant", "\nAssistant",
            "user:", "assistant:", "<|im_start|>", "<|im_end|>",
        ]
        cut = len(text)
        for stop in stops:
            idx = text.find(stop)
            if idx != -1:
                cut = min(cut, idx)
        return text[:cut].strip()
    
    def chat_fn(message, history, max_new_tokens, temperature):
        # 构建prompt
        messages = []
        system_prompt = f"你正在扮演《{config.novel_title}》中的{config.canonical_role}。严格保持{config.canonical_role}的语气、性格、说话习惯和价值观，根据对话上下文自然回应，不要跳出角色，不要续写其他角色的发言。"
        messages.append({"role": "system", "content": system_prompt})
        
        for h in history:
            messages.append({"role": "user", "content": h[0]})
            messages.append({"role": "assistant", "content": h[1]})
        
        messages.append({"role": "user", "content": message})
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "top_p": 0.9,
            "repetition_penalty": 1.15,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
        
        if not config.stream_output:
            with torch.no_grad():
                outputs = model.generate(**inputs, **generation_kwargs)
            reply = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            return clean_reply(reply)
        else:
            # 流式输出
            from transformers import TextIteratorStreamer
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs["streamer"] = streamer
            
            from threading import Thread
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            
            partial_reply = ""
            for new_text in streamer:
                partial_reply += new_text
                yield clean_reply(partial_reply)
    
    # 构建Gradio界面
    demo = gr.ChatInterface(
        fn=chat_fn,
        title=f"👤 {config.canonical_role} 角色聊天 - TaleTalk",
        description=f"📚 来自《{config.novel_title}》的{config.canonical_role}，LoRA模型: {config.run_name}",
        additional_inputs=[
            gr.Slider(32, 2048, value=128, step=8, label="最大生成字数"),
            gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="温度系数"),
        ],
        examples=[
            ["如果一个规则看起来互相矛盾，你会怎么判断？", 128, 0.7],
            ["我现在很慌，怎么办？", 128, 0.7],
            ["你最讨厌什么样的人？", 128, 0.7],
        ],
        fill_height=True,
    )
    
    logger.info(f"Gradio服务启动，端口: {config.gradio_port}")
    if config.share:
        logger.info("公网共享链接会在启动后生成")
    
    # 启动服务
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=config.gradio_port,
        share=config.share,
        inline=False,
    )
