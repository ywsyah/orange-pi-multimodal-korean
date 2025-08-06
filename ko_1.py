#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gradio chat-demo for Qwen2-VL on MindSpore
-----------------------------------------
pip install "gradio==4.40.0" mindspore-cpu mindnlp pillow
python app.py
"""

import os
import base64
import time
import gradio as gr
import mindspore
from mindspore import context

from mindnlp.transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info

# ────────────────── MindSpore 环境 ──────────────────
context.set_context(enable_debug_runtime=True)
context.set_context(mode=context.GRAPH_MODE, max_call_depth=2000)
print("Device target:", mindspore.get_context("device_target"))

# ────────────────── 模型与处理器 ──────────────────
MODEL_PATH = "/home/HwHiAiUser/.cache/modelscope/hub/models/Qwen/Qwen2-VL-2B-Instruct"
FP16 = mindspore.float16

print("Loading model ... (首次可能较慢)")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    ms_dtype=FP16,
    trust_remote_code=True,
)

min_pixels = 256 * 28 * 28
max_pixels = 512 * 28 * 28
processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    min_pixels=min_pixels,
    max_pixels=max_pixels,
    ms_dtype=FP16,
)

# ────────────────── 工具函数 ──────────────────
def _pil_to_data_uri(pil_img):
    """将 PIL.Image 转为 data-URI，方便在 browser 端回显。"""
    import io
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"

def build_messages(history):
    """
    将内部 history( list[dict] ) 转为 Qwen2-VL 期望的 messages 格式。
    history 结构见 chat() 下方说明。
    """
    messages = []
    for turn in history:
        user_obj = {"role": "user", "content": []}

        if turn["user_img"] is not None:
            user_obj["content"].append(
                {"type": "image", "image": turn["user_img"]}
            )
        if turn["user_text"].strip():
            user_obj["content"].append(
                {"type": "text", "text": turn["user_text"]}
            )

        messages.append(user_obj)
        messages.append(
            {
                "role": "assistant",
                "content": turn["assistant"],
            }
        )
    return messages

def qwen2vl_generate(history):
    """
    1. 根据 history 重新拼装完整 messages
    2. 调用模型生成
    3. 返回 assistant 文本
    """
    messages = build_messages(history)

    # Qwen2-VL 专用模板
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)

    inputs = processor(
        text=[prompt],
        images=image_inputs,
        padding=True,
        return_tensors="ms",
    )

    # MindSpore int64→int32
    for k in ("input_ids", "attention_mask", "position_ids"):
        if k in inputs and inputs[k].dtype == mindspore.int64:
            inputs[k] = inputs[k].astype(mindspore.int32)

    generated = model.generate(**inputs, max_new_tokens=256)

    # 只取新增 token
    trimmed = [
        out[len(src) :] for src, out in zip(inputs.input_ids, generated)
    ]
    text = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return text.strip()

# ────────────────── Gradio 回调 ──────────────────
def chat(user_text, user_image, state):
    """
    Gradio handler
    Parameters
    ----------
    user_text : str
    user_image : PIL.Image or None
    state : list[dict], gr.State
        [{
            "user_text": "...",
            "user_img": "/tmp/xxx.png" or None,
            "assistant": "..."
        }, ...]
    """
    state = state or []

    # 保存上传图片到临时文件（路径传给模型）
    img_path = None
    if user_image is not None:
        tmp_dir = "tmp_uploads"
        os.makedirs(tmp_dir, exist_ok=True)
        img_path = os.path.join(
            tmp_dir, f"{int(time.time()*1000)}.png"
        )
        user_image.save(img_path)

    # 新 turn（assistant 暂置空，先用于显示 user）
    state.append(
        {
            "user_text": user_text or "",
            "user_img": img_path,
            "assistant": "",
        }
    )

    # 让 chatbot 先显示用户消息
    chatbot_messages = []
    for t in state:
        user_display = ""
        if t["user_img"] is not None:
            user_display += f'<img src="{_pil_to_data_uri(gr.processing_utils.load_image(t["user_img"]))}" width="128"/> '
        user_display += t["user_text"]
        chatbot_messages.append((user_display, t["assistant"]))

    yield chatbot_messages, state  # 立即刷新 UI

    # 调用模型
    assistant_reply = qwen2vl_generate(state)
    state[-1]["assistant"] = assistant_reply

    # 更新 UI
    chatbot_messages[-1] = (chatbot_messages[-1][0], assistant_reply)
    yield chatbot_messages, state

def clear_fn():
    return [], []

# ────────────────── Gradio UI ──────────────────
with gr.Blocks(title="Qwen2-VL Demo 🚀") as demo:
    gr.Markdown(
        "<h1 style='text-align:center;'>Qwen2-VL 🖼️🗣️ MindSpore Demo</h1>"
    )
    chatbot = gr.Chatbot(
        [], elem_id="chatbot", height=480, bubble_full_width=False
    )
    with gr.Row():
        txt = gr.Textbox(
            lines=2,
            placeholder="请输入文字…",
            scale=4,
            container=False,
        )
        img = gr.Image(
            type="pil",
            label="上传图片",
            scale=1,
            height=120,
            tool=None,
        )
    submit = gr.Button("发送 🚀")
    clear = gr.Button("清空 💫")
    state = gr.State([])

    submit.click(
        chat,
        inputs=[txt, img, state],
        outputs=[chatbot, state],
        show_progress="minimal",
    )
    clear.click(clear_fn, None, [chatbot, state])

demo.launch(server_name="0.0.0.0", server_port=7860, share=False)