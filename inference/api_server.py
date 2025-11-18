# FILE: inference/api_server.py
# -*- coding: utf-8 -*-
"""
[v2.5 - Final Cleaned Version]

- 移除所有调试用的 print 语句，提供一个干净的、可部署的版本。
- 最终确认：服务器逻辑正确，客户端接收问题由客户端工具（如 PowerShell）
  的处理方式导致。本服务器正确地返回了模型的原始、未经修改的输出。
"""
import torch
import argparse
from pathlib import Path
import sys
import time
import asyncio
from typing import List, Dict, Any
from contextlib import asynccontextmanager
import uvicorn
from dataclasses import dataclass

# --- 路径修复 ---
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI
from pydantic import BaseModel, Field
from tokenizers import Tokenizer

from utils.config_loader import load_config
from utils.builders import build_model
from inference.engine.paged_engine import PagedInferenceEngine


# --- 1. 生产者-消费者队列与请求对象 ---

@dataclass
class APIRequest:
    """封装一个API请求的所有信息"""
    seq_id: int
    prompt: str
    prompt_tokens: List[int]
    future: asyncio.Future


# 全局请求队列
request_queue: asyncio.Queue = None


# --- 2. Lifespan 上下文管理器 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 服务器启动中... 正在加载模型和初始化推理引擎...")
    global engine, request_queue

    request_queue = asyncio.Queue()
    args = app.state.args
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    project_base_path = Path(__file__).parent.parent.resolve()
    cfg = load_config(args.config_path, project_base_path)

    model = build_model(cfg.model)
    model.load_state_dict(checkpoint['model_state_dict'])
    tokenizer = Tokenizer.from_file(cfg.data.tokenizer_name)

    device = 'cpu'
    model.to(device)
    try:
        model = model.to(torch.bfloat16)
        print("   -> 模型已转换为 bfloat16。")
    except Exception:
        print("   -> CPU 不支持 bfloat16，将使用 float32。")

    engine = PagedInferenceEngine(model, tokenizer, block_size=16, num_blocks=256)
    print("✅ PagedInferenceEngine 初始化完成！")

    loop = asyncio.get_running_loop()
    app.state.inference_task = loop.create_task(inference_loop())
    print("🔥 推理后台任务已启动。服务器准备就绪！")

    yield

    print("👋 服务器正在关闭... 正在取消推理任务...")
    app.state.inference_task.cancel()
    try:
        await app.state.inference_task
    except asyncio.CancelledError:
        print("   -> 推理任务已成功取消。")
    print("✅ 服务器已关闭。")


# --- 3. FastAPI 应用与 Pydantic 数据模型 ---
app = FastAPI(lifespan=lifespan)


class ChatCompletionRequest(BaseModel):
    model: str = "llm-from-scratch"
    messages: List[Dict[str, str]]


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Dict[str, str]
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]


# --- 4. 核心后台任务：推理循环 ---
async def inference_loop():
    global engine, request_queue
    active_requests: Dict[int, APIRequest] = {}

    while True:
        try:
            while not request_queue.empty():
                new_request = await request_queue.get()
                engine.add_request(prompt=new_request.prompt, seq_id=new_request.seq_id)
                active_requests[new_request.seq_id] = new_request

            if engine.has_unfinished_requests():
                finished_sequences_tokens = engine.step()
                for seq_id, output_tokens in finished_sequences_tokens.items():
                    if seq_id in active_requests:
                        request = active_requests.pop(seq_id)
                        request.future.set_result(output_tokens)
            else:
                await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"🔥 推理循环中出现严重错误: {e}")
            for request in active_requests.values():
                request.future.set_exception(e)
            active_requests.clear()
            await asyncio.sleep(1)


# --- 5. API 端点实现 ---
@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    global request_queue

    user_message = next((msg["content"] for msg in reversed(request.messages) if msg["role"] == "user"), None)
    if user_message is None:
        return {"error": "No user message found."}

    formatted_prompt = f"<|im_start|>{user_message}<|im_end|>"

    loop = asyncio.get_running_loop()
    future = loop.create_future()

    seq_id = int(time.time() * 1000)

    prompt_tokens = engine.tokenizer.encode(formatted_prompt).ids

    api_request = APIRequest(
        seq_id=seq_id,
        prompt=formatted_prompt,
        prompt_tokens=prompt_tokens,
        future=future
    )

    await request_queue.put(api_request)
    output_tokens = await future

    completion_tokens = output_tokens

    if completion_tokens and completion_tokens[-1] == engine.eos_id:
        completion_tokens = completion_tokens[:-1]

    if len(completion_tokens) >= len(prompt_tokens) and completion_tokens[:len(prompt_tokens)] == prompt_tokens:
        completion_tokens = completion_tokens[len(prompt_tokens):]

    # 直接解码，不进行任何字符串处理
    completion_text = engine.tokenizer.decode(completion_tokens)

    response = ChatCompletionResponse(
        id=f"chatcmpl-{seq_id}",
        model=request.model,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message={"role": "assistant", "content": completion_text},
                finish_reason="stop"
            )
        ]
    )
    return response


# --- 6. 启动器 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="启动符合OpenAI标准的FastAPI推理服务器。")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="服务器监听的主机地址。")
    parser.add_argument("--port", type=int, default=8000, help="服务器监听的端口。")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="模型检查点 (.pth) 的路径。")
    parser.add_argument("--config_path", type=str, required=True, help="模型配置文件 (.yaml) 的路径。")
    args = parser.parse_args()

    app.state.args = args
    uvicorn.run(app, host=args.host, port=args.port)

# END FILE: inference/api_server.py