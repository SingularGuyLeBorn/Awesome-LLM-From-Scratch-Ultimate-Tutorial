# FILE: inference/api_server.py
# -*- coding: utf-8 -*-
"""
[v2.6 - Robustness Check] API Server
- 增加架构兼容性检查，防止 Linear/NSA 模型在 PagedEngine 中崩溃。
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
# [新增]
from utils.model_utils import check_architecture_compatibility


# ... (APIRequest, request_queue definitions ... keep unchanged) ...
@dataclass
class APIRequest:
    seq_id: int
    prompt: str
    prompt_tokens: List[int]
    future: asyncio.Future


request_queue: asyncio.Queue = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 服务器启动中... 正在加载模型和初始化推理引擎...")
    global engine, request_queue

    request_queue = asyncio.Queue()
    args = app.state.args

    project_base_path = Path(__file__).parent.parent.resolve()
    cfg = load_config(args.config_path, project_base_path)

    # [核心新增] 兼容性检查
    if not check_architecture_compatibility(cfg.model, 'inference_paged'):
        print("❌ 错误: 当前模型架构不支持 PagedAttention 推理引擎。")
        print("   请使用 inference/chat.py 进行标准推理，或更换为 MHA/GQA/MLA 架构。")
        # 强制退出，避免后续报错
        sys.exit(1)

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
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

    # ... (Initialization continues) ...
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


# ... (Rest of the file: app definition, models, endpoints, inference_loop, main ... keep unchanged) ...
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


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    global request_queue
    user_message = next((msg["content"] for msg in reversed(request.messages) if msg["role"] == "user"), None)
    if user_message is None: return {"error": "No user message found."}
    formatted_prompt = f"<|im_start|>{user_message}<|im_end|>"
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    seq_id = int(time.time() * 1000)
    prompt_tokens = engine.tokenizer.encode(formatted_prompt).ids
    api_request = APIRequest(seq_id=seq_id, prompt=formatted_prompt, prompt_tokens=prompt_tokens, future=future)
    await request_queue.put(api_request)
    output_tokens = await future
    completion_tokens = output_tokens
    if completion_tokens and completion_tokens[-1] == engine.eos_id: completion_tokens = completion_tokens[:-1]
    if len(completion_tokens) >= len(prompt_tokens) and completion_tokens[:len(prompt_tokens)] == prompt_tokens:
        completion_tokens = completion_tokens[len(prompt_tokens):]
    completion_text = engine.tokenizer.decode(completion_tokens)
    response = ChatCompletionResponse(
        id=f"chatcmpl-{seq_id}", model=request.model,
        choices=[ChatCompletionResponseChoice(index=0, message={"role": "assistant", "content": completion_text},
                                              finish_reason="stop")]
    )
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="启动符合OpenAI标准的FastAPI推理服务器。")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="服务器监听的主机地址。")
    parser.add_argument("--port", type=int, default=8000, help="服务器监听的端口。")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="模型检查点 (.pth) 的路径。")
    parser.add_argument("--config_path", type=str, required=True, help="模型配置文件 (.yaml) 的路径。")
    args = parser.parse_args()
    app.state.args = args
    uvicorn.run(app, host=args.host, port=args.port)
# END OF FILE: inference/api_server.py