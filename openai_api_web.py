# coding=utf-8
# Implements API for Qwen-7B in OpenAI's format. (https://platform.openai.com/docs/api-reference/chat)
# Usage: python openai_api.py
# Visit http://localhost:8000/docs for documents.

from argparse import ArgumentParser
import time
import torch
import uvicorn
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig
# from auto_gptq import AutoGPTQForCausalLM
from sse_starlette.sse import EventSourceResponse

import gradio as gr
import mdtex2html

# codes for OpenAI API

@asynccontextmanager
async def lifespan(app: FastAPI): # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    global model_args
    model_card = ModelCard(id="gpt-3.5-turbo")
    return ModelList(data=[model_card])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer

    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")
    query = request.messages[-1].content

    prev_messages = request.messages[:-1]
    # Temporarily, the system role does not work as expected. We advise that you write the setups for role-play in your query.
    # if len(prev_messages) > 0 and prev_messages[0].role == "system":
    #     query = prev_messages.pop(0).content + query

    history = []
    if len(prev_messages) % 2 == 0:
        for i in range(0, len(prev_messages), 2):
            if prev_messages[i].role == "user" and prev_messages[i+1].role == "assistant":
                history.append([prev_messages[i].content, prev_messages[i+1].content])
            else:
                raise HTTPException(status_code=400, detail="Invalid request.")
    else:
        raise HTTPException(status_code=400, detail="Invalid request.")

    if request.stream:
        generate = predict(query, history, request.model)
        return EventSourceResponse(generate, media_type="text/event-stream")

    response, _ = model.chat_stream(tokenizer, query, history=history)
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response),
        finish_reason="stop"
    )

    return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion")


async def predict(query: str, history: List[List[str]], model_id: str):
    global model, tokenizer

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    current_length = 0

    for new_response in model.chat_stream(tokenizer, query, history):
        if len(new_response) == current_length:
            continue

        new_text = new_response[current_length:]
        current_length = len(new_response)

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(content=new_text),
            finish_reason=None
        )
        chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))


    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    yield '[DONE]'

# codes for web UI
def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert(message),
            None if response is None else mdtex2html.convert(response),
        )
    return y

gr.Chatbot.postprocess = postprocess

def _parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text

def _build_demo(args, model, tokenizer):
    task_history = []

    def predict(_query, _chatbot):
        print("User: " + _parse_text(_query))
        _chatbot.append((_parse_text(_query), ""))
        full_response = ""

        for response in model.chat_stream(tokenizer, _query, history=task_history):
            _chatbot[-1] = (_parse_text(_query), _parse_text(response))

            yield _chatbot
            full_response = _parse_text(response)

        task_history.append((_query, full_response))
        print("Qwen-7B-Chat: " + _parse_text(full_response))

    def regenerate(_chatbot):
        if not task_history:
            yield _chatbot
            return
        item = task_history.pop(-1)
        _chatbot.pop(-1)
        yield from predict(item[0], _chatbot)

    def reset_user_input():
        return gr.update(value="")

    def reset_state():
        task_history.clear()
        return []

    with gr.Blocks() as demo:
        gr.Markdown("""\
<p align="center"><img src="https://modelscope.cn/api/v1/models/qwen/Qwen-7B-Chat/repo?
Revision=master&FilePath=assets/logo.jpeg&View=true" style="height: 80px"/><p>""")
        gr.Markdown("""<center><font size=8>Qwen-7B-Chat Bot</center>""")
        gr.Markdown(
            """\
<center><font size=3>This WebUI is based on Qwen-7B-Chat, developed by Alibaba Cloud. \
(本WebUI基于Qwen-7B-Chat打造，实现聊天机器人功能。)</center>""")
        gr.Markdown("""\
<center><font size=4>Qwen-7B <a href="https://modelscope.cn/models/qwen/Qwen-7B/summary">🤖 </a> 
| <a href="https://huggingface.co/Qwen/Qwen-7B">🤗</a>&nbsp ｜ 
Qwen-7B-Chat <a href="https://modelscope.cn/models/qwen/Qwen-7B-Chat/summary">🤖 </a> | 
<a href="https://huggingface.co/Qwen/Qwen-7B-Chat">🤗</a>&nbsp ｜ 
&nbsp<a href="https://github.com/QwenLM/Qwen-7B">Github</a></center>""")

        chatbot = gr.Chatbot(label='Qwen-7B-Chat', elem_classes="control-height")
        query = gr.Textbox(lines=2, label='Input')

        with gr.Row():
            empty_btn = gr.Button("🧹 Clear History (清除历史)")
            submit_btn = gr.Button("🚀 Submit (发送)")
            regen_btn = gr.Button("🤔️ Regenerate (重试)")

        submit_btn.click(predict, [query, chatbot], [chatbot], show_progress=True)
        submit_btn.click(reset_user_input, [], [query])
        empty_btn.click(reset_state, outputs=[chatbot], show_progress=True)
        regen_btn.click(regenerate, [chatbot], [chatbot], show_progress=True)

        gr.Markdown("""\
<font size=2>Note: This demo is governed by the original license of Qwen-7B. \
We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, \
including hate speech, violence, pornography, deception, etc. \
(注：本演示受Qwen-7B的许可协议限制。我们强烈建议，用户不应传播及不应允许他人传播以下内容，\
包括但不限于仇恨言论、暴力、色情、欺诈相关的有害信息。)""")

    demo.queue()
    return demo

# main codes

def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str,
                        default='Qwen/Qwen-7B-Chat',
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("--cpu-only", action="store_true", help="Run demo with CPU only")
    parser.add_argument("--server-port", type=int, default=8080,
                        help="Demo server port.")
    parser.add_argument("--server-name", type=str, default="127.0.0.1",
                        help="Demo server name.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _get_args()
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "auto"

    if args.checkpoint_path == "Qwen/Qwen-7B-Chat-Int4":
        from auto_gptq import AutoGPTQForCausalLM
        model = AutoGPTQForCausalLM.from_quantized(
            args.checkpoint_path,
            device_map=device_map,
            trust_remote_code=True,
            resume_download=True,
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint_path,
            device_map=device_map,
            trust_remote_code=True,
            resume_download=True,
        ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

    demo = _build_demo(args, model, tokenizer)
    app = gr.mount_gradio_app(app, demo, path="/demo")

    uvicorn.run(app, host=args.server_name, port=args.server_port, workers=1)
