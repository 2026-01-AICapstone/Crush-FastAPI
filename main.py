import time
from contextlib import asynccontextmanager
from typing import Literal

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ─── 모델 로딩 ────────────────────────────────────────────────────────────────

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
_pipe = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipe
    print(f"[startup] Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    _pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    print("[startup] Model ready")
    yield
    _pipe = None


app = FastAPI(lifespan=lifespan)


# ─── 스키마 ───────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    session_id: str
    user_message: str
    conversation_history: list[Message] = []
    mode: Literal["baseline", "safety"] = "baseline"


class ChatResponse(BaseModel):
    session_id: str
    risk_score: float
    risk_category: str
    detected_layer: int | None
    intervention_triggered: bool
    intervention_type: str
    final_response: str
    processing_time_ms: int


# ─── 추론 ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = "You are a helpful assistant."

def build_messages(request: ChatRequest) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in request.conversation_history:
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": request.user_message})
    return messages


def generate(messages: list[dict]) -> str:
    tokenizer = _pipe.tokenizer
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    output = _pipe(
        prompt,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = output[0]["generated_text"]
    # 프롬프트 이후 생성된 텍스트만 추출
    return generated[len(prompt):].strip()


# ─── 엔드포인트 ───────────────────────────────────────────────────────────────

@app.post("/ai/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    start = time.time()
    messages = build_messages(request)
    final_response = generate(messages)
    elapsed_ms = int((time.time() - start) * 1000)

    # baseline 모드: risk 계산 없음
    # safety 모드: TODO — 실제 모델 적용 후 danger_vector.pt 로드해서 코사인 유사도 계산
    return ChatResponse(
        session_id=request.session_id,
        risk_score=0.0,
        risk_category="none",
        detected_layer=None,
        intervention_triggered=False,
        intervention_type="NONE",
        final_response=final_response,
        processing_time_ms=elapsed_ms,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME}
