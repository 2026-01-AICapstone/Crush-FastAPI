import time
from contextlib import asynccontextmanager
from typing import Literal

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import get_config

# ─── 모델 로딩 ────────────────────────────────────────────────────────────────

_model = None
_tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer
    cfg = get_config()

    print(f"[startup] Loading model: {cfg.model_name}")

    _tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    load_kwargs: dict = {
        "device_map": cfg.device_map,
    }

    dtype = torch.float32 if cfg.torch_dtype == "float32" else torch.float16
    load_kwargs["torch_dtype"] = dtype

    _model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **load_kwargs)
    _model.eval()

    print("[startup] Model ready")
    yield

    _model = None
    _tokenizer = None


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

SYSTEM_PROMPT = (
    "당신은 공감 능력이 뛰어난 정신건강 상담 보조 AI입니다. "
    "사용자의 이야기를 경청하고 따뜻하고 안전하게 응답하세요."
)


def build_messages(request: ChatRequest) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in request.conversation_history:
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": request.user_message})
    return messages


def generate(messages: list[dict]) -> str:
    input_ids = _tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    # 모델과 같은 디바이스로 이동
    device = next(_model.parameters()).device
    input_ids = input_ids.to(device)

    with torch.no_grad():
        output_ids = _model.generate(
            input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=_tokenizer.eos_token_id,
            pad_token_id=_tokenizer.eos_token_id,
        )

    # 입력 토큰 이후 생성 부분만 디코딩
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ─── 엔드포인트 ───────────────────────────────────────────────────────────────

@app.post("/ai/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    start = time.time()
    messages = build_messages(request)
    final_response = generate(messages)
    elapsed_ms = int((time.time() - start) * 1000)

    # baseline 모드: risk 계산 없음
    # safety 모드: TODO — AI팀에게 danger_vector.pt + 레이어 번호 받은 후
    #              PyTorch forward hook으로 activation 추출 → 코사인 유사도 계산
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
    from config import MODEL_MODE
    return {
        "status": "ok",
        "mode": MODEL_MODE,
        "model": get_config().model_name,
    }
