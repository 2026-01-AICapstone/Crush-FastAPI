import os
from dataclasses import dataclass

MODEL_MODE = os.getenv("MODEL_MODE", "local")  # "local" | "cloud"


@dataclass
class ModelConfig:
    model_name: str
    load_in_4bit: bool   # cloud=True (bitsandbytes INT4), local=False
    device_map: str      # cloud="auto", local="cpu"
    torch_dtype: str     # "float16" | "float32"


_CONFIGS: dict[str, ModelConfig] = {
    # ── AWS g4dn.xlarge (T4 16GB) ──────────────────────────────────────────
    # EXAONE-3.5-2.4B-Instruct: LG AI, LLaMA 기반, 한국어 강세
    # GPU fp16 → VRAM ~2.5GB, T4에서 매우 여유 있음
    "cloud": ModelConfig(
        model_name="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
        load_in_4bit=False,
        device_map="auto",
        torch_dtype="float16",
    ),
    # ── 로컬 개발 / CPU 환경 ────────────────────────────────────────────────
    # 동일 모델 → hook 레이어 번호 cloud와 완전히 일치
    "local": ModelConfig(
        model_name="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
        load_in_4bit=False,
        device_map="cpu",
        torch_dtype="float32",
    ),
}


def get_config() -> ModelConfig:
    if MODEL_MODE not in _CONFIGS:
        raise ValueError(
            f"[config] 알 수 없는 MODEL_MODE='{MODEL_MODE}'. "
            f"유효값: {list(_CONFIGS.keys())}"
        )
    cfg = _CONFIGS[MODEL_MODE]
    print(f"[config] MODE={MODEL_MODE} | model={cfg.model_name} | 4bit={cfg.load_in_4bit}")
    return cfg
