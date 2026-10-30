from pathlib import Path
from pydantic_settings import BaseSettings

# Resolve .env relative to project root (2 levels up from this file)
_ENV_PATH = Path(__file__).resolve().parent.parent.parent.parent / ".env"


class Settings(BaseSettings):
    model_config = {"env_file": str(_ENV_PATH), "env_file_encoding": "utf-8", "extra": "ignore"}

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    acestep_model_dir: str = "./models/ace-step-1.5"
    output_dir: str = "./outputs"
    database_url: str = "sqlite:///./data/app.db"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    acestep_backend: str = "subprocess"
    acestep_api_url: str = "http://127.0.0.1:8765"
    acestep_dir: str = "D:/Model/AI_Music/vendor/ACE-Step-1.5"
    acestep_lora_path: str = ""
    acestep_lora_scale: float = 0.8
    acestep_lora_adapter_name: str = "ai_music"
    acestep_inference_steps: int = 25
    acestep_guidance_scale: float = 7.0
    # Post-generation vocal humanization (demucs -> frame repair -> de_buzz EQ).
    # Disable to get the raw ACE-Step output. (Only used by the ACE-Step backend.)
    vocal_humanize_enabled: bool = True
    vocal_humanize_variant: str = "de_buzz"
    # Generation backend: "heartmula" (redesigned source-clean pipeline,
    # recommended) or "acestep" (legacy draft backend).
    generation_backend: str = "heartmula"


settings = Settings()
