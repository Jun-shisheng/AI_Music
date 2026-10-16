from pathlib import Path
from pydantic_settings import BaseSettings

# Resolve .env relative to project root (4 levels up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"


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
    # Defaults to the checkout sitting next to this repository, so a fresh clone
    # works wherever it was cloned instead of inheriting the path of the machine
    # this was developed on. Override in .env.
    acestep_dir: str = str(_PROJECT_ROOT / "vendor" / "ACE-Step-1.5")


settings = Settings()
