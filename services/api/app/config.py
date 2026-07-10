from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    acestep_model_dir: str = "./models/ace-step-1.5"
    output_dir: str = "./outputs"
    database_url: str = "sqlite:///./data/app.db"
    api_host: str = "127.0.0.1"
    api_port: int = 8000


settings = Settings()
