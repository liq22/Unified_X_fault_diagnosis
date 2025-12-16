from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


def load_dotenv(dotenv_path: str | Path | None = None, override: bool = False) -> Dict[str, str]:
    """
    读取项目根目录 `.env`（或指定路径）并写入 os.environ。

    - 不依赖第三方库（避免离线环境安装依赖）。
    - 默认不覆盖已存在的环境变量（override=False）。
    - `.env` 必须 gitignore；仓库仅提供 `.env.example`。
    """

    if dotenv_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        dotenv_path = repo_root / ".env"
    dotenv_path = Path(dotenv_path)
    if not dotenv_path.exists():
        return {}

    parsed: Dict[str, str] = {}
    for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")  # 简单去引号
        if not key:
            continue
        parsed[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return parsed


@dataclass(frozen=True)
class LLMSettings:
    """
    LLM 配置：必须来自环境变量（支持 `.env`），缺失时自动降级 mock，不中断非 LLM 流程。
    """

    provider: str
    openai_api_key: Optional[str]
    anthropic_api_key: Optional[str]
    deepseek_api_key: Optional[str]
    glm_api_key: Optional[str]
    base_url: Optional[str]

    @classmethod
    def from_env(cls) -> "LLMSettings":
        load_dotenv()
        provider = os.getenv("LLM_PRIMARY_PROVIDER", "mock")
        return cls(
            provider=provider,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
            glm_api_key=os.getenv("GLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
        )

    def is_configured(self) -> bool:
        if self.provider in {"mock", "template"}:
            return True
        if self.provider == "openai":
            return bool(self.openai_api_key)
        if self.provider == "anthropic":
            return bool(self.anthropic_api_key)
        if self.provider == "deepseek":
            return bool(self.deepseek_api_key)
        if self.provider == "glm":
            return bool(self.glm_api_key)
        return False

