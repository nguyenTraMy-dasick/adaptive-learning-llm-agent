"""Client gọi LLM cho LLMPolicy, không phụ thuộc thư viện ngoài (chỉ dùng urllib).

`OpenAICompatibleClient` gọi endpoint `POST {base_url}/chat/completions` theo định dạng
OpenAI Chat Completions. Nhiều nhà cung cấp và server local hỗ trợ định dạng này;
đổi nhà cung cấp chỉ cần đổi biến môi trường, không sửa code policy.

Biến môi trường (xem .env.example):
    LLM_BASE_URL     ví dụ https://api.openai.com/v1 hoặc http://localhost:11434/v1
    LLM_API_KEY      khóa API (có thể để trống với server local)
    LLM_MODEL        tên model
    LLM_TEMPERATURE  mặc định 0 để kết quả ổn định nhất có thể
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Mapping, Sequence


class LLMClientError(RuntimeError):
    pass


def load_env_file(path: str | os.PathLike[str] = ".env", override: bool = False) -> dict[str, str]:
    """Đọc file .env đơn giản (KEY=VALUE, bỏ dòng trống và #) vào os.environ.

    Không ghi đè biến đã có trừ khi override=True. Trả về các biến đã đọc.
    """
    loaded: dict[str, str] = {}
    if not os.path.exists(path):
        return loaded
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip('"').strip("'")
            loaded[key] = value
            if override or key not in os.environ:
                os.environ[key] = value
    return loaded


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout: float = 60.0,
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        if not base_url or not model:
            raise ValueError("base_url and model are required")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.extra_body = dict(extra_body or {})

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> OpenAICompatibleClient:
        env = os.environ if environ is None else environ
        missing = [k for k in ("LLM_BASE_URL", "LLM_MODEL") if not env.get(k)]
        if missing:
            raise LLMClientError(f"Missing environment variables: {', '.join(missing)}")
        return cls(
            base_url=env["LLM_BASE_URL"],
            model=env["LLM_MODEL"],
            api_key=env.get("LLM_API_KEY") or None,
            temperature=float(env.get("LLM_TEMPERATURE", "0") or 0),
        )

    def build_request(self, messages: Sequence[Mapping[str, str]]) -> urllib.request.Request:
        body = {
            "model": self.model,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "temperature": self.temperature,
            **self.extra_body,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )

    def _send(self, request: urllib.request.Request) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise LLMClientError(f"HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMClientError(f"Cannot reach {self.base_url}: {exc.reason}") from exc

    def __call__(self, messages: Sequence[Mapping[str, str]]) -> str:
        data = self._send(self.build_request(messages))
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError(f"Unexpected response format: {str(data)[:300]}") from exc
        if not isinstance(content, str):
            raise LLMClientError("Response content is not text")
        return content
