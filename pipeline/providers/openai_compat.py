"""OpenAI 兼容的 Chat Completions 客户端（纯标准库）。

可对接：Ollama（/v1）、LM Studio、vLLM、llama.cpp server、以及任何 OpenAI 兼容云服务。
只用到 urllib，无需 requests。
"""
from __future__ import annotations

import base64
import json
import mimetypes
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ..utils import LOG


class OpenAICompatError(RuntimeError):
    pass


def _resolve_key(explicit: str = "", env_name: str = "") -> str:
    import os
    if explicit:
        return explicit
    if env_name:
        return os.environ.get(env_name, "")
    return ""


class OpenAICompatClient:
    def __init__(self, base_url: str, model: str, api_key: str = "",
                 api_key_env: str = "", timeout: float = 180.0, temperature: float = 0.6,
                 max_retry: int = 2, disable_thinking: bool = True):
        self.base = base_url.rstrip("/")
        self.model = model
        self.api_key = _resolve_key(api_key, api_key_env) or "sk-no-key"
        self.timeout = timeout
        self.temperature = temperature
        self.max_retry = max_retry
        # Qwen3 这类模型默认开思维链：出题时会先"想"一大段，既慢又容易把 JSON 带歪。
        # llama.cpp 支持用 chat_template_kwargs 关掉；不支持的提供方会自动降级重试。
        self.disable_thinking = disable_thinking

    # ------------------------------------------------------------------ #
    def chat(self, messages: list[dict[str, Any]], temperature: float | None = None,
             json_mode: bool = False, max_tokens: int = 2048) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            return self._post(payload)
        except OpenAICompatError as exc:
            # 有些提供方不认识 chat_template_kwargs，会直接 400；退一步再试一次
            if self.disable_thinking and "HTTP 400" in str(exc):
                LOG.debug("服务端不接受 chat_template_kwargs，去掉后重试")
                payload.pop("chat_template_kwargs", None)
                return self._post(payload)
            raise

    def _post(self, payload: dict[str, Any]) -> str:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base}/chat/completions", data=data,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                out = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:500]
            raise OpenAICompatError(f"HTTP {exc.code}: {body}") from exc
        except Exception as exc:  # noqa: BLE001
            raise OpenAICompatError(f"请求失败: {exc}") from exc
        try:
            msg = out["choices"][0]["message"]
            content = msg.get("content") or ""
            if not content and msg.get("reasoning_content"):
                content = msg["reasoning_content"]
            return content
        except (KeyError, IndexError) as exc:
            raise OpenAICompatError(f"响应结构异常: {str(out)[:300]}") from exc

    def health(self) -> tuple[bool, str]:
        try:
            req = urllib.request.Request(
                f"{self.base}/models",
                headers={"Authorization": f"Bearer {self.api_key}"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                out = json.loads(resp.read().decode("utf-8"))
            ids = [m.get("id") for m in (out.get("data") or [])][:6]
            return True, f"{self.base} 可用，模型示例：{ids}"
        except Exception as exc:  # noqa: BLE001
            return False, f"无法连接 {self.base}: {exc}"


def image_content(path: Path, detail: str = "high") -> dict[str, Any]:
    """把本地图片编码成 OpenAI 兼容的 image_url 内容块。"""
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}", "detail": detail}}


def extract_json(text: str) -> Any:
    """从模型输出里抠出 JSON（容忍 ```json 围栏与前后废话）。"""
    if not text:
        raise ValueError("空响应")
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1] if "```" in t[3:] else t[3:]
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    # 关键：按"{ 或 [ 哪个先出现"来决定解析顺序。
    # 否则模型输出 "如下：[{...}]" 时会被误抠成里面的单个对象，静默丢掉整批结果。
    candidates: list[tuple[int, int, str]] = []
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = t.find(opener), t.rfind(closer)
        if i >= 0 and j > i:
            candidates.append((i, j, opener))
    candidates.sort()
    for i, j, _opener in candidates:
        try:
            return json.loads(t[i:j + 1])
        except json.JSONDecodeError:
            continue
    raise ValueError(f"无法解析 JSON: {text[:200]}")


def build_llm_client(cfg: Any, prefix: str = "plan.llm") -> OpenAICompatClient:
    return OpenAICompatClient(
        base_url=cfg.get(f"{prefix}.base_url", "http://127.0.0.1:8080/v1"),
        model=cfg.get(f"{prefix}.model", "Qwen3-8B-Q5_K_M.gguf"),
        api_key=cfg.get(f"{prefix}.api_key", ""),
        api_key_env=cfg.get(f"{prefix}.api_key_env", "HV_LLM_API_KEY"),
        timeout=float(cfg.get(f"{prefix}.timeout_s", 180)),
        temperature=float(cfg.get(f"{prefix}.temperature", 0.6)),
        max_retry=int(cfg.get(f"{prefix}.max_retry", 2)),
        disable_thinking=bool(cfg.get(f"{prefix}.disable_thinking", True)),
    )


def build_vlm_client(cfg: Any) -> OpenAICompatClient:
    return OpenAICompatClient(
        base_url=cfg.get("qc.vlm.base_url", "http://127.0.0.1:8080/v1"),
        model=cfg.get("qc.vlm.model", ""),
        api_key=cfg.get("qc.vlm.api_key", ""),
        api_key_env=cfg.get("qc.vlm.api_key_env", "HV_VLM_API_KEY"),
        timeout=float(cfg.get("qc.vlm.timeout_s", 240)),
        temperature=0.1,
        max_retry=int(cfg.get("qc.vlm.max_retry", 2)),
        disable_thinking=bool(cfg.get("qc.vlm.disable_thinking", True)),
    )
