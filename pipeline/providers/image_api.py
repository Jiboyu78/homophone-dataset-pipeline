"""云端出图适配器：只要 API Key + 请求地址 + 模型名就能用。

支持两种接口形态（大多数服务商属于其中之一）：

  mode=openai     POST {base_url}/images/generations
                  {"model":…, "prompt":…, "n":1, "size":"1024x1024",
                   "response_format":"b64_json"}
                  适用：OpenAI(gpt-image-1/dall-e-3)、火山方舟 Seedream、
                        硅基流动、各类 one-api/302 网关、以及一切标称
                        "OpenAI 兼容" 的出图服务。

  mode=dashscope  阿里云百炼「通义万相」异步接口：
                  POST {base_url}/api/v1/services/aigc/text2image/image-synthesis
                       header X-DashScope-Async: enable
                  → {"output":{"task_id":…}} → 轮询
                    GET {base_url}/api/v1/tasks/{task_id}
                  → {"output":{"task_status":"SUCCEEDED",
                               "results":[{"url":…}]}}

两种模式都支持"返回 b64"或"返回 url"，
落盘时统一转成真 PNG（数据集里格式不一致会给后续训练添麻烦）。
"""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from ..schema import Item, Plan
from ..utils import LOG
from .image_base import ImageProvider, ImageResult


class ImageAPIError(RuntimeError):
    pass


def _request(url: str, payload: dict | None, api_key: str, timeout: float,
             method: str = "", headers: dict | None = None) -> Any:
    hdrs = {"Content-Type": "application/json"}
    if api_key:
        hdrs["Authorization"] = f"Bearer {api_key}"
    hdrs.update(headers or {})
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=hdrs,
                                method=method or ("POST" if data else "GET"))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:600]
        raise ImageAPIError(f"HTTP {exc.code} {url} -> {detail}") from exc
    except Exception as exc:  # noqa: BLE001
        raise ImageAPIError(f"请求失败 {url}: {exc}") from exc
    try:
        return json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {"_raw": body}


def _save_png(data: bytes, out_path: Path) -> None:
    """统一落盘为真 PNG（服务端可能返回 jpg/webp）。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import io
        from PIL import Image
        with Image.open(io.BytesIO(data)) as im:
            im.convert("RGB").save(out_path, "PNG")
    except Exception:  # noqa: BLE001
        out_path.write_bytes(data)


class ImageAPIProvider(ImageProvider):
    name = "api"

    def __init__(self, cfg: Any):
        super().__init__(cfg)
        import os
        self.mode = str(cfg.get("image.api.mode", "openai")).lower()
        self.base = str(cfg.get("image.api.base_url", "")).rstrip("/")
        self.model = str(cfg.get("image.api.model", ""))
        # 允许只给环境变量，不必把 key 写进配置文件
        self.api_key = (str(cfg.get("image.api.api_key", "") or "")
                        or os.environ.get(str(cfg.get("image.api.api_key_env",
                                                      "HV_IMAGE_API_KEY")), ""))
        self.size = str(cfg.get("image.api.size", "1024x1024"))
        self.timeout = float(cfg.get("image.api.timeout_s", 300))
        self.poll_interval = float(cfg.get("image.api.poll_interval_s", 3.0))
        self.max_poll = float(cfg.get("image.api.max_poll_s", 600))
        self.extra = cfg.get("image.api.extra", {}) or {}
        # 有些服务商的图片接口地址不带 /v1，这里允许两种写法
        self.images_url = self._join(cfg.get("image.api.images_path", ""), default_openai=True)
        self.task_url = self.base + "/api/v1/tasks/{task_id}"

    def _join(self, path: str, default_openai: bool) -> str:
        if path:
            return path if path.startswith("http") else f"{self.base}{path}"
        if self.mode == "dashscope":
            return f"{self.base}/api/v1/services/aigc/text2image/image-synthesis"
        if self.base.endswith("/v1"):
            return f"{self.base}/images/generations"
        return f"{self.base}/v1/images/generations" if default_openai \
            else f"{self.base}/images/generations"

    # ------------------------------------------------------------------ #
    def health(self) -> tuple[bool, str]:
        problems = []
        if not self.base:
            problems.append("image.api.base_url 没填")
        if not self.model:
            problems.append("image.api.model 没填")
        if not self.api_key:
            problems.append("image.api.api_key 没填（或环境变量 HV_IMAGE_API_KEY 为空）")
        if problems:
            return False, "；".join(problems)
        return True, (f"{self.mode} 模式 {self.images_url} 模型 {self.model}"
                      f"（连通性要等第一张图才能确认）")

    # ------------------------------------------------------------------ #
    def generate(self, item: Item, plan: Plan, out_path: Path, seed: int = -1,
                 feedback: str = "") -> ImageResult:
        prompt = self.compose_prompt(plan, feedback)
        negative = self.compose_negative(plan, self.cfg)
        if self.mode == "dashscope":
            raw, used = self._dashscope(prompt, negative)
        else:
            raw, used = self._openai(prompt, negative)
        _save_png(raw, out_path)
        LOG.debug("api 出图: %s (%s, %d 字节)", out_path.name, used, len(raw))
        return ImageResult(path=out_path, seed=-1, provider=f"api:{self.model}",
                           meta={"prompt_used": used, "mode": self.mode})

    # ------------------------------------------------------------------ #
    def _openai(self, prompt: str, negative: str) -> tuple[bytes, str]:
        # OpenAI 的图像接口没有 negative_prompt 字段，把负向词作为
        # "avoid ..." 追加进正向提示词——这是目前唯一可行的传递方式。
        full = prompt
        if negative:
            full = f"{prompt}. Avoid: {negative}"
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": full,
            "n": 1,
            "size": self.size,
        }
        payload.update(self.extra)
        # 同时要 b64 与 url：有的网关只支持其中一种
        payload.setdefault("response_format", "b64_json")
        out = _request(self.images_url, payload, self.api_key, self.timeout)
        if isinstance(out, dict) and out.get("_raw"):
            raise ImageAPIError(f"返回的不是 JSON（{len(out['_raw'])} 字节）")
        if isinstance(out, dict) and out.get("error"):
            raise ImageAPIError(f"服务端报错：{json.dumps(out['error'], ensure_ascii=False)[:400]}")
        data = (out or {}).get("data") or []
        if not data:
            raise ImageAPIError(f"响应里没有 data 字段：{json.dumps(out, ensure_ascii=False)[:400]}")
        first = data[0]
        if first.get("b64_json"):
            return base64.b64decode(first["b64_json"]), "b64"
        if first.get("url"):
            return self._download(first["url"]), "url"
        raise ImageAPIError(f"data[0] 既无 b64_json 也无 url：{str(first)[:200]}")

    def _dashscope(self, prompt: str, negative: str) -> tuple[bytes, str]:
        payload = {
            "model": self.model,
            "input": {"prompt": prompt},
            "parameters": {"size": self.size.replace("x", "*"), "n": 1},
        }
        if negative:
            payload["input"]["negative_prompt"] = negative
        payload.update(self.extra)
        out = _request(self.images_url, payload, self.api_key, self.timeout,
                       headers={"X-DashScope-Async": "enable"})
        task_id = ((out or {}).get("output") or {}).get("task_id")
        if not task_id:
            raise ImageAPIError(f"没拿到 task_id：{json.dumps(out, ensure_ascii=False)[:400]}")
        deadline = time.time() + self.max_poll
        while time.time() < deadline:
            time.sleep(self.poll_interval)
            st = _request(self.task_url.format(task_id=task_id), None,
                          self.api_key, min(self.timeout, 60))
            output = (st or {}).get("output") or {}
            status = str(output.get("task_status", ""))
            if status == "SUCCEEDED":
                results = output.get("results") or []
                if not results:
                    raise ImageAPIError("任务成功但没有 results")
                url = results[0].get("url")
                if not url:
                    raise ImageAPIError(f"results[0] 没有 url：{str(results[0])[:200]}")
                return self._download(url), "url"
            if status in ("FAILED", "CANCELED", "UNKNOWN"):
                raise ImageAPIError(f"任务 {status}："
                                    f"{json.dumps(output, ensure_ascii=False)[:400]}")
        raise ImageAPIError(f"轮询超时（{self.max_poll:.0f}s），task_id={task_id}")

    def _download(self, url: str) -> bytes:
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            raise ImageAPIError(f"下载图片失败 {url}: {exc}") from exc
