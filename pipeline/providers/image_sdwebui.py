"""Stable Diffusion WebUI (AUTOMATIC1111 / Forge) 生图后端。

比 ComfyUI 简单：POST /sdapi/v1/txt2img，返回 base64 图。
适合只想快速试跑、不想配工作流的场景。启动参数建议见 README。
"""
from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from ..schema import Item, Plan
from ..utils import LOG
from .image_base import ImageProvider, ImageResult


class SDWebUIProvider(ImageProvider):
    name = "sdwebui"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.base = str(cfg.get("image.sdwebui.base_url", "http://127.0.0.1:7860")).rstrip("/")

    def health(self) -> tuple[bool, str]:
        try:
            req = urllib.request.Request(f"{self.base}/sdapi/v1/options")
            with urllib.request.urlopen(req, timeout=5) as resp:
                opt = json.loads(resp.read().decode("utf-8"))
            return True, f"SD WebUI 在线 {self.base}，当前模型 {opt.get('sd_model_checkpoint','?')}"
        except Exception as exc:  # noqa: BLE001
            return False, f"无法连接 SD WebUI {self.base}：{exc}"

    def generate(self, item: Item, plan: Plan, out_path: Path, seed: int = -1,
                 feedback: str = "") -> ImageResult:
        if seed < 0:
            seed = int.from_bytes(uuid.uuid4().bytes[:4], "big") & 0x7FFFFFFF
        payload = {
            "prompt": self.compose_prompt(plan, feedback),
            "negative_prompt": self.compose_negative(plan, self.cfg),
            "width": int(self.cfg.get("image.sdwebui.width") or self.cfg.get("plan.width") or 768),
            "height": int(self.cfg.get("image.sdwebui.height") or self.cfg.get("plan.height") or 768),
            "steps": int(self.cfg.get("image.sdwebui.steps", 28)),
            "cfg_scale": float(self.cfg.get("image.sdwebui.cfg_scale", 6.5)),
            "sampler_name": self.cfg.get("image.sdwebui.sampler_name", "DPM++ 2M"),
            "scheduler": self.cfg.get("image.sdwebui.scheduler", "Karras"),
            "seed": seed,
            "batch_size": 1,
            "save_images": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(f"{self.base}/sdapi/v1/txt2img", data=data,
                                     headers={"Content-Type": "application/json"})
        timeout = float(self.cfg.get("image.sdwebui.timeout_s", 600))
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                out = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"SD WebUI 请求失败 HTTP {exc.code}: "
                               f"{exc.read().decode('utf-8', 'replace')[:400]}") from exc
        imgs = out.get("images") or []
        if not imgs:
            raise RuntimeError(f"SD WebUI 未返回图像: {str(out)[:300]}")
        b64 = imgs[0].split(",", 1)[-1]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        raw = base64.b64decode(b64)
        # A1111 会按 samples_format 设置返回 jpg/webp，而我们的产物约定是 png。
        # 直接写字节会出现"文件名叫 .png 实际是 jpeg"的情况，所以统一转码成真 PNG。
        try:
            from PIL import Image
            import io
            with Image.open(io.BytesIO(raw)) as im:
                im.convert("RGB").save(out_path, "PNG")
            fmt = "PNG(转码)"
        except Exception:  # noqa: BLE001
            out_path.write_bytes(raw)
            fmt = "raw"
        LOG.debug("sdwebui 出图: %s (seed=%s, %s, %d 字节)",
                  out_path.name, seed, fmt, out_path.stat().st_size)
        return ImageResult(path=out_path, seed=seed, provider=self.name,
                           meta={"info": str(out.get("info", ""))[:200], "format": fmt})
