"""生图后端抽象接口。"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..schema import Item, Plan


@dataclass
class ImageResult:
    path: Path
    seed: int = -1
    provider: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


class ImageProvider(abc.ABC):
    name = "base"

    def __init__(self, cfg: Any):
        self.cfg = cfg

    @abc.abstractmethod
    def generate(self, item: Item, plan: Plan, out_path: Path, seed: int = -1,
                 feedback: str = "") -> ImageResult:
        """按 plan 生成一张图并保存到 out_path。

        feedback: 上一轮质检给出的修正建议，会追加到提示词里（重试闭环）。
        """

    def health(self) -> tuple[bool, str]:
        """后端可用性自检，返回 (是否可用, 说明)。"""
        return True, "ok"

    # 任何生图后端都必须带上的"禁止文字"约束
    NO_TEXT_CLAUSE = ("absolutely no text, no letters, no chinese characters, no numbers, "
                      "no captions, no labels, no watermark, no signature")

    @classmethod
    def compose_prompt(cls, plan: Plan, feedback: str = "") -> str:
        """组装正向提示词。

        兜底保证"画面不得出现任何文字"这条硬约束一定在——
        计划可能来自 LLM 或人工编辑，未必带这句，而文字泄漏是这类谜面的头号失败模式。
        """
        prompt = (plan.prompt_en or "").strip()
        low = prompt.lower()
        if "no text" not in low and "no chinese characters" not in low:
            prompt = f"{prompt.rstrip('.')}. {cls.NO_TEXT_CLAUSE}." if prompt \
                else f"{cls.NO_TEXT_CLAUSE}."
        if feedback:
            prompt = f"{prompt} IMPORTANT CORRECTION: {feedback}."
        return prompt

    @staticmethod
    def compose_negative(plan: Plan, cfg: Any) -> str:
        base = plan.negative_prompt or cfg.get("plan.negative_prompt", "")
        # 汉字/字母是这类谜面最常见的失败模式，永远加进负向词
        extra = "text, letters, words, chinese characters, hanzi, kanji, calligraphy"
        return f"{base}, {extra}" if base else extra


def build_provider(cfg: Any):
    """按配置构造生图后端。"""
    name = (cfg.get("image.provider") or "mock").lower()
    if name in ("external", "none", "off"):
        # 外部出图模式：图片不由本程序生成，而是拿 tasks 导出的清单去别处生成，
        # 再用 import-images 导回。这里给明确报错，避免静默什么都不做。
        raise RuntimeError(
            "image.provider=external：本模式不出图。请用 "
            "`pipeline.cli tasks` 导出描述清单，生成后用 "
            "`pipeline.cli import-images --dir <目录>` 导回，"
            "再跑 `run --stages qc,package`。")
    if name in ("api", "openai", "dashscope", "cloud"):
        from .image_api import ImageAPIProvider
        return ImageAPIProvider(cfg)
    if name == "comfyui":
        from .image_comfyui import ComfyUIProvider
        return ComfyUIProvider(cfg)
    if name in ("sdwebui", "a1111", "webui"):
        from .image_sdwebui import SDWebUIProvider
        return SDWebUIProvider(cfg)
    from .image_mock import MockProvider
    return MockProvider(cfg)
