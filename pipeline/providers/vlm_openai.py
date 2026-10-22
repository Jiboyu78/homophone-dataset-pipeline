"""视觉大模型质检器（可选，但强烈建议开）。

做两件独立的事：
  A. 可解性测试（不看答案）：只给图片，让模型猜这是哪个成语/词语。
     —— 衡量"这张图到底能不能被猜出来"，是数据集最核心的质量指标。
  B. 一致性核验（给答案）：检查必备元素是否出现、图上是否写了字、
     是否出现多格/水印/无关主体。

两次调用刻意分开，避免"给了答案再问有没有"的谄媚偏差。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils import LOG
from .openai_compat import (OpenAICompatClient, OpenAICompatError, build_vlm_client,
                            extract_json, image_content)

VERIFY_SYSTEM = (
    "你是一个严格的图像数据集质检员。你只根据画面本身判断，不猜测作者意图。"
    "如果画面里出现任何汉字、字母、数字、水印、签名、对话框，必须如实报告。"
    "只输出 JSON，不要输出任何解释文字。"
)

VERIFY_TEMPLATE = """这是一张给"看图猜成语/词语"数据集用的谜面图。
标准答案（不要在判断中假定它一定对）：{answer}
这张图按设计应当出现的关键视觉元素：{elements}
该谜面采用的构图范式：{composition}
{allowed_text}
请核验并只返回如下 JSON：
{{
  "text_present": true/false,          // 画面中是否存在"计划外"的文字（汉字/字母/数字/水印/签名）
  "text_content": "画面中出现的计划外文字内容，没有则空字符串",
  "elements_present": ["确实出现的关键元素"],
  "elements_missing": ["应当出现但缺失或无法辨认的关键元素"],
  "matches_answer": true/false,        // 仅看图，能否合理指向标准答案
  "problems": ["其他质量问题，如多格拼贴、主体被裁切、画面过乱、风格不统一等"],
  "confidence": 0.0-1.0                // 你对自己判断的信心
}}"""

ALLOWED_TEXT_HINT = ("注意：这类题目属于「字形参与」式，画面中**应当**出现这些汉字：{chars}。"
                     "它们不算问题；只有这些字之外的其他文字才算问题。")

SOLVE_SYSTEM = (
    "你正在参加一个看图猜成语/词语的游戏。看图片，猜出它想表达的那个成语或常用词语。"
    "只输出 JSON。"
)

SOLVE_TEMPLATE = """请先客观描述你在这张图里看到的物体与它们的关系，然后给出你的猜测。
只返回 JSON：
{{
  "objects": ["你看到的物体"],
  "layout": "这些物体是怎么排布的",
  "guess": "你最可能的一个答案（四字成语或常用词语）",
  "alternatives": ["其他可能答案，最多3个"],
  "confidence": 0.0-1.0
}}"""


class VLMChecker:
    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.enabled = str(cfg.get("qc.vlm.provider", "none")).lower() == "openai"
        self.client: OpenAICompatClient | None = build_vlm_client(cfg) if self.enabled else None
        self.check_solvability = bool(cfg.get("qc.vlm.check_solvability", True))

    def health(self) -> tuple[bool, str]:
        if not self.enabled:
            return False, "未启用 VLM 语义质检（qc.vlm.provider=none）"
        assert self.client
        return self.client.health()

    # ------------------------------------------------------------------ #
    def verify(self, image_path: Path, answer: str, elements: list[str],
               composition: str, allowed_text: list[str] | None = None) -> dict[str, Any]:
        assert self.client
        elem_txt = "、".join(elements[:8]) if elements else "（未提供，可自行判断）"
        allowed = [t for t in (allowed_text or []) if t]
        allowed_line = ALLOWED_TEXT_HINT.format(chars="、".join(allowed)) if allowed else ""
        msg = [{"role": "system", "content": VERIFY_SYSTEM},
               {"role": "user", "content": [
                   {"type": "text", "text": VERIFY_TEMPLATE.format(
                       answer=answer, elements=elem_txt, composition=composition,
                       allowed_text=allowed_line)},
                   image_content(image_path, detail="high")]}]
        try:
            raw = self.client.chat(msg, temperature=0.0, json_mode=False, max_tokens=900)
            data = extract_json(raw)
            return data if isinstance(data, dict) else {"error": "非对象 JSON"}
        except (OpenAICompatError, ValueError) as exc:
            LOG.warning("VLM 核验失败 %s: %s", image_path.name, exc)
            return {"error": str(exc)}

    def solve(self, image_path: Path) -> dict[str, Any]:
        assert self.client
        msg = [{"role": "system", "content": SOLVE_SYSTEM},
               {"role": "user", "content": [
                   {"type": "text", "text": SOLVE_TEMPLATE},
                   image_content(image_path, detail="high")]}]
        try:
            raw = self.client.chat(msg, temperature=0.2, json_mode=False, max_tokens=700)
            data = extract_json(raw)
            return data if isinstance(data, dict) else {"error": "非对象 JSON"}
        except (OpenAICompatError, ValueError) as exc:
            LOG.warning("VLM 猜题失败 %s: %s", image_path.name, exc)
            return {"error": str(exc)}
