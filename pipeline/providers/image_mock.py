"""Mock 生图后端：用 Pillow 画"构图示意卡"。

它不生成美术图，而是把 plan 的构图意图画成示意图（元素框 + 位置 + 标签），
用途有两个：
  1) 没有 GPU / 还没装 ComfyUI 时，先把整条流水线跑通、验证数据结构；
  2) 在烧显卡之前，先人工审一遍"出题构图"是否合理。
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .. import fonts
from ..schema import Item, Plan
from ..utils import LOG
from .image_base import ImageProvider, ImageResult

PALETTE = [
    (232, 180, 160), (170, 200, 180), (180, 190, 220), (225, 205, 160),
    (200, 170, 200), (170, 205, 215), (215, 190, 170), (190, 215, 180),
]


def _load_font(size: int):
    # 跨平台：先前硬编码 C:\Windows\Fonts\，在 macOS/Linux 上中文标签会变方框
    font, _ = fonts.load_font(size)
    return font


class MockProvider(ImageProvider):
    name = "mock"

    def health(self) -> tuple[bool, str]:
        return True, "mock 后端始终可用（生成构图示意卡，不是美术图）"

    def generate(self, item: Item, plan: Plan, out_path: Path, seed: int = -1,
                 feedback: str = "") -> ImageResult:
        w = int(self.cfg.get("image.mock.width") or self.cfg.get("plan.width") or 768)
        h = int(self.cfg.get("image.mock.height") or self.cfg.get("plan.height") or 768)
        draw_labels = bool(self.cfg.get("image.mock.draw_labels", True))

        # 调用序号参与种子：这样"质检不通过 → 重出图"真的会得到一张不同的示意图，
        # 与真实扩散后端（每次都换随机种子）行为一致。
        self._seq = getattr(self, "_seq", 0) + 1
        if seed < 0:
            seed = int(hashlib.sha1(
                f"{item.id}{plan.prompt_en}{feedback}{self._seq}".encode()
            ).hexdigest()[:8], 16)
        rng = _Rng(seed)

        img = Image.new("RGB", (w, h), (250, 248, 244))
        d = ImageDraw.Draw(img)

        # 背景网格，示意"画布"
        for x in range(0, w, 48):
            d.line([(x, 0), (x, h)], fill=(240, 238, 233), width=1)
        for y in range(0, h, 48):
            d.line([(0, y), (w, y)], fill=(240, 238, 233), width=1)
        d.rectangle([8, 8, w - 8, h - 8], outline=(200, 195, 185), width=3)

        elems = [e for e in plan.elements if getattr(e, "en", "")]
        slots = self._slots(plan, len(elems), w, h)
        # 加一点确定性抖动，避免不同词条的示意卡在感知哈希上互相"撞车"
        slots = [(cx + (rng.next() % 25 - 12), cy + (rng.next() % 25 - 12),
                  bw * (0.78 + (rng.next() % 45) / 100.0),
                  bh * (0.78 + (rng.next() % 45) / 100.0))
                 for (cx, cy, bw, bh) in slots]
        font = _load_font(max(14, w // 34))
        small = _load_font(max(12, w // 46))

        for i, e in enumerate(elems):
            cx, cy, bw, bh = slots[i]
            color = PALETTE[i % len(PALETTE)]
            self._draw_shape(d, cx, cy, bw, bh, color, getattr(e, "role", ""),
                             getattr(e, "count", 1) or 1, rng)
            if draw_labels:
                label = f"{e.morpheme} = {getattr(e, 'en', '')}"
                tw = d.textlength(label, font=small)
                tx, ty = cx - tw / 2, cy + bh / 2 + 6
                ty = min(ty, h - 30)
                d.rectangle([tx - 4, ty - 2, tx + tw + 4, ty + small.size + 4],
                            fill=(255, 255, 255))
                d.text((tx, ty), label, font=small, fill=(70, 70, 80))

        # 顶部：构图范式（不写答案，方便盲审）
        # 注意：draw_labels=False 时整张示意卡必须是"零文字"的，
        # 否则想用 mock 跑一遍干净流程的人会被自己的标签坑到。
        if draw_labels:
            head = f"[{plan.composition}] {len(elems)} elements   difficulty {plan.difficulty}"
            d.rectangle([8, 8, w - 8, 46], fill=(255, 255, 255))
            d.text((18, 16), head, font=font, fill=(60, 60, 70))
            if feedback:
                d.text((18, h - 30), f"retry feedback: {feedback[:90]}", font=small,
                       fill=(180, 60, 60))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, "PNG")
        LOG.debug("mock 出图: %s", out_path)
        return ImageResult(path=out_path, seed=seed, provider=self.name,
                           meta={"mock": True, "elements": len(elems)})

    # ------------------------------------------------------------------ #
    def _slots(self, plan: Plan, n: int, w: int, h: int) -> list[tuple[float, float, float, float]]:
        pad = w * 0.12
        avail_w = w - 2 * pad
        avail_h = h - 2 * pad
        comp = plan.composition
        if n <= 0:
            return []
        if comp == "quantity":
            cols = min(n, 3)
            rows = math.ceil(n / cols)
            bw, bh = avail_w / (cols * 1.35), avail_h / (rows * 1.8)
            out = []
            for i in range(n):
                r, c = divmod(i, cols)
                cx = pad + avail_w * (c + 0.5) / cols
                cy = pad + avail_h * (r + 0.5) / rows
                out.append((cx, cy, bw, bh))
            return out
        if comp == "spatial":
            # 上半 / 下半
            half = math.ceil(n / 2)
            out = []
            for i in range(n):
                cy = pad + avail_h * (0.25 if i < half else 0.75)
                cx = pad + avail_w * (0.5 if half == 1 else ((i % half) + 0.5) / half)
                out.append((cx, cy, avail_w / (half * 1.6), avail_h * 0.28))
            return out
        if comp == "contrast":
            out = []
            for i in range(n):
                left = i % 2 == 0
                cx = w * (0.28 if left else 0.72)
                cy = pad + avail_h * ((i // 2) + 0.5) / max(1, math.ceil(n / 2))
                out.append((cx, cy, avail_w * 0.22, avail_h * 0.26))
            return out
        if comp == "literal":
            out = []
            for i in range(n):
                cx = w * (0.34 if i == 0 else 0.66)
                cy = h * 0.55
                out.append((cx, cy, avail_w * 0.26, avail_h * 0.34))
            return out
        # juxtapose / action：横向排列
        bw = avail_w / (n * 1.25)
        return [(pad + avail_w * (i + 0.5) / n, h * 0.55, bw, avail_h * 0.4)
                for i in range(n)]

    @staticmethod
    def _draw_shape(d: ImageDraw.ImageDraw, cx: float, cy: float, bw: float, bh: float,
                    color: tuple[int, int, int], role: str, count: int, rng: "_Rng") -> None:
        bw, bh = max(24.0, bw), max(24.0, bh)
        reps = max(1, min(count, 6))
        for k in range(reps):
            off = (k - (reps - 1) / 2) * (bw * 0.18)
            x0, y0 = cx - bw / 2 + off, cy - bh / 2
            x1, y1 = cx + bw / 2 + off, cy + bh / 2
            shape = rng.choice(["ellipse", "rect", "roundrect", "tri"])
            outline = (90, 90, 100)
            if shape == "ellipse":
                d.ellipse([x0, y0, x1, y1], fill=color, outline=outline, width=3)
            elif shape == "rect":
                d.rectangle([x0, y0, x1, y1], fill=color, outline=outline, width=3)
            elif shape == "roundrect":
                d.rounded_rectangle([x0, y0, x1, y1], radius=min(bw, bh) * 0.2,
                                    fill=color, outline=outline, width=3)
            else:
                d.polygon([(cx + off, y0), (x0, y1), (x1, y1)], fill=color,
                          outline=outline)
        if role == "action":
            d.line([(cx - bw * 0.7, cy + bh * 0.7), (cx + bw * 0.7, cy + bh * 0.7)],
                   fill=(150, 90, 90), width=4)


class _Rng:
    """极简可复现随机数（避免依赖 random 的全局状态）。"""

    def __init__(self, seed: int):
        self.s = (seed or 1) & 0xFFFFFFFF

    def next(self) -> int:
        self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
        return self.s

    def choice(self, seq):
        return seq[self.next() % len(seq)]
