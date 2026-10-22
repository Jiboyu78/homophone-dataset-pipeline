"""验证 OCR 文字泄漏检测：造三张图，看判定是否正确。

    python scripts/check_ocr.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from pipeline.config import load_config  # noqa: E402
from pipeline.providers.ocr import OCRChecker  # noqa: E402

TMP = ROOT / ".selftest-tmp" / "ocr"
TMP.mkdir(parents=True, exist_ok=True)


def _font(size: int):
    for p in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make(kind: str) -> Path:
    p = TMP / f"{kind}.png"
    img = Image.new("RGB", (512, 512), (245, 244, 240))
    d = ImageDraw.Draw(img)
    if kind == "clean":
        # 纯图形：一个圆、一个方块，没有任何文字
        d.ellipse([80, 120, 240, 280], fill=(200, 170, 160), outline=(90, 90, 100), width=3)
        d.rectangle([280, 140, 430, 290], fill=(170, 200, 180), outline=(90, 90, 100), width=3)
    elif kind == "chinese":
        d.ellipse([80, 220, 240, 380], fill=(200, 170, 160), outline=(90, 90, 100), width=3)
        d.text((300, 240), "画蛇添足", font=_font(56), fill=(40, 40, 50))
    else:  # latin
        d.rectangle([80, 220, 240, 380], fill=(170, 200, 180), outline=(90, 90, 100), width=3)
        d.text((260, 250), "CAT", font=_font(64), fill=(40, 40, 50))
        d.text((260, 330), "watermark", font=_font(28), fill=(120, 120, 130))
    img.save(p)
    return p


def main() -> int:
    cfg = load_config(str(ROOT / "config.json"))
    checker = OCRChecker(cfg)
    checker._load()  # noqa: SLF001
    print(f"OCR 引擎：{checker.note}")
    if not checker.available:
        print("OCR 不可用：请先执行")
        print('  python -m pip install --target .deps rapidocr-onnxruntime')
        return 2

    cases = [("clean", False), ("chinese", True), ("latin", True)]
    ok = True
    for kind, expect_text in cases:
        path = make(kind)
        out = checker.detect(path)
        got = out.get("has_text")
        texts = [t for t, _ in out.get("texts", [])]
        flag = "✓" if got == expect_text else "✗"
        if got != expect_text:
            ok = False
        print(f"{flag} {kind:8s} 期望有文字={expect_text!s:5s} 实际={got!s:5s} "
              f"字符数={out.get('chars')} 识别内容={texts}")
    print("\nOCR 检测链路验证通过 ✅" if ok else "\nOCR 检测链路存在问题 ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
