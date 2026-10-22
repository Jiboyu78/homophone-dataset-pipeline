"""检查图片是否含文字（交付前抽查 / 排查 OCR 漏检）。

    python scripts/check_text_leak.py <图片...>
    python scripts/check_text_leak.py --dir output_sd_tags/images
    python scripts/check_text_leak.py --dir output_sd_tags/images --expect 劳

--expect 用于字形参与式：指定"计划内允许出现的字"，其余文字都算问题。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.config import load_config  # noqa: E402
from pipeline.providers.ocr import OCRChecker, score_text_evidence  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*")
    ap.add_argument("--dir", default="")
    ap.add_argument("--expect", default="", help="计划内允许出现的字（字形参与式），其余算泄漏")
    ap.add_argument("--conf", type=float, default=0.6)
    args = ap.parse_args()

    cfg = load_config(str(ROOT / "config.json"))
    if args.conf:
        cfg.set("qc.ocr.min_conf", args.conf)
    ocr = OCRChecker(cfg)
    ocr._load()
    print(f"OCR 引擎：{ocr.note}")
    if not ocr.available:
        return 2

    targets: list[Path] = [Path(p) for p in args.images]
    if args.dir:
        targets += sorted(Path(args.dir).glob("*.png")) + sorted(Path(args.dir).glob("*.jpg"))
    if not targets:
        print("没有指定图片。用法见文件头注释。")
        return 1

    expect = [c for c in args.expect if c]
    bad = 0
    for p in targets:
        if not p.exists():
            print(f"  跳过（不存在）：{p}")
            continue
        out = ocr.detect(p)
        texts = [t for t, _ in out.get("texts", [])]
        filtered = [t for t in texts if not any(e in t for e in expect)] if expect else texts
        ev = score_text_evidence([(t, 1.0) for t in filtered],
                                 float(cfg.get("qc.ocr.weak_char_weight", 0.34)))
        flag = "❌" if filtered else "✓"
        if filtered:
            bad += 1
        print(f"  {flag} {p.name:34s} 检出={texts} "
              f"{'计划外=' + str(filtered) if expect and filtered else ''} "
              f"证据={ev['effective_chars']}")
    print(f"\n共 {len(targets)} 张，可疑 {bad} 张")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
