"""续跑：把待重试的条目重出图并再次质检（真实 SD + 真实 OCR）。

    python scripts/sd_retry_round.py --dir output_sd_tags
    python scripts/sd_retry_round.py --dir output_sd_tags --rounds 2

质检判 retry 的条目会带着"修正建议"回灌提示词重出图；这里就是执行那一轮。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import generate, package, qc, review  # noqa: E402
from pipeline.config import build_config  # noqa: E402
from pipeline.state import State  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="output_sd_tags")
    ap.add_argument("--base-url", default="http://127.0.0.1:7860")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--target-width", type=int, default=1152)
    ap.add_argument("--target-height", type=int, default=768)
    args = ap.parse_args()

    out = (ROOT / args.dir).resolve()
    if not (out / "run.db").exists():
        print(f"找不到状态库 {out / 'run.db'}，请先把 --dir 指向跑过的输出目录")
        return 2

    raw = {
        "paths": {"data_dir": str(out), "output_dir": str(out),
                  "images_dir": str(out / "images"), "log_file": str(out / "run.log"),
                  "db": str(out / "run.db")},
        "plan": {"prompt_style": "tags", "width": 768, "height": 512},
        "image": {
            "provider": "sdwebui", "concurrency": 1,
            "sdwebui": {"base_url": args.base_url, "width": 0, "height": 0,
                        "steps": args.steps, "cfg_scale": 7.0,
                        "sampler_name": "DPM++ 2M", "scheduler": "Karras",
                        "timeout_s": 1200},
            "glyph_overlay": {"enabled": True, "mode": "composite"},
            "postprocess": {"enabled": True, "target_width": args.target_width,
                            "target_height": args.target_height},
        },
        "qc": {"ocr": {"enabled": True, "skip_providers": []},
               "vlm": {"provider": "none"}, "max_retry": 3},
        "package": {"contact_sheets": False, "splits": {"train": 1.0}},
    }
    cfg = build_config(raw, "<retry-round>")
    state = State(cfg.db_path)
    try:
        for r in range(1, args.rounds + 1):
            pending = generate.pending_retries(cfg, state)
            print(f"--- 第 {r} 轮：待重试 {pending} 条 ---")
            if not pending:
                break
            print("重出图：", generate.run(cfg, state, mode="retry"))
            print("质检  ：", qc.run(cfg, state))
        package.run(cfg, state)
        records = package.collect(cfg, state)
        if records:
            review.write_review(cfg, records)
        print(f"\n最终 {len(records)} 条：")
        for rec in records:
            print(f"  {rec['answer']:10s} {rec['composition']:10s} "
                  f"分{rec['qc']['score']:<5} 无文字={rec['qc']['text_free']} "
                  f"通道={rec['qc']['checker']}  {Path(rec['image']).name}")
    finally:
        state.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
