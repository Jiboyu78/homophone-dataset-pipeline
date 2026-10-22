"""真实出图：用本机 SD WebUI 生成一批真图，走完整流水线（真实 OCR 质检 + 放大 + 打包）。

    python scripts/demo_real_sd.py --limit 8

产物：output_sd/（images/ + dataset.jsonl + review.html + summary.txt）

默认只跑 8 个词：4 个是用户给的参考题，另 4 个是同类机制的成语，方便逐张对照。
显存/内存按"安全配置"：SD 用 --upcast-sampling 起（权重 fp16），本脚本按 768x512 生成、
再 Lanczos 放大到 1152x768 交付；llama.cpp 必须处于停止状态。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import generate, ingest, package, plan_rule, qc, review, select  # noqa: E402
from pipeline.config import build_config  # noqa: E402
from pipeline.providers.image_base import build_provider  # noqa: E402
from pipeline.state import State  # noqa: E402

OUT = ROOT / "output_sd"
WORDS = ["举手之劳", "九牛一毛", "人面兽心", "马首是瞻",
         "守株待兔", "一石二鸟", "火上浇油", "胸有成竹"]
REFERENCE = {
    "举手之劳": "中央巨大「劳」字 + 一只举起的手（字形参与）",
    "九牛一毛": "九牛 3×3 网格 + 一侧一根细毛（数量式）",
    "人面兽心": "人微笑站立，影子却是长角猛兽带红心（表里式）",
    "马首是瞻": "高举马头，众人仰望（动作式）",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=len(WORDS))
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--prompt-style", default="tags", choices=["tags", "prose"],
                    help="tags=短标签串（SD1.5 用这个）；prose=长句描述")
    args = ap.parse_args()

    # 两种提示词风格分开输出，方便并排比对
    out_dir = ROOT / f"output_sd_{args.prompt_style}"
    global OUT
    OUT = out_dir
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.db"):
        old.unlink()
    src = OUT / "_input.json"
    src.write_text(json.dumps(
        [{"id": f"{i:03d}", "答案词": w, "拼音": "", "类型": "成语"}
         for i, w in enumerate(WORDS)], ensure_ascii=False), encoding="utf-8")

    raw = {
        "paths": {"data_dir": str(OUT), "output_dir": str(OUT),
                  "images_dir": str(OUT / "images"), "log_file": str(OUT / "run.log"),
                  "db": str(OUT / "run.db")},
        "ingest": {"files": [str(src)]},
        "select": {"min_score": 0.3, "min_coverage": 0.3},
        "plan": {"provider": "rule", "style_preset": "rebus_cartoon",
                 "prompt_style": args.prompt_style,
                 "width": 768, "height": 512},
        "image": {
            "provider": "sdwebui",
            "concurrency": 1,
            "sdwebui": {"base_url": "http://127.0.0.1:7860", "width": 0, "height": 0,
                        "steps": args.steps, "cfg_scale": 7.0,
                        "sampler_name": "DPM++ 2M", "scheduler": "Karras",
                        "timeout_s": 1200},
            "glyph_overlay": {"enabled": True, "mode": "composite",
                              "position": "center-left", "size_ratio": 0.55},
            "postprocess": {"enabled": True, "target_width": 1152,
                            "target_height": 768, "method": "lanczos"},
        },
        "qc": {"ocr": {"enabled": True, "skip_providers": []},
               "vlm": {"provider": "none"}, "max_retry": 2},
        "package": {"contact_sheets": True, "splits": {"train": 1.0}},
    }
    cfg = build_config(raw, "<real-sd-demo>")

    prov = build_provider(cfg)
    ok, msg = prov.health()
    print(f"生图后端：{'✅ ' if ok else '❌ '}{msg}")
    if not ok:
        print("\nSD WebUI 没起来。按安全配置启动（权重 fp16，省一半显存）：\n"
              "  cd G:\\sd-webui-aki\\sd-webui-aki-v4.11.1-cu128\n"
              "  set GIT_CONFIG_COUNT=1 & set GIT_CONFIG_KEY_0=safe.directory & "
              "set GIT_CONFIG_VALUE_0=*\n"
              "  set HF_ENDPOINT=https://hf-mirror.com\n"
              "  python\\python.exe launch.py --api --upcast-sampling --no-half-vae "
              "--medvram --skip-install --skip-torch-cuda-check --port 7860")
        return 2

    state = State(cfg.db_path)
    try:
        ingest.run(cfg, state)
        select.run(cfg, state, rescore=True)
        plan_rule.run(cfg, state, redo=True)
        res = generate.run(cfg, state, limit=args.limit, mode="all")
        print(f"出图：{res}")
        qcres = qc.run(cfg, state)
        print(f"质检：{qcres}")
        package.run(cfg, state)
        records = package.collect(cfg, state)
        if records:
            review.write_review(cfg, records)
            review.write_contact_sheets(cfg, records)

        lines: list[str] = []
        for text in WORDS:
            item = next((i for i in state.load_items() if i.text == text), None)
            if item is None:
                continue
            plan = state.get_plan(item.id)
            img = state.latest_image(item.id)
            qc_row = state.get_qc(img["id"]) if img else None
            lines.append("=" * 92)
            lines.append(f"【{text}】{plan.composition}"
                         f"{'／字形=' + plan.glyph if plan.glyph else ''}")
            if text in REFERENCE:
                lines.append(f"  参考题：{REFERENCE[text]}")
            lines.append(f"  元素  ：" + " | ".join(
                f"{e.morpheme}={e.en}×{e.count}" for e in plan.elements))
            lines.append(f"  提示词：" + plan.prompt_en[:150] + "...")
            if img:
                lines.append(f"  图片  ：{img['path']}")
            if qc_row:
                lines.append(f"  质检  ：{qc_row.verdict} 分{qc_row.score} "
                             f"通道={qc_row.checker} 无文字={qc_row.text_free} "
                             f"元素覆盖={qc_row.element_coverage}")
                if qc_row.issues:
                    lines.append(f"  问题  ：" + "；".join(qc_row.issues[:2]))
        report = "\n".join(lines)
        (OUT / "summary.txt").write_text(report, encoding="utf-8")
        print(report)
        print(f"\n产物目录：{OUT}")
    finally:
        state.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
