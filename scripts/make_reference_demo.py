"""按参考题生成一组对照样例，用来和你给的 4 张真实题目逐一比对。

    python scripts/make_reference_demo.py

产物：output_reference/
    images/*.png       4 张图（举手之劳 会走"字形叠加"，画面里会有正确的「劳」字）
    review.html        复核页
    dataset.jsonl      4 条记录
    summary.txt        逐题对照说明
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import generate, ingest, package, plan_rule, qc, review, runner, select  # noqa: E402
from pipeline.config import build_config  # noqa: E402
from pipeline.state import State  # noqa: E402

OUT = ROOT / "output_reference"
WORDS = ["举手之劳", "九牛一毛", "人面兽心", "马首是瞻"]
# 参考题给的人工说明，用于逐条对照（仅作打印，不参与生成）
REFERENCE = {
    "举手之劳": "画面中央大字「劳」+ 一只举起的手；举+手+之+劳",
    "九牛一毛": "九牛排成三行三列，右侧一条黑色细线（一根毛）",
    "人面兽心": "男子又腰微笑，影子是长角利爪、黄眼、胸口红心的黑色怪物",
    "马首是瞻": "大个子士兵持马头形木槌在前，三名小兵跟随",
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.db"):
        old.unlink()
    src = OUT / "_input.json"
    src.write_text(json.dumps(
        [{"id": f"{i:03d}", "答案词": w, "拼音": "", "类型": "成语"}
         for i, w in enumerate(WORDS)], ensure_ascii=False), encoding="utf-8")

    raw = {
        "paths": {**{k: str(OUT / v) for k, v in
                     {"output_dir": "", "images_dir": "images", "log_file": "run.log",
                      "db": "run.db"}.items()}},
        "ingest": {"files": [str(src)]},
        "select": {"min_score": 0.3, "min_coverage": 0.3},
        "plan": {"provider": "rule", "style_preset": "rebus_cartoon"},
        "image": {"provider": "mock",
                  "mock": {"width": 0, "height": 0, "draw_labels": False},
                  "glyph_overlay": {"enabled": True, "mode": "composite"}},
        "qc": {"ocr": {"enabled": True, "skip_providers": []},
               "vlm": {"provider": "none"}, "max_retry": 2},
        "package": {"contact_sheets": False, "splits": {"train": 1.0}},
    }
    raw["paths"]["output_dir"] = str(OUT)
    raw["paths"]["images_dir"] = str(OUT / "images")
    cfg = build_config(raw, "<reference-demo>")

    state = State(cfg.db_path)
    lines: list[str] = []
    try:
        ingest.run(cfg, state)
        select.run(cfg, state, rescore=True)
        plan_rule.run(cfg, state, redo=True)
        generate.run(cfg, state, mode="all")
        qc.run(cfg, state)
        package.run(cfg, state)

        for text in WORDS:
            item = next((i for i in state.load_items() if i.text == text), None)
            if item is None:
                lines.append(f"【{text}】未入库")
                continue
            plan = state.get_plan(item.id)
            lines.append("=" * 88)
            lines.append(f"【{text}】参考题：{REFERENCE[text]}")
            lines.append(f"  构图范式 : {plan.composition}"
                         f"（{plan.extra.get('composition_zh')}）"
                         f"   手法：{plan.extra.get('technique_zh')}")
            if plan.glyph:
                lines.append(f"  字形参与 : 画面中会写入「{plan.glyph}」"
                             f"（{plan.extra['glyph'].get('mode')} 模式，"
                             f"位置 {plan.extra['glyph'].get('position')}）")
            lines.append(f"  构图说明 : {plan.layout}")
            lines.append(f"  元素     : " + " | ".join(
                f"{e.morpheme}={e.en}×{e.count}" for e in plan.elements))
            lines.append(f"  正向提示 : {plan.prompt_en[:200]}...")
            if plan.rationale:
                lines.append(f"  rationale: {plan.rationale}")
        records = package.collect(cfg, state)
        if records:
            review.write_review(cfg, records)
        lines.append("=" * 88)
        lines.append(f"图片目录：{cfg.images_dir}")
        lines.append(f"复核页  ：{OUT / 'review.html'}")
    finally:
        state.close()

    text = "\n".join(lines)
    (OUT / "summary.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
