"""用本机 llama.cpp 的真实大模型出题，并和规则出题器逐条对比。

    python scripts/demo_local_llm.py --limit 24
    python scripts/demo_local_llm.py --limit 24 --base-url http://127.0.0.1:8080/v1

产物：output_localllm/（含 plans_compare.txt：同一批词条，规则版 vs 大模型版）

为什么值得单跑一个目录：大模型出题慢（Qwen3-8B 约 35 tok/s，8 条一批要十几秒），
全量 7800 条要一个多小时；先在小批上确认质量再决定要不要全量换。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import ingest, plan_llm, plan_rule, select  # noqa: E402
from pipeline.config import build_config  # noqa: E402
from pipeline.providers.openai_compat import build_llm_client  # noqa: E402
from pipeline.state import State  # noqa: E402

OUT = ROOT / "output_localllm"
# 参考题 + 几个多义字/长词，专门看大模型有没有比规则版更懂语义
WATCH = ["举手之劳", "九牛一毛", "人面兽心", "马首是瞻",
         "人云亦云", "光阴似箭", "胸有成竹", "守株待兔"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=24, help="先跑多少条（大模型出题慢）")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="Qwen3-8B-Q5_K_M.gguf")
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.db"):
        f.unlink()

    raw = {
        "paths": {"data_dir": str(ROOT / "data"),
                  "output_dir": str(OUT), "images_dir": str(OUT / "images"),
                  "log_file": str(OUT / "run.log"), "db": str(OUT / "run.db")},
        "ingest": {"files": ["成语大全_整理版(1).json", "homophone_answer_vocab.json"]},
        "select": {"max_items": 0},
        "plan": {"provider": "llm",
                 "llm": {"base_url": args.base_url, "model": args.model,
                         "batch": args.batch, "timeout_s": 300,
                         "disable_thinking": True}},
        "image": {"provider": "mock"},
    }
    cfg = build_config(raw, "<local-llm-demo>")

    client = build_llm_client(cfg)
    ok, msg = client.health()
    print(f"本机大模型：{'✅ ' if ok else '❌ '}{msg}")
    if not ok:
        print("\n服务没起来。请先双击 C:\\Users\\1\\llama-cpp\\start-server.bat，"
              "等日志出现 'server is listening' 再重跑本脚本。")
        return 2

    state = State(cfg.db_path)
    try:
        ingest.run(cfg, state)
        select.run(cfg, state)
        acc = state.load_items(status=["accepted"], order_by="drawable_score DESC, text")
        print(f"候选池：接受 {len(acc)} 条")
        # 关键：不要按分数取前 N（那样只会拿到最简单的"字面直译"词），
        # 而是点名跑一批**语义上有难度**的词，这才是大模型该发挥的地方。
        watch = [t for t in WATCH if any(i.text == t for i in acc)]
        extra_limit = max(0, args.limit - len(watch))
        extra = [i.text for i in acc[:extra_limit] if i.text not in watch]
        targets = set(watch) | set(extra)
        n = plan_llm.run(cfg, state, only=targets)
        print(f"大模型出题完成：{n} 条（点名 {len(watch)} 条难题 + {len(extra)} 条高分词）\n")

        lines: list[str] = []
        texts = watch + extra[:8]
        for text in texts:
            item = next((i for i in state.load_items() if i.text == text), None)
            if item is None:
                continue
            llm_plan = state.get_plan(item.id)
            rule_plan = plan_rule.build_plan(item, cfg)
            lines.append("=" * 96)
            lines.append(f"【{text}】切分：" + " ".join(
                f"{s['morpheme']}({s['kind']})" for s in item.segments))
            for tag, p in (("大模型", llm_plan), ("规则版", rule_plan)):
                if p is None:
                    lines.append(f"  [{tag}] 无")
                    continue
                lines.append(f"  [{tag}] {p.composition}"
                             f"{' 字形=' + p.glyph if p.glyph else ''}"
                             f" 难度{p.difficulty} 元素{len(p.elements)}")
                lines.append(f"         场景: {p.extra.get('scene_en', '')[:170]}")
                lines.append(f"         元素: " + " | ".join(
                    f"{e.morpheme}={e.en}" for e in p.elements[:5]))
        report = "\n".join(lines)
        (OUT / "plans_compare.txt").write_text(report, encoding="utf-8")
        print(report)
        print(f"\n对照报告：{OUT / 'plans_compare.txt'}")
    finally:
        state.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
