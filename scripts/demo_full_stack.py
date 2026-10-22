"""全栈联调：LLM 出题 + 出图 + OCR/VLM 质检 + 打包，一条命令验证所有链路。

它会在本地起一个 OpenAI 兼容的 mock 服务当真模型用，
因此不需要 GPU、不需要 API Key，也能把"计划中"的所有代码路径真实跑一遍。

    python scripts/demo_full_stack.py             # 默认 24 条
    python scripts/demo_full_stack.py --limit 60

产物目录：output_stackdemo/（可以直接打开里面的 review.html 看效果）

注意：mock 大模型只是"接线测试替身"，它的出题质量与猜测结果没有实际意义；
这个脚本证明的是**链路能通、字段能落、闭环能转**，不代表数据集质量。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mock_llm_server import start_server  # noqa: E402

from pipeline import package, runner  # noqa: E402
from pipeline.config import Config, DEFAULTS, _deep_merge  # noqa: E402
from pipeline.state import State  # noqa: E402


def build_config(out_dir: Path, base_url: str) -> Config:
    raw = {
        "paths": {
            "data_dir": str(ROOT / "data"),
            "output_dir": str(out_dir),
            "images_dir": str(out_dir / "images"),
            "log_file": str(out_dir / "demo.log"),
            "db": str(out_dir / "demo.db"),
        },
        "ingest": {"files": ["成语大全_整理版(1).json", "homophone_answer_vocab.json"]},
        # 只跑一部分词，节省时间；真实生产去掉这一行
        "select": {"max_items": 4000},
        "plan": {
            "provider": "llm",
            "llm": {"base_url": base_url, "model": "mock-llm", "batch": 6,
                    "timeout_s": 30, "max_retry": 1},
        },
        "image": {"provider": "mock", "mock": {"width": 512, "height": 512}},
        "qc": {
            "ocr": {"enabled": True, "skip_providers": ["mock"]},
            "vlm": {"provider": "openai", "base_url": base_url, "model": "mock-vlm",
                    "timeout_s": 30, "max_retry": 1, "check_solvability": True},
            "max_retry": 2,
        },
        "package": {"contact_sheets": False, "splits": {"train": 0.7, "val": 0.15,
                                                        "test": 0.15}},
    }
    return Config(_deep_merge(DEFAULTS, raw), "<stackdemo>")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=24, help="出图条数")
    ap.add_argument("--out", default="output_stackdemo")
    args = ap.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.db"):
        f.unlink()

    httpd, base = start_server(0)
    print(f"Mock LLM/VLM 服务：{base}")
    cfg = build_config(out_dir, base)
    cfg.set("paths.output_dir", str(out_dir))
    state = State(cfg.db_path)
    try:
        res = runner.run(cfg, state, limit=args.limit, rounds=3)

        print("\n" + "=" * 74)
        print("链路自检")
        print("=" * 74)
        report = runner.health(cfg, state)
        for k in ("生图后端(mock)", "LLM 出题器", "VLM 质检", "OCR"):
            print(f"  {k}: {report.get(k)}")

        plans = [state.get_plan(i.id) for i in state.load_items(status=["accepted"])]
        plans = [p for p in plans if p]
        print(f"\n  出题器来源分布: {dict(Counter(p.planner for p in plans))}")
        print(f"  构图范式分布: {dict(Counter(p.composition for p in plans))}")

        qc_rows = [json.loads(r["result_json"]) for r in state.all_qc()]
        print(f"\n  质检结论分布: {dict(Counter(r['verdict'] for r in qc_rows))}")
        checker_mix = Counter(r["checker"] for r in qc_rows)
        print(f"  质检通道分布: {dict(checker_mix)}")
        solved = [r for r in qc_rows if r.get("solvable") is not None]
        print(f"  可解性被测过的图: {len(solved)} "
              f"（其中判定可解 {sum(1 for r in solved if r['solvable'])}）")
        tl = [r for r in qc_rows if r.get("text_free") is not None]
        print(f"  文字核验过的图: {len(tl)} "
              f"（其中无文字 {sum(1 for r in tl if r['text_free'])}）")
        retried = [i for i in state.list_images() if i["attempt"] > 1]
        print(f"  触发过重出图的条目: {len({i['item_id'] for i in retried})}")

        records = package.collect(cfg, state)
        print(f"\n  最终数据集记录数: {len(records)}")
        if records:
            r = records[0]
            print(f"  样例字段: answer={r['answer']} composition={r['composition']} "
                  f"difficulty={r['difficulty']} distractors={r['distractors']}")
            print(f"  样例 qc : {json.dumps(r['qc'], ensure_ascii=False)[:220]}")
        print("=" * 74)
        print(f"产物目录：{out_dir}")
        print(f"复核页  ：{out_dir / 'review.html'}")
        ok = (report.get("LLM 出题器", "").startswith("OK")
              and report.get("VLM 质检", "").startswith("OK")
              and report.get("OCR", "").startswith("OK")
              and len(records) > 0
              and any(p.planner == "llm" for p in plans))
        if ok:
            print("\n全栈联调通过 ✅（LLM 出题 / 出图 / OCR / VLM / 闭环 / 打包 均真实执行）")
        else:
            print("\n全栈联调存在问题 ❌")
        return 0 if ok else 1
    finally:
        state.close()
        httpd.shutdown()


if __name__ == "__main__":
    logging.getLogger("hv").setLevel(logging.INFO)
    raise SystemExit(main())
