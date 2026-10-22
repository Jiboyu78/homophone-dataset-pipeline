"""端到端验证"一键流程"：用 mock 服务仿真描述模型与出图模型，跑完整个流水线。

    python scripts/test_oneclick.py
    python scripts/test_oneclick.py --limit 12 --mode dashscope

它做的事与 scripts/oneclick.ps1 完全一致（同样的配置结构、同样的阶段顺序），
只是把两个模型的地址指向本地 mock 服务——所以**不需要任何 API Key**
就能确认：出题 → 出图（两种接口形态）→ OCR 质检 → 打包训练格式 全部可用。
真实使用时把地址和 key 换成你自己的即可。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mock_llm_server import start_server  # noqa: E402

from pipeline import runner  # noqa: E402
from pipeline.config import build_config  # noqa: E402
from pipeline.state import State  # noqa: E402

OUT = ROOT / "output_mockrun"
TRAIN_FILES = ["sft_chat", "multiple_choice", "llava", "classification"]


def build_raw(base_url: str, mode: str) -> dict:
    """与 oneclick.ps1 生成的 config.run.json 同构。"""
    return {
        "paths": {"data_dir": str(ROOT / "data"), "output_dir": str(OUT),
                  "images_dir": str(OUT / "images"),
                  "log_file": str(OUT / "pipeline.log"), "db": str(OUT / "pipeline.db")},
        "ingest": {"files": ["成语大全_整理版(1).json", "homophone_answer_vocab.json"]},
        "select": {"min_score": 0.6, "min_coverage": 0.6, "max_items": 240},
        "plan": {"provider": "llm", "prompt_style": "prose",
                 "style_preset": "rebus_cartoon",
                 "llm": {"base_url": base_url, "api_key": "mock-key",
                         "model": "mock-llm", "batch": 4,
                         "timeout_s": 60, "max_retry": 1, "disable_thinking": True,
                         "hard_max": 60}},
        "image": {"provider": "api", "concurrency": 1,
                  "api": {"mode": mode, "base_url": base_url if mode == "dashscope"
                          else base_url.rsplit("/v1", 1)[0] + "/v1",
                          "api_key": "mock-key", "model": "mock-image",
                          "size": "1024x1024",
                          "timeout_s": 60, "poll_interval_s": 0.2, "max_poll_s": 30},
                  "glyph_overlay": {"enabled": True, "mode": "composite"}},
        "qc": {"ocr": {"enabled": True, "skip_providers": []},
               "vlm": {"provider": "none"}, "min_accept_score": 0.75, "max_retry": 2},
        "package": {"splits": {"train": 0.8, "val": 0.1, "test": 0.1}, "seed": 42,
                    "only_accepted": True, "contact_sheets": False,
                    "export_formats": ["raw", "sft_chat", "multiple_choice",
                                       "llava", "classification"],
                    "question": "这张图表达的是哪个成语？"},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--mode", default="openai", choices=["openai", "dashscope"])
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.db"):
        old.unlink()

    httpd, base = start_server(0)
    print(f"mock 服务：{base}（同时仿真 chat / images.generations / dashscope 异步）")
    cfg = build_config(build_raw(base, args.mode), "<oneclick-test>")
    print(f"出图形态：{args.mode}   出图请求地址：{cfg.get('image.api.base_url')}")

    state = State(cfg.db_path)
    problems: list[str] = []
    try:
        res = runner.run(cfg, state, limit=args.limit, rounds=3)
        print("\n各阶段结果：")
        for k, v in res.items():
            if k != "counts":
                print(f"  {k}: {json.dumps(v, ensure_ascii=False)[:160]}")

        # 1) 出题是否真的走了大模型
        plans = [state.get_plan(i.id) for i in state.load_items(status=["accepted"])]
        plans = [p for p in plans if p]
        n_llm = sum(1 for p in plans if p.planner == "llm")
        print(f"\n方案总数 {len(plans)}，其中大模型出题 {n_llm}")
        if n_llm == 0:
            problems.append("没有任何方案标记为 llm 出题——大模型链路没走通")

        # 2) 出图是否真的走了 api 后端
        imgs = state.list_images()
        provs = {i["provider"] for i in imgs}
        print(f"图片 {len(imgs)} 张，来源 {provs}")
        if not any(p.startswith("api:") for p in provs):
            problems.append("没有图片来自 api 出图后端")

        # 3) 训练格式是否齐全且字段正确
        print("\n训练格式文件：")
        for stem in TRAIN_FILES:
            for split in ("train", "val", "test"):
                p = OUT / f"{stem}_{split}.jsonl"
                if p.exists():
                    n = len(p.read_text(encoding="utf-8").strip().splitlines())
                    print(f"  {p.name:34s} {n} 条")
                    break
            else:
                problems.append(f"缺少 {stem}_*.jsonl")
        for name in ("data_card.md", "dataset.jsonl", "manifest.csv", "stats.md",
                     "review.html", "labels.json"):
            if not (OUT / name).exists():
                problems.append(f"缺少 {name}")

        # 4) 抽查格式内容
        sft = OUT / "sft_chat_train.jsonl"
        if sft.exists():
            rec = json.loads(sft.read_text(encoding="utf-8").splitlines()[0])
            msgs = rec.get("messages") or []
            if not (len(msgs) == 2 and isinstance(msgs[0].get("content"), list)
                    and msgs[0]["content"][0].get("type") == "image"):
                problems.append(f"sft_chat 结构不对：{json.dumps(rec, ensure_ascii=False)[:200]}")
            else:
                print(f"  sft_chat 样例：问题={msgs[0]['content'][1]['text']!r} "
                      f"答案={msgs[1]['content']!r}")
        mc = OUT / "multiple_choice_train.jsonl"
        if mc.exists():
            rec = json.loads(mc.read_text(encoding="utf-8").splitlines()[0])
            opts = rec.get("options") or []
            if rec["answer"] != opts[rec["answer_index"]]:
                problems.append("multiple_choice 的 answer_index 指错了")
            if len(opts) < 2:
                problems.append("multiple_choice 选项不足 2 个")
            else:
                print(f"  选择题样例：{rec['question']} {opts} -> "
                      f"{rec['answer_index']} ({rec['answer']})")

        print("\n" + "=" * 74)
        if problems:
            print("存在问题：")
            for p in problems:
                print("  ✗", p)
            return 1
        print("一键流程端到端验证通过 ✅（出题 / 出图 / 质检 / 训练格式 全部跑通）")
        print(f"产物目录：{OUT}")
        return 0
    finally:
        state.close()
        httpd.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
