"""根据用户配置生成完整的运行配置（Windows 与 macOS/Linux 共用）。

    python scripts/make_run_config.py --user config.user.json --out config.run.json \
        --out-dir output --limit 50 --llm-batch 4

为什么单独做成脚本：入口脚本（oneclick.ps1 / oneclick.sh）各写一份配置构造逻辑
必然漂移——之前就出现过"用户配置和运行配置写成同一个文件"把 key 覆盖成 null 的事故。
这里集中一处，两边都只是薄薄一层壳。

--user 指向用户配置（只存 key 与地址），校验规则：
  · 六个必填项都不能为空
  · 地址必须以 http:// 或 https:// 开头
  · API Key 不能是网址（实测有人把两项填反）
校验不过就退出码 2 并打印具体哪一项有问题。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.utils import read_json, write_json  # noqa: E402

REQUIRED = [
    ("llm_base_url", "描述模型 请求地址"),
    ("llm_api_key", "描述模型 API Key"),
    ("llm_model", "描述模型 模型名"),
    ("image_base_url", "出图模型 请求地址"),
    ("image_api_key", "出图模型 API Key"),
    ("image_model", "出图模型 模型名"),
]


def validate(user: dict, require_image: bool = True) -> list[str]:
    """校验用户配置。

    require_image=False 用于"只出图片描述不出图"模式：那次运行根本不会调用
    出图模型，所以出图模型的地址与 key 可以留空。强行要求会让用户卡住——
    他只能胡乱填个占位值（实测有人填了 `0`，然后被格式校验拒掉）。
    """
    bad: list[str] = []
    keys = list(REQUIRED)
    if not require_image:
        keys = [k for k in REQUIRED if not k[0].startswith("image_")]
    for key, label in keys:
        if not str(user.get(key) or "").strip():
            bad.append(f"{label} 为空")
    if bad:
        return bad
    pairs = [("llm_base_url", "llm_api_key", "描述模型")]
    if require_image:
        pairs.append(("image_base_url", "image_api_key", "出图模型"))
    for base_key, key_key, who in pairs:
        base = str(user.get(base_key) or "").strip()
        key = str(user.get(key_key) or "").strip()
        if not base.startswith(("http://", "https://")):
            bad.append(f"{who} 的请求地址必须以 http:// 或 https:// 开头（当前是「{base[:28]}」）")
        if key.startswith(("http://", "https://")):
            bad.append(f"{who} 的 API Key 看起来是个网址——很可能把地址和 key 填反了")
    return bad


def build(user: dict, args: argparse.Namespace) -> dict:
    prompt_style = args.prompt_style or str(user.get("prompt_style") or "prose")
    image_mode = args.image_mode or str(user.get("image_mode") or "openai")
    image_size = args.image_size or str(user.get("image_size") or "1024x1024")

    # 费用控制：只把 目标条数×3（下限 40）条送去大模型出题，
    # 而不是让模型给整个词库（2 万词）出题。
    max_words = max(40, args.limit * 3)

    plan_provider = "rule" if args.plan_provider == "rule" else "llm"
    return {
        "paths": {
            "data_dir": "data",
            "output_dir": args.out_dir,
            "images_dir": f"{args.out_dir}/images",
            "log_file": f"{args.out_dir}/pipeline.log",
            "db": f"{args.out_dir}/pipeline.db",
        },
        "ingest": {"files": ["成语大全_整理版(1).json", "homophone_answer_vocab.json"]},
        "select": {"min_score": 0.55, "min_coverage": 0.6, "max_items": max_words},
        "plan": {
            "provider": plan_provider,
            # 大模型不可用时不许悄悄退回规则出题器：用户要的就是模型读语义。
            "fallback_to_rule": plan_provider == "rule",
            "prompt_style": prompt_style,
            "style_preset": "rebus_cartoon",
            "llm": {
                "base_url": user.get("llm_base_url"),
                "api_key": user.get("llm_api_key"),
                "model": user.get("llm_model"),
                "batch": args.llm_batch,
                "timeout_s": args.llm_timeout,
                "max_retry": 2,
                "disable_thinking": True,
                "hard_max": 200,
            },
        },
        "image": {
            # 只出描述模式把 provider 设成 external：万一有人误跑 generate 阶段，
            # 会拿到一句明确的中文提示，而不是一个莫名其妙的 HTTP 错误。
            "provider": "external" if getattr(args, "no_images", False) else "api",
            "concurrency": args.concurrency,
            "api": {
                "mode": image_mode,
                "base_url": user.get("image_base_url"),
                "api_key": user.get("image_api_key"),
                "model": user.get("image_model"),
                "size": image_size,
                "timeout_s": 300,
                "poll_interval_s": 3.0,
                "max_poll_s": 600,
            },
            "glyph_overlay": {"enabled": True, "mode": "composite"},
            "postprocess": {"enabled": False},
        },
        "qc": {
            "ocr": {"enabled": True, "skip_providers": ["mock"]},
            "vlm": {"provider": "none"},
            "min_accept_score": 0.75,
            "max_retry": 2,
        },
        "package": {
            "splits": {"train": 0.8, "val": 0.1, "test": 0.1},
            "seed": 42,
            "only_accepted": True,
            "contact_sheets": True,
            "export_formats": ["raw", "sft_chat", "multiple_choice", "llava",
                               "classification"],
            "exclude_degraded": bool(args.exclude_degraded),
            "question_idiom": "这张图表达的是哪个成语？",
            "question_word": "这张图表达的是哪个词语？",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="config.user.json")
    ap.add_argument("--out", default="config.run.json")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--llm-batch", type=int, default=4)
    ap.add_argument("--llm-timeout", type=float, default=300)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--prompt-style", default="", choices=["", "prose", "tags"])
    ap.add_argument("--image-mode", default="", choices=["", "openai", "dashscope"])
    ap.add_argument("--image-size", default="")
    ap.add_argument("--plan-provider", default="llm", choices=["llm", "rule"])
    ap.add_argument("--exclude-degraded", action="store_true")
    ap.add_argument("--no-images", action="store_true",
                    help="只出图片描述不出图（不校验也不需要出图模型的地址与 key）")
    args = ap.parse_args()

    user_path = Path(args.user)
    if not user_path.is_absolute():
        user_path = ROOT / user_path
    user = read_json(user_path, default=None)
    if not isinstance(user, dict):
        print(f"读不到用户配置：{user_path}", file=sys.stderr)
        print("请先不带 -Reuse 运行一次，把两个模型的地址与 key 填进去。", file=sys.stderr)
        return 2

    bad = validate(user, require_image=not args.no_images)
    if bad:
        print("配置有问题，未生成运行配置：", file=sys.stderr)
        for b in bad:
            print(f"  - {b}", file=sys.stderr)
        print("\n正确填法：请求地址形如 https://api.deepseek.com/v1", file=sys.stderr)
        print("          API Key  形如 sk-xxxxxxxx（一串密钥，不是网址）", file=sys.stderr)
        return 2

    cfg = build(user, args)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    write_json(out_path, cfg)
    print(f"运行配置已生成：{out_path}")
    print(f"  目标条数 {args.limit}，送去出题 {cfg['select']['max_items']} 条，"
          f"出题方式 {cfg['plan']['provider']}，每批 {args.llm_batch} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
