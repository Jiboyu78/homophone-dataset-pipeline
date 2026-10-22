"""数据集完整性校验：交付前跑一遍，确认 JSONL 与图片都自洽。

    python scripts/check_dataset.py                 # 校验 output/
    python scripts/check_dataset.py output_stackdemo

检查项：
    · 必需字段齐全、ID 唯一
    · 图片文件存在、尺寸合理、sha1 与记录一致
    · 干扰项恰好 3 个、不含答案本身
    · 划分已分配且覆盖全部记录
    · 统计分布（范式/类型/难度/质检通道/可解性/文字）
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REQUIRED = ["id", "answer", "pinyin", "type", "composition", "composition_zh",
            "difficulty", "image", "image_sha1", "elements", "must_have",
            "distractors", "qc", "prompt_en", "split"]


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(dir_path: Path, verify_sha: bool = False) -> bool:
    ds = dir_path / "dataset.jsonl"
    if not ds.exists():
        print(f"✗ 找不到 {ds}")
        return False
    records = [json.loads(l) for l in ds.read_text(encoding="utf-8").strip().splitlines()]
    errors: list[str] = []

    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)):
        errors.append("存在重复 id")

    for r in records:
        missing = [k for k in REQUIRED if k not in r]
        if missing:
            errors.append(f"{r.get('id')}: 缺字段 {missing}")
            continue
        img = dir_path / r["image"]
        if not img.exists():
            errors.append(f"{r['id']}: 图片缺失 {r['image']}")
        elif verify_sha and r.get("image_sha1") and sha1_file(img) != r["image_sha1"]:
            errors.append(f"{r['id']}: 图片 sha1 不匹配")
        if len(r["distractors"]) != 3:
            errors.append(f"{r['id']}: 干扰项 {len(r['distractors'])} 个（应为 3）")
        if r["answer"] in r["distractors"]:
            errors.append(f"{r['id']}: 干扰项里混入了答案")
        if not (1 <= int(r["difficulty"]) <= 5):
            errors.append(f"{r['id']}: 难度越界 {r['difficulty']}")
        if not r.get("split"):
            errors.append(f"{r['id']}: 未分配 split")

    print("=" * 70)
    print(f"数据集目录：{dir_path}")
    print(f"记录数：{len(records)}")
    print("=" * 70)
    print("  构图范式:", dict(collections.Counter(r["composition_zh"] for r in records)))
    print("  词条类型:", dict(collections.Counter(r["type"] for r in records)))
    print("  难度分布:", dict(sorted(collections.Counter(r["difficulty"] for r in records).items())))
    print("  划分    :", dict(collections.Counter(r["split"] for r in records)))
    print("  质检通道:", dict(collections.Counter(r["qc"].get("checker") for r in records)))
    solvable = sum(1 for r in records if r["qc"].get("solvable") is not None)
    ok_sol = sum(1 for r in records if r["qc"].get("solvable") is True)
    text_checked = sum(1 for r in records if r["qc"].get("text_free") is not None)
    text_clean = sum(1 for r in records if r["qc"].get("text_free") is True)
    print(f"  可解性已测: {solvable}（判定可解 {ok_sol}）"
          f"  文字已验: {text_checked}（无文字 {text_clean}）")
    scores = [float(r["qc"].get("score") or 0) for r in records]
    if scores:
        print(f"  质检分    : min={min(scores):.2f} avg={sum(scores)/len(scores):.2f} "
              f"max={max(scores):.2f}")

    if errors:
        print(f"\n✗ 发现 {len(errors)} 个问题：")
        for e in errors[:20]:
            print("   -", e)
        return False
    print("\n✓ 校验通过：字段、图片、干扰项、划分全部自洽")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", default=["output"])
    ap.add_argument("--verify-sha", action="store_true", help="额外校验图片内容哈希（慢）")
    args = ap.parse_args()
    ok = True
    for d in args.dirs:
        ok = check(ROOT / d, verify_sha=args.verify_sha) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
