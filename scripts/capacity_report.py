"""产能报告：这份词表最多能做出多少条「会意式单图」数据。

    python scripts/capacity_report.py
    python scripts/capacity_report.py --db output/pipeline.db
    python scripts/capacity_report.py --llm-reject-rate 0.15

核心约束：**会意题至少需要 2 个"画得出来"的语素**。
只有一个可画语素的词（如「蚯蚓」「葡萄」）没法做元素组合题，只能单物直画，
不属于本数据集的题型。所以真正的上限不是"筛选通过数"，而是
"可画语素数 ≥ 2" 的数量。
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.lexicon import LEX  # noqa: E402
from pipeline.pinyin import normalize  # noqa: E402


def _table_ok(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "select count(*) from sqlite_master where type='table' and name=?", (name,)
    ).fetchone()
    return bool(row and row[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="output/pipeline.db")
    ap.add_argument("--min-score", type=float, default=0.55,
                    help="与 config 里 select.min_score 保持一致")
    ap.add_argument("--min-coverage", type=float, default=0.60,
                    help="与 config 里 select.min_coverage 保持一致")
    ap.add_argument("--llm-reject-rate", type=float, default=0.12,
                    help="预估大模型判定 reject 的比例（纯抽象/专名之类）")
    args = ap.parse_args()

    db = ROOT / args.db
    if not db.exists():
        hint = ("python3 scripts/oneclick.py --stages ingest,select"
                if sys.platform != "win32"
                else r"powershell -ExecutionPolicy Bypass -File .\scripts\oneclick.ps1 -Stages ingest,select")
        print(f"找不到状态库：{db}\n先在仓库根目录跑一次：\n  {hint}")
        return 2
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    if not _table_ok(conn, "items"):
        print("状态库里没有 items 表，先跑 ingest 阶段")
        return 2

    rows = conn.execute("select * from items").fetchall()
    by_status = Counter(r["select_status"] for r in rows)
    accepted = [r for r in rows if r["select_status"] == "accepted"]
    # "超出 select.max_items 限制" 被拒的词并不是质量不合格，只是上一轮 -Limit 太小
    # 被截断了。做产能评估时必须把它们算回来，否则会严重低估上限。
    capped = [r for r in rows
              if r["select_status"] == "rejected"
              and "max_items" in (r["select_reason"] or "")]
    rejected_quality = [r for r in rows
                        if r["select_status"] == "rejected"
                        and "max_items" not in (r["select_reason"] or "")]

    # 逐条算"可画语素数"——这才是会意题的真实上限。
    # 优先用库里已存的 segments_json，避免重复切分。
    import json as _json

    def drawable_count(r) -> int:
        try:
            segs = _json.loads(r["segments_json"] or "[]")
        except (ValueError, TypeError):
            segs = []
        if not segs:
            segs = LEX.segment(r["text"], apply_suffix=(r["type"] == "word"))
        return sum(1 for s in segs if s.get("drawable"))

    # 产能要在"全量"上评估，而不是只看上一轮被 cap 之后剩下的那些
    min_score = float(args.min_score)
    min_cov = float(args.min_coverage)
    pool = [r for r in rows
            if float(r["drawable_score"] or 0) >= min_score
            and float(r["coverage"] or 0) >= min_cov]

    buckets = Counter()
    need_glyph = 0
    usable_rows = []
    for r in pool:
        n = drawable_count(r)
        buckets[n] += 1
        if n >= 2:
            usable_rows.append(r)
        try:
            segs = _json.loads(r["segments_json"] or "[]")
        except (ValueError, TypeError):
            segs = []
        if any((not s.get("drawable")) and s.get("kind") not in
               ("function", "number", "direction", "measure", "time", "sound")
               and len(str(s.get("morpheme", ""))) == 1 for s in segs):
            need_glyph += 1

    ge3 = sum(v for k, v in buckets.items() if k >= 3)
    eq2 = buckets.get(2, 0)
    eq1 = buckets.get(1, 0)
    eq0 = buckets.get(0, 0)
    usable = len(usable_rows)

    by_type = Counter(r["type"] for r in usable_rows)
    by_comp = Counter(r["composition"] or "?" for r in usable_rows)
    by_cat = Counter(r["category"] or "未分类" for r in usable_rows)

    print("=" * 74)
    print("产能报告")
    print("=" * 74)
    print(f"状态库：{db}")
    print()
    print("【各阶段条数】")
    print(f"  读入词条        : {len(rows):>7,}")
    print(f"  筛选 accepted   : {by_status.get('accepted', 0):>7,}"
          f"   ← 受上一轮 -Limit 限制（max_items）")
    print(f"  因上限截断而拒  : {len(capped):>7,}   （质量没问题，只是没轮到）")
    print(f"  质量不合格而拒  : {len(rejected_quality):>7,}")
    print(f"  待处理          : {by_status.get('pending', 0):>7,}")
    print()
    print(f"【按全量评估产量】阈值：可图化分≥{min_score} 且 元素覆盖率≥{min_cov}")
    print(f"  进入评估的词    : {len(pool):>7,}")
    print(f"  ≥3 个可画语素   : {ge3:>7,}   （元素丰富，最好做）")
    print(f"  =2 个可画语素   : {eq2:>7,}   （标准两元素组合）")
    print(f"  =1 个可画语素   : {eq1:>7,}   ← 做不了会意题（只能单物直画）")
    if eq0:
        print(f"  =0 个可画语素   : {eq0:>7,}   ← 做不了")
    print("  ---------------------------------------------")
    print(f"  可用于会意题    : {usable:>7,}   ← 这是真正的上限")
    print()
    print("【构成】")
    print(f"  成语 : {by_type.get('idiom', 0):>7,}      词语 : {by_type.get('word', 0):>7,}")
    print(f"  含'画不出的实义字'、可走字形参与式的：约 {need_glyph:,} 条")
    print()
    print("【构图范式分布（筛选阶段预估）】")
    for k, v in by_comp.most_common():
        print(f"  {k:<12} {v:>7,}")
    print()
    print("【原始分类 Top 12】")
    for k, v in by_cat.most_common(12):
        print(f"  {k:<14} {v:>7,}")

    # 已有产出与质检通过率
    img_rows = []
    if _table_ok(conn, "images"):
        img_rows = conn.execute("select * from images").fetchall()
    qc_rows = conn.execute("select * from qc").fetchall() if _table_ok(conn, "qc") else []
    verdicts = Counter((q["verdict"] if "verdict" in qc_rows[0].keys() else "?")
                       for q in qc_rows) if qc_rows else Counter()
    if img_rows or qc_rows:
        print()
        print("【已有产出】")
        print(f"  已出图        : {len(img_rows):>7,}")
        if verdicts:
            tot = sum(verdicts.values())
            print(f"  已质检        : {tot:>7,}")
            for k, v in verdicts.most_common():
                print(f"    {k:<10} {v:>6,}  ({v / tot * 100:.1f}%)")
            acc = verdicts.get("accept", 0)
            if tot:
                print(f"  → 首轮通过率约 {acc / tot * 100:.1f}%"
                      f"（未通过的重试后还能救回一部分）")

    print()
    print("=" * 74)
    print("【结论】")
    print(f"  理论上限（≥2 个可画语素）      : 约 {usable:,} 条")
    print(f"  大模型预计判定画不出来（≈{args.llm_reject_rate * 100:.0f}%）: "
          f"约 {int(usable * args.llm_reject_rate):,} 条")
    plan_ok = int(usable * (1 - args.llm_reject_rate))
    print(f"  出题可用                        : 约 {plan_ok:,} 条")
    print(f"  → 建议按 8:1:1 划分，训练集约 {int(plan_ok * 0.8):,} 条")
    print()
    print("  说明：出图后还有质检关卡。未通过质检的条目会自动带修正建议重试，")
    print("  所以最终产出数量略低于上面数字，但重试不额外消耗额度。")
    print("  想扩产就再给一份词表放进 data/（同一批文件里的词会合并去重）。")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
