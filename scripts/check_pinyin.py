"""扫描词表的拼音字段：报告非标准/错误之处，并给出改正后的结果。

    python scripts/check_pinyin.py                 # 只报告
    python scripts/check_pinyin.py --fix           # 顺带写出改正后的 JSON
    python scripts/check_pinyin.py --fix --limit 5000

产物：output/pinyin_fixed_idioms.json / pinyin_fixed_words.json
      output/pinyin_report.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.pinyin import check_item, mark_syllable, normalize  # noqa: E402

SOURCES = [
    ("成语表", ROOT / "成语大全_整理版(1).json", "答案词"),
    ("词语表", ROOT / "homophone_answer_vocab.json", "答案词"),
]

# 自检用例：这些是标调规则的"生死线"
SELF_TEST = [
    ("jiu3", "jiǔ"), ("gui4", "guì"), ("liu2", "liú"), ("shui3", "shuǐ"),
    ("hua4", "huà"), ("xue2", "xué"), ("lv4", "lǜ"), ("nv3", "nǚ"),
    ("er2", "ér"), ("yi1", "yī"), ("wu3", "wǔ"), ("zhong1", "zhōng"),
]


def self_test() -> bool:
    ok = True
    print("标调规则自检：")
    for src, want in SELF_TEST:
        got = mark_syllable(src.rstrip("012345"), int(src[-1]))
        flag = "✓" if got == want else "✗"
        if got != want:
            ok = False
        print(f"  {flag} {src:8s} -> {got:8s} (期望 {want})")
    print()
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true", help="写出改正后的 JSON")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    lines: list[str] = []
    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    say("=" * 78)
    say("拼音规范化报告")
    say("=" * 78)

    self_ok = self_test()
    lines.extend(["", "标调自检：" + ("通过" if self_ok else "失败")])

    total = 0
    changed = 0
    kinds: Counter[str] = Counter()
    examples: list[str] = []

    for name, path, key in SOURCES:
        if not path.exists():
            say(f"跳过 {name}：找不到 {path.name}")
            continue
        rows = json.loads(path.read_text(encoding="utf-8-sig"))
        say(f"\n--- {name}（{len(rows)} 条）---")
        fixed_rows = []
        n_bad = 0
        n_syllable_fix = 0
        for i, r in enumerate(rows):
            if args.limit and i >= args.limit:
                fixed_rows.append(r)
                continue
            text = str(r.get(key, ""))
            py = str(r.get("拼音", ""))
            total += 1
            # 只规范化一次，并且带上字数——否则"粘连音节拆分"不会生效（要与 check_item 一致）
            res = normalize(py, expect_chars=len(text) if text else 0)
            problems = list(res.issues)
            if text and res.syllables and abs(len(res.syllables) - len(text)) > 1:
                problems.append(
                    f"音节数 {len(res.syllables)} 与字数 {len(text)} 不匹配：{res.numbered}")
            if res.marked:
                fixed_marked = " ".join(
                    mark_syllable(s, t if t != 5 else 0)
                    for s, t in zip(res.syllables, res.tones))
                if fixed_marked != res.marked:
                    n_syllable_fix += 1
            if problems:
                n_bad += 1
                for p in problems:
                    kinds[p.split("：")[0].split("「")[0][:18]] += 1
                if len(examples) < 25:
                    examples.append(f"  {text}  原「{py}」 -> 规范「{res.numbered}」"
                                    f" / 「{res.marked}」  {'；'.join(problems)}")
            new_row = dict(r)
            new_row["拼音"] = res.numbered
            new_row["拼音_符号"] = res.marked
            if problems:
                new_row["拼音_问题"] = problems
            if res.numbered != py:
                changed += 1
            fixed_rows.append(new_row)
        say(f"  规范化后有变化：{changed} 条（累计）")
        say(f"  本表存在问题：{n_bad} 条；标调位置被纠正：{n_syllable_fix} 条")
        if args.fix:
            out = ROOT / "output" / f"pinyin_fixed_{'idioms' if '成语' in name else 'words'}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(fixed_rows, ensure_ascii=False, indent=1),
                           encoding="utf-8")
            say(f"  已写出：{out}")

    say("")
    say(f"合计扫描 {total} 条，规范后有变化 {changed} 条")
    say("问题类型分布：")
    for k, v in kinds.most_common(10):
        say(f"  {k}: {v}")
    if examples:
        say("\n样例（最多 25 条）：")
        lines.extend(examples)
        for e in examples:
            print(e)

    (ROOT / "output").mkdir(parents=True, exist_ok=True)
    (ROOT / "output" / "pinyin_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已写入：{ROOT / 'output' / 'pinyin_report.txt'}")
    return 0 if self_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
