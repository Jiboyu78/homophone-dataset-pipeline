"""步骤 1：从词表里挑候选答案词，输出 AI 好判断的结构化信息。

    # 从 pipeline 已算好的状态库里取（推荐，含可图化打分与语素切分）
    python scripts/pick_words.py --db ../../../output/pipeline.db --limit 40 --out candidates.json

    # 直接从原始 JSON 词表取（不依赖 pipeline）
    python scripts/pick_words.py --words ../../../成语大全_整理版(1).json --limit 40

    # 查单个字/词的义项与英文提示词（设计谜面时随时查）
    python scripts/pick_words.py --lookup 面 心 云

输出 JSON 数组，每条包含 AI 设计谜面所需的全部客观信息：
    text / pinyin / type / category / meaning
    drawable_score     可图化分（越高越适合做会意谜面）
    composition_hint   规则推荐的构图范式（AI 可以改，但改了要给理由）
    segments           语素切分：字 / 语义类 / 英文提示词 / 是否可画 / 是否有义项纠偏
    flags              多义字风险、否定字、词典外字、需要字形参与
    glyph_candidates   可以走"字形参与式"的字（承载语义但画不出来）
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# 允许直接引用同仓库的 pipeline 包（本 skill 与其同源）
_HERE = Path(__file__).resolve().parent
for cand in (_HERE.parent.parent.parent, _HERE.parent.parent):
    if (cand / "pipeline").is_dir():
        sys.path.insert(0, str(cand))
        break

from pipeline.lexicon import LEX, FUNCTION_CHARS, POLYSEMY_RISK  # noqa: E402
from pipeline.pinyin import normalize  # noqa: E402

try:
    from pipeline.plan_rule import glyph_candidates  # noqa: E402
except Exception:  # noqa: BLE001
    glyph_candidates = None


def segment_item(text: str, is_word: bool) -> list[dict]:
    segs = LEX.segment(text, apply_suffix=is_word)
    out = []
    for s in segs:
        out.append({
            "char": s["morpheme"],
            "kind": s["kind"],
            "en": s["en"],
            "drawable": bool(s["drawable"]),
            "weight": round(float(s["weight"]), 2),
            "sense_forced": s.get("sense_forced", ""),
        })
    return out


def glyph_cands(text: str, is_word: bool) -> list[str]:
    if glyph_candidates is not None:
        # 注意：不能在 class 体里写 `text = text`——那里的 text 是类作用域的名字，
        # 取不到外层变量。用 SimpleNamespace 更清楚。
        from types import SimpleNamespace
        item = SimpleNamespace(id="x", text=text,
                               type=("word" if is_word else "idiom"))
        return glyph_candidates(item, LEX.segment(text, apply_suffix=is_word))
    return [c for c in text if c not in FUNCTION_CHARS
            and LEX.entries.get(c, {}).get("mode") == "no"]


def enrich(text: str, is_word: bool, extra: dict | None = None) -> dict:
    segs = segment_item(text, is_word)
    drawable = [s for s in segs if s["drawable"]]
    flags = []
    if sum(1 for c in set(text) if c in POLYSEMY_RISK) >= 2:
        flags.append("多义字风险：先查释义再定元素，别按字面直译")
    if any(c in "不无莫非未勿" for c in text):
        flags.append("含否定字：画面很难表达'没有'，需要留空/断裂/打叉等手法")
    unk = [s["char"] for s in segs if s["kind"] == "unknown"]
    if unk:
        flags.append(f"词典外字：{'、'.join(unk)}（需要你自己给英文提示词）")
    gc = glyph_cands(text, is_word)
    if gc:
        flags.append(f"可考虑字形参与式：{'、'.join(gc)}")
    rec = {
        "text": text,
        "type": "word" if is_word else "idiom",
        "segments": segs,
        "drawable_count": len(drawable),
        "flags": flags,
        "glyph_candidates": gc,
    }
    if extra:
        rec.update(extra)
    return rec


def from_db(db: Path, limit: int, status: str, order: str, only_type: str,
            min_score: float, offset: int, seed: int) -> list[dict]:
    import sqlite3
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    sql = "SELECT * FROM items WHERE select_status=?"
    args: list = [status]
    if only_type:
        sql += " AND type=?"
        args.append(only_type)
    if min_score > 0:
        sql += " AND drawable_score>=?"
        args.append(min_score)
    sql += " ORDER BY drawable_score DESC, text"
    rows = conn.execute(sql, args).fetchall()
    conn.close()
    if order == "random":
        random.Random(seed).shuffle(rows)
    picked = rows[offset:offset + limit] if limit > 0 else rows[offset:]
    out = []
    for r in picked:
        extra = {
            "pinyin": normalize(r["pinyin"]).marked if r["pinyin"] else "",
            "pinyin_numbered": r["pinyin"],
            "category": r["category"],
            "meaning": r["meaning"],
            "drawable_score": round(r["drawable_score"], 3),
            "composition_hint": r["composition"],
            "select_reason": r["select_reason"],
        }
        out.append(enrich(r["text"], r["type"] == "word", extra))
    return out


def from_json(path: Path, limit: int, offset: int, key: str) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(rows, dict):
        rows = next((v for v in rows.values() if isinstance(v, list)), [])
    out = []
    for i, r in enumerate(rows[offset:offset + limit] if limit > 0 else rows[offset:]):
        if not isinstance(r, dict):
            continue
        text = str(r.get(key) or r.get("word") or r.get("成语") or "").strip()
        if not text:
            continue
        py = normalize(str(r.get("拼音") or r.get("pinyin") or "")).marked
        out.append(enrich(text, len(text) <= 3, {"pinyin": py,
                                                "meaning": r.get("释义") or r.get("meaning", "")}))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="", help="pipeline 状态库路径（推荐）")
    ap.add_argument("--words", default="", help="原始 JSON 词表路径")
    ap.add_argument("--key", default="答案词", help="JSON 里词条的字段名")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--status", default="accepted")
    ap.add_argument("--type", default="", choices=["", "idiom", "word"])
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--order", default="score", choices=["score", "random"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lookup", nargs="*", default=[],
                    help="查询字/词的义项与英文提示词")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.lookup:
        print("义项查询（同一字有多个义项时，用同现字判断该取哪个）：")
        for ch in args.lookup:
            senses = LEX.senses.get(ch)
            if not senses:
                print(f"  {ch}: 词典里没有（词典外字，需自己写英文提示词）")
                continue
            primary = LEX.entries[ch]
            print(f"  {ch}: 主条目 kind={primary['kind']} en={primary['en']!r}")
            for s in senses:
                mark = " ←主" if s is primary else ""
                print(f"      - kind={s['kind']:<10} en={s['en']!r}{mark}")
        return 0

    if args.db:
        data = from_db(Path(args.db), args.limit, args.status, args.order,
                       args.type, args.min_score, args.offset, args.seed)
    elif args.words:
        data = from_json(Path(args.words), args.limit, args.offset, args.key)
    else:
        print("请给 --db 或 --words（或 --lookup 查词）")
        return 2

    text = json.dumps(data, ensure_ascii=False, indent=1)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"已写出 {len(data)} 条候选 -> {args.out}")
        print(f"字段：text / pinyin / type / category / meaning / drawable_score /"
              f" composition_hint / segments / flags / glyph_candidates")
        print(f"提示：先用 read 工具看这个文件，再逐条设计谜面。")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
