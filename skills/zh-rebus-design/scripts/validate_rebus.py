"""步骤 4：机械校验 AI 产出的谜面 JSONL，把机器能查的错全部查出来。

    python scripts/validate_rebus.py out.jsonl
    python scripts/validate_rebus.py out.jsonl --strict --report report.md

退出码 0 = 全部通过；1 = 有 ERROR（必须改）；2 = 只有 WARNING。
每条记录的必需字段：
    answer, composition, technique, elements[], layout, prompt_en, negative_prompt,
    must_have[], difficulty
可选：
    glyph, panels, distractors[], rationale, meaning, pinyin, style
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for cand in (_HERE.parent.parent.parent, _HERE.parent.parent):
    if (cand / "pipeline").is_dir():
        sys.path.insert(0, str(cand))
        break

from pipeline.lexicon import LEX, FUNCTION_CHARS  # noqa: E402

try:
    from pipeline.plan_rule import glyph_candidates as _gc  # noqa: E402
except Exception:  # noqa: BLE001
    _gc = None

COMPOSITIONS = {"literal", "juxtapose", "spatial", "quantity", "contrast",
                "action", "inner", "metaphor", "homophone"}
TECHNIQUES = {"ideographic", "glyph", "homophone"}
# 不该被当成"画面元素"的语义类：数字/方位/度量是修饰语，虚词和抽象字画不出来
NON_ELEMENT_KINDS = {"number", "direction", "measure", "time", "sound",
                     "function", "abstract", "unknown"}
TEXT_WORDS = ("text", "word", "letter", "character", "chinese", "caption",
              "subtitle", "watermark", "signature", "logo", "stamp", "seal",
              "hanzi", "kanji", "number", "文字", "汉字", "字母", "数字", "水印")
# 结构性构图必须写出的版面关键词
LAYOUT_KEYS = {
    "quantity": ("grid", "row", "arranged", "grid of", "in a row", "网格", "排列"),
    "spatial": ("top", "bottom", "above", "below", "upper", "lower", "inside",
                "between", "left", "right", "上", "下", "中", "里"),
    "inner": ("shadow", "silhouette", "inside", "reflection", "影子", "内心"),
    "contrast": ("left", "right", "contrast", "opposite", "左", "右", "对比"),
    "literal": ("merged", "combined", "made of", "one single", "合成", "融为"),
}


def _segments(text: str, is_word: bool) -> dict[str, dict]:
    return {s["morpheme"]: s for s in LEX.segment(text, apply_suffix=is_word)}


def check_record(rec: dict, idx: int,
                 allowed_extra: set[str] | None = None) -> tuple[list[str], list[str]]:
    errs: list[str] = []
    warns: list[str] = []
    tag = f"[{idx}] {rec.get('answer', '?')}"

    for f in ("answer", "composition", "technique", "elements", "prompt_en",
              "negative_prompt", "must_have", "difficulty"):
        if f not in rec or rec[f] in ("", None, []):
            errs.append(f"{tag}: 缺字段 {f}")
    answer = str(rec.get("answer", ""))
    if not answer:
        return errs, warns
    if not re.fullmatch(r"[\u4e00-\u9fff]+", answer):
        errs.append(f"{tag}: answer 必须是纯汉字")

    comp = str(rec.get("composition", ""))
    if comp not in COMPOSITIONS:
        errs.append(f"{tag}: composition「{comp}」不在 {sorted(COMPOSITIONS)}")
    tech = str(rec.get("technique", ""))
    if tech not in TECHNIQUES:
        errs.append(f"{tag}: technique「{tech}」不在 {sorted(TECHNIQUES)}")

    is_word = len(answer) <= 3
    segs = _segments(answer, is_word)
    elems = rec.get("elements") or []
    if not isinstance(elems, list) or len(elems) < 2:
        errs.append(f"{tag}: elements 至少要 2 个（元素组合才成立）")
    names: list[str] = []
    for e in elems:
        if not isinstance(e, dict):
            errs.append(f"{tag}: elements 项必须是对象")
            continue
        m = str(e.get("morpheme", "")).strip()
        en = str(e.get("en", "")).strip()
        names.append(m)
        if not m:
            errs.append(f"{tag}: 元素缺 morpheme")
            continue
        if m not in answer:
            errs.append(f"{tag}: 元素「{m}」不在答案里（不许凭空造元素）")
            continue
        seg = segs.get(m)
        if seg is not None:
            if seg["kind"] in NON_ELEMENT_KINDS:
                errs.append(f"{tag}: 「{m}」是 {seg['kind']}（修饰语/虚词），"
                            f"不能当画面元素")
            elif not seg["drawable"]:
                errs.append(f"{tag}: 「{m}」词典判为不可画，不能当画面元素")
        elif len(m) > 1 and m in answer and all(
                segs.get(c, {}).get("drawable") for c in m):
            # 合并语素（马首 = 马+首、兽心 = 兽+心）：逐字都可画且连续，视为合法
            pass
        else:
            warns.append(f"{tag}: 元素「{m}」是词典外字，请确认 en 写得足够具体")
        if not en:
            errs.append(f"{tag}: 元素「{m}」缺英文描述 en")
        elif len(en.split()) < 2:
            warns.append(f"{tag}: 元素「{m}」的 en「{en}」太短，"
                         f"建议写成 a/an + 形容词 + 名词")
        if not e.get("role"):
            warns.append(f"{tag}: 元素「{m}」没写 role（subject/object/action/…）")

    # 字形参与式
    glyph = str(rec.get("glyph") or "").strip()
    if tech == "glyph":
        if not glyph or len(glyph) != 1:
            errs.append(f"{tag}: technique=glyph 时必须给单个汉字 glyph")
        elif glyph not in answer:
            errs.append(f"{tag}: glyph「{glyph}」不在答案里")
        else:
            allowed = _gc_for(answer, is_word) if _gc else None
            if allowed is not None:
                if not allowed:
                    # 空集合意味着"这个词没有画不出来的实义字"，即根本不该走字形参与式。
                    # 这里不能跳过检查——早期版本因为 allowed 为空而放行，漏掉了"仓鼠+字形"这种错。
                    errs.append(f"{tag}: 「{answer}」的字全都画得出来，"
                                f"不该用字形参与式（没有承载语义却画不出的字）")
                elif glyph not in allowed:
                    errs.append(f"{tag}: glyph「{glyph}」不是「承载语义但画不出」的字"
                                f"（允许的是 {sorted(allowed)}）")
    elif glyph:
        errs.append(f"{tag}: 给了 glyph 但 technique 不是 glyph")

    # 提示词
    prompt = str(rec.get("prompt_en", ""))
    neg = str(rec.get("negative_prompt", ""))
    if len(prompt.split()) > 80:
        warns.append(f"{tag}: prompt_en 有 {len(prompt.split())} 个词，"
                     f"SD1.5 建议 ≤40、SDXL ≤75")
    low = prompt.lower()
    if tech != "glyph" and not any(w in low for w in ("no text", "no letter",
                                                      "no chinese", "no word")):
        errs.append(f"{tag}: 非字形模式下，prompt_en 必须显式声明不得出现文字")
    if not any(w in neg.lower() for w in TEXT_WORDS):
        errs.append(f"{tag}: negative_prompt 必须包含文字/水印类负向词")
    if tech == "glyph" and any(w in neg.lower() for w in
                               ("chinese character", "hanzi", "kanji", "汉字", "calligraphy")):
        warns.append(f"{tag}: 字形参与式的负向词里禁了汉字，会和 glyph「{glyph}」冲突，"
                     f"建议改成 no other text")
    if "multiple" not in neg.lower() and "panel" not in neg.lower() \
            and "拼贴" not in neg:
        warns.append(f"{tag}: negative_prompt 建议加 multiple panels/collage（防拼贴）")
    if comp in LAYOUT_KEYS:
        keys = LAYOUT_KEYS[comp]
        if not any(k in low for k in keys):
            errs.append(f"{tag}: {comp} 属结构性构图，提示词必须写出版面"
                        f"（{keys[:4]}…），否则模型会画错")

    # must_have / 难度 / 干扰项
    mh = rec.get("must_have") or []
    if mh and elems:
        en_list = {str(e.get("en", "")).lower() for e in elems if isinstance(e, dict)}
        missing = [x for x in mh if str(x).lower() not in en_list]
        if missing and not allowed_extra:
            warns.append(f"{tag}: must_have 里有 {missing[:2]} 与 elements 的 en 不一致"
                         f"（质检核对时会比不出来）")
    try:
        d = int(rec.get("difficulty"))
        if not 1 <= d <= 5:
            errs.append(f"{tag}: difficulty 必须在 1-5")
    except (TypeError, ValueError):
        errs.append(f"{tag}: difficulty 必须是整数")
    dis = rec.get("distractors") or []
    if dis:
        if len(dis) != 3:
            warns.append(f"{tag}: 干扰项建议 3 个，现在是 {len(dis)} 个")
        if answer in dis:
            errs.append(f"{tag}: 干扰项里混入了答案")
        if len(set(dis)) != len(dis):
            errs.append(f"{tag}: 干扰项有重复")
    else:
        warns.append(f"{tag}: 没有干扰项（做选择题时后面可以补）")
    return errs, warns


def _gc_for(answer: str, is_word: bool) -> list[str]:
    if _gc is None:
        return []
    from types import SimpleNamespace
    item = SimpleNamespace(id="x", text=answer,
                           type=("word" if is_word else "idiom"))
    return _gc(item, LEX.segment(answer, apply_suffix=is_word))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--report", default="")
    ap.add_argument("--allow-extra-must-have", action="store_true")
    args = ap.parse_args()

    path = Path(args.jsonl)
    if not path.exists():
        print(f"找不到文件：{path}")
        return 2
    records = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(f"第 {i + 1} 行不是合法 JSON：{exc}")
            return 1
    if not records:
        print("文件里没有记录")
        return 2

    allowed = set() if args.allow_extra_must_have else None
    all_err: list[str] = []
    all_warn: list[str] = []
    for i, rec in enumerate(records, 1):
        e, w = check_record(rec, i, allowed)
        all_err += e
        all_warn += w

    lines = [f"# 谜面校验报告", "", f"记录数：{len(records)}",
             f"ERROR：{len(all_err)}　WARNING：{len(all_warn)}", ""]
    if all_err:
        lines += ["## ERROR（必须修）", ""] + [f"- {x}" for x in all_err] + [""]
    if all_warn:
        lines += ["## WARNING（建议改）", ""] + [f"- {x}" for x in all_warn] + [""]
    if not all_err and not all_warn:
        lines += ["全部通过 ✅", ""]
    report = "\n".join(lines)
    print(report)
    if args.report:
        Path(args.report).write_text(report, encoding="utf-8")
        print(f"报告已写入：{args.report}")
    if all_err:
        return 1
    return 2 if all_warn else 0


if __name__ == "__main__":
    raise SystemExit(main())
