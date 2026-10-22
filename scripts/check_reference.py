"""按参考题验收：把 4 张真实题目的机制固化成断言，防止以后再跑偏。

    python scripts/check_reference.py

参考样本（来自用户提供的真实题目截图）：
    IMG_07181 举手之劳  画面中央巨大「劳」字 + 一只举起的手   -> 字形参与式
    IMG_07182 九牛一毛  九头牛排成 3×3 + 一侧一根细毛        -> 数量式
    IMG_07183 人面兽心  男孩微笑站立，影子却是长角怪物+红心  -> 表里式（外表 vs 内心）
    IMG_07184 马首是瞻  前排士兵高举马头木槌，三名小兵跟随仰望 -> 动作式（马的头 + 仰望）

这些断言覆盖了参考图暴露出来的 4 类真实缺陷：
    · 多义字解错（人面兽心的"面"被当成"面条"）
    · 缺"表里式"构图
    · 缺"字形参与式"（原来硬禁汉字，这类题根本画不出来）
    · 动物+身体部位不合并（马首 -> "马"+"人头"）
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import plan_rule  # noqa: E402
from pipeline.config import load_config  # noqa: E402
from pipeline.lexicon import LEX  # noqa: E402
from pipeline.schema import Item  # noqa: E402

# (词条, 期望构图, 期望手法, 期望字形, 必须出现在提示词里的片段, 不得出现的片段)
CASES = [
    ("举手之劳", {"action", "inner", "juxtapose"}, "glyph", "劳",
     ["hand"],
     ["a bowl of noodles"]),
    ("九牛一毛", {"quantity"}, "ideographic", "",
     ["nine identical oxen", "hair", "3 by 3"],
     ["noodles", "oxes"]),
    ("人面兽心", {"inner"}, "ideographic", "",
     ["shadow", "human face", "beast", "heart"],
     ["noodles", "a horse"]),
    ("马首是瞻", {"action", "juxtapose", "spatial"}, "ideographic", "",
     ["head of", "horse", "gaz"],
     ["a human head"]),
]

_results: list[tuple[str, bool, list[str]]] = []


def check_one(cfg, text: str, comps: set[str], technique: str, glyph: str,
              must: list[str], must_not: list[str]) -> tuple[bool, list[str]]:
    item = Item(id="ref_" + text, text=text, type="idiom", pinyin="",
                category="成语", segments=LEX.segment(text))
    plan = plan_rule.build_plan(item, cfg)
    mode = str((cfg.get("image.glyph_overlay") or {}).get("mode", "composite")).lower()
    hay = (plan.prompt_en + " " + plan.extra.get("scene_en", "")).lower()
    problems: list[str] = []

    # 字形参与式：合成模式下提示词应"预留空白"（汉字由字体叠加），
    # 提示词模式下才应让模型自己写这个字。两种模式的期望不同，不能混用。
    must = list(must)
    if glyph and mode == "composite":
        must = [f for f in must if f != "no other text besides"]
        must.append("clean empty area")
    elif glyph and mode == "prompt":
        must.append("劳")

    if plan.composition not in comps:
        problems.append(f"构图范式 {plan.composition} 不在期望集合 {sorted(comps)}")
    if plan.technique != technique:
        problems.append(f"手法 {plan.technique} != 期望 {technique}")
    if glyph and plan.glyph != glyph:
        problems.append(f"字形字段 {plan.glyph!r} != 期望 {glyph!r}")
    if not glyph and plan.glyph:
        problems.append(f"不该出现字形，却给了 {plan.glyph!r}")
    for frag in must:
        if frag.lower() not in hay:
            problems.append(f"提示词缺少关键片段：{frag!r}")
    for frag in must_not:
        if frag.lower() in hay:
            problems.append(f"提示词出现错误内容：{frag!r}")

    print("=" * 92)
    print(f"【{text}】构图={plan.composition} 手法={plan.technique} "
          f"字形={plan.glyph or '无'} 难度={plan.difficulty}")
    print(f"  场景: {plan.extra.get('scene_en', '')[:230]}")
    print(f"  元素: {[(e.morpheme, e.en, e.count) for e in plan.elements]}")
    if problems:
        for p in problems:
            print(f"  ✗ {p}")
    else:
        print("  ✓ 与参考题机制一致")
    return (not problems), problems


def main() -> int:
    cfg = load_config(str(ROOT / "config.json"))
    print(f"画风预设：{cfg.get('plan.style_preset')}  画幅：{cfg.get('plan.width')}x{cfg.get('plan.height')}")
    for case in CASES:
        ok, problems = check_one(cfg, *case)
        _results.append((case[0], ok, problems))

    passed = sum(1 for _t, ok, _p in _results if ok)
    print("=" * 92)
    print(f"参考题验收：{passed}/{len(_results)} 通过")
    for text, ok, problems in _results:
        if not ok:
            print(f"  ✗ {text}: {problems[:2]}")
    if passed == len(_results):
        print("全部与参考题机制一致 ✅")
        return 0
    print("存在不一致 ❌")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
