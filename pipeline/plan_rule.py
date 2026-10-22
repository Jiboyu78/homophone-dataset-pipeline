"""plan 阶段（规则版，无需 LLM）：把语素切分结果组合成"可画的谜面方案"。

核心思路：会意谜面 = 元素 + 元素间关系。
    literal  字面直译  河马   -> 一匹马站在河水之中
    spatial  方位式    火上浇油 -> 画面纵向分区，上方是火，下方油浇下去
    quantity 数量式    三长两短 -> 三根长物、两根短物并排陈列
    contrast 反义并置  有眼无珠 -> 左半：眼中有珠；右半：同样的眼珠却缺席
    action   动作式    画蛇添足 -> 一只手用毛笔给蛇上色，又给蛇加上多余的脚
    juxtapose 并列式   杯弓蛇影 -> 杯、弓、蛇同框并列

产出 Plan：元素表 + 构图说明 + 中英提示词 + 负向提示词 + 难度 + 质检要点。
有本地 LLM 时（plan.provider=llm）走 plan_llm.py，语言质量更高；本模块是离线兜底。
"""
from __future__ import annotations

import re
from typing import Any

from .config import Config
from .lexicon import (ACTION_CHARS, ANTONYM_PAIRS, BODY_NOUN, LEX,
                      MATERIAL_CHARS, MATERIAL_NOUN, NEGATION_CHARS,
                      POLYSEMY_RISK, SPATIAL_CHARS, VERB_EN)
from .schema import COMPOSITION_ZH, TECHNIQUE_ZH, Element, Item, Plan
from .utils import LOG

PLACE_KINDS = {"nature", "place", "sky", "weather"}
WEATHER_KINDS = {"weather"}
OBJECT_KINDS = {"object", "tool", "weapon", "cloth", "material", "building",
                "vehicle", "food"}
CREATURE_KINDS = {"animal", "person", "plant"}
BODY_KINDS = {"body", "organ"}
ATTR_KINDS = {"color", "state", "measure", "number", "direction", "time", "sound",
              "dimension", "property"}
# 数量/对比句里，运动类语素当作"方式限定语"处理（七上八下 -> 七个向上、八个向下）
ATTR_LIKE_KINDS = ATTR_KINDS | {"motion"}

SPATIAL_EN = {
    "上": "above", "下": "below", "中": "in the centre of", "内": "inside",
    "里": "inside", "外": "outside", "间": "between", "旁": "beside",
    "前": "in front of", "后": "behind", "左": "to the left of", "右": "to the right of",
    "东": "to the east of", "南": "to the south of", "西": "to the west of",
    "北": "to the north of", "底": "underneath", "顶": "on top of",
    "侧": "beside", "边": "at the edge of",
}
SPATIAL_ZH = {
    "上": "上方", "下": "下方", "中": "正中", "内": "内部", "里": "里面",
    "外": "外面", "间": "之间", "旁": "旁边", "前": "前面", "后": "后面",
    "左": "左边", "右": "右边", "底": "底下", "顶": "顶上", "侧": "侧面", "边": "边上",
}
NUMBER_ZH = {"一": "一", "二": "二", "两": "两", "三": "三", "四": "四", "五": "五",
             "六": "六", "七": "七", "八": "八", "九": "九", "十": "十", "百": "百",
             "千": "千", "万": "万", "半": "半", "双": "一对", "对": "一对",
             "单": "单", "满": "满", "众": "一群", "群": "一群", "数": "若干"}
NUMBER_EN = {"一": "one", "二": "two", "两": "two", "三": "three", "四": "four",
             "五": "five", "六": "six", "七": "seven", "八": "eight", "九": "nine",
             "十": "ten", "百": "one hundred", "千": "one thousand",
             "万": "ten thousand", "半": "half a", "双": "a matched pair of",
             "对": "a matched pair of", "单": "a single", "满": "a full",
             "众": "a whole crowd of", "群": "a whole herd of", "数": "several"}
NUMBER_VALUE = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6,
                "七": 7, "八": 8, "九": 9, "十": 10, "百": 100, "千": 1000,
                "万": 10000, "双": 2, "对": 2, "半": 1, "单": 1}


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _bare(en: str) -> str:
    """去掉英文短语开头的不定冠词，便于自己加冠词或数量词。"""
    for art in ("a ", "an ", "the "):
        if en.lower().startswith(art):
            return en[len(art):]
    return en


def _art(en: str) -> str:
    """按需补不定冠词（复数/物质名词保持裸用，避免 'a leaping flames'）。"""
    if not en:
        return ""
    low = en.lower()
    if low.startswith(("a ", "an ", "the ", "exactly", "several", "some")):
        return en
    if en[:1].isdigit():
        return en
    if _is_plural(en):
        return en
    return ("an " if en[0].lower() in "aeiou" else "a ") + en


def _is_plural(en: str) -> bool:
    w = en.strip().split()[-1].lower().rstrip(".,")
    if w.endswith(("ss", "us", "is", "as", "ous", "ics")):
        return False
    return w.endswith("s")


IRREGULAR_PLURAL = {
    "ox": "oxen", "mouse": "mice", "goose": "geese", "foot": "feet",
    "tooth": "teeth", "man": "men", "woman": "women", "child": "children",
    "fish": "fish", "sheep": "sheep", "deer": "deer", "swine": "swine",
}


def _plural(en: str) -> str:
    """把名词短语变复数（够用即可，不做完整英语形态学）。"""
    words = en.strip().split()
    if not words:
        return en
    last = words[-1]
    m = re.search(r"([A-Za-z]+)([^A-Za-z]*)$", last)
    if not m:
        return en
    stem, tail = m.group(1), m.group(2)
    low = stem.lower()
    if low in IRREGULAR_PLURAL:
        plural = IRREGULAR_PLURAL[low]
        if stem[:1].isupper():
            plural = plural.capitalize()
    elif low.endswith(("s", "x", "z", "ch", "sh")):
        plural = stem + "es"
    elif low.endswith("y") and len(stem) > 1 and stem[-2].lower() not in "aeiou":
        plural = stem[:-1] + "ies"
    elif low.endswith("f"):
        plural = stem[:-1] + "ves"
    else:
        plural = stem + "s"
    words[-1] = plural + tail
    return " ".join(words)


def _mk_element(seg: dict[str, Any], role: str = "", position: str = "",
                scale: str = "", note: str = "") -> Element:
    return Element(
        morpheme=seg["morpheme"],
        zh=seg["morpheme"],
        en=seg.get("en") or seg["morpheme"],
        count=1,
        role=role or seg.get("role", "object"),
        position=position,
        scale=scale,
        note=note or seg.get("kind", ""),
    )


def _dedupe_elements(elems: list[Element]) -> list[Element]:
    out: dict[tuple[str, str], Element] = {}
    for e in elems:
        if not e.en or e.en.startswith("("):
            continue
        key = (e.morpheme, e.en)
        if key in out:
            out[key].count += 1
        else:
            out[key] = e
    return list(out.values())


MASS_NOUN_TAIL = ("oil", "sauce", "sugar", "salt", "water", "rice", "soup", "milk",
                  "tea", "wine", "vinegar", "honey", "meat", "juice", "cheese",
                  "flour", "wheat", "sand", "dust", "smoke", "blood")


def _plural_unless_mass(en: str) -> str:
    """可数名词变复数，物质名词保持原样（'cooking oil' 不该变成 oils）。"""
    low = en.lower()
    if " of " in low or low.endswith(MASS_NOUN_TAIL) or low.endswith("food"):
        return en
    return _plural(en)


def _pick_relation(a: dict[str, Any], b: dict[str, Any]) -> tuple[str, str]:
    """推导两个字面语素之间的关系，返回 (英文场景片段, 中文说明)。"""
    ka, kb = a["kind"], b["kind"]
    ca, cb = a["morpheme"], b["morpheme"]
    ea, eb = _bare(a.get("en") or ca), _bare(b.get("en") or cb)

    # 材质 + 器物 -> 用…做成（冰箱、雪糕、石桥）
    if (ka in WEATHER_KINDS or ca in MATERIAL_CHARS) and kb in OBJECT_KINDS:
        if ca in MATERIAL_CHARS:
            return (f"{_art(eb)} made entirely of {MATERIAL_NOUN.get(ca, ea)}",
                    f"用{ca}做成的{cb}")
        return (f"{_art(eb)} in the middle of {_art(ea)}", f"{cb}处在{ca}之中")
    # 雨伞/火车 这类"在…之中"的场景
    if ka in PLACE_KINDS and kb in OBJECT_KINDS:
        return (f"{_art(eb)} in the middle of {_art(ea)}", f"{cb}处在{ca}之中")
    # 河马/海豹/山羊 -> 动物位于环境之中
    if ka in PLACE_KINDS and kb in CREATURE_KINDS:
        return (f"{_art(eb)} standing in the middle of {_art(ea)}", f"{cb}位于{ca}之中")
    # 鸡腿/鸭舌 -> 动物的部位
    if ka in CREATURE_KINDS and kb in BODY_KINDS:
        return (f"the {BODY_NOUN.get(cb, eb)} of {_art(ea)}", f"{ca}的{cb}")
    # 菜包/肉夹馍 -> 食物里塞了另一种食物
    if ka in ("plant", "food", "animal") and kb in ("food",):
        return (f"{_art(eb)} stuffed with {_plural_unless_mass(ea)}", f"塞满{ca}的{cb}")
    # 刀鱼/剑龙 -> 形似
    if ka in OBJECT_KINDS and kb in CREATURE_KINDS:
        return (f"{_art(eb)} shaped like {_art(ea)}", f"形似{ca}的{cb}")
    if ka in ATTR_KINDS:
        return (_art(f"{ea} {eb}"), f"{ca}的{cb}")
    if kb in ATTR_KINDS:
        return (_art(f"{eb} {ea}"), f"{cb}的{ca}")
    return (f"{_art(ea)} combined with {_art(eb)}", f"{ca}与{cb}组合")


# --------------------------------------------------------------------------- #
# 各构图范式
# --------------------------------------------------------------------------- #
def _modifier_sense(seg: dict[str, Any]) -> dict[str, Any]:
    """修饰语位置上的多义字消歧：香醋的"香"取"芳香的"而不是"香烛"。"""
    alt = LEX.sense_in(seg["morpheme"], ATTR_KINDS)
    if alt and seg["kind"] not in ATTR_KINDS:
        merged = dict(seg)
        merged.update({"en": alt["en"], "kind": alt["kind"], "role": alt["role"]})
        return merged
    return seg


def _plan_literal(item: Item, segs: list[dict[str, Any]]) -> tuple[list[Element], str, str]:
    parts = [s for s in segs if s["drawable"]]
    if len(parts) < 2:
        return _plan_juxtapose(item, segs)
    a, b = _modifier_sense(parts[0]), parts[1]
    relation_en, relation_zh = _pick_relation(a, b)
    elems = [_mk_element(a, role="object", position="左"),
             _mk_element(b, role="object", position="右")]
    scene, scene_zh = relation_en, relation_zh
    if len(parts) > 2:
        extra_en = ", ".join(_art(_bare(p.get("en") or p["morpheme"])) for p in parts[2:])
        scene += f", together with {extra_en}"
        scene_zh += "，并配上" + "、".join(p["morpheme"] for p in parts[2:])
        elems.extend(_mk_element(p) for p in parts[2:])
    return elems, scene, f"把「{item.text}」逐字具象为：{scene_zh}"


def _plan_juxtapose(item: Item, segs: list[dict[str, Any]]) -> tuple[list[Element], str, str]:
    parts = [s for s in segs if s["drawable"]]
    elems = [_mk_element(p, position="从左到右依次排列") for p in parts]
    en = ", ".join(_art(_bare(p.get("en") or p["morpheme"])) for p in parts)
    scene = f"a single still-life composition containing, side by side: {en}"
    zh = "、".join(p["morpheme"] for p in parts)
    return elems, scene, f"画面并置「{zh}」几个元素"


# 方位字 -> 构图关系族，决定提示词用哪种版面语言描述
REL_FAMILY = {
    "上": "vertical", "顶": "vertical", "下": "vertical", "底": "vertical",
    "内": "contain", "里": "contain", "中": "contain",
    "间": "between", "旁": "side", "侧": "side", "边": "side",
    "前": "depth", "后": "depth",
    "左": "hsplit", "右": "hsplit",
    "东": "hsplit", "西": "hsplit", "南": "hsplit", "北": "hsplit",
}


def _join_art(parts: list[dict[str, Any]]) -> str:
    return " and ".join(_art(_bare(p.get("en") or p["morpheme"])) for p in parts)


def _plan_spatial(item: Item, segs: list[dict[str, Any]]) -> tuple[list[Element], str, str]:
    text = item.text
    sp_idx = next((i for i, c in enumerate(text) if c in SPATIAL_CHARS), -1)
    if sp_idx < 0:
        return _plan_juxtapose(item, segs)
    rel_char = text[sp_idx]
    rel_en = SPATIAL_EN.get(rel_char, "beside")
    rel_zh = SPATIAL_ZH.get(rel_char, "旁边")
    family = REL_FAMILY.get(rel_char, "side")
    # 方位字本身不是画面元素
    parts = [s for s in segs if s["drawable"] and s["morpheme"] != rel_char]
    if len(parts) < 2:
        return _plan_juxtapose(item, segs)
    names = "、".join(p["morpheme"] for p in parts)

    if sp_idx >= len(text) - 1:
        # 方位字收尾（瓜田李下、江河日下）：画面整体处在某个方位的视角/位置
        en = _join_art(parts)
        scene = (f"the whole scene is situated {rel_en}, the camera looks from {rel_en} "
                 f"towards it, the image contains {en}")
        elems = [_mk_element(p, position=f"整体处于{rel_zh}") for p in parts]
        return elems, scene, f"用视角表意：画面整体处于{rel_zh}，内容为「{names}」"

    before = [p for p in parts if text.index(p["morpheme"]) < sp_idx]
    after = [p for p in parts if text.index(p["morpheme"]) > sp_idx]
    if not before and after:
        before, after = after[:1], after[1:]
    if not after and before:
        after, before = before[-1:], before[:-1]

    a_en = _join_art(before) or _join_art(after)
    b_en = _join_art(after)
    verb = "are" if (" and " in b_en or " and " in a_en) else "is"
    if family == "vertical":
        v = "occupy" if (" and " in a_en or " and " in b_en) else "occupies"
        scene = (f"the frame is divided into an upper half and a lower half: {a_en} "
                 f"{v} the upper half, {b_en} {v} the lower half, so that the "
                 f"first clearly sits above the second")
    elif family == "contain":
        scene = (f"{b_en} {verb} placed inside {a_en} and clearly enclosed within it, "
                 f"the containment itself is the clue")
    elif family == "between":
        scene = f"{b_en} {verb} placed between two identical instances of {a_en}"
    elif family == "depth":
        scene = (f"the frame reads from front to back: {a_en} in the foreground and "
                 f"{b_en} behind it, {rel_en} one another")
    elif family == "hsplit":
        scene = (f"the frame is divided into a left half and a right half: {a_en} on the "
                 f"left, {b_en} on the right, {rel_en} one another")
    else:
        scene = f"{a_en} and {b_en} stand side by side, {rel_en} one another"
    elems = [_mk_element(p, position=f"{rel_zh}一侧") for p in before + after]
    zh = "；".join(filter(None, [
        f"「{'、'.join(p['morpheme'] for p in before)}」在上/外侧" if before else "",
        f"「{'、'.join(p['morpheme'] for p in after)}」在下/内侧" if after else ""]))
    return elems, scene, f"用方位关系表意：{zh or rel_zh}"


def _phrase_group(parts: list[dict[str, Any]], quantifier: str = "") -> str:
    """把一组语素合成一个通顺的英文名词短语。

    属性类语素（长/短/大/小/上/下…）作限定语，没有实体名词时用 "objects" 兜底。
    """
    attrs = [p for p in parts if p["kind"] in ATTR_LIKE_KINDS]
    nouns = [p for p in parts if p["kind"] not in ATTR_LIKE_KINDS]
    attr_en = " and ".join(_bare(p.get("en") or p["morpheme"]) for p in attrs)
    noun_en = " and ".join(_bare(p.get("en") or p["morpheme"]) for p in nouns)
    q = (quantifier + " ") if quantifier else ""
    if nouns:
        body = _plural(noun_en) if quantifier and quantifier not in ("exactly one", "half of one") else noun_en
        if attrs:
            return f"{q}{body} that are {attr_en}"
        return f"{q or ''}{_art(body) if not q else body}"
    if attrs:
        # quantifier 里可能已经带了 identical，避免重复
        q_clean = re.sub(r"\bidentical\b\s*", "", quantifier).strip()
        q_clean = (q_clean + " identical ") if q_clean else ""
        return f"{q_clean}objects that are {attr_en}"
    return f"{q}objects"


def _join_art(parts: list[dict[str, Any]]) -> str:
    return " and ".join(_art(_bare(p.get("en") or p["morpheme"])) for p in parts)


def _plan_quantity(item: Item, segs: list[dict[str, Any]]) -> tuple[list[Element], str, str]:
    parts = [s for s in segs if s["drawable"]]
    mods = [s for s in parts if s["kind"] == "number"]
    objects = [s for s in parts if s["kind"] != "number"]
    if not objects:
        return _plan_juxtapose(item, segs)

    elems: list[Element] = []
    bits_en: list[str] = []
    bits_zh: list[str] = []
    for i, obj in enumerate(objects):
        num = mods[i] if i < len(mods) else None
        if num is None:
            bits_en.append(_phrase_group([obj]))
            bits_zh.append(obj["morpheme"])
            e = _mk_element(obj, position="并排陈列")
            elems.append(e)
            continue
        ch = num["morpheme"]
        val = min(NUMBER_VALUE.get(ch, 1), 12)
        if ch == "一":
            quant = "exactly one"
        elif ch == "半":
            quant = "half of one"
        elif val == 1:
            quant = "one"
        else:
            quant = NUMBER_EN.get(ch, str(val))
            if ch not in ("双", "对", "众", "群"):
                quant = f"{quant} identical"
        bits_en.append(_phrase_group([obj], quant))
        bits_zh.append(f"{NUMBER_ZH.get(ch, ch)}{obj['morpheme']}")
        e = _mk_element(obj, position="并排陈列", note=f"数量：{NUMBER_ZH.get(ch, ch)}")
        e.count = val
        elems.append(e)

    rest = [p for p in parts if p["kind"] == "number" and p not in mods[:len(objects)]]
    max_count = max((e.count for e in elems), default=1)
    grid_hint = ""
    if max_count >= 4:
        # 参考题「九牛一毛」：9 头牛排成 3×3，单独一根毛放在一侧
        grid_hint = (f", the {max_count} identical ones arranged "
                     f"{GRID_LAYOUT.get(max_count, 'in a neat grid')} filling the left side, "
                     f"while the single different element stands alone on the right, "
                     f"visibly tiny by comparison")
    scene = "a countable arrangement of " + ", ".join(bits_en) + grid_hint
    if rest:
        scene += ", the counting itself is the clue"
    return elems, scene, "用数量表意：" + "、".join(bits_zh)


def _plan_contrast(item: Item, segs: list[dict[str, Any]]) -> tuple[list[Element], str, str]:
    chars = set(item.text)
    pair = next(((a, b) for a, b in ANTONYM_PAIRS if a in chars and b in chars), None)
    parts = [s for s in segs if s["drawable"]]
    neg = [c for c in dict.fromkeys(item.text) if c in NEGATION_CHARS]
    idx_of = {p["morpheme"]: i for i, p in enumerate(parts)}

    # 有否定字且首元素是实体名词：用"有 X"对"缺 X"
    if len(parts) >= 2 and neg and parts[0]["kind"] not in ATTR_KINDS:
        subject = _bare(parts[0].get("en") or parts[0]["morpheme"])
        attrs = " and ".join(_art(_bare(p.get("en") or p["morpheme"])) for p in parts[1:])
        attrs_bare = " and ".join(_bare(p.get("en") or p["morpheme"]) for p in parts[1:])
        elems = [_mk_element(p, position="左半" if i == 0 else "右半")
                 for i, p in enumerate(parts)]
        scene = (f"the image is split left and right as an explicit contrast: the left "
                 f"half shows {_art(subject)} together with {attrs}, while the right half "
                 f"shows the very same {subject} with the {attrs_bare} conspicuously "
                 f"absent, leaving an obvious empty gap")
        zh = (f"左右对比：左边「{parts[0]['morpheme']}」带有"
              f"{'、'.join(p['morpheme'] for p in parts[1:])}，右边同样的主体却缺少它")
        return elems, scene, zh

    # 反义对：把含反义字的两组元素分置左右
    if pair and pair[0] in idx_of and pair[1] in idx_of:
        ia, ib = idx_of[pair[0]], idx_of[pair[1]]
        if ia <= ib:
            left, right = parts[:ib], parts[ib:]
        else:
            left, right = parts[:ia], parts[ia:]
    else:
        half = max(1, len(parts) // 2)
        left, right = parts[:half], parts[half:]

    elems = ([_mk_element(p, position="左半") for p in left]
             + [_mk_element(p, position="右半") for p in right])
    l_en, r_en = _phrase_group(left), _phrase_group(right)
    pair_txt = f"（{'／'.join(pair)}）" if pair else ""
    scene = (f"the frame is split into a left half and a right half showing an explicit "
             f"contrast: on the left {l_en}, on the right {r_en}, the two halves are "
             f"mirror images of each other so the opposition is unmistakable")
    zh = f"左右对比表意{pair_txt}：左半为「{''.join(p['morpheme'] for p in left)}」，右半为「{''.join(p['morpheme'] for p in right)}」"
    return elems, scene, zh


def _action_element(seg: dict[str, Any]) -> Element:
    """动作字作元素时用"动作义"而不是名词义：画 -> painting，而不是"卷轴画"。"""
    alt = LEX.sense_in(seg["morpheme"], {"action", "motion"})
    merged = dict(seg)
    if alt:
        merged.update({"en": alt["en"], "kind": alt["kind"], "role": "action"})
    return _mk_element(merged, role="action", position="画面中心")


def _plan_action(item: Item, segs: list[dict[str, Any]]) -> tuple[list[Element], str, str]:
    parts = [s for s in segs if s["drawable"]]
    actions = [s for s in parts
               if s["morpheme"] in ACTION_CHARS or s["kind"] == "action"]
    objects = [s for s in parts if s not in actions]
    elems = [_action_element(a) for a in actions]
    elems += [_mk_element(o, role="object", position="被作用的对象") for o in objects]

    if actions:
        clauses: list[str] = []
        for i, a in enumerate(actions):
            verb = VERB_EN.get(a["morpheme"],
                               f"performs the action of {_bare(a.get('en') or '')}".strip())
            obj = objects[i] if i < len(objects) else (objects[0] if objects else None)
            if obj is not None:
                obj_bare = _bare(obj.get("en") or obj["morpheme"])
                # 动词本身已带冠词（adds an extra）时不再补冠词
                obj_phrase = obj_bare if verb.endswith(("extra", " a", " an", " the")) \
                    else _art(obj_bare)
                clauses.append(f"{verb} {obj_phrase}")
            else:
                clauses.append(verb)
        # 未被动词消耗的对象补在后面
        used = {objects[i]["morpheme"] for i in range(min(len(actions), len(objects)))}
        leftover = [o for o in objects if o["morpheme"] not in used]
        tail = ""
        if leftover:
            tail = ", together with " + " and ".join(
                _art(_bare(o.get("en") or o["morpheme"])) for o in leftover)
        subj = "a simple human figure" if len(actions) == 1 or not objects else "a simple human figure"
        scene = (f"{subj} who {', and then '.join(clauses)}{tail}, "
                 f"the gesture is unmistakable and occupies the centre of the frame")
    else:
        en = ", ".join(_art(_bare(p.get("en") or p["morpheme"])) for p in parts)
        scene = f"a scene containing {en} in the middle of an unmistakable action"
    zh = ("动作表意：" + "、".join(f"{a['morpheme']}({VERB_EN.get(a['morpheme'], '')})" for a in actions)
          + ("，作用于" + "、".join(o["morpheme"] for o in objects) if objects else ""))
    return elems, scene, zh


INNER_MARKERS = "心里行实"

# 数量多时的版面：参考题「九牛一毛」把九头牛排成 3×3 网格
GRID_LAYOUT = {4: "in a 2 by 2 grid", 5: "in a neat grid", 6: "in a 3 by 2 grid",
               7: "in a neat grid", 8: "in a 4 by 2 grid", 9: "in a 3 by 3 grid",
               10: "in a neat grid", 11: "in a neat grid", 12: "in a 4 by 3 grid"}


def _plan_inner(item: Item, segs: list[dict[str, Any]]) -> tuple[list[Element], str, str]:
    """表里式：同一个主体的"外表"与"内心/影子"并置。

    参考题「人面兽心」：一个男孩笑着站着，他的影子却是有角有爪、胸口一颗红心的黑色怪物。
    """
    parts = [s for s in segs if s["drawable"]]
    if len(parts) < 3:
        return _plan_contrast(item, segs)
    # 找承载"里/心"标记的元素。注意它可能已经被合并过（兽+心 -> 兽心），
    # 这时内心一侧从该元素**本身**开始，而不是从它的前一个元素开始。
    idx = next((i for i, s in enumerate(parts)
                if any(m in s["morpheme"] for m in INNER_MARKERS)), -1)
    if idx < 0:
        idx = len(parts) - 1
    merged_marker = len(parts[idx]["morpheme"]) > 1
    split = idx if merged_marker else max(1, idx - 1)
    outer, inner = parts[:split], parts[split:]
    if not outer or not inner:
        return _plan_contrast(item, segs)

    outer_en = " and ".join(_art(_bare(s.get("en") or s["morpheme"])) for s in outer)
    inner_en = " and ".join(_art(_bare(s.get("en") or s["morpheme"])) for s in inner)
    elems = ([_mk_element(s, position="外表一侧", role="subject") for s in outer]
             + [_mk_element(s, position="影子/内心一侧", role="object") for s in inner])
    scene = (f"a single figure standing on the left looking completely ordinary, "
             f"presenting {outer_en} to the world; the long dark shadow stretching to "
             f"the right is not his own shape at all, inside that shadow {inner_en} is "
             f"plainly visible, the shadow drawn as a separate menacing silhouette; "
             f"the gap between outward appearance and inner nature is the clue")
    zh = (f"表里对比：外表是「{''.join(s['morpheme'] for s in outer)}」，"
          f"影子/内心却是「{''.join(s['morpheme'] for s in inner)}」")
    return elems, scene, zh


def _plan_metaphor(item: Item, segs: list[dict[str, Any]]) -> tuple[list[Element], str, str]:
    parts = [s for s in segs if s["drawable"]]
    if len(parts) >= 2:
        return _plan_juxtapose(item, segs)
    single = parts[0] if parts else None
    elems = [_mk_element(single, role="object")] if single else []
    en = _bare(single.get("en")) if single else item.text
    scene = (f"a single symbolic object that stands for the abstract idea of "
             f"'{item.text}', rendered as {_art(en)}, with one small visual metaphor "
             f"element that hints at its figurative meaning")
    return elems, scene, f"用具象物「{en}」隐喻抽象含义"


COMPOSITION_BUILDERS = {
    "literal": _plan_literal,
    "juxtapose": _plan_juxtapose,
    "spatial": _plan_spatial,
    "quantity": _plan_quantity,
    "contrast": _plan_contrast,
    "action": _plan_action,
    "inner": _plan_inner,
    "metaphor": _plan_metaphor,
    "homophone": _plan_juxtapose,
}


# --------------------------------------------------------------------------- #
# 难度与标记
# --------------------------------------------------------------------------- #
def _difficulty(item: Item, plan: Plan) -> int:
    k = len([e for e in plan.elements if e.en])
    d = 2.0
    if k >= 3:
        d += 0.8
    if k >= 4:
        d += 0.4
    if k == 2:
        d -= 0.4
    if sum(1 for c in set(item.text) if c in POLYSEMY_RISK) >= 2:
        d += 0.7
    if any(c in NEGATION_CHARS for c in item.text):
        d += 0.6
    if len(item.text) >= 5:
        d += 0.5
    if item.composition == "metaphor":
        d += 0.8
    if item.type == "word":
        d -= 0.3
    return max(1, min(5, int(round(d))))


def _flags(item: Item) -> list[str]:
    flags: list[str] = []
    negs = [c for c in dict.fromkeys(item.text) if c in NEGATION_CHARS]
    if negs:
        flags.append("negation:" + "".join(negs))
    if len(item.text) >= 5:
        flags.append("long_text")
    if item.type == "word":
        flags.append("literal_compound")
    if any(ch.isascii() for ch in item.text):
        flags.append("non_han")
    if sum(1 for c in set(item.text) if c in POLYSEMY_RISK) >= 2:
        flags.append("polysemy_risk")
    return flags


def _merge_animal_body(segs: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    """把"动物+身体部位"合成一个元素：马首 -> 马的头，猪蹄 -> 猪的蹄。

    只在词组里还有其他成分时合并（马首是瞻），像"鸡腿"这种整词本身就是
    动物+部位的情况交给字面直译式处理，不在这里动。
    """
    if len(text) <= 2:
        return segs
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(segs):
        cur = segs[i]
        nxt = segs[i + 1] if i + 1 < len(segs) else None
        if (nxt and cur["kind"] in ("animal", "person")
                and nxt["kind"] in ("body", "organ") and len(cur["morpheme"]) == 1
                and len(nxt["morpheme"]) == 1):
            noun = BODY_NOUN.get(nxt["morpheme"], _bare(nxt.get("en") or nxt["morpheme"]))
            art = cur.get("en") or cur["morpheme"]
            merged = dict(cur)
            merged.update({
                "morpheme": cur["morpheme"] + nxt["morpheme"],
                "en": f"the {noun} of {_art(art)}",
                "kind": "body_part",
                "role": "object",
                "weight": max(cur["weight"], nxt["weight"]),
                "drawable": True,
            })
            out.append(merged)
            i += 2
            continue
        out.append(cur)
        i += 1
    return out


def glyph_candidates(item: Item, segs: list[dict[str, Any]]) -> list[str]:
    """可以充当"字形参与"的字：承载语义但画不出来的实义字（且不是纯虚词）。

    参考题「举手之劳」的"劳"命中；「九牛一毛」「人面兽心」这类全可画的词条不会命中。
    这个判据同时用来**校验大模型自己声明的 glyph**，防止它给"仓鼠"这类词乱塞字形。
    """
    from .lexicon import FUNCTION_CHARS
    cands: list[str] = []
    for s in segs:
        ch = s["morpheme"]
        if len(ch) != 1 or ch in FUNCTION_CHARS:
            continue
        if s["kind"] == "abstract" and s["mode"] == "no":
            cands.append(ch)
    return cands


def _pick_glyph_char(item: Item, segs: list[dict[str, Any]]) -> str:
    cands = glyph_candidates(item, segs)
    return cands[0] if cands else ""


# --------------------------------------------------------------------------- #
# 标签式提示词（SD1.5 友好）
#
# 实测教训：SD1.5 + Counterfeit 这类底模读不懂长句散文。给它
# "the frame is divided into an upper half and a lower half: a burning fire
#  occupies the upper half..." 会直接画错；而 "9 oxen in a 3 by 3 grid,
#  one hair on the right" 这种"前几个词就是主体+数量+布局"的标签串才吃得住。
# SD1.5 对前 ~20 个 token 最敏感，所以顺序必须是：主体 → 数量/布局 → 风格 → 禁文字。
# --------------------------------------------------------------------------- #
LAYOUT_TAGS = {
    "juxtapose": "{items}, side by side on a plain background, evenly spaced",
    "literal": "{items}, merged into one single object",
    "spatial": "{items}, vertical composition, top and bottom halves",
    "quantity": "{items}, countable, arranged in a neat grid",
    "contrast": "{items}, left half and right half, mirrored contrast",
    "action": "{items}, a person performing the action at the centre",
    "inner": "{items}, a smiling young man standing, his long black shadow "
             "behind him drawn as a separate menacing silhouette",
    "metaphor": "{items}, one symbolic object in the centre",
}
STYLE_TAGS_SD15 = ("flat cartoon illustration, bold black outlines, simple cel shading, "
                   "warm cream paper texture background, clean composition")
NO_TEXT_TAGS = ("no text, no letters, no words, no watermark, no signature, "
                "no speech bubble, single image")


def tag_prompt(elements: list[Element], composition: str, cfg: Config,
               glyph: str = "") -> str:
    """把元素表拼成 SD1.5 吃得住的短标签串（主体在前，风格在后）。"""
    items: list[str] = []
    for e in elements:
        en = _bare(e.en or e.morpheme)
        cnt = int(getattr(e, "count", 1) or 1)
        if getattr(e, "role", "") == "action":
            # 动作类语素的词典释义本来就是动名词短语（lifting overhead），
            # 加个主体才像一张画：a person lifting overhead
            items.append(f"a person {en}")
        elif cnt > 1:
            items.append(f"{cnt} {_plural(en)}")
        elif cnt == 1 and composition == "quantity" and e is elements[-1]:
            items.append(f"exactly one {en}")
        else:
            items.append(_art(en))

    if composition == "quantity" and len(elements) >= 2:
        # 数量式要写成"主体+数量+网格+次要元素位置"，否则 SD1.5 会画一大群、还漏掉小的那个
        big = max(elements, key=lambda e: e.count)
        small = [e for e in elements if e is not big]
        grid = GRID_LAYOUT.get(big.count, "in a neat grid")
        head = (f"{big.count} {_plural(_bare(big.en or big.morpheme))} arranged {grid} "
                f"filling the left side")
        tail = ", ".join(f"one single {_bare(e.en or e.morpheme)}" for e in small) or ""
        layout = f"{head}, {tail} alone on the far right side" if tail else head
    else:
        head = ", ".join(items[:4])      # 超过 4 个元素 SD1.5 就顾不过来
        layout = LAYOUT_TAGS.get(composition,
                                 LAYOUT_TAGS["juxtapose"]).format(items=head)

    style = cfg.get("plan.style_tags_sd15") or STYLE_TAGS_SD15
    parts = [layout, style]
    if glyph:
        parts.append(f"one single large brush-written chinese character {glyph}")
    else:
        parts.append(NO_TEXT_TAGS)
    return ", ".join(p for p in parts if p)


def build_plan(item: Item, cfg: Config) -> Plan:
    # 注意：这里总是用"当前词典"重新切分，不使用库里缓存的 segments，
    # 否则改过 lexicon 之后出题仍会沿用旧义项。
    segs = LEX.segment(item.text, apply_suffix=(item.type == "word"))
    comp = item.composition or LEX.composition_hint(item.text, segs)
    if comp not in ("literal",):
        segs = _merge_animal_body(segs, item.text)
    builder = COMPOSITION_BUILDERS.get(comp, _plan_juxtapose)
    elems, scene_en, rationale = builder(item, segs)
    elems = _dedupe_elements(elems)

    style_en = cfg.get("plan.style_en", "")
    style_zh = cfg.get("plan.style_zh", "")
    avoid = ("absolutely no text, no letters, no chinese characters, no numbers, "
             "no captions, no labels, no watermark, no signature")

    # ---------------- 字形参与式 ---------------- #
    # 参考题「举手之劳」：画面中央是一个巨大的"劳"字 + 一只举起的手。
    # 本地扩散模型写不对汉字，所以默认走"确定性合成"：生成时预留空白，
    # 出图后用字体把字叠上去；也支持让模型自己写（云模型可用）。
    glyph = _pick_glyph_char(item, segs)
    glyph_cfg = cfg.get("image.glyph_overlay", {}) or {}
    glyph_mode = str(glyph_cfg.get("mode", "composite")).lower()
    glyph_pos = str(glyph_cfg.get("position", "center-left"))
    technique = "glyph" if glyph else "ideographic"
    reserve = ""
    if glyph:
        if glyph_mode == "prompt":
            # 交给模型写：必须解除"禁止汉字"的约束，否则自相矛盾
            avoid = ("no other text besides the single large character, no captions, "
                     "no labels, no watermark, no signature, no letters, no numbers")
            glyph_clause = (f'Render one single huge brush-written Chinese character '
                            f'"{glyph}" as the visual centrepiece, occupying most of the '
                            f'frame, drawn in confident black ink strokes on the paper '
                            f'background')
            prompt_en = f"{glyph_clause}. {scene_en}. {style_en}. {avoid}."
        else:
            reserve = (f" leave a large clean empty area at the {glyph_pos} of the frame, "
                       f"keep that area completely blank and uncluttered")
            prompt_en = f"{scene_en}.{reserve} {style_en}. {avoid}."
    else:
        prompt_en = f"{scene_en}. {style_en}. {avoid}."

    prompt_zh = f"{style_zh}。画面内容：{scene_en}。"
    if reserve:
        prompt_zh += f"（画面{glyph_pos}处留出大片空白）"
    prompt_zh += avoid + "。"

    # 标签式（SD1.5 友好）：短、主体在前、不要从句
    if str(cfg.get("plan.prompt_style", "prose")).lower() == "tags":
        prompt_en = tag_prompt(elems, comp, cfg, glyph=glyph)
        # 合成字形模式下要额外说明留白（字由字体叠加，不是让模型写）
        if glyph and glyph_mode != "prompt":
            prompt_en += f", large clean empty space at the {glyph_pos}"

    # 降级标记：规则出题器对词语类只能并列铺开元素（刀鱼 = 刀 + 鱼），
    # 没能力判断两个语素能否合成，所以这类一律标为降级。
    # 成语用并列式通常是设计如此（杯弓蛇影），不标。
    degraded = bool(item.type == "word" and comp == "juxtapose")
    degraded_reason = "规则出题器只能并列铺开元素，未做语义合成" if degraded else ""

    plan = Plan(
        item_id=item.id,
        answer=item.text,
        composition=comp,
        elements=elems,
        layout=rationale,
        prompt_zh=prompt_zh,
        prompt_en=prompt_en,
        negative_prompt=cfg.get("plan.negative_prompt", ""),
        must_have=[e.en for e in elems if e.en],
        must_not=["其他文字", "字母", "数字", "水印", "签名", "多格/拼贴"],
        difficulty=3,
        distractors=[],
        rationale=rationale,
        planner="rule",
        glyph=glyph,
        technique=technique,
        panels=1,
        degraded=degraded,
        degraded_reason=degraded_reason,
        extra={
            "composition_zh": COMPOSITION_ZH.get(comp, comp),
            "technique_zh": TECHNIQUE_ZH.get(technique, technique),
            "category": item.category,
            "pinyin": item.pinyin,
            "scene_en": scene_en,
            "flags": _flags(item) + (["glyph:" + glyph] if glyph else []),
            "glyph": ({"char": glyph, "mode": glyph_mode, "position": glyph_pos,
                       "reserve_blank": bool(reserve)} if glyph else {}),
        },
    )
    plan.difficulty = _difficulty(item, plan)
    return plan


def run(cfg: Config, state: State, limit: int = 0, redo: bool = False,
        only: set[str] | None = None) -> int:
    from .lexicon import lexicon_hash
    lh = lexicon_hash()
    if state.get_meta("lexicon_hash") != lh:
        LOG.info("检测到词典已更新（%s -> %s），重新生成全部构图方案",
                 state.get_meta("lexicon_hash"), lh)
        redo = True
    if redo:
        state.reset_stage("plan")
    items = state.items_without_plan(limit=0)
    if only:
        items = [it for it in items if it.text in only]
    if limit and limit > 0:
        items = items[:limit]
    if not items:
        LOG.info("plan(rule)：没有待生成的词条")
        state.set_meta("lexicon_hash", lh)
        return 0
    for it in items:
        state.save_plan(build_plan(it, cfg))
    state.set_meta("lexicon_hash", lh)
    n = len(items)
    LOG.info("plan(rule) 完成：生成 %d 个构图方案", n)
    p = state.get_plan(items[0].id)
    if p:
        LOG.info("样例「%s」%s：%s", items[0].text, p.composition, p.prompt_en[:160])
    return n
