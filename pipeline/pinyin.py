"""拼音规范化与校验：把数据表里各式各样的拼音统一成标准形式，并报出问题。

要解决的问题（真实数据里都出现过）：
    · 数字声调（"hua4 she2"）与符号声调（"huà shé"）混用
    · 标调位置错误：iu 标在 i 上（jǐu ✗ → jiǔ ✓）、ui 标在 u 上（gùi ✗ → guì ✓）
    · ü 写成 v / u: / u（"lv4" / "nu:3"）
    · 音节拼写非法（gi、siu 之类不存在的音节）
    · 音节数与字数不匹配（数据切分错误）
    · 大小写、全角、多空格

对外接口：
    normalize(s)            -> PinyinResult(numbered, marked, issues)
    check_item(text, py)    -> [问题字符串]
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# 标准音节表
#
# 生成原则：宁可"漏判"也不"误判"——表里只放确保合法的组合，
# 这样它只会漏掉个别非法音节，绝不会把 ju/qu/xu/jun 这种合法音节报成错误。
# 内部用 v 表示 ü；j/q/x/y 后的 ü 系按规定写成 u（ju/jue/juan/jun）。
# --------------------------------------------------------------------------- #
_PLAIN = {
    "b": "a o i u ai ei ao ou an en ang eng ie iao ian in ing",
    "p": "a o i u ai ei ao ou an en ang eng ie iao ian in ing",
    "m": "a o e i u ai ei ao ou an en ang eng ie iao ian in ing iu",
    "f": "a o e u ei ou an en ang eng",
    "d": "a e i u ai ei ao ou an en ang eng ong ia ie iao iu ian ing uo ui uan un",
    "t": "a e i u ai ei ao ou an en ang eng ong ie iao ian ing uo ui uan un",
    "n": "a e i u ai ei ao ou an en ang eng ong ia ie iao iu ian in iang ing uo uan un",
    "l": "a e i u ai ei ao ou an en ang eng ong ia ie iao iu ian in iang ing uo uan un",
    "g": "a e ai ei ao ou an en ang eng ong u ua uo uai ui uan un uang",
    "k": "a e ai ei ao ou an en ang eng ong u ua uo uai ui uan un uang",
    "h": "a e ai ei ao ou an en ang eng ong u ua uo uai ui uan un uang",
    "j": "i ia ie iao iu ian in iang ing iong",
    "q": "i ia ie iao iu ian in iang ing iong",
    "x": "i ia ie iao iu ian in iang ing iong",
    "zh": "a e i ai ei ao ou an en ang eng ong u ua uo uai ui uan un uang",
    "ch": "a e i ai ei ao ou an en ang eng ong u ua uo uai ui uan un uang",
    "sh": "a e i ai ei ao ou an en ang eng u ua uo uai ui uan un uang",
    "r": "e i ao ou an en ang eng ong u ua uo ui uan un",
    "z": "a e i ai ei ao ou an en ang eng ong u uo ui uan un",
    "c": "a e i ai ei ao ou an en ang eng ong u uo ui uan un",
    "s": "a e i ai ei ao ou an en ang eng ong u uo ui uan un",
    "y": "i a e o ai ao ou an en in un ang eng ing ong",
    "w": "u a o ai ei an en ang eng",
    "": "a o e ai ei ao ou an en ang eng er",
}
# ü 系韵母：j/q/x/y 写成 u 系（ju/jue/juan/jun），n/l 写成 v（nv/nve）
_V_FINALS = {
    "j": "v ve van vn", "q": "v ve van vn", "x": "v ve van vn",
    "y": "v ve van vn", "n": "v ve", "l": "v ve",
}


def _spell(ini: str, fin: str) -> str:
    """把 (声母, 韵母) 拼成标准写法。fin 里的 v 代表 ü。"""
    if fin.startswith("v"):
        rest = fin[1:]
        if ini in ("j", "q", "x", "y"):
            return ini + "u" + rest          # jü->ju, jüan->juan, jün->jun
        return ini + "v" + rest              # nü->nv, lüe->lve
    return ini + fin


def _build_syllables() -> set[str]:
    ok: set[str] = set()
    for ini, finals in _PLAIN.items():
        for fin in finals.split():
            ok.add(_spell(ini, fin))
    for ini, finals in _V_FINALS.items():
        for fin in finals.split():
            ok.add(_spell(ini, fin))
    # 少数整体认读/叹词；lue/nue 是"省略 ü 上两点"的常见简写，一并接受
    ok.update({"er", "hm", "hng", "ng", "n", "m", "o", "e", "a", "ê",
               "lue", "nue", "lüe", "nüe"})
    return {s for s in ok if s}


SYLLABLES = _build_syllables()

# 生成器的自检清单：这些合法音节必须全部在表内，否则说明拼合规则写错了。
# 注意 ong 不是独立音节（只能作 weng/带声母形式），所以不列入。
_MUST_HAVE = (
    "ju qu xu yu jun quan xun yun yuan yue jue que xue nv lv nve lve "
    "zhi chi shi ri zi ci si yi ya ye yao you yan yin ying yong wu wa wo wai "
    "wei wan wen wang weng er ang eng hua gui jiu liu shui niu hui zhui "
    "juan quan xuan lia nian liang guang kuang huang"
)


def _self_check() -> list[str]:
    missing = [s for s in _MUST_HAVE.split() if s not in SYLLABLES]
    return missing


MISSING_SYLLABLES = _self_check()


SYLLABLES = _build_syllables()

# 带调元音 <-> 基本元音
_TONE_MAP = {
    "a": "āáǎà", "o": "ōóǒò", "e": "ēéěè", "i": "īíǐì",
    "u": "ūúǔù", "v": "ǖǘǚǜ", "ü": "ǖǘǚǜ",
}
_MARK_TO_BASE = {}
for _base, _marked in _TONE_MAP.items():
    for _i, _ch in enumerate(_marked):
        _MARK_TO_BASE[_ch] = (_base, _i + 1)
_MARK_TO_BASE["ê"] = ("e", 0)

# 标调优先级：a o e 优先；其余情况按"iu 标 u、ui 标 i"的规则处理
_TONE_PRIORITY = "aoe"


@dataclass
class PinyinResult:
    numbered: str = ""      # 标准数字声调形式：jiu3 niu2 yi4 mao2
    marked: str = ""        # 标准符号形式：jiǔ niú yì máo
    syllables: list[str] = field(default_factory=list)
    tones: list[int] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


def mark_syllable(base: str, tone: int) -> str:
    """给一个不带调的普通话音节加声调符号（正确处理 iu / ui / ü）。"""
    if tone <= 0 or tone > 4:
        return base
    low = base.lower()
    target = ""
    for v in _TONE_PRIORITY:            # a > o > e
        if v in low:
            target = v
            break
    if not target:
        if "iu" in low:                 # jiu -> jiǔ（标在 u 上）
            target = "u"
        elif "ui" in low:               # gui -> guì（标在 i 上）
            target = "i"
        elif "ü" in low or "v" in low:
            target = "ü" if "ü" in low else "v"
        elif "i" in low:
            target = "i"
        elif "u" in low:
            target = "u"
    if not target:
        return base
    marked = _TONE_MAP.get(target, "")
    if not marked:
        return base
    idx = low.rindex(target) if target in _TONE_PRIORITY else low.index(target)
    if target == "ü" or target == "v":
        idx = low.index(target)
    return base[:idx] + marked[tone - 1] + base[idx + 1:]


def _strip_marks(tok: str) -> tuple[str, int, bool]:
    """把 'huà' / 'hua4' / 'lv4' / 'nu:3' 统一解析成 (基本音节, 声调, 是否有标记)。"""
    base_chars: list[str] = []
    tone = 0
    had_mark = False
    t = unicodedata.normalize("NFC", tok.strip().lower())
    t = t.replace("u:", "v").replace("ü", "v").replace("ǖ", "v").replace("ǘ", "v") \
         .replace("ǚ", "v").replace("ǜ", "v")
    for ch in t:
        if ch.isdigit():
            tone = int(ch)
            continue
        if ch in _MARK_TO_BASE:
            b, tn = _MARK_TO_BASE[ch]
            base_chars.append("v" if b == "ü" else b)
            tone = tn
            had_mark = True
            continue
        if ch.isalpha() or ch == "ê":
            base_chars.append(ch)
    return "".join(base_chars), tone, had_mark


def _canonicalize(base: str) -> str:
    """把 v 形式的 ü 按声母写成标准拼法：jü->ju、jüe->jue、jüan->juan、jün->jun。"""
    if base[:1] in ("j", "q", "x", "y") and "v" in base:
        return base.replace("v", "u")
    return base


def _try_split(base: str) -> tuple[str, str] | None:
    """把一个粘连的音节（zhizhao）拆成两个合法音节（zhi + zhao）。

    只在两半都是合法音节时返回，宁可拆不出来也不乱拆。
    """
    for i in range(1, len(base)):
        left, right = base[:i], base[i:]
        if _canonicalize(left) in SYLLABLES and _canonicalize(right) in SYLLABLES:
            return left, right
    return None


def normalize(s: str, expect_chars: int = 0) -> PinyinResult:
    """把任意写法的拼音规范成标准数字形式与符号形式，并记录发现的问题。

    expect_chars > 0 时会尝试修复"音节粘连"（zhizhao4 -> zhi + zhao4）。
    """
    res = PinyinResult()
    if not s:
        return res
    raw = unicodedata.normalize("NFKC", str(s)).strip()
    if not raw:
        return res
    tokens = [t for t in re.split(r"[\s,;/|]+", raw) if t]
    if len(tokens) == 1 and len(tokens[0]) > 6:
        tokens = re.findall(r"[A-Za-züÜ:êāáǎàōóǒòēéěèīíǐìūúǔùǖǘǚǜ]+[0-5]?", tokens[0])

    parsed: list[tuple[str, int, str]] = []      # (base, tone, 原 token)
    for tok in tokens:
        base, tone, had_mark = _strip_marks(tok)
        if not base:
            continue
        digit = re.search(r"([0-5])$", tok.strip())
        if had_mark and digit:
            res.issues.append(f"「{tok}」同时有调号与数字声调")
        parsed.append((base, tone, tok))

    # 粘连音节修复：只有当音节数明显少于字数时才动手
    if expect_chars and len(parsed) < expect_chars:
        repaired: list[tuple[str, int, str]] = []
        for base, tone, tok in parsed:
            if _canonicalize(base) not in SYLLABLES and len(base) >= 2:
                sp = _try_split(base)
                if sp:
                    left, right = sp
                    repaired.append((left, 0, left + "?"))
                    repaired.append((right, tone, tok))
                    res.issues.append(
                        f"音节粘连已拆分：「{tok}」-> 「{left}(?)+{right}{tone or ''}」"
                        f"（首音节声调缺失，需人工确认）")
                    continue
            repaired.append((base, tone, tok))
        parsed = repaired

    numbered: list[str] = []
    marked: list[str] = []
    for base, tone, tok in parsed:
        base = _canonicalize(base)
        if base not in SYLLABLES:
            res.issues.append(f"音节「{base}」不在标准普通话音节表内")
        if tone == 0:
            tone = 5                      # 轻声
        numbered.append(f"{base}{tone if tone != 5 else ''}")
        marked.append(mark_syllable(base, tone if tone != 5 else 0))
        res.syllables.append(base)
        res.tones.append(tone)
    res.numbered = " ".join(numbered)
    res.marked = " ".join(marked)
    return res


def check_item(text: str, pinyin: str) -> list[str]:
    """校验一条词条的拼音，返回问题列表（空表示没问题）。"""
    problems: list[str] = []
    if not pinyin:
        return problems
    res = normalize(pinyin, expect_chars=len(text) if text else 0)
    problems.extend(res.issues)
    if text and res.syllables:
        # 儿化音（一丁点儿）会多一个 er，允许多 1 个音节
        n_han = len(text)
        if abs(len(res.syllables) - n_han) > 1:
            problems.append(
                f"音节数 {len(res.syllables)} 与字数 {n_han} 不匹配：{res.numbered}")
        elif len(res.syllables) != n_han and "儿" not in text:
            problems.append(
                f"音节数 {len(res.syllables)} 与字数 {n_han} 相差 1（无儿化字）：{res.numbered}")
    return problems
