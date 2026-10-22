"""ingest 阶段：读取任意 JSON 成语表/词语表 -> 统一 Item。

字段自动识别策略：
  1) 若顶层是字符串数组，直接当词条。
  2) 否则对每条记录，按"候选键名 + 值形态"打分，选出 text / pinyin / meaning 三列。
  3) 词条里的 '画蛇添足(huà shé tiān zú)' 会被拆出正文与拼音。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .config import Config
from .pinyin import normalize
from .schema import Item
from .state import State
from .utils import (LOG, clean_text, is_all_han, load_json_lenient, resolve,
                    stable_id, strip_pinyin)

TEXT_KEYS = [
    "word", "words", "idiom", "idioms", "chengyu", "phrase", "text", "content",
    "name", "title", "entry", "item", "w", "key", "词条", "词语", "成语", "词",
    "条目", "汉字词", "内容", "名称", "题目", "answer", "谜底",
]
PINYIN_KEYS = ["pinyin", "py", "pronunciation", "spell", "spelling", "拼音", "读音", "注音"]
MEANING_KEYS = [
    "meaning", "definition", "def", "explanation", "explain", "description", "desc",
    "gloss", "释义", "解释", "意思", "含义", "注解", "注释", "说明", "翻译", "译文",
    "example", "例句", "用法", "出处", "典故",
]
TYPE_KEYS = ["type", "category", "kind", "类型", "类别", "分类", "词性"]
FREQ_KEYS = ["freq", "frequency", "count", "weight", "score", "rank", "频次", "频率",
             "词频", "权重", "热度", "常见度"]


def _key_score(key: str, values: list[Any]) -> float:
    """给"这个键是不是目标列"打个分。"""
    k = str(key).strip().lower()
    score = 0.0
    for i, cand in enumerate(TEXT_KEYS):
        if k == cand:
            score += 3.0 - i * 0.02
            break
        if cand in k and len(k) <= 8:
            score += 1.5 - i * 0.01
            break
    return score


def _han_ratio(values: list[Any]) -> float:
    vals = [v for v in values if isinstance(v, str) and v.strip()]
    if not vals:
        return 0.0
    ok = sum(1 for v in vals if is_all_han(strip_pinyin(clean_text(v))))
    return ok / len(vals)


def detect_columns(rows: list[dict]) -> dict[str, str]:
    """返回 {'text': key, 'pinyin': key|'', 'meaning': key|''}。"""
    keys: list[str] = []
    for r in rows[:200]:
        for k in r:
            if k not in keys:
                keys.append(k)
    if not keys:
        return {"text": "", "pinyin": "", "meaning": ""}

    scored: list[tuple[float, str]] = []
    for k in keys:
        vals = [r.get(k) for r in rows[:200]]
        ratio = _han_ratio(vals)
        if ratio < 0.5:
            continue
        ks = _key_score(k, vals)
        # 值的平均长度落在 2-8 更像词条本体
        strs = [strip_pinyin(clean_text(v)) for v in vals if isinstance(v, str)]
        avg_len = sum(len(s) for s in strs) / max(1, len(strs))
        len_bonus = 1.0 if 2 <= avg_len <= 8 else 0.0
        scored.append((ks + ratio * 2.0 + len_bonus, k))
    scored.sort(reverse=True)
    text_key = scored[0][1] if scored else keys[0]

    def pick(cands: list[str], exclude: str = "") -> str:
        for k in keys:
            if k == exclude:
                continue
            lk = str(k).strip().lower()
            if any(c == lk or c in lk for c in cands):
                return k
        return ""

    pinyin_key = pick(PINYIN_KEYS, text_key)
    meaning_key = pick(MEANING_KEYS, text_key)
    return {"text": text_key, "pinyin": pinyin_key, "meaning": meaning_key}


def _guess_type(path: Path, text: str, raw: dict, explicit: str = "") -> str:
    if explicit:
        e = explicit.lower()
        if "成语" in e or "idiom" in e or "chengyu" in e:
            return "idiom"
        if "词" in e or "word" in e:
            return "word"
    name = path.name.lower()
    if "成语" in path.name or "idiom" in name or "chengyu" in name or "cheng" in name:
        return "idiom"
    if "词语" in path.name or "word" in name or "lexicon" in name or "vocab" in name:
        return "word"
    return "idiom" if len(text) == 4 else "word"


def discover_files(cfg: Config) -> list[Path]:
    files = [resolve(f) for f in (cfg.get("ingest.files") or [])]
    if files:
        return [f for f in files if f.exists()]
    data_dir = cfg.data_dir
    found = sorted(p for p in data_dir.glob("*.json") if p.is_file())
    if found:
        return found
    # 兜底：数据表被直接放在项目根目录
    root = resolve(".")
    skip = {"config.json", "package.json"}
    return sorted(p for p in root.glob("*.json")
                  if p.is_file() and p.name not in skip)


def run(cfg: Config, state: State, limit: int = 0, include_sample: bool = True) -> int:
    files = discover_files(cfg)
    if not include_sample:
        files = [f for f in files if "sample" not in f.name.lower()]
    if not files:
        LOG.error("在 %s 下没有找到任何 .json 数据文件", cfg.data_dir)
        return 0

    min_len = int(cfg.get("ingest.min_len", 2))
    max_idiom = int(cfg.get("ingest.max_len_idiom", 8))
    max_word = int(cfg.get("ingest.max_len_word", 6))

    # text -> (优先级, Item)；跨表去重，优先保留"成语表/带释义/带类型"的那条
    picked: dict[str, tuple[tuple[int, int, int], Item]] = {}
    scanned = 0
    for path in files:
        try:
            rows = load_json_lenient(path)
        except Exception as exc:  # noqa: BLE001
            LOG.error("解析失败 %s: %s", path, exc)
            continue

        # 纯字符串数组的情况
        if rows and all(len(r) == 1 for r in rows) and all(
                isinstance(next(iter(r.values())), str) for r in rows):
            rows = [{"word": next(iter(r.values()))} for r in rows]

        cols = detect_columns(rows)
        if not cols["text"]:
            LOG.warning("无法识别词条列: %s", path.name)
            continue
        LOG.info("%s -> 词条列=%s 拼音列=%s 释义列=%s（%d 条）",
                 path.name, cols["text"], cols["pinyin"] or "-", cols["meaning"] or "-", len(rows))

        for raw in rows:
            text_value = raw.get(cols["text"])
            if not isinstance(text_value, str):
                continue
            raw_text = clean_text(text_value)
            pinyin = clean_text(raw.get(cols["pinyin"])) if cols["pinyin"] else ""
            if not pinyin:
                m = re.search(r"[\(（\[]([A-Za-züÜ\u0100-\u01ff' ,;·]+)[\)）\]]", raw_text)
                if m:
                    pinyin = m.group(1).strip()
            text = clean_text(strip_pinyin(raw_text))
            if not text or not is_all_han(text):
                continue

            explicit_type = ""
            for k in TYPE_KEYS:
                if k in raw and isinstance(raw[k], str):
                    explicit_type = raw[k]
                    break
            itype = _guess_type(path, text, raw, explicit_type)
            maxlen = max_idiom if itype == "idiom" else max_word
            if not (min_len <= len(text) <= maxlen):
                continue
            scanned += 1

            meaning = clean_text(raw.get(cols["meaning"])) if cols["meaning"] else ""
            category = clean_text(raw.get("类型") or raw.get("category") or "")

            # 拼音规范化：数据表里混着"数字声调/符号声调/标调位置错/ü 写成 v/
            # 音节粘连（zhizhao4）"等各种写法，统一成标准数字声调形式再入库。
            py_raw = pinyin
            py_res = normalize(pinyin, expect_chars=len(text)) if pinyin else None
            if py_res and py_res.numbered:
                pinyin = py_res.numbered
            if py_res and py_res.issues:
                raw = dict(raw)
                raw["_pinyin_issues"] = py_res.issues
                raw["_pinyin_raw"] = py_raw

            item = Item(
                id=stable_id(itype[:3], text, 12),
                text=text, type=itype, pinyin=pinyin, meaning=meaning[:400],
                category=category, source=path.name, raw=raw,
            )
            priority = (1 if itype == "idiom" else 0, 1 if meaning else 0, 1 if category else 0)
            prev = picked.get(text)
            if prev is None or priority > prev[0]:
                picked[text] = (priority, item)
            if limit and len(picked) >= limit:
                break
        if limit and len(picked) >= limit:
            break

    batch = [v[1] for v in picked.values()]
    if batch:
        state.upsert_items(batch)
    state.dedupe_by_text()
    LOG.info("ingest 完成：词条 %d 条（扫描 %d 条，去重 %d 条，来源 %d 个文件）",
             len(batch), scanned, scanned - len(batch), len(files))
    return len(batch)
