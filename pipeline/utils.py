"""通用工具：日志、JSON IO、哈希、文件名、重试。"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

T = TypeVar("T")

LOG = logging.getLogger("hv")


# --------------------------------------------------------------------------- #
# 路径
# --------------------------------------------------------------------------- #
def repo_root() -> Path:
    """项目根目录（pipeline/ 的上一级）。"""
    return Path(__file__).resolve().parent.parent


def resolve(p: str | os.PathLike[str]) -> Path:
    """把配置里的相对路径解析为相对项目根目录的绝对路径。"""
    path = Path(p)
    return path if path.is_absolute() else (repo_root() / path)


def ensure_dir(p: str | os.PathLike[str]) -> Path:
    d = resolve(p)
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# 日志
# --------------------------------------------------------------------------- #
def setup_logging(log_file: str | os.PathLike[str] | None = None, verbose: bool = False) -> None:
    if LOG.handlers:
        return
    LOG.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    LOG.addHandler(sh)

    if log_file:
        lp = resolve(log_file)
        lp.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(lp, encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(logging.DEBUG)
        LOG.addHandler(fh)


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #
def read_json(path: str | os.PathLike[str], default: Any = None) -> Any:
    """读 JSON。用 utf-8-sig 容忍 BOM。

    很多 JSON 带 UTF-8 BOM（记事本、Excel 导出、PowerShell 的 Set-Content
    都会加），json.load 遇到 BOM 会直接抛 "Unexpected UTF-8 BOM"。
    utf-8-sig 对没有 BOM 的文件同样有效，所以统一用它。
    """
    p = resolve(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: str | os.PathLike[str], obj: Any, indent: int = 2) -> Path:
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    return p


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[dict]) -> int:
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | os.PathLike[str]) -> list[dict]:
    p = resolve(path)
    if not p.exists():
        return []
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def read_text_any_encoding(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk", "big5", "utf-16"):
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, UnicodeError, ValueError):
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def load_json_lenient(path: Path) -> list[dict]:
    """尽量宽容地读入一个 JSON 文件，返回记录列表。

    支持：顶层数组 / 顶层对象内含数组字段 / 单条对象 / JSONL / 多段拼接 JSON。
    """
    raw = read_text_any_encoding(path)
    data: Any = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = None

    if data is None:
        rows: list[dict] = []
        for line in raw.splitlines():
            line = line.strip().rstrip(",")
            if not line or line[0] not in "[{":
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
        if rows:
            return rows
        m = re.search(r"\[.*\]", raw, re.S)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                data = None
        if data is None:
            raise ValueError(f"无法解析 JSON 文件: {path}")

    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    if isinstance(data, dict):
        best: list[dict] = []
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                if len(v) > len(best):
                    best = [d for d in v if isinstance(d, dict)]
        return best or [data]
    raise ValueError(f"不支持的 JSON 顶层类型: {type(data)} in {path}")


# --------------------------------------------------------------------------- #
# 字符串 / 文件名
# --------------------------------------------------------------------------- #
_PINYIN_RE = re.compile(r"\s*[\(（\[][A-Za-züÜ\u0100-\u01ff' ,;·]*[\)）\]]")


def strip_pinyin(s: str) -> str:
    """去掉 '画蛇添足(huà shé tiān zú)' 里的拼音注解。"""
    return _PINYIN_RE.sub("", s).strip()


def clean_text(s: Any) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u3000", " ").replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()


def is_all_han(s: str) -> bool:
    if not s:
        return False
    for ch in s:
        code = ord(ch)
        if not (0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF):
            return False
    return True


def slugify(text: str, maxlen: int = 40) -> str:
    s = re.sub(r"[\\/:*?\"<>|\r\n\t ]+", "_", text.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:maxlen] or "item"


def stable_id(prefix: str, text: str, n: int = 8) -> str:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]
    return f"{prefix}_{h}"


def sha1_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha1_text(*parts: Any) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(json.dumps(p, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# 拼音：统一交给 pipeline.pinyin 处理（那里实现了完整的标调规则与音节校验）
# --------------------------------------------------------------------------- #
def pinyin_tone_marks(s: str) -> str:
    """把 'hua4 she2 tian1 zu2' 转成 'huà shé tiān zú'。

    注意 iu/ui 的标调位置：jiu3 -> jiǔ（标在 u 上）、gui4 -> guì（标在 i 上）。
    早期版本按 a>o>e>i>u 的简单优先级标，会把 jiu3 标成 jǐu——那是错的。
    """
    from .pinyin import normalize
    if not s:
        return ""
    return normalize(s).marked


def pinyin_numbered(s: str) -> str:
    """规范成数字声调形式（数据表统一的存储形式）。"""
    from .pinyin import normalize
    if not s:
        return ""
    return normalize(s).numbered


# --------------------------------------------------------------------------- #
# 重试
# --------------------------------------------------------------------------- #
def retry_call(fn: Callable[[], T], attempts: int = 3, base_delay: float = 1.5,
               label: str = "") -> T:
    last: BaseException | None = None
    for i in range(1, max(1, attempts) + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if i >= attempts:
                break
            delay = base_delay * (2 ** (i - 1))
            LOG.warning("%s 第 %d/%d 次失败: %s —— %.1fs 后重试",
                        label or "调用", i, attempts, exc, delay)
            time.sleep(delay)
    assert last is not None
    raise last
