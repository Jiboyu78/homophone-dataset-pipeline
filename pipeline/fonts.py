"""跨平台中文字体查找。

字形参与式（如举手之劳的「劳」）必须用中文字体渲染，联系表和 mock 占位图也要。
早期版本把候选路径硬编码成 C:\\Windows\\Fonts\\...，在 macOS / Linux 上会**静默失效**：
`Path(f).exists()` 全为 False，字不叠加、或退化成一个空心方框，而流程照样"成功"。
这个模块按平台给候选清单，并提供目录扫描与 fc-match 兜底。

指定优先级的顺序：
    调用方显式指定 > 环境变量 HV_CJK_FONT > 平台候选清单 > 目录扫描 > fc-match
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .utils import LOG

# --------------------------------------------------------------------------- #
# 各平台候选：越靠前越接近"毛笔/楷体"的感觉，参考题用的就是行楷
# --------------------------------------------------------------------------- #
WINDOWS = [
    r"C:\Windows\Fonts\STXINGKA.TTF",   # 华文行楷，最接近参考题的毛笔感
    r"C:\Windows\Fonts\STKAITI.TTF",    # 华文楷体
    r"C:\Windows\Fonts\simkai.ttf",     # 楷体
    r"C:\Windows\Fonts\msyh.ttc",       # 微软雅黑
    r"C:\Windows\Fonts\simhei.ttf",     # 黑体
    r"C:\Windows\Fonts\simsun.ttc",     # 宋体
]
# macOS 自带：楷体-简 / 宋体 / 苹方 / 华文黑体；另外系统里可能有华文行楷
MACOS = [
    "/System/Library/Fonts/Supplemental/Kaiti.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/System/Library/Fonts/Supplemental/STKaiti.ttf",
    "/System/Library/Fonts/Supplemental/STXingkai.ttf",
    "/System/Library/Fonts/Supplemental/Hannotate.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/Library/Fonts/Kaiti.ttc",
    "/Library/Fonts/Songti.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
]
LINUX = [
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJKsc-Regular.otf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]

_SCAN_DIRS = {
    "darwin": ["/System/Library/Fonts", "/System/Library/Fonts/Supplemental",
               "/Library/Fonts", str(Path.home() / "Library/Fonts")],
    "linux": ["/usr/share/fonts", "/usr/local/share/fonts",
              str(Path.home() / ".fonts"), str(Path.home() / ".local/share/fonts")],
}
_SCAN_GLOBS = ["*Kaiti*", "*Xingkai*", "*STKaiti*", "*Songti*", "*PingFang*",
               "*Heiti*", "*Hiragino*GB*", "*Noto*CJK*", "*SourceHan*",
               "*wqy*", "*ukai*", "*msyh*", "*simhei*", "*simsun*", "*STXINGKA*"]


def _platform_key() -> str:
    if sys.platform.startswith("win"):
        return "win32"
    if sys.platform == "darwin":
        return "darwin"
    return "linux"


def _candidates() -> list[str]:
    return {"win32": WINDOWS, "darwin": MACOS}.get(_platform_key(), LINUX)


def _scan() -> str:
    for d in _SCAN_DIRS.get(_platform_key(), []):
        base = Path(d)
        if not base.is_dir():
            continue
        for pat in _SCAN_GLOBS:
            try:
                hits = sorted(p for p in base.rglob(pat) if p.is_file())
            except (OSError, PermissionError):
                continue
            if hits:
                return str(hits[0])
    return ""


def _fc_match() -> str:
    """Linux 上用 fontconfig 问一句（mac 没有 fc-match，有也不影响）。"""
    if not shutil.which("fc-match"):
        return ""
    for family in ("Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei",
                   "sans-serif:lang=zh"):
        try:
            out = subprocess.run(
                ["fc-match", "-f", "%{file}", family],
                capture_output=True, text=True, timeout=5, check=False)
        except (OSError, subprocess.SubprocessError):
            continue
        p = (out.stdout or "").strip()
        if p and Path(p).is_file():
            return p
    return ""


def find_cjk_font(explicit: str | Path = "") -> str:
    """返回一个可用的中文字体路径；找不到返回空字符串。"""
    for cand in (explicit, os.environ.get("HV_CJK_FONT", "")):
        if cand and Path(str(cand)).is_file():
            return str(cand)
    for p in _candidates():
        if Path(p).is_file():
            return p
    p = _scan()
    if p:
        return p
    p = _fc_match()
    if p:
        return p
    return ""


def describe_missing() -> str:
    """找不到字体时给用户的提示（不同平台给不同的安装建议）。"""
    key = _platform_key()
    if key == "darwin":
        hint = ("macOS 自带楷体/宋体/苹方，正常不该找不到。"
                "若被裁剪过，可安装字体或指定：export HV_CJK_FONT=/System/Library/Fonts/Supplemental/Kaiti.ttc")
    elif key == "linux":
        hint = ("安装一款中文字体即可，例如：\n"
                "  Debian/Ubuntu: sudo apt install fonts-noto-cjk\n"
                "  CentOS/Fedora: sudo dnf install google-noto-sans-cjk-fonts\n"
                "  或指定：export HV_CJK_FONT=/usr/share/fonts/.../your-cjk-font.ttc")
    else:
        hint = "Windows 自带中文字体，正常不该找不到；可用 image.glyph_overlay.font 指定。"
    return hint


def load_font(size: int, explicit: str | Path = "", index: int = 0):
    """加载中文字体；找不到时退回 Pillow 默认位图字体（能跑但中文会变方框）。

    返回 (font, path)。调用方可以据此判断是否要告警。
    """
    from PIL import ImageFont

    path = find_cjk_font(explicit)
    if path:
        try:
            return ImageFont.truetype(path, size, index=index), path
        except OSError as exc:  # .ttc 索引不对 / 文件损坏
            LOG.warning("加载字体失败 %s: %s，改用默认字体", path, exc)
    try:
        return ImageFont.load_default(size), ""
    except TypeError:      # 老版本 Pillow 的 load_default 不接受 size
        return ImageFont.load_default(), ""
