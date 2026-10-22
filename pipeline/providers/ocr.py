"""OCR 文字泄漏检测。

会意类谜面的第一失败模式是：扩散模型在画面里"写出"一堆伪汉字、字母、水印、签名。
本模块用可选 OCR 引擎把这种图自动揪出来（缺引擎时会降级为启发式检测，不影响主流程）。

支持的引擎（按可用性自动选择，装哪个用哪个）：
    rapidocr_onnxruntime  推荐：纯 CPU、离线、中英混排效果好、pip 依赖轻
    paddleocr             精度最高，但依赖重
    easyocr               容易装，速度一般
    pytesseract           需要另装 Tesseract 程序与 chi_sim 语言包
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from ..utils import LOG

# 允许把可选依赖装在项目内（推荐做法：pip install --target <repo>/.deps
# rapidocr-onnxruntime），这样不污染系统 Python，换机器只需删掉 .deps。
# 追加到 sys.path 末尾：让自带的 Pillow/numpy 优先，只从这里取缺失的包（onnxruntime 等）。
_DEPS_DIR = Path(__file__).resolve().parent.parent.parent / ".deps"
if _DEPS_DIR.is_dir() and str(_DEPS_DIR) not in sys.path:
    sys.path.append(str(_DEPS_DIR))
    LOG.debug("已加载项目内可选依赖目录：%s", _DEPS_DIR)


# 会被判为"图上写了字"的字符类：汉字 / 拉丁字母 / 日韩假名
def _is_text_char(ch: str) -> bool:
    code = ord(ch)
    if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
        return True
    if 0x3040 <= code <= 0x30FF or 0xAC00 <= code <= 0xD7AF:
        return True
    if ch.isascii() and ch.isalpha():
        return True
    return False


def _is_text_char_any(tok: str) -> bool:
    """整串里只要含一个"像文字"的字符就算候选（含数字，数字在水印里很常见）。"""
    for ch in tok:
        if _is_text_char(ch) or ch.isdigit():
            return True
    return False


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF
            or 0x3040 <= code <= 0x30FF or 0xAC00 <= code <= 0xD7AF)


def _is_single_latin(token: str) -> bool:
    """单个 ASCII 字母/数字——这是几何图形最常见的误识别（圆->O、0，叉->X）。"""
    t = token.strip()
    return len(t) == 1 and t.isascii() and t.isalnum()


def score_text_evidence(texts: list[tuple[str, float]],
                        weak_weight: float = 0.34) -> dict[str, Any]:
    """把 OCR 结果折算成"文字证据强度"。

    单个孤立拉丁字符只算弱证据（几何形状误报太多）；汉字、多字符单词、
    数字串都算强证据。这样"圆被读成 O"不会误杀一张好图。
    """
    strong = weak = 0
    cjk = 0
    for token, _conf in texts:
        t = (token or "").strip()
        if not t:
            continue
        if _is_single_latin(t):
            weak += 1
            continue
        for ch in t:
            if not _is_text_char(ch):
                continue
            if _is_cjk(ch):
                cjk += 1
            strong += 1
    effective = strong + weak * weak_weight
    return {"strong_chars": strong, "weak_chars": weak, "cjk_chars": cjk,
            "effective_chars": round(effective, 3)}


class OCRChecker:
    """统一的 OCR 接口。"""

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.enabled = bool(cfg.get("qc.ocr.enabled", True))
        self.engine_name = str(cfg.get("qc.ocr.engine", "auto")).lower()
        self.min_conf = float(cfg.get("qc.ocr.min_conf", 0.6))
        self._engine = None
        self._loaded = False
        self.available = False
        self.note = ""

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.enabled or self.engine_name == "none":
            self.note = "OCR 已关闭"
            return
        order = (["rapidocr", "paddleocr", "easyocr", "pytesseract"]
                 if self.engine_name == "auto" else [self.engine_name])
        for name in order:
            try:
                if name == "rapidocr":
                    from rapidocr_onnxruntime import RapidOCR  # type: ignore
                    self._engine = ("rapidocr", RapidOCR())
                elif name == "paddleocr":
                    from paddleocr import PaddleOCR  # type: ignore
                    self._engine = ("paddleocr", PaddleOCR(use_angle_cls=True, lang="ch",
                                                           show_log=False))
                elif name == "easyocr":
                    import easyocr  # type: ignore
                    self._engine = ("easyocr", easyocr.Reader(["ch_sim", "en"], gpu=False,
                                                              verbose=False))
                elif name == "pytesseract":
                    import pytesseract  # type: ignore
                    self._engine = ("pytesseract", pytesseract)
                else:
                    continue
                self.available = True
                self.note = f"OCR 引擎：{name}"
                LOG.info("质检 OCR 引擎就绪：%s", name)
                return
            except Exception as exc:  # noqa: BLE001
                LOG.debug("OCR 引擎 %s 不可用：%s", name, exc)
        self.note = ("未安装任何 OCR 引擎（可 pip install rapidocr-onnxruntime），"
                     "文字泄漏将退化为启发式检测")
        LOG.warning("%s", self.note)

    # ------------------------------------------------------------------ #
    def detect(self, image_path: Path,
               expected_text: list[str] | None = None) -> dict[str, Any]:
        """返回 {'available','texts','chars','effective_chars','has_text'}

        实测教训：只对整图跑一次 OCR 会漏掉"物体上的小字"——生成图里
        杯子上的金色伪汉字、角落里的小水印都漏过。所以默认做**多尺度 + 分块**：
        原图、2 倍放大图、以及把图切成四块分别放大后再各跑一次，最后合并去重。

        expected_text 非空表示这是"字形参与式"题目（画面里本来就有一个汉字）：
        此时**不做分块**——分块会把那个大字切开，OCR 会把半个字读成别的字
        （实测把「劳」的下半读成了「力」），从而误报成文字泄漏。
        """
        self._load()
        if not self.available:
            return {"available": False, "texts": [], "chars": 0, "has_text": None,
                    "note": self.note}
        allow_tiles = not [e for e in (expected_text or []) if e]
        try:
            variants = self._variants(image_path, allow_tiles=allow_tiles)
            merged: list[tuple[str, float]] = []
            seen: set[str] = set()
            for tag, img in variants:
                for tok, conf in self._run_image(img):
                    key = tok.strip()
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    merged.append((key, conf))
        except Exception as exc:  # noqa: BLE001
            LOG.warning("OCR 失败 %s: %s", image_path.name, exc)
            return {"available": True, "texts": [], "chars": 0, "has_text": None,
                    "note": f"OCR 异常: {exc}"}
        texts = [(t, c) for t, c in merged
                 if c >= self.min_conf and t.strip() and _is_text_char_any(t)]
        ev = score_text_evidence(texts, float(self.cfg.get("qc.ocr.weak_char_weight", 0.34)))
        chars = sum(1 for t, _ in texts for ch in t if _is_text_char(ch))
        # 需要"至少一个实打实的字符"才算有文字：max_text_chars=0 时阈值为 0.5
        threshold = float(self.cfg.get("qc.ocr.max_text_chars", 0)) + 0.5
        return {"available": True, "texts": texts, "chars": chars,
                "has_text": ev["effective_chars"] > threshold,
                "evidence": ev, "variants": [t for t, _ in variants],
                "note": self.note}

    # ------------------------------------------------------------------ #
    def _variants(self, image_path: Path,
                  allow_tiles: bool = True) -> list[tuple[str, Any]]:
        """生成多尺度 + 分块的待检测图（纯内存，不落盘）。"""
        try:
            from PIL import Image
        except Exception:  # noqa: BLE001
            return [("orig", image_path)]
        out: list[tuple[str, Any]] = [("orig", image_path)]
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:  # noqa: BLE001
            return out
        if not bool(self.cfg.get("qc.ocr.multiscale", True)):
            return out
        scale = float(self.cfg.get("qc.ocr.upscale", 2.0))
        big = img.resize((int(img.width * scale), int(img.height * scale)),
                         Image.LANCZOS)
        out.append(("2x", big))
        # 四象限：物体上的小字在整图里字号太小，切开再放大更容易检出；
        # 但字形参与式的图里有个大字，切开会被读成碎片，所以那种情况跳过。
        if allow_tiles and bool(self.cfg.get("qc.ocr.tiles", True)):
            w, h = big.size
            for i in range(2):
                for j in range(2):
                    box = (j * w // 2, i * h // 2, (j + 1) * w // 2, (i + 1) * h // 2)
                    out.append((f"q{i}{j}", big.crop(box)))
        return out

    def _run_image(self, img: Any) -> list[tuple[str, float]]:
        """对单张图（路径或 PIL 图像）跑一次 OCR。"""
        if isinstance(img, Path) or isinstance(img, str):
            return self._run(Path(img))
        import numpy as np
        arr = np.asarray(img)[:, :, ::-1]        # PIL RGB -> OCR 需要的 BGR
        kind, engine = self._engine
        if kind == "rapidocr":
            result, _ = engine(arr)
            return [(str(r[1]), float(r[2])) for r in (result or []) if len(r) >= 3]
        if kind == "easyocr":
            return [(str(t), float(c)) for _b, t, c in engine.readtext(np.asarray(img))]
        if kind == "paddleocr":
            res = engine.ocr(np.asarray(img), cls=True)
            out = []
            for page in (res or []):
                for line in (page or []):
                    try:
                        out.append((str(line[1][0]), float(line[1][1])))
                    except Exception:  # noqa: BLE001
                        continue
            return out
        # 其他引擎退回"先落临时文件"的老路
        import tempfile
        tmp = Path(tempfile.gettempdir()) / "hv_ocr_variant.png"
        img.save(tmp)
        return self._run(tmp)

    def _run(self, image_path: Path) -> list[tuple[str, float]]:
        kind, engine = self._engine
        p = str(image_path)
        if kind == "rapidocr":
            result, _elapse = engine(p)
            out = []
            for row in (result or []):
                # [box, text, score]
                if len(row) >= 3:
                    out.append((str(row[1]), float(row[2])))
            return out
        if kind == "paddleocr":
            res = engine.ocr(p, cls=True)
            out = []
            for page in (res or []):
                for line in (page or []):
                    try:
                        txt, conf = line[1][0], float(line[1][1])
                        out.append((str(txt), conf))
                    except Exception:  # noqa: BLE001
                        continue
            return out
        if kind == "easyocr":
            res = engine.readtext(p)
            return [(str(t), float(c)) for _b, t, c in res]
        if kind == "pytesseract":
            from PIL import Image
            data = engine.image_to_data(Image.open(p), lang="chi_sim+eng",
                                        output_type=engine.Output.DICT)
            out = []
            for txt, conf in zip(data.get("text", []), data.get("conf", [])):
                try:
                    c = float(conf)
                except (TypeError, ValueError):
                    c = 0.0
                if c >= 0:
                    out.append((str(txt), c / 100.0))
            return out
        return []


# --------------------------------------------------------------------------- #
# 无 OCR 时的启发式兜底
# --------------------------------------------------------------------------- #
def heuristic_text_hint(img) -> dict[str, Any]:
    """粗略估计"画面是否含大量文字"：文字通常是高对比、细、水平重复的笔画。

    只在没有 OCR 引擎时作为弱信号使用，不作为判定依据。
    """
    try:
        from PIL import ImageFilter, ImageOps
        g = ImageOps.grayscale(img).resize((512, 512))
        edges = g.filter(ImageFilter.FIND_EDGES)
        hist = edges.histogram()
        total = sum(hist) or 1
        strong = sum(hist[80:]) / total          # 强边缘像素占比
        mid = sum(hist[30:80]) / total
        return {
            "strong_edge_ratio": round(strong, 4),
            "mid_edge_ratio": round(mid, 4),
            "suspicious": strong > 0.10 and mid < 0.10,
        }
    except Exception:  # noqa: BLE001
        return {"suspicious": False}
