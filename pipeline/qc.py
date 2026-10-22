"""qc 阶段：三路质检 + 打分定级 + 重试反馈。

    技术质检  Pillow 直方图/边缘/尺寸/近似重复(pHash)  —— 永远可用
    文字泄漏  可选 OCR（汉字/字母/水印），无引擎时给弱启发式提示
    语义质检  可选 VLM：可解性测试（不看答案猜）+ 一致性核验（给答案查元素）

定级：accept / retry / reject / needs_human
retry 会把 feedback_for_retry 写回状态库，generate 阶段下一轮生成时自动拼进提示词。
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .config import Config
from .providers.ocr import OCRChecker, heuristic_text_hint, score_text_evidence
from .providers.vlm_openai import VLMChecker
from .schema import QCResult
from .state import State
from .utils import LOG

try:
    from PIL import Image, ImageFilter, ImageOps
    PIL_OK = True
except Exception:  # noqa: BLE001
    PIL_OK = False


# --------------------------------------------------------------------------- #
# 技术质检
# --------------------------------------------------------------------------- #
def dhash(img, size: int = 8) -> int:
    """64 位差异哈希，用于近似重复检测。"""
    g = ImageOps.grayscale(img).resize((size + 1, size), Image.Resampling.LANCZOS)
    px = list(g.getdata())
    bits = 0
    for row in range(size):
        for col in range(size):
            left = px[row * (size + 1) + col]
            right = px[row * (size + 1) + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def technical_check(path: Path, cfg: Config) -> tuple[dict[str, Any], Any]:
    info: dict[str, Any] = {"ok": False, "issues": []}
    if not PIL_OK:
        info["issues"].append("Pillow 不可用，跳过技术质检")
        info["ok"] = True
        return info, None
    if not path.exists():
        info["issues"].append("图像文件不存在")
        return info, None
    try:
        img = Image.open(path)
        img.load()
    except Exception as exc:  # noqa: BLE001
        info["issues"].append(f"图像损坏无法解码: {exc}")
        return info, None

    img = img.convert("RGB")
    w, h = img.size
    info.update({"width": w, "height": h, "bytes": path.stat().st_size})

    min_side = int(cfg.get("qc.technical.min_side", 256))
    if min(w, h) < min_side:
        info["issues"].append(f"分辨率过低 {w}x{h}")
    aspect = max(w, h) / max(1, min(w, h))
    info["aspect"] = round(aspect, 2)
    if aspect > float(cfg.get("qc.technical.max_aspect", 3.0)):
        info["issues"].append(f"长宽比异常 {aspect:.2f}")

    gray = ImageOps.grayscale(img)
    small = gray.resize((256, 256))
    px = list(small.getdata())
    mean = sum(px) / len(px)
    std = (sum((v - mean) ** 2 for v in px) / len(px)) ** 0.5
    info["gray_std"] = round(std, 2)
    info["gray_mean"] = round(mean, 2)
    if std < float(cfg.get("qc.technical.min_std", 12.0)):
        info["issues"].append(f"画面接近纯色/空白（灰度标准差 {std:.1f}）")

    edges = gray.resize((256, 256)).filter(ImageFilter.FIND_EDGES)
    ep = list(edges.getdata())
    edge_density = sum(1 for v in ep if v > 24) / len(ep)
    info["edge_density"] = round(edge_density, 4)
    if edge_density < float(cfg.get("qc.technical.min_edge_density", 0.005)):
        info["issues"].append("画面几乎没有细节（疑似空白或纯色块）")

    info["dhash"] = dhash(img)
    info["ok"] = not info["issues"]
    return info, img


# --------------------------------------------------------------------------- #
# 答案比对
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    return "".join(ch for ch in (s or "") if ch.isalnum())


def answer_matches(answer: str, guess: str) -> bool:
    a, g = _norm(answer), _norm(guess)
    if not a or not g:
        return False
    if a == g or a in g or g in a:
        return True
    # 允许只答对一部分字（如"守株待兔"答成"守株"）
    common = len(set(a) & set(g))
    return common >= max(2, len(a) - 1)


# --------------------------------------------------------------------------- #
# 单张图的质检
# --------------------------------------------------------------------------- #
def check_image(cfg: Config, image_row: dict, plan_elements: list[str], answer: str,
                composition: str, attempt: int, max_attempt: int,
                ocr: OCRChecker, vlm: VLMChecker, seen_hashes: dict[int, tuple[str, str]],
                expected_text: list[str] | None = None) -> QCResult:
    res = QCResult(item_id=image_row["item_id"], checker="technical")
    path = Path(image_row["path"])
    issues: list[str] = []
    expected = [e for e in (expected_text or []) if e]

    tech, img = technical_check(path, cfg)
    res.technical = {k: v for k, v in tech.items() if k != "issues"}

    # 近似重复检测：默认只查"跨词条"重复（同一词条的不同尝试本来就该相似）
    scope = str(cfg.get("qc.technical.dedup_scope", "cross_item")).lower()
    skip_providers = [str(p).lower() for p in
                      (cfg.get("qc.technical.skip_dedup_providers") or [])]
    provider = str(image_row.get("provider", "")).lower()
    do_dedup = scope != "off" and provider not in skip_providers
    if do_dedup and isinstance(tech.get("dhash"), int):
        threshold = int(cfg.get("qc.technical.phash_hamming_dup", 6))
        for prev_hash, (prev_id, prev_item) in seen_hashes.items():
            if hamming(prev_hash, tech["dhash"]) > threshold:
                continue
            if scope == "cross_item" and prev_item == image_row["item_id"]:
                continue
            issues.append(f"与已有图片高度相似（{prev_id}）")
            res.technical["duplicate_of"] = prev_id
            break
    if isinstance(tech.get("dhash"), int):
        seen_hashes.setdefault(tech["dhash"], (image_row["id"], image_row["item_id"]))

    if not tech["ok"]:
        issues.extend(tech.get("issues", []))
        res.verdict = "reject"
        res.score = 0.0
        res.issues = issues
        res.feedback_for_retry = (
            "the image is blank, corrupted or too plain; draw a clear, detailed "
            "composition with easily recognisable objects")
        return res
    score = 1.0
    checkers = ["technical"]

    # ---------------- 文字泄漏 ---------------- #
    # Mock 后端画的是"构图示意卡"，上面给人看的标签不算生成缺陷，默认跳过 OCR。
    ocr_skip = [str(p).lower() for p in (cfg.get("qc.ocr.skip_providers") or [])]
    provider_name = str(image_row.get("provider", "")).lower()
    if img is not None and provider_name in ocr_skip:
        res.technical["ocr_skipped"] = (
            f"provider={provider_name} 在 qc.ocr.skip_providers 中（示意卡标签不计为文字泄漏）")
    elif img is not None:
        ocr_out = ocr.detect(path, expected_text=expected)
        if ocr_out.get("available"):
            texts = list(ocr_out.get("texts") or [])
            detected = [t for t, _ in texts]
            joined = "".join(detected)
            # 字形参与式（举手之劳的"劳"）是**计划内**的文字，先把它摘掉再判泄漏；
            # 这一步不做，合法的字形题会被自己的质检全部误杀。
            filtered = [(t, c) for t, c in texts
                        if not any(e in t for e in expected)] if expected else texts
            ev = score_text_evidence(
                filtered, float(cfg.get("qc.ocr.weak_char_weight", 0.34)))
            threshold = float(cfg.get("qc.ocr.max_text_chars", 0)) + 0.5
            if ev["effective_chars"] > threshold:
                res.text_free = False
                res.text_found = [t for t, _ in filtered][:6]
                issues.append(f"画面出现计划外的文字：{res.text_found}")
                score -= 0.5
                checkers.append("ocr")
            else:
                res.text_free = True
                checkers.append("ocr")
                res.technical["ocr_evidence"] = ev
                if expected:
                    missing = [e for e in expected if e not in joined]
                    if missing:
                        res.technical["glyph_missing"] = missing
                        issues.append(f"期望出现的字形未检出：{missing}")
                        score -= 0.15
                    else:
                        res.technical["glyph_detected"] = expected
        else:
            hint = heuristic_text_hint(img)
            res.technical["text_heuristic"] = hint
            if hint.get("suspicious"):
                issues.append("疑似有文字笔画（未装 OCR，无法确认）")

    # ---------------- 语义质检 ---------------- #
    if vlm.enabled:
        checkers.append("vlm")
        # allowed_text 是后加的形参：自定义的 VLM 实现可能还没有，退回旧签名调用，
        # 不要因为一个可选特性把整条链路打断。
        try:
            ver = vlm.verify(path, answer, plan_elements, composition,
                             allowed_text=expected)
        except TypeError:
            ver = vlm.verify(path, answer, plan_elements, composition)
        res.technical["vlm_verify"] = ver
        if "error" not in ver:
            if ver.get("text_present"):
                res.text_free = False
                content = str(ver.get("text_content", ""))[:80]
                res.text_found = list({*(res.text_found), content}) if content else res.text_found
                issues.append(f"VLM 报告画面有文字：{content}")
                score -= 0.5
            elif res.text_free is None:
                res.text_free = True

            present = [str(x) for x in (ver.get("elements_present") or [])]
            missing = [str(x) for x in (ver.get("elements_missing") or [])]
            total = len(present) + len(missing)
            if total:
                res.element_coverage = round(len(present) / total, 3)
                res.missing_elements = missing
                score = score * 0.55 + res.element_coverage * 0.45
                if res.element_coverage < 0.6:
                    issues.append(f"关键元素缺失：{missing}")
            if ver.get("matches_answer") is False:
                issues.append("VLM 认为画面与标准答案不吻合")
                score -= 0.2
            for p in (ver.get("problems") or [])[:4]:
                issues.append(f"VLM：{p}")

        if vlm.check_solvability:
            sol = vlm.solve(path)
            res.technical["vlm_solve"] = sol
            if "error" not in sol:
                guess = str(sol.get("guess", ""))
                res.guessed_answer = guess
                res.solvable = answer_matches(answer, guess)
                if res.solvable:
                    score = min(1.0, score + 0.10)
                else:
                    score -= 0.18
                    issues.append(f"模型猜不出（猜成「{guess}」）")
    else:
        issues.append("未启用 VLM 语义质检：元素齐备性与可解性未经机器验证")
        score = min(score, 0.85)

    res.score = round(max(0.0, min(1.0, score)), 3)
    res.issues = issues
    res.checker = "+".join(checkers)

    # ---------------- 定级 ---------------- #
    # 关键区分：哪些失败"重出图能救"，哪些不能。
    #   不可修复（reject）  ：图损坏、空白、纯色、分辨率/长宽比异常 —— 通常是工作流坏了，重试纯属浪费
    #   可修复（retry）     ：画面有文字、与别的图撞车、元素不全、模型猜不出 —— 换种子/加反馈就能改
    #   到顶了（needs_human/reject）：重试次数用完
    min_accept = float(cfg.get("qc.min_accept_score", 0.75))
    # "机器猜不出来"只是扣分并记录；打开这个开关才会因此拒绝入库
    require_solvable = bool(cfg.get("qc.require_solvable", False))
    joined = " ".join(issues)
    unfixable = any(k in joined for k in
                    ("损坏", "无法解码", "空白", "纯色", "分辨率", "长宽比", "几乎没有细节"))
    duplicate = res.technical.get("duplicate_of")
    text_leak = res.text_free is False

    if unfixable:
        res.verdict = "reject"
    elif (res.score >= min_accept and not text_leak and not duplicate
          and not (require_solvable and res.solvable is False)):
        res.verdict = "accept"
    elif attempt >= max_attempt:
        # 到顶仍带文字 -> 这种图对"看图猜谜"不可用，直接淘汰
        res.verdict = "reject" if (text_leak or res.score < 0.5) else "needs_human"
    else:
        res.verdict = "retry"

    res.feedback_for_retry = _make_feedback(res)
    return res


def _make_feedback(res: QCResult) -> str:
    parts: list[str] = []
    joined = " ".join(res.issues)
    if "文字" in joined or res.text_free is False:
        parts.append("remove every trace of text, characters, letters, numbers and "
                     "watermarks; the image must contain pictures only")
    if res.missing_elements:
        parts.append("make these elements large, obvious and clearly recognisable: "
                     + ", ".join(res.missing_elements[:5]))
    if res.element_coverage is not None and res.element_coverage < 0.6:
        parts.append("simplify the composition, use fewer and bigger objects")
    if res.solvable is False:
        parts.append("strengthen the visual clue for the answer: exaggerate the key "
                     "objects and their spatial relation, remove decorative clutter")
    if "与标准答案不吻合" in joined:
        parts.append("re-check that the depicted scene literally matches the intended "
                     "answer's imagery")
    if res.technical.get("duplicate_of"):
        # 撞车时的修正方向：提示词层面能做的是换构图、视角与配色
        parts.append("the composition is too similar to another item: change the "
                     "arrangement, the viewing angle and the colour scheme, and vary "
                     "the size and position of the main objects so the two are clearly "
                     "distinguishable")
    if not parts:
        # 兜底：只要判了 retry，就必须给出一条可执行的修正方向。
        # 否则提示词没变，重试会反复拿到同一张图（实测踩过这个坑）。
        return ("revise the composition: change the arrangement, viewing angle and "
                "colour scheme, and make the key objects larger and more obvious")
    return "; ".join(parts[:3])


# --------------------------------------------------------------------------- #
# 批处理
# --------------------------------------------------------------------------- #
def run(cfg: Config, state: State, limit: int = 0, vlm: VLMChecker | None = None) -> dict:
    ocr = OCRChecker(cfg)
    vlm = vlm or VLMChecker(cfg)
    if vlm.enabled:
        ok, msg = vlm.health()
        LOG.info("VLM 语义质检：%s", msg if ok else f"不可用（{msg}），将只做技术+OCR质检")
        if not ok:
            vlm.enabled = False
    if ocr.enabled:
        ocr._load()  # noqa: SLF001  预加载并打印引擎情况
        LOG.info("OCR：%s", ocr.note)

    # 跨轮次去重：先载入已有 dhash
    seen: dict[int, tuple[str, str]] = {}
    for row in state.all_qc():
        try:
            import json as _json
            d = _json.loads(row["result_json"])
            h = (d.get("technical") or {}).get("dhash")
            if isinstance(h, int):
                seen.setdefault(h, (row["image_id"], row["item_id"]))
        except Exception:  # noqa: BLE001
            continue

    todo = state.items_needing_qc(limit=limit)
    if not todo:
        LOG.info("qc：没有待质检的图片")
        return {"checked": 0}

    max_attempt = int(cfg.get("qc.max_retry", 2)) + 1
    verdicts: Counter[str] = Counter()
    for row in todo:
        item = state.get_item(row["item_id"])
        plan = state.get_plan(row["item_id"])
        if item is None or plan is None:
            continue
        res = check_image(cfg, row, [e.en for e in plan.elements if e.en],
                          item.text, plan.composition, int(row["attempt"]),
                          max_attempt, ocr, vlm, seen,
                          expected_text=[plan.glyph] if plan.glyph else None)
        state.save_qc(row["id"], item.id, res, int(row["attempt"]))
        verdicts[res.verdict] += 1
        if res.verdict == "accept":
            state.mark_image_selected(row["id"], True)
            LOG.info("✓ %s 通过（%.2f，%s）", item.text, res.score, res.checker)
        else:
            LOG.info("✗ %s -> %s（%.2f）：%s", item.text, res.verdict, res.score,
                     "；".join(res.issues[:2])[:110])

    LOG.info("qc 完成：%s（共 %d 张）", dict(verdicts), len(todo))
    return {"checked": len(todo), **{f"verdict:{k}": v for k, v in verdicts.items()}}
