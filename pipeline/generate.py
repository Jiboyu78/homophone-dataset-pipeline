"""generate 阶段：按 plan 批量出图，支持并发、重试、提示词级缓存与质检反馈回灌。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import fonts
from .config import Config
from .providers.image_base import ImageProvider, build_provider
from .schema import Item, Plan
from .state import State
from .utils import LOG, retry_call, sha1_file, sha1_text


def _prompt_hash(cfg: Config, plan: Plan, feedback: str) -> str:
    return sha1_text(
        plan.prompt_en, plan.negative_prompt, feedback,
        cfg.get("image.provider"),
        cfg.get("image.comfyui.width"), cfg.get("image.comfyui.height"),
        cfg.get("image.sdwebui.width"), cfg.get("image.sdwebui.height"),
        cfg.get("image.mock.width"), cfg.get("image.mock.height"),
    )


def apply_glyph_overlay(cfg: Config, plan: Plan, path: Path) -> bool:
    """字形参与式：把 plan.glyph 指定的汉字确定性地叠加到图上。

    为什么不让扩散模型自己写？本地 SD1.5/SDXL 写汉字基本必然写错，
    参考题要的是"画面中央一个正确的『劳』字"，所以生成时留白、出图后合成，
    结果 100% 可控。想让模型自己写就把 image.glyph_overlay.mode 改成 prompt。
    """
    g = (plan.extra or {}).get("glyph") or {}
    ch = plan.glyph or g.get("char") or ""
    if not ch or str(g.get("mode", "composite")).lower() != "composite":
        return False
    ov = cfg.get("image.glyph_overlay", {}) or {}
    if not ov.get("enabled", True):
        return False
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:  # noqa: BLE001
        LOG.warning("Pillow 不可用，跳过字形叠加")
        return False

    # 跨平台字体查找（先前硬编码 C:\Windows\Fonts\，在 macOS/Linux 上会静默失效）
    font_path = fonts.find_cjk_font(ov.get("font") or "")
    if not font_path:
        LOG.warning("找不到可用的中文字体，跳过字形叠加。%s", fonts.describe_missing())
        return False

    img = Image.open(path).convert("RGB")
    w, h = img.size
    size = max(24, int(h * float(ov.get("size_ratio", 0.55))))
    try:
        font = ImageFont.truetype(font_path, size)
    except OSError as exc:
        LOG.warning("加载字体失败 %s: %s", font_path, exc)
        return False

    d = ImageDraw.Draw(img)
    box = d.textbbox((0, 0), ch, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    pos = str(g.get("position") or ov.get("position", "center-left"))
    fx, fy = {"center": (0.5, 0.5), "center-left": (0.30, 0.50),
              "left": (0.24, 0.50), "center-right": (0.72, 0.50),
              "top": (0.5, 0.26), "bottom": (0.5, 0.74)}.get(pos, (0.30, 0.5))
    x = w * fx - tw / 2 - box[0]
    y = h * fy - th / 2 - box[1]

    ink = tuple(ov.get("ink_color", [26, 26, 30]))
    halo = tuple(ov.get("halo_color", [250, 248, 240]))
    # 白色描边提高可读性，再叠墨色字
    for dx, dy in ((-3, 0), (3, 0), (0, -3), (0, 3), (-2, -2), (2, 2), (-2, 2), (2, -2)):
        d.text((x + dx, y + dy), ch, font=font, fill=halo)
    d.text((x, y), ch, font=font, fill=ink)
    img.save(path)
    LOG.debug("已叠加字形「%s」到 %s", ch, path.name)
    return True


def apply_postprocess(cfg: Config, path: Path) -> bool:
    """把生成结果缩放到交付尺寸。

    SD1.5 的原生分辨率是 512，硬拉 1152x768 既费显存又会出重复构图；
    所以用小尺寸生成、再用 Lanczos 放大到交付尺寸（P100 显存有限时的常规做法）。
    """
    pp = cfg.get("image.postprocess", {}) or {}
    if not pp.get("enabled", False):
        return False
    tw = int(pp.get("target_width") or 0)
    th = int(pp.get("target_height") or 0)
    if tw <= 0 or th <= 0:
        return False
    try:
        from PIL import Image
    except Exception:  # noqa: BLE001
        return False
    img = Image.open(path).convert("RGB")
    if (img.width, img.height) == (tw, th):
        return False
    method = str(pp.get("method", "lanczos")).lower()
    resample = {"lanczos": Image.LANCZOS, "bicubic": Image.BICUBIC,
                "nearest": Image.NEAREST}.get(method, Image.LANCZOS)
    img = img.resize((tw, th), resample)
    img.save(path)
    LOG.debug("已放大到 %dx%d：%s", tw, th, path.name)
    return True


def _gen_one(cfg: Config, state: State, provider: ImageProvider, item: Item,
             plan: Plan, attempt: int, force: bool = False) -> tuple[bool, str]:
    # 重试时把上一轮质检的修正建议回灌进提示词
    feedback = ""
    if attempt > 1:
        prev = state.latest_qc_for_item(item.id)
        if prev and prev.feedback_for_retry:
            feedback = prev.feedback_for_retry

    phash = _prompt_hash(cfg, plan, feedback)
    # 重试必须真的重出图：如果仍按提示词哈希查缓存，那么"反馈为空的重试"
    # （例如只判了"与别的图撞车"）会直接命中上一次的图，重试轮次白耗。
    is_retry = attempt > 1
    if not force and not is_retry:
        cached = state.find_image_by_prompt_hash(item.id, phash)
        if cached and Path(cached["path"]).exists():
            LOG.debug("命中缓存，跳过出图：%s", item.text)
            return True, "cached"

    image_id = f"{item.id}_a{attempt}"
    out_path = cfg.images_dir / f"{item.id}_a{attempt}.png"
    max_retry = int(cfg.get("image.max_retry", 2))

    try:
        res = retry_call(
            lambda: provider.generate(item, plan, out_path, seed=-1, feedback=feedback),
            attempts=max(1, max_retry + 1), base_delay=2.0,
            label=f"出图[{item.text}]")
    except Exception as exc:  # noqa: BLE001
        LOG.error("出图失败「%s」: %s", item.text, exc)
        return False, str(exc)

    # 顺序很重要：先放大到交付尺寸，再叠字形——反过来的话毛笔字会被放大糊掉
    resized = apply_postprocess(cfg, res.path)
    overlaid = apply_glyph_overlay(cfg, plan, res.path)
    sha = sha1_file(res.path) if res.path.exists() else ""
    state.add_image(image_id, item.id, str(res.path), sha, attempt,
                    res.provider or provider.name, phash, res.seed)
    LOG.info("出图 %s [%s] attempt=%d seed=%s%s%s", item.text, plan.composition, attempt,
             res.seed, " +放大" if resized else "",
             " +字形" + plan.glyph if overlaid else "")
    return True, "ok"


def _pick_with_quota(items: list[tuple[Item, Plan, int]], quota: int,
                     glyph_cap: float) -> list[tuple[Item, Plan, int]]:
    """从候选池里挑 quota 条，并限制"字形参与式"的占比。

    轮询取样会把手法的自然比例（约 7%）抬得过高；参考题的经验值是 4 张里 1 张，
    所以按上限交替挑选：非字形式先取，字形式受配额限制，超出的留在池子里下批再取。
    """
    if quota <= 0:
        return []
    if not glyph_cap or glyph_cap >= 1:
        return items[:quota]
    allowed = max(0, int(round(quota * glyph_cap)))
    glyphs = [t for t in items if t[1].technique == "glyph"]
    others = [t for t in items if t[1].technique != "glyph"]
    head: list[tuple[Item, Plan, int]] = []
    g_i = o_i = 0
    while len(head) < quota and (o_i < len(others) or (g_i < allowed and g_i < len(glyphs))):
        if o_i < len(others):
            head.append(others[o_i])
            o_i += 1
        if len(head) < quota and g_i < allowed and g_i < len(glyphs):
            head.append(glyphs[g_i])
            g_i += 1
    # 池子里非字形式的也不够时，用剩余字形式补齐，别让配额饿死整批
    if len(head) < quota:
        for t in glyphs[g_i:]:
            head.append(t)
            if len(head) >= quota:
                break
    return head


def run(cfg: Config, state: State, limit: int = 0, provider: ImageProvider | None = None,
        force: bool = False, check_health: bool = True, mode: str = "new") -> dict:
    """出图。

    mode="new"   出图预算 limit 只作用于"首次出图"的条目；已判 retry 的条目无条件重出
    mode="retry" 只处理待重试条目（闭环第 2 轮起用）
    mode="all"   两者混合，并按 limit 统一截断
    """
    provider = provider or build_provider(cfg)
    if check_health:
        ok, msg = provider.health()
        if not ok:
            LOG.error("生图后端不可用：%s", msg)
            LOG.error("提示：把 image.provider 改成 mock 可先跑通全流程（生成构图示意卡）")
            return {"generated": 0, "failed": 0, "error": msg}
        LOG.info("生图后端：%s", msg)

    max_rounds = int(cfg.get("qc.max_retry", 2)) + 1
    balanced = str(cfg.get("image.selection_strategy", "balanced")).lower() == "balanced"
    candidates = state.items_needing_image(max_attempt=max_rounds, balanced=balanced)

    # 关键：重试不吃 limit 预算，否则 --limit N 会让闭环永远转不起来
    fresh = [t for t in candidates if t[2] <= 1]
    retries = [t for t in candidates if t[2] > 1]
    glyph_cap = float(cfg.get("image.max_glyph_share", 0.30))
    if mode == "retry":
        todo = retries
    elif mode == "all":
        picked = _pick_with_quota(fresh + retries, limit, glyph_cap) if limit > 0 \
            else (fresh + retries)
        todo = picked
    else:                                   # "new"：新条目按配额挑，重试全放行
        todo = _pick_with_quota(fresh, limit, glyph_cap) + retries
    if not todo:
        LOG.info("generate：没有待出图的词条")
        return {"generated": 0, "failed": 0, "todo": 0}

    LOG.info("generate：待出图 %d 条（最多 %d 轮尝试）", len(todo), max_rounds)
    concurrency = max(1, int(cfg.get("image.concurrency", 1)))
    ok_n = fail_n = cached_n = 0

    if concurrency == 1:
        results = []
        for item, plan, attempt in todo:
            results.append(_gen_one(cfg, state, provider, item, plan, attempt, force))
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = {pool.submit(_gen_one, cfg, state, provider, item, plan, attempt, force): item
                    for item, plan, attempt in todo}
            results = []
            for fut in as_completed(futs):
                try:
                    results.append(fut.result())
                except Exception as exc:  # noqa: BLE001
                    LOG.error("任务异常 %s: %s", futs[fut].text, exc)
                    results.append((False, str(exc)))

    for ok, tag in results:
        if not ok:
            fail_n += 1
        elif tag == "cached":
            cached_n += 1
        else:
            ok_n += 1

    LOG.info("generate 完成：新出图 %d，命中缓存 %d，失败 %d", ok_n, cached_n, fail_n)
    return {"generated": ok_n, "cached": cached_n, "failed": fail_n,
            "todo": len(todo), "new_images": ok_n}


def pending_retries(cfg: Config, state: State) -> int:
    """还有多少条目等着重出图（闭环是否该继续转）。"""
    max_rounds = int(cfg.get("qc.max_retry", 2)) + 1
    return sum(1 for t in state.items_needing_image(max_attempt=max_rounds)
               if t[2] > 1)


def export_prompts(cfg: Config, state: State, limit: int = 0) -> Path:
    """把提示词导出成 CSV，便于丢给别的出图工具或人工筛选。"""
    import csv
    todo = state.items_needing_image(max_attempt=1, limit=limit)
    out = cfg.output_dir / "prompts.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "答案", "类型", "拼音", "构图范式", "难度", "元素",
                     "positive_en", "negative", "layout_zh"])
        for item, plan, _ in todo:
            wr.writerow([item.id, item.text, item.type, item.pinyin, plan.composition,
                         plan.difficulty,
                         " | ".join(f"{e.morpheme}={e.en}" for e in plan.elements),
                         plan.prompt_en, plan.negative_prompt, plan.layout])
    LOG.info("提示词已导出：%s（%d 条）", out, len(todo))
    return out
