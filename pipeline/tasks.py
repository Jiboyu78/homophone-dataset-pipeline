"""图片描述（生图任务）的导出与外图导入。

用途：批量生图用 API 太贵时，可以**只出描述不出图**，把提示词清单拿去别处生成
（例如充了会员的桌面端生图工具），再把图片导回流水线继续质检与打包。

    导出任务：pipeline.cli tasks            （或一键脚本 --no-images）
    导入图片：pipeline.cli import-images --dir <你生成图片的目录>

导出会写这几样，覆盖"人读 / 机读 / 逐条喂"三种用法：
    image_tasks.csv        表格：序号、文件名、答案、提示词、负向词、尺寸、字形、降级
    image_tasks.jsonl      机读：每条一个对象，字段最全
    prompts/<文件名>.txt   每条一个纯文本文件，只含提示词本身
    README_image_tasks.md  给协作者/桌面 agent 的说明

文件名是**约定**而不是随便取的：`001_九牛一毛_idi_dc81dc47d2d5.png`。
导入时按文件名里的 item_id 或答案匹配，所以图片怎么改名都不容易丢；
完全匹配不上时会明确列出来，而不是静默跳过。
"""
from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path
from typing import Any

from .config import Config
from .schema import Item
from .state import State
from .utils import LOG, resolve, sha1_file, sha1_text, write_jsonl

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_ID_RE = re.compile(r"((?:idi|wor)_[0-9a-f]{6,})")


def _safe(text: str) -> str:
    """把答案做成安全的文件名片段（去掉路径分隔符与奇怪字符）。"""
    out = re.sub(r'[\\/:*?"<>|\s]+', "", text or "")
    return out[:16] or "item"


def target_name(index: int, item: Item) -> str:
    """约定的任务文件名：序号_答案_稳定id.png（人可读、机器可匹配）。"""
    return f"{index:03d}_{_safe(item.text)}_{item.id}.png"


def export_image_tasks(cfg: Config, state: State, limit: int = 0,
                       only_missing: bool = True) -> dict[str, Any]:
    """导出"待生图"清单。only_missing=True 时跳过已经出过图的条目。"""
    out_dir = cfg.output_dir
    prompt_dir = out_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    have_image: set[str] = set()
    if only_missing:
        have_image = {r["item_id"] for r in state.list_images()}

    items = state.load_items(status=["accepted"], order_by="drawable_score DESC, text")
    rows: list[dict[str, Any]] = []
    index = 0
    for it in items:
        if it.id in have_image:
            continue
        plan = state.get_plan(it.id)
        if plan is None:
            continue
        index += 1
        rows.append({
            "index": index,
            "item_id": it.id,
            "answer": it.text,
            "pinyin": it.pinyin,
            "type": it.type,
            "category": it.category,
            "composition": plan.composition,
            "technique": plan.technique,
            "degraded": bool(getattr(plan, "degraded", False)),
            "degraded_reason": getattr(plan, "degraded_reason", ""),
            "difficulty": plan.difficulty,
            "target_filename": target_name(index, it),
            "prompt": plan.prompt_en,
            "negative_prompt": plan.negative_prompt,
            "size": str(cfg.get("image.api.size") or "1024x1024"),
            "glyph": plan.glyph or "",
            "must_have": plan.must_have,
            "layout": plan.layout,
        })
        if limit and len(rows) >= limit:
            break

    if not rows:
        LOG.warning("没有待生图的条目（可能都已经出过图了，或还没出题）")
        return {"tasks": 0}

    write_jsonl(out_dir / "image_tasks.jsonl", rows)

    # 人读表格（utf-8-sig 让 Excel 正确识别中文）
    csv_path = out_dir / "image_tasks.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["序号", "文件名", "答案", "类型", "构图", "降级", "尺寸",
                     "提示词", "负向词", "需叠加的字", "必须出现的元素"])
        for r in rows:
            wr.writerow([r["index"], r["target_filename"], r["answer"], r["type"],
                         r["composition"], "是" if r["degraded"] else "", r["size"],
                         r["prompt"], r["negative_prompt"], r["glyph"],
                         " / ".join(r["must_have"])])

    # 逐条纯文本（很多生图工具支持"按文件批量读提示词"）
    for r in rows:
        (prompt_dir / f"{Path(r['target_filename']).stem}.txt").write_text(
            r["prompt"], encoding="utf-8")

    _write_readme(out_dir, rows)
    LOG.info("生图任务已导出：%d 条 -> %s", len(rows), csv_path)
    return {"tasks": len(rows), "csv": str(csv_path),
            "jsonl": str(out_dir / "image_tasks.jsonl"),
            "prompts_dir": str(prompt_dir)}


def _write_readme(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    glyph_rows = [r for r in rows if r["glyph"]]
    degraded_rows = [r for r in rows if r["degraded"]]
    lines = [
        "# 生图任务清单",
        "",
        f"共 **{len(rows)}** 条待生图。三份内容一致，用哪份都行：",
        "",
        "| 文件 | 用途 |",
        "| --- | --- |",
        "| `image_tasks.csv` | 表格，肉眼扫或导入 Excel |",
        "| `image_tasks.jsonl` | 机读，字段最全（含 must_have，供后续质检核对） |",
        "| `prompts/*.txt` | 每条一个纯文本文件，只含提示词 |",
        "",
        "## 生图时请注意",
        "",
        "1. **按 `target_filename` 命名输出文件**"
        "（如 `001_九牛一毛_idi_dc81dc47d2d5.png`）。",
        "   文件名里带了稳定 id，导回时能准确对上答案。",
        f"2. 尺寸建议 `{rows[0]['size']}`（按需自定，pipeline 不限制）。",
        "3. 大多数工具没有负向词输入框。本清单的提示词里**已经包含**"
        " `no text, no letters, no watermark` 这类要求，",
        "   所以即使不填负向词也能用。",
    ]
    if glyph_rows:
        lines += [
            "",
            f"4. 其中 **{len(glyph_rows)} 条需要叠加一个汉字**"
            "（字形参与式，见 CSV 的「需叠加的字」列）。",
            "   **出图时请让画面留出空白，不要指望模型写对汉字**——",
            "   把图片导回后，pipeline 会用系统字体把该字确定性地叠加上去。",
        ]
    if degraded_rows:
        lines += [
            "",
            f"5. 其中 **{len(degraded_rows)} 条是降级样本**（元素只是并排铺开、",
            "   没做语义合成），画面上两个物体并列即可，不要强求互动。",
        ]
    lines += [
        "",
        "## 生成完之后：导回 pipeline",
        "",
        "```bash",
        "# 把图片都放进一个目录（文件名尽量保持上面的约定），然后：",
        "python3 -m pipeline.cli import-images --dir <你放图片的目录>",
        "python3 -m pipeline.cli run --stages qc,package",
        "```",
        "",
        "导入时会自动完成：格式统一成 PNG、按文件名匹配答案、给字形参与式的图叠字、",
        "登记进状态库。之后质检与打包跟正常流程完全一样。",
        "",
    ]
    (out_dir / "README_image_tasks.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# 导入外图
# --------------------------------------------------------------------------- #
def _build_index(state: State) -> tuple[dict[str, Item], dict[str, Item]]:
    items = state.load_items()
    by_id = {it.id: it for it in items}
    by_text: dict[str, Item] = {}
    for it in items:
        old = by_text.get(it.text)
        if old is None or len(it.text) >= len(old.text):
            by_text[it.text] = it
    return by_id, by_text


def _match(stem: str, by_id: dict[str, Item], by_text: dict[str, Item]) -> Item | None:
    # 1) 文件名里有稳定 id（推荐做法）
    m = _ID_RE.search(stem)
    if m and m.group(1) in by_id:
        return by_id[m.group(1)]
    # 2) 整个文件名就是 id 或 id_aN
    head = stem.split("_a")[0]
    if head in by_id:
        return by_id[head]
    # 3) 按答案匹配：取能匹配上的最长答案，避免"牛"命中"九牛一毛"
    hits = [t for t in by_text if t and t in stem]
    if hits:
        return by_text[max(hits, key=len)]
    return None


def import_images(cfg: Config, state: State, src_dir: str | Path,
                  recursive: bool = True, apply_glyph: bool = True,
                  copy_mode: str = "copy") -> dict[str, Any]:
    """把外部生成的图片导回流水线。

    匹配 → 规范化成 PNG → （字形参与式则叠字）→ 登记进状态库。
    之后直接跑 qc,package 即可。
    """
    src = Path(src_dir)
    if not src.is_absolute():
        src = resolve(str(src_dir))
    if not src.is_dir():
        raise FileNotFoundError(f"目录不存在：{src}")

    by_id, by_text = _build_index(state)
    files = [p for p in (src.rglob("*") if recursive else src.glob("*"))
             if p.is_file() and p.suffix.lower() in IMAGE_EXT]
    if not files:
        LOG.warning("目录里没有图片文件：%s", src)
        return {"imported": 0, "unmatched": 0, "skipped": 0, "glyph_applied": 0}

    already = {r["id"] for r in state.list_images()}
    imported: list[str] = []
    unmatched: list[str] = []
    skipped: list[str] = []
    glyph_applied = 0
    glyph_failed: list[str] = []

    try:
        from PIL import Image
    except Exception:  # noqa: BLE001
        Image = None  # type: ignore[assignment]

    for p in sorted(files):
        item = _match(p.stem, by_id, by_text)
        if item is None:
            unmatched.append(p.name)
            continue
        dest = cfg.images_dir / f"{item.id}_a1.png"
        image_id = f"{item.id}_a1"
        if image_id in already and dest.exists():
            skipped.append(p.name)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)

        same_file = p.resolve() == dest.resolve()
        if not same_file:
            if copy_mode == "move":
                shutil.move(str(p), str(dest))
            elif Image is not None and p.suffix.lower() != ".png":
                # 统一转成真 PNG：数据集里格式不一致会给后续训练添麻烦
                with Image.open(p) as im:
                    im.convert("RGB").save(dest, "PNG")
            else:
                shutil.copy2(p, dest)
        elif Image is not None and p.suffix.lower() != ".png":
            with Image.open(p) as im:
                im.convert("RGB").save(dest, "PNG")

        plan = state.get_plan(item.id)
        if apply_glyph and plan is not None and plan.glyph:
            from . import generate
            if generate.apply_glyph_overlay(cfg, plan, dest):
                glyph_applied += 1
            else:
                glyph_failed.append(item.text)

        state.add_image(image_id, item.id, str(dest), sha1_file(dest), 1,
                        "external", sha1_text(str(dest)), -1)
        imported.append(item.text)

    LOG.info("导入外图：成功 %d，跳过（已存在）%d，匹配不上 %d，叠字 %d",
             len(imported), len(skipped), len(unmatched), glyph_applied)
    if unmatched:
        LOG.warning("这些文件名匹配不上任何词条（请把稳定 id 或答案放进文件名）：%s",
                    "、".join(unmatched[:12]))
    if glyph_failed:
        LOG.warning("这些条目需要叠字但失败了（检查中文字体）：%s",
                    "、".join(glyph_failed[:8]))

    return {"imported": len(imported), "skipped": len(skipped),
            "unmatched": len(unmatched), "unmatched_files": unmatched[:50],
            "glyph_applied": glyph_applied, "glyph_failed": glyph_failed,
            "answers": imported[:50]}
