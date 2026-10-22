"""package 阶段：把通过质检的图片打包成数据集。

产出：
    output/dataset.jsonl        全量记录（一行一条）
    output/train|val|test.jsonl 划分后的子集
    output/dataset.json         全量记录（数组形式，便于直接喂给别的脚本）
    output/manifest.csv         便于 Excel 浏览的清单
    output/stats.md             数据统计报告
    output/review.html          人工复核页（含图，可导出决策）
    output/contact_sheet_*.png  联系表（按构图范式分组预览）
    训练格式（见 pipeline/export.py）：sft_chat / multiple_choice / llava / classification
    output/data_card.md         数据卡
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import export
from .config import Config
from .schema import COMPOSITION_ZH, TECHNIQUE_ZH, Record
from .state import State
from .utils import LOG, pinyin_tone_marks, write_json, write_jsonl


# --------------------------------------------------------------------------- #
# 干扰项（多选题选项）
# --------------------------------------------------------------------------- #
def build_distractors(answer: str, composition: str, itype: str, pool: list[dict],
                      n: int = 3) -> list[str]:
    """从同池词条里挑"最像但不对"的干扰项。

    优先级：同长度 + 有共同字 > 同构图范式 > 同词条类型 > 随机。
    """
    cands: list[tuple[float, str]] = []
    a_set = set(answer)
    a_len = len(answer)
    for rec in pool:
        t = rec["answer"]
        if t == answer or abs(len(t) - a_len) > 1:
            continue
        share = len(a_set & set(t))
        score = (share * 2.0
                 + (1.0 if len(t) == a_len else 0.0)
                 + (0.3 if rec["composition"] == composition else 0.0)
                 + (0.2 if rec["type"] == itype else 0.0))
        cands.append((score, t))
    cands.sort(key=lambda x: (-x[0], x[1]))
    out: list[str] = []
    seen: set[str] = set()

    def take(pred) -> None:
        for rec in pool:
            t = rec["answer"]
            if t in seen or t == answer or not pred(t):
                continue
            seen.add(t)
            out.append(t)
            if len(out) >= n:
                return

    take(lambda t: True)                       # 优先：上面算出的相似候选
    if len(out) < n:
        take(lambda t: len(t) == a_len)        # 其次：同字数
    if len(out) < n:
        take(lambda t: abs(len(t) - a_len) <= 1)   # 再次：字数相近
    if len(out) < n:
        take(lambda t: True)                   # 最后：词池里任意其他词
    return out


# --------------------------------------------------------------------------- #
def collect(cfg: Config, state: State) -> list[dict[str, Any]]:
    only_accepted = bool(cfg.get("package.only_accepted", True))
    min_score = float(cfg.get("qc.min_accept_score", 0.75)) if only_accepted else 0.0
    pairs = state.accepted_pairs(min_score=min_score, only_accepted=only_accepted)
    records: list[dict[str, Any]] = []
    images_dir = cfg.images_dir
    for p in pairs:
        item = state.get_item(p["item_id"])
        plan = state.get_plan(p["item_id"])
        if item is None or plan is None:
            continue
        img = Path(p["path"])
        try:
            rel = img.relative_to(cfg.output_dir).as_posix()
        except ValueError:
            rel = img.as_posix()
        qc_d = json.loads(p["result_json"]) if p.get("result_json") else {}
        rec = Record(
            id=item.id,
            answer=item.text,
            type=item.type,
            pinyin=pinyin_tone_marks(item.pinyin) or item.pinyin,
            meaning=item.meaning,
            composition=plan.composition,
            difficulty=plan.difficulty,
            image=rel,
            image_sha1=p.get("sha1", ""),
            prompt_en=plan.prompt_en,
            prompt_zh=plan.prompt_zh,
            negative_prompt=plan.negative_prompt,
            elements=[e.to_dict() for e in plan.elements],
            must_have=list(plan.must_have),
            distractors=list(plan.distractors or []),
            qc={
                "verdict": qc_d.get("verdict"),
                "score": qc_d.get("score"),
                "text_free": qc_d.get("text_free"),
                "element_coverage": qc_d.get("element_coverage"),
                "solvable": qc_d.get("solvable"),
                "guessed_answer": qc_d.get("guessed_answer"),
                "checker": qc_d.get("checker"),
                "issues": qc_d.get("issues", []),
            },
            planner=plan.planner,
            glyph=plan.glyph,
            technique=plan.technique,
            panels=plan.panels,
            source=item.source,
            degraded=bool(getattr(plan, "degraded", False)),
            degraded_reason=str(getattr(plan, "degraded_reason", "") or ""),
        )
        d = rec.to_dict()
        d["category"] = item.category
        d["composition_zh"] = COMPOSITION_ZH.get(plan.composition, plan.composition)
        d["technique_zh"] = TECHNIQUE_ZH.get(plan.technique, plan.technique)
        d["layout"] = plan.layout
        d["rationale"] = plan.rationale
        d["flags"] = plan.extra.get("flags", [])
        if d["degraded"] and "degraded:并列铺开" not in d["flags"]:
            d["flags"] = list(d["flags"]) + ["degraded:并列铺开"]
        records.append(d)

    # 干扰项在这里补齐：collect() 的所有调用方（复核页、runner、外部脚本）
    # 都应该拿到"可直接用"的完整记录，而不是半成品。
    pool = [{"answer": r["answer"], "composition": r["composition"], "type": r["type"]}
            for r in records]
    for r in records:
        if not r.get("distractors"):
            r["distractors"] = build_distractors(r["answer"], r["composition"],
                                                 r["type"], pool, 3)
    return records


def assign_splits(records: list[dict[str, Any]], splits: dict[str, float],
                  seed: int) -> None:
    rnd = random.Random(seed)
    # 按答案去重后打乱，保证同一答案不会跨集合
    rnd.shuffle(records)
    n = len(records)
    names = [k for k, v in splits.items() if v > 0]
    total_w = sum(splits[k] for k in names) or 1.0
    idx = 0
    for i, name in enumerate(names):
        if i == len(names) - 1:
            count = n - idx
        else:
            count = int(round(n * splits[name] / total_w))
        for rec in records[idx:idx + count]:
            rec["split"] = name
        idx += count


def run(cfg: Config, state: State) -> dict[str, Any]:
    records = collect(cfg, state)
    if not records:
        LOG.warning("package：没有通过质检的记录，先跑 generate + qc")
        return {"records": 0}

    assign_splits(records, cfg.get("package.splits", {}), int(cfg.get("package.seed", 42)))

    out_dir = cfg.output_dir
    write_jsonl(out_dir / "dataset.jsonl", records)
    write_json(out_dir / "dataset.json", records)
    for name in (cfg.get("package.splits") or {}):
        sub = [r for r in records if r.get("split") == name]
        if sub:
            write_jsonl(out_dir / f"{name}.jsonl", sub)

    # manifest.csv
    with (out_dir / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "答案", "拼音", "类型", "分类", "构图范式", "难度", "split",
                     "降级", "降级原因", "质检分", "可解", "文字泄漏", "元素覆盖率",
                     "图片", "提示词"])
        for r in records:
            wr.writerow([r["id"], r["answer"], r["pinyin"], r["type"],
                         r.get("category", ""), r["composition_zh"], r["difficulty"],
                         r.get("split", ""),
                         "是" if r.get("degraded") else "",
                         r.get("degraded_reason", ""),
                         r["qc"].get("score"),
                         r["qc"].get("solvable"), r["qc"].get("text_free"),
                         r["qc"].get("element_coverage"), r["image"], r["prompt_en"]])

    stats = write_stats(cfg, records, out_dir)
    exported = export.export_training_formats(cfg, records)
    LOG.info("package 完成：%d 条记录 -> %s", len(records), out_dir)
    return {"records": len(records), "stats": stats, "exported": exported}


def write_stats(cfg: Config, records: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    by_comp = Counter(r["composition_zh"] for r in records)
    by_split = Counter(r.get("split", "") for r in records)
    by_type = Counter(r["type"] for r in records)
    by_diff = Counter(r["difficulty"] for r in records)
    by_cat = Counter(r.get("category") or "成语表" for r in records)
    by_src = Counter(r.get("source", "") for r in records)
    solvable = sum(1 for r in records if r["qc"].get("solvable") is True)
    unsolvable = sum(1 for r in records if r["qc"].get("solvable") is False)
    text_leak = sum(1 for r in records if r["qc"].get("text_free") is False)
    degraded = [r for r in records if r.get("degraded")]
    degraded_words = sum(1 for r in degraded if r.get("type") == "word")
    cov = [r["qc"]["element_coverage"] for r in records
           if isinstance(r["qc"].get("element_coverage"), (int, float))]
    avg_score = sum(float(r["qc"].get("score") or 0) for r in records) / max(1, len(records))
    total_bytes = 0
    for r in records:
        p = cfg.output_dir / r["image"]
        if p.exists():
            total_bytes += p.stat().st_size

    lines = [
        "# 会意猜谜数据集统计",
        "",
        f"- 记录总数：**{len(records)}**",
        f"- 平均质检分：**{avg_score:.3f}**",
        f"- 图像总体积：**{total_bytes / 2 ** 20:.1f} MB**",
        f"- 机器判定可解：**{solvable}** / 判定不可解：**{unsolvable}**"
        f"（未做 VLM 质检：{len(records) - solvable - unsolvable}）",
        f"- 检出文字泄漏：**{text_leak}**",
        f"- 降级样本（元素仅并排铺开、未做语义合成）：**{len(degraded)}**"
        f"（占 {len(degraded) / max(1, len(records)) * 100:.1f}%，"
        f"其中词语类 {degraded_words} 条）",
        f"- 平均元素覆盖率：**{(sum(cov) / len(cov)):.3f}**" if cov else "- 元素覆盖率：未质检",
        "",
        "## 按构图范式",
        "",
        "| 范式 | 数量 |",
        "| --- | --- |",
    ]
    for k, v in by_comp.most_common():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## 按难度", "", "| 难度 | 数量 |", "| --- | --- |"]
    for k in sorted(by_diff):
        lines.append(f"| {k} | {by_diff[k]} |")
    lines += ["", "## 按划分", "", "| split | 数量 |", "| --- | --- |"]
    for k, v in by_split.most_common():
        lines.append(f"| {k or '-'} | {v} |")
    lines += ["", "## 按词条类型", "", "| 类型 | 数量 |", "| --- | --- |"]
    for k, v in by_type.most_common():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## 按原始分类（前 25）", "", "| 分类 | 数量 |", "| --- | --- |"]
    for k, v in by_cat.most_common(25):
        lines.append(f"| {k} | {v} |")
    lines += ["", "## 数据来源", "", "| 来源文件 | 数量 |", "| --- | --- |"]
    for k, v in by_src.most_common():
        lines.append(f"| {k} | {v} |")

    if degraded:
        lines += ["", "## 降级样本清单（元素仅并排铺开，未做语义合成）", "",
                  "这类样本的谜面是「把元素分开画」而不是「把元素合成一个画面」，",
                  "语义关联较弱。训练时可按需过滤或降权。", "",
                  "| 答案 | 类型 | 构图 | 降级原因 |", "| --- | --- | --- | --- |"]
        for r in degraded[:60]:
            lines.append(f"| {r['answer']} | {r.get('type', '')} | "
                         f"{r.get('composition_zh', '')} | {r.get('degraded_reason', '')} |")
        if len(degraded) > 60:
            lines.append(f"| … | | | 共 {len(degraded)} 条，其余见 manifest.csv 的「降级」列 |")
    lines += ["", "## 字段说明", "", "```",
              "id           唯一标识（稳定哈希，可跨轮次对齐）",
              "answer       谜底（成语/词语）",
              "pinyin       带声调符号的拼音",
              "type         idiom=成语 word=词语",
              "composition  元素组合范式：literal/juxtapose/spatial/quantity/contrast/action/metaphor",
              "difficulty   1(易)-5(难)，由元素数、多义字、否定字、词长推算",
              "image        相对 output/ 的图片路径",
              "elements     构图元素表 [{morpheme,zh,en,count,role,position,scale}]",
              "must_have    质检/标注时可核对的关键元素英文短语",
              "distractors  3 个干扰答案（同长度、有共同字优先）",
              "qc           质检明细：verdict/score/text_free/element_coverage/solvable",
              "```", ""]
    (out_dir / "stats.md").write_text("\n".join(lines), encoding="utf-8")
    return {"by_composition": dict(by_comp), "by_split": dict(by_split),
            "by_difficulty": {str(k): v for k, v in by_diff.items()},
            "avg_score": round(avg_score, 4), "total_mb": round(total_bytes / 2 ** 20, 2)}
