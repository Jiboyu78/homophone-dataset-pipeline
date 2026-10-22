"""把打包结果导出成"可直接用于训练"的多种格式。

一次导出五份，训练哪一类就取哪一份：

  raw                完整字段（含提示词、元素表、质检明细）—— 自己写 dataloader 用
  sft_chat           多模态对话 SFT（OpenAI/通用 chat 格式，带图）
  multiple_choice    选择题（选项已打乱，给正确项下标）
  llava              LLaVA 风格 conversations
  classification      图像分类（图片 → 答案 id），附 labels.json

另外写一份 data_card.md：来源、字段、数量、质检统计、各格式样例、许可提示。
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .config import Config
from .utils import LOG, write_jsonl


def _shuffle_options(answer: str, distractors: list[str], seed: int) -> tuple[list[str], int]:
    opts = [answer] + [d for d in distractors if d and d != answer]
    rnd = random.Random(seed)
    rnd.shuffle(opts)
    return opts, opts.index(answer)


def _question_for(cfg: Config, rec: dict[str, Any]) -> str:
    """按词条类型选问法：成语表来的问"哪个成语"，词语表来的问"哪个词语"。

    一开始统一写"哪个成语"，结果答案是「果酱」「墨鱼」这种词时问题就说不通了。
    """
    if str(rec.get("type") or "idiom") == "word":
        q = cfg.get("package.question_word")
        if q:
            return str(q)
    q = cfg.get("package.question_idiom") or cfg.get("package.question")
    return str(q or "这张图表达的是哪个成语？")


def export_training_formats(cfg: Config, records: list[dict[str, Any]]) -> dict[str, Any]:
    formats = [str(x) for x in (cfg.get("package.export_formats") or ["raw"])]
    out = cfg.output_dir
    base_seed = int(cfg.get("package.seed", 42))
    # 降级样本（元素只是并排铺开、没做语义合成）可以整体排除：
    # 设 package.exclude_degraded=true。默认保留，但每条都带 degraded 标记，
    # 训练侧可以自行过滤或降权。
    if bool(cfg.get("package.exclude_degraded", False)):
        kept = [r for r in records if not r.get("degraded")]
        if len(kept) != len(records):
            LOG.info("按 package.exclude_degraded 排除 %d 条降级样本",
                     len(records) - len(kept))
        records = kept
    splits = sorted({r.get("split") or "train" for r in records})
    written: dict[str, int] = {}

    for split in splits:
        rows = [r for r in records if (r.get("split") or "train") == split]
        if not rows:
            continue
        if "sft_chat" in formats:
            data = []
            for r in rows:
                data.append({
                    "id": r["id"],
                    "image": r["image"],
                    "degraded": bool(r.get("degraded")),
                    "messages": [
                        {"role": "user", "content": [
                            {"type": "image", "image": r["image"]},
                            {"type": "text", "text": _question_for(cfg, r)},
                        ]},
                        {"role": "assistant", "content": r["answer"]},
                    ],
                })
            written[f"sft_chat_{split}.jsonl"] = write_jsonl(
                out / f"sft_chat_{split}.jsonl", data)
        if "multiple_choice" in formats:
            data = []
            for i, r in enumerate(rows):
                opts, idx = _shuffle_options(r["answer"], r.get("distractors") or [],
                                             base_seed + i)
                if len(opts) < 2:
                    continue
                data.append({
                    "id": r["id"],
                    "image": r["image"],
                    "question": _question_for(cfg, r),
                    "options": opts,
                    "answer_index": idx,
                    "answer": r["answer"],
                    "degraded": bool(r.get("degraded")),
                })
            written[f"multiple_choice_{split}.jsonl"] = write_jsonl(
                out / f"multiple_choice_{split}.jsonl", data)
        if "llava" in formats:
            data = []
            for r in rows:
                data.append({
                    "id": r["id"],
                    "image": r["image"],
                    "degraded": bool(r.get("degraded")),
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{_question_for(cfg, r)}"},
                        {"from": "gpt", "value": r["answer"]},
                    ],
                })
            written[f"llava_{split}.jsonl"] = write_jsonl(
                out / f"llava_{split}.jsonl", data)
        if "classification" in formats:
            labels = sorted({r["answer"] for r in rows})
            label_id = {a: i for i, a in enumerate(labels)}
            data = [{"image": r["image"], "label": r["answer"],
                     "label_id": label_id[r["answer"]],
                     "degraded": bool(r.get("degraded"))} for r in rows]
            written[f"classification_{split}.jsonl"] = write_jsonl(
                out / f"classification_{split}.jsonl", data)

    if "classification" in formats:
        labels = sorted({r["answer"] for r in records})
        (out / "labels.json").write_text(
            json.dumps({a: i for i, a in enumerate(labels)}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        written["labels.json"] = len(labels)

    write_data_card(cfg, records, written)
    for name, n in sorted(written.items()):
        LOG.info("导出 %-32s %d 条", name, n)
    return written


def write_data_card(cfg: Config, records: list[dict[str, Any]],
                    written: dict[str, int]) -> Path:
    by_comp = Counter(r.get("composition_zh") or r.get("composition") for r in records)
    by_tech = Counter(r.get("technique_zh") or r.get("technique") for r in records)
    by_split = Counter(r.get("split") or "" for r in records)
    by_cat = Counter(r.get("category") or "成语表" for r in records)
    by_src = Counter(r.get("source", "") for r in records)
    scores = [float(r["qc"].get("score") or 0) for r in records]
    solvable = [r for r in records if r["qc"].get("solvable") is not None]
    text_checked = [r for r in records if r["qc"].get("text_free") is not None]
    glyphs = [r for r in records if r.get("glyph")]
    degraded = [r for r in records if r.get("degraded")]

    sample_raw = json.dumps(records[0], ensure_ascii=False, indent=1)[:1800] if records else "{}"
    sample_mc = ""
    mc = cfg.output_dir / "multiple_choice_train.jsonl"
    if mc.exists():
        sample_mc = mc.read_text(encoding="utf-8").splitlines()[0][:900]
    sample_sft = ""
    sft = cfg.output_dir / "sft_chat_train.jsonl"
    if sft.exists():
        sample_sft = sft.read_text(encoding="utf-8").splitlines()[0][:900]

    lines = [
        f"# 数据卡：{cfg.get('package.dataset_name')} v{cfg.get('package.version')}",
        "",
        "## 一句话说明",
        "",
        "「会意式单图猜成语/词语」数据集：每张图由若干具体元素按特定排布组合而成，",
        "用来表达一个成语或常用词语。可用于多模态 SFT、看图猜谜选择题、图像分类。",
        "",
        "## 规模",
        "",
        f"- 记录数：**{len(records)}**",
        f"- 图片目录：`images/`（PNG）",
        f"- 平均质检分：**{(sum(scores) / len(scores)) if scores else 0:.3f}**",
        f"- 字形参与式（画面含指定汉字）：**{len(glyphs)}** 条",
        f"- 做过可解性盲测：**{len(solvable)}** 条"
        f"（判定可解 {sum(1 for r in solvable if r['qc'].get('solvable'))}）",
        f"- 做过文字核验：**{len(text_checked)}** 条"
        f"（无文字 {sum(1 for r in text_checked if r['qc'].get('text_free'))}）",
        f"- **降级样本：{len(degraded)} 条**"
        f"（占 {len(degraded) / max(1, len(records)) * 100:.1f}%）"
        f"——元素仅并排铺开，未做语义合成，详见下文专节",
        "",
        "## 分布",
        "",
        "| 构图范式 | 数量 |",
        "| --- | --- |",
    ]
    lines += [f"| {k} | {v} |" for k, v in by_comp.most_common()]
    lines += ["", "| 出题手法 | 数量 |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in by_tech.most_common()]
    lines += ["", "| 划分 | 数量 |", "| --- | --- |"]
    lines += [f"| {k or '-'} | {v} |" for k, v in by_split.most_common()]
    lines += ["", "| 原始分类（前 15） | 数量 |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in by_cat.most_common(15)]
    lines += ["", "| 来源文件 | 数量 |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in by_src.most_common()]

    lines += [
        "",
        "## 降级样本说明（重要）",
        "",
        "非成语类词语常常没法把语素合成一个合理画面。例如「刀鱼」是一种鱼，",
        "无法让「刀」和「鱼」合成一个有意义的物体，也无法让它们发生自然互动。",
        "这类词条允许**降级处理**：不合成，直接把元素并排铺开——左边一把刀、右边一条鱼。",
        "",
        f"本数据集包含 **{len(degraded)} 条**降级样本。它们的特征：",
        "",
        "- `degraded = true`，`degraded_reason` 写明为什么合成不了",
        "- `composition = juxtapose`，提示词里是并列关系（side by side）",
        "- 语义关联弱于正常样本，**训练时建议过滤或降权**",
        "",
        "过滤方法（任选）：",
        "",
        "1. 导出时就整体剔除：在配置里设 `\"package\": {\"exclude_degraded\": true}`",
        "2. 训练侧过滤：读各格式里的 `degraded` 字段（四种训练格式都带这个字段）",
        "3. 看清单人工筛：`stats.md` 的「降级样本清单」节，或 `manifest.csv` 的「降级」列",
        "",
    ]
    if degraded:
        lines += ["| 答案 | 类型 | 构图 | 降级原因 |", "| --- | --- | --- | --- |"]
        for r in degraded[:40]:
            lines.append(f"| {r['answer']} | {r.get('type', '')} | "
                         f"{r.get('composition_zh', r.get('composition', ''))} | "
                         f"{r.get('degraded_reason', '')} |")
        if len(degraded) > 40:
            lines.append(f"| … | | | 共 {len(degraded)} 条，完整清单见 stats.md 与 manifest.csv |")
        lines.append("")

    lines += [
        "",
        "## 导出的文件",
        "",
        "| 文件 | 说明 | 条数 |",
        "| --- | --- | --- |",
        "| `dataset.jsonl` | 全量原始记录（字段最全） | %d |" % len(records),
        "| `train|val|test.jsonl` | 按划分拆开的原始记录 | - |",
        "| `sft_chat_*.jsonl` | 多模态对话 SFT（user 带图 + assistant 答答案） | %s |"
        % written.get("sft_chat_train.jsonl", "-"),
        "| `multiple_choice_*.jsonl` | 选择题（选项已打乱，含 answer_index） | %s |"
        % written.get("multiple_choice_train.jsonl", "-"),
        "| `llava_*.jsonl` | LLaVA conversations 格式 | %s |"
        % written.get("llava_train.jsonl", "-"),
        "| `classification_*.jsonl` + `labels.json` | 图像分类 | %s |"
        % written.get("classification_train.jsonl", "-"),
        "| `manifest.csv` | 便于 Excel 浏览的清单 | - |",
        "| `review.html` | 人工复核页（答案默认隐藏，可盲测） | - |",
        "",
        "图片路径都相对于本目录，例如 `images/idi_xxxxxxxx_a1.png`。",
        "",
        "## 字段说明（dataset.jsonl）",
        "",
        "```",
        "id            稳定哈希 ID，跨批次一致，可用于对齐标注",
        "answer        谜底（成语或词语）",
        "pinyin        带声调符号的拼音（已做规范化和错误校正）",
        "type          idiom=成语  word=词语",
        "category      原始词表里的分类（食物/动物/生活用品…）",
        "composition   构图范式：literal/juxtapose/spatial/quantity/contrast/action/inner/metaphor",
        "technique     出题手法：ideographic 会意图解 / glyph 字形参与",
        "glyph         字形参与式要写进画面的那个汉字，其余为空",
        "difficulty    1(易)-5(难)",
        "image         相对路径",
        "elements      元素表 [{morpheme,zh,en,count,role,position}]",
        "must_have     质检时可核对的关键元素英文短语",
        "distractors   3 个干扰答案（同长度、有共同字优先）",
        "prompt_en     生成该图用的正向提示词（可复现）",
        "negative_prompt 负向提示词",
        "layout        中文构图说明",
        "rationale     为什么这样构图能被猜出来",
        "qc            质检明细：verdict/score/text_free/element_coverage/solvable/checker",
        "degraded      true = 降级样本（元素仅并排铺开，未做语义合成），训练时可过滤",
        "degraded_reason 降级原因说明",
        "split         train/val/test",
        "flags         风险标记：negation:不、polysemy_risk、literal_compound、glyph:劳",
        "```",
        "",
        "## 样例",
        "",
        "`dataset.jsonl`：",
        "",
        "```json",
        sample_raw,
        "```",
    ]
    if sample_sft:
        lines += ["", "`sft_chat_train.jsonl`：", "", "```json", sample_sft, "```"]
    if sample_mc:
        lines += ["", "`multiple_choice_train.jsonl`：", "", "```json", sample_mc, "```"]
    lines += [
        "",
        "## 生成方式（可复现）",
        "",
        f"- 答案词来自：{'、'.join(by_src) or '用户词表'}",
        "- 谜面由大模型按 `skills/zh-rebus-design` 的规则设计，并经机械校验",
        "- 图片由文生图模型生成；字形参与式的汉字由字体确定性叠加（不依赖模型写对汉字）",
        "- 质检：技术质检（分辨率/空白/近似重复）+ OCR 文字泄漏检测"
        + ("+ VLM 语义与可解性核验" if solvable else "（未启用 VLM，语义未机器验证）"),
        "",
        "## 使用限制与许可",
        "",
        "- 生成图像的版权取决于你使用的文生图服务条款，商用前自行核对。",
        "- 原始词表来自你自己的数据文件，注意其来源与授权。",
        "- `qc.solvable=false` 的条目表示视觉模型盲猜没猜出来，训练时可考虑过滤或降权。",
        "",
    ]
    path = cfg.output_dir / "data_card.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    LOG.info("数据卡：%s", path)
    return path
