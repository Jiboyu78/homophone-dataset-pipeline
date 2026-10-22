"""编排层：把各阶段串起来，处理 generate→qc→重试 的闭环。"""
from __future__ import annotations

from typing import Any, Iterable

from . import (generate, ingest, package, plan_llm, plan_rule, qc, review,
               select, tasks)
from .config import Config
from .providers.image_base import build_provider
from .providers.ocr import OCRChecker
from .providers.openai_compat import build_llm_client, build_vlm_client
from .state import State
from .utils import LOG

# tasks 紧跟 plan：导出"待生图清单"，用于"只出描述、图在别处生成"的省钱路线
STAGES = ["ingest", "select", "plan", "tasks", "generate", "qc", "package"]


def run_stage(name: str, cfg: Config, state: State, limit: int = 0,
              force: bool = False) -> dict[str, Any]:
    if name == "ingest":
        return {"ingest": ingest.run(cfg, state, limit=limit)}
    if name == "select":
        return {"select": select.run(cfg, state, limit=limit, rescore=force)}
    if name == "plan":
        provider = str(cfg.get("plan.provider", "rule")).lower()
        if provider in ("llm", "hybrid"):
            return {"plan": plan_llm.run(cfg, state, limit=limit, redo=force)}
        return {"plan": plan_rule.run(cfg, state, limit=limit, redo=force)}
    if name == "generate":
        return {"generate": generate.run(cfg, state, limit=limit, force=force)}
    if name == "tasks":
        return {"tasks": tasks.export_image_tasks(cfg, state, limit=limit)}
    if name == "qc":
        return {"qc": qc.run(cfg, state, limit=limit)}
    if name == "package":
        res = package.run(cfg, state)
        records = package.collect(cfg, state)
        if records:
            review.write_review(cfg, records)
            if bool(cfg.get("package.contact_sheets", True)):
                review.write_contact_sheets(cfg, records)
        return {"package": res}
    raise ValueError(f"未知阶段：{name}")


def run(cfg: Config, state: State, stages: Iterable[str] | None = None,
        limit: int = 0, limit_words: int = 0, force: bool = False,
        rounds: int = 0) -> dict[str, Any]:
    """按阶段顺序执行。

    limit        出图-质检闭环的条目预算（花钱/花显卡的那部分）
    limit_words  ingest/select/plan 的处理上限（0 = 全部）
    """
    names = [n for n in (stages or STAGES)]
    for n in names:
        if n not in STAGES:
            raise ValueError(f"未知阶段：{n}（可选：{', '.join(STAGES)}）")
    result: dict[str, Any] = {}
    if not rounds:
        rounds = int(cfg.get("qc.max_retry", 2)) + 1

    # 1) 非闭环阶段
    for name in names:
        if name in ("generate", "qc", "package"):
            continue
        result.update(run_stage(name, cfg, state, limit=limit_words, force=force))

    # 2) 出图-质检闭环
    #    --limit 是"首次出图"的条目预算；重试是同一批条目，不吃预算，
    #    否则第 1 轮就把额度花光，闭环永远走不起来。
    if "generate" in names or "qc" in names:
        budget = limit if limit > 0 else 0
        for r in range(1, rounds + 1):
            LOG.info("=========== 出图-质检 第 %d/%d 轮 ===========", r, rounds)
            if "generate" in names:
                gen = generate.run(cfg, state, limit=budget if r == 1 else 0,
                                   mode="new" if r == 1 else "retry",
                                   force=(force and r == 1))
                result[f"generate_round_{r}"] = gen
            if "qc" in names:
                result[f"qc_round_{r}"] = qc.run(cfg, state, limit=0)
            if r < rounds and "generate" in names:
                pending = generate.pending_retries(cfg, state)
                if not pending:
                    LOG.info("没有需要重试的条目，提前结束闭环")
                    break
                LOG.info("待重试 %d 条，进入下一轮", pending)

    # 3) 打包
    if "package" in names:
        result.update(run_stage("package", cfg, state, limit=limit_words, force=force))

    result["counts"] = state.counts()
    return result


# --------------------------------------------------------------------------- #
def health(cfg: Config, state: State | None = None) -> dict[str, Any]:
    """自检：数据、生图后端、LLM、VLM、OCR、Pillow 是否就绪。"""
    report: dict[str, Any] = {}

    files = ingest.discover_files(cfg)
    report["数据文件"] = [f"{f.name} ({f.stat().st_size / 1024:.0f} KB)" for f in files] or "未找到"

    try:
        import PIL  # noqa: F401
        report["Pillow"] = "OK"
    except Exception as exc:  # noqa: BLE001
        report["Pillow"] = f"缺失（{exc}）—— 联系表与质检会降级"

    # 外部出图模式（只出描述、图在别处生成）下没有生图后端可查，
    # 直接说清楚，而不是抛异常或报一个吓人的"不可用"。
    if str(cfg.get("image.provider", "")).lower() in ("external", "none", "off"):
        report["生图后端"] = ("外部出图模式：本次不出图。"
                            "用 `pipeline.cli tasks` 导出描述清单，"
                            "生成后用 `import-images` 导回。")
    else:
        prov = build_provider(cfg)
        ok, msg = prov.health()
        report[f"生图后端({prov.name})"] = ("OK " if ok else "不可用 ") + msg
        if prov.name == "comfyui" and hasattr(prov, "list_checkpoints"):
            ckpts = prov.list_checkpoints()  # type: ignore[attr-defined]
            report["可用 checkpoint"] = ckpts[:10] or "读取失败或为空"

    if str(cfg.get("plan.provider", "rule")).lower() == "llm":
        ok, msg = build_llm_client(cfg).health()
        report["LLM 出题器"] = ("OK " if ok else "不可用 ") + msg
    else:
        report["LLM 出题器"] = "未启用（plan.provider=rule，使用内置规则出题器）"

    if str(cfg.get("qc.vlm.provider", "none")).lower() == "openai":
        ok, msg = build_vlm_client(cfg).health()
        report["VLM 质检"] = ("OK " if ok else "不可用 ") + msg
    else:
        report["VLM 质检"] = "未启用（qc.vlm.provider=none，只做技术+OCR质检）"

    ocr = OCRChecker(cfg)
    ocr._load()  # noqa: SLF001
    report["OCR"] = ("OK " if ocr.available else "不可用 ") + ocr.note

    if state is not None:
        report["状态库统计"] = state.counts()
    return report
