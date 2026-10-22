"""命令行入口。

常用：
    python -m pipeline.cli health                       # 自检各后端
    python -m pipeline.cli run --limit 50               # 小批量试跑全流程
    python -m pipeline.cli run --stages select,plan     # 只跑到出题
    python -m pipeline.cli run --mock                   # 强制用 Mock 后端跑通闭环
    python -m pipeline.cli inspect 画蛇添足              # 看某条词的分词/打分/方案
    python -m pipeline.cli stats                        # 看当前统计
    python -m pipeline.cli prompts --limit 200          # 导出提示词 CSV
    python -m pipeline.cli review                       # 重新生成复核页
    python -m pipeline.cli reset --stage generate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import ingest, package, plan_llm, plan_rule, review, runner, select
from .config import load_config
from .lexicon import LEX
from .providers.image_base import build_provider
from .schema import COMPOSITION_ZH
from .state import State
from .utils import LOG, ensure_dir, pinyin_tone_marks, setup_logging


def _mk_state(cfg) -> State:
    return State(cfg.db_path)


def _parse_stages(s: str | None, cfg) -> list[str]:
    if not s:
        return list(runner.STAGES)
    names = [x.strip() for x in s.replace("，", ",").split(",") if x.strip()]
    if "all" in names:
        return list(runner.STAGES)
    return names


def cmd_health(args, cfg) -> int:
    state = _mk_state(cfg)
    rep = runner.health(cfg, state)
    print("=" * 78)
    print("环境自检")
    print("=" * 78)
    for k, v in rep.items():
        if isinstance(v, (list, dict)):
            print(f"{k}:")
            if isinstance(v, dict):
                for kk, vv in v.items():
                    print(f"    {kk}: {vv}")
            else:
                for item in v:
                    print(f"    - {item}")
        else:
            print(f"{k}: {v}")
    print("=" * 78)
    print(f"词典规模：{len(LEX.entries)} 条语素，{len(LEX.stats())} 个语义类")
    state.close()
    return 0


def cmd_run(args, cfg) -> int:
    if args.mock:
        cfg.set("image.provider", "mock")
        LOG.info("已强制使用 mock 生图后端")
    if args.provider:
        cfg.set("image.provider", args.provider)
    if args.plan_provider:
        cfg.set("plan.provider", args.plan_provider)
    if args.llm_base_url:
        cfg.set("plan.llm.base_url", args.llm_base_url)
    if args.llm_model:
        cfg.set("plan.llm.model", args.llm_model)
    if args.vlm:
        cfg.set("qc.vlm.provider", args.vlm)
    # 并发分片：多进程各跑 I/N 之一。必须配合"每路一个独立输出目录/状态库"，
    # 否则 worker0 会把不属于自己的候选标成 rejected，其他路就无词可做了。
    if args.shard:
        try:
            idx_s, cnt_s = str(args.shard).split("/")
            idx, cnt = int(idx_s), int(cnt_s)
            if not (1 <= idx <= cnt) or cnt < 1:
                raise ValueError
        except ValueError:
            print("--shard 的格式应为 I/N，例如 --shard 1/4", file=sys.stderr)
            return 2
        cfg.set("select.shard_index", idx - 1)
        cfg.set("select.shard_count", cnt)
        LOG.info("已启用分片：第 %d/%d 路", idx, cnt)
    stages = _parse_stages(args.stages, cfg)
    state = _mk_state(cfg)
    try:
        res = runner.run(cfg, state, stages=stages, limit=args.limit,
                         limit_words=args.limit_words, force=args.force,
                         rounds=args.rounds)
    except plan_llm.LLMUnavailable as exc:
        state.close()
        print("=" * 78, file=sys.stderr)
        print(f"失败：{exc}", file=sys.stderr)
        print("=" * 78, file=sys.stderr)
        return 3
    finally:
        state.close()
    print("=" * 78)
    print(json.dumps(res, ensure_ascii=False, indent=2, default=str)[:4000])
    print("=" * 78)
    print(f"产出目录：{cfg.output_dir}")
    # 一条图都没产出时不能说"完成"：调用方（一键脚本/CI）要能拿到非零退出码。
    # 注意只在"确实请求了出图阶段"时才判定，否则"只出描述"模式会被误判为失败。
    if "generate" in stages and int((res.get("counts") or {}).get("images") or 0) == 0:
        print("注意：本轮没有产出任何图片，数据集体积为 0。"
              "请看上面的错误信息（通常是出图后端不可用）。", file=sys.stderr)
        return 4
    return 0


def cmd_inspect(args, cfg) -> int:
    state = _mk_state(cfg)
    texts = args.text
    items = state.load_items(order_by="id")
    found = [it for it in items if it.text in texts]
    if not found and len(texts) == 1 and not args.exact:
        found = [it for it in items if texts[0] in it.text][:20]
    if not found:
        print("未找到词条。提示：先跑 `run --stages ingest,select`")
        state.close()
        return 1
    for it in found:
        plan = state.get_plan(it.id)
        print("=" * 78)
        print(f"【{it.text}】 {it.type}  拼音 {pinyin_tone_marks(it.pinyin) or it.pinyin}"
              f"  分类 {it.category or '-'}")
        print(f"  状态 {it.select_status}  可图化分 {it.drawable_score}  "
              f"覆盖率 {it.coverage}  构图 {COMPOSITION_ZH.get(it.composition, it.composition)}")
        print(f"  理由 {it.select_reason}")
        print("  语素切分：")
        for s in it.segments:
            flag = "可画" if s["drawable"] else "  - "
            print(f"    {s['morpheme']:<3} {flag} w={s['weight']:.2f} kind={s['kind']:<10} en={s['en']}")
        if plan:
            print(f"  难度 {plan.difficulty}  出题器 {plan.planner}")
            print(f"  构图说明：{plan.layout}")
            print(f"  正向提示词：{plan.prompt_en}")
            print(f"  负向提示词：{plan.negative_prompt[:120]}...")
            print("  元素表：")
            for e in plan.elements:
                print(f"    {e.morpheme:<3} {e.role:<9} x{e.count} {e.position or '-':<10} {e.en}")
    state.close()
    return 0


def cmd_stats(args, cfg) -> int:
    state = _mk_state(cfg)
    counts = state.counts()
    print("=" * 78)
    print("状态库统计")
    print("=" * 78)
    for k, v in counts.items():
        print(f"  {k:<22} {v}")
    accepted = state.load_items(status=["accepted"], order_by="drawable_score DESC")
    if accepted:
        from collections import Counter
        print(f"\n  已接受条目按构图范式：")
        for k, v in Counter(i.composition for i in accepted).most_common():
            print(f"    {COMPOSITION_ZH.get(k, k):<10} {v}")
    qc_rows = state.all_qc()
    if qc_rows:
        from collections import Counter
        print(f"\n  质检结论分布：")
        for k, v in Counter(r["verdict"] for r in qc_rows).most_common():
            print(f"    {k:<12} {v}")
    state.close()
    return 0


def cmd_prompts(args, cfg) -> int:
    from . import generate
    state = _mk_state(cfg)
    p = generate.export_prompts(cfg, state, limit=args.limit)
    state.close()
    print(f"已导出：{p}")
    return 0


def cmd_review(args, cfg) -> int:
    state = _mk_state(cfg)
    records = package.collect(cfg, state)
    if not records:
        print("没有可复核的记录（先跑 generate + qc）")
        state.close()
        return 1
    p = review.write_review(cfg, records)
    sheets = review.write_contact_sheets(cfg, records)
    state.close()
    print(f"复核页：{p}\n联系表：{len(sheets)} 张")
    return 0


def cmd_reset(args, cfg) -> int:
    state = _mk_state(cfg)
    state.reset_stage(args.stage)
    print(f"已重置阶段 {args.stage}；当前统计：{state.counts()}")
    state.close()
    return 0


def cmd_apply_decisions(args, cfg) -> int:
    """把复核页导出的 decisions.json 落库：通过->保留，驳回->标记 rejected。"""
    path = Path(args.file)
    if not path.exists():
        print(f"文件不存在：{path}")
        return 1
    decisions = json.loads(path.read_text(encoding="utf-8"))
    state = _mk_state(cfg)
    ok = bad = 0
    for item_id, dec in decisions.items():
        item = state.get_item(item_id)
        if item is None:
            continue
        if dec == "ok":
            ok += 1
        else:
            item.select_status = "rejected"
            item.select_reason = "人工复核驳回"
            state.update_select_bulk([item])
            bad += 1
    state.close()
    print(f"已应用人工决策：通过 {ok} 条，驳回 {bad} 条")
    return 0


def cmd_fix_degraded(args, cfg) -> int:
    """给旧方案回填「降级」标记。

    旧版本没有 degraded 字段。已经出过题的词在重跑时会被跳过（不会重新出题），
    所以标记一直是空的。这里按与出题器一致的判据回填：
    词语类 + 并列式 = 元素只是并排铺开，属于降级。
    已在 plan_json 里显式写过 degraded 的方案不动。
    """
    state = _mk_state(cfg)
    try:
        rows = state.conn.execute(
            "SELECT p.item_id, p.plan_json FROM plans p").fetchall()
        items_by_id = {it.id: it for it in state.load_items()}
        changed = already = 0
        for r in rows:
            d = json.loads(r["plan_json"])
            if "degraded" in d:
                already += 1
                continue
            item = items_by_id.get(r["item_id"])
            is_word = (item.type == "word") if item else False
            if is_word and d.get("composition") == "juxtapose":
                d["degraded"] = True
                d["degraded_reason"] = "（回填）词语类且为并列式，未做语义合成"
                flag = 1
                changed += 1
            else:
                d["degraded"] = False
                d["degraded_reason"] = ""
                flag = 0
                changed += 1
            state.conn.execute(
                "UPDATE plans SET plan_json=?, degraded=? WHERE item_id=?",
                (json.dumps(d, ensure_ascii=False), flag, r["item_id"]))
        state.conn.commit()
        n_deg = state.conn.execute(
            "SELECT COUNT(*) FROM plans WHERE degraded=1").fetchone()[0]
        print(f"已回填降级标记：更新 {changed} 条方案（其中标记为降级 {n_deg} 条），"
              f"已是新格式未改动 {already} 条")
        print("提示：重新跑 package 阶段即可让数据集带上该标记："
              "python -m pipeline.cli run --stages package")
        return 0
    finally:
        state.close()


def cmd_tasks(args, cfg) -> int:
    """导出"待生图"清单（只出描述不出图时用）。"""
    from . import tasks
    state = _mk_state(cfg)
    try:
        res = tasks.export_image_tasks(cfg, state, limit=args.limit,
                                       only_missing=not args.all)
        print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
        if res.get("tasks"):
            print(f"\n共 {res['tasks']} 条待生图。把 image_tasks.csv 拿去生图，"
                  f"\n生成完用 import-images 导回："
                  f"\n  python -m pipeline.cli import-images --dir <图片目录>"
                  f"\n  python -m pipeline.cli run --stages qc,package")
        return 0
    finally:
        state.close()


def cmd_import_images(args, cfg) -> int:
    """把外部生成的图片导回流水线，之后跑 qc,package 即可。"""
    from . import tasks
    state = _mk_state(cfg)
    try:
        res = tasks.import_images(cfg, state, args.dir,
                                  recursive=not args.no_recursive,
                                  apply_glyph=not args.no_glyph,
                                  copy_mode="move" if args.move else "copy")
        print(json.dumps(res, ensure_ascii=False, indent=2, default=str)[:3000])
        if res.get("imported"):
            print(f"\n已导入 {res['imported']} 张。接着跑质检与打包："
                  f"\n  python -m pipeline.cli run --stages qc,package")
            return 0
        print("\n没有导入任何图片。请检查文件名是否包含稳定 id 或答案。")
        return 1
    finally:
        state.close()


def cmd_merge(args, cfg) -> int:
    """把多路并发跑出来的结果合并成一份数据集。"""
    from . import merge as merge_mod
    dirs: list[str] = []
    for chunk in (args.dirs or []):
        dirs += [x.strip() for x in str(chunk).split(",") if x.strip()]
    if not dirs:
        print("请用 --dirs 指定要合并的目录，例如 --dirs output/shard1,output/shard2",
              file=sys.stderr)
        return 2
    if args.out:
        cfg.set("paths.output_dir", args.out)
        cfg.set("paths.images_dir", str(Path(args.out) / "images"))
        cfg.set("paths.db", str(Path(args.out) / "pipeline.db"))
        cfg.set("paths.log_file", str(Path(args.out) / "pipeline.log"))
    res = merge_mod.merge_runs(cfg, dirs)
    print(json.dumps(res, ensure_ascii=False, indent=2, default=str)[:3000])
    if res.get("mode") in ("records", "tasks"):
        print(f"\n合并结果在：{cfg.output_dir}")
        return 0
    return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline.cli", description="会意式中文单图猜谜数据集生成 pipeline")
    p.add_argument("--config", default=None, help="配置文件路径（默认 config.json）")
    p.add_argument("-v", "--verbose", action="store_true", help="输出 DEBUG 日志")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("health", help="环境与后端自检")
    sp.set_defaults(func=cmd_health)

    sp = sub.add_parser("run", help="执行流水线")
    sp.add_argument("--stages", default=None,
                    help="逗号分隔：ingest,select,plan,generate,qc,package（默认全部）")
    sp.add_argument("--limit", type=int, default=0,
                    help="出图-质检闭环最多处理多少条（0=不限）")
    sp.add_argument("--limit-words", type=int, default=0,
                    help="ingest/select/plan 最多处理多少条词（0=全部，默认全部）")
    sp.add_argument("--rounds", type=int, default=0, help="出图-质检闭环轮数（默认 qc.max_retry+1）")
    sp.add_argument("--shard", default=None,
                    help="并发分片，格式 I/N（例如 4 路并发时分别传 1/4 2/4 3/4 4/4）。"
                         "每路必须用不同的 --config（不同输出目录与状态库）")
    sp.add_argument("--force", action="store_true", help="忽略缓存重算/重出图")
    sp.add_argument("--mock", action="store_true", help="强制使用 Mock 生图后端")
    sp.add_argument("--provider", default=None, help="临时指定生图后端 comfyui|sdwebui|mock")
    sp.add_argument("--plan-provider", default=None, help="临时指定出题器 rule|llm")
    sp.add_argument("--llm-base-url", default=None,
                    help="OpenAI 兼容接口地址，如 http://127.0.0.1:8080/v1")
    sp.add_argument("--llm-model", default=None, help="模型名，如 Qwen3-8B-Q5_K_M.gguf")
    sp.add_argument("--vlm", default=None, help="临时指定语义质检 openai|none")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("inspect", help="查看某条词的分词、打分与构图方案")
    sp.add_argument("text", nargs="+", help="词条，如 画蛇添足")
    sp.add_argument("--exact", action="store_true", help="只精确匹配")
    sp.set_defaults(func=cmd_inspect)

    sp = sub.add_parser("stats", help="查看统计")
    sp.set_defaults(func=cmd_stats)

    sp = sub.add_parser("prompts", help="导出提示词 CSV（可丢给别的出图工具）")
    sp.add_argument("--limit", type=int, default=0)
    sp.set_defaults(func=cmd_prompts)

    sp = sub.add_parser("review", help="生成/刷新人工复核页与联系表")
    sp.set_defaults(func=cmd_review)

    sp = sub.add_parser("reset", help="重置某个阶段")
    sp.add_argument("--stage", required=True,
                    choices=["ingest", "select", "plan", "generate", "qc", "all"])
    sp.set_defaults(func=cmd_reset)

    sp = sub.add_parser("apply-decisions", help="应用复核页导出的 decisions.json")
    sp.add_argument("--file", required=True)
    sp.set_defaults(func=cmd_apply_decisions)

    sp = sub.add_parser("fix-degraded",
                        help="给旧方案回填「降级」标记（元素仅并排铺开、未做语义合成）")
    sp.set_defaults(func=cmd_fix_degraded)

    sp = sub.add_parser("tasks",
                        help="导出待生图清单（只出描述不出图，图在别处生成）")
    sp.add_argument("--limit", type=int, default=0)
    sp.add_argument("--all", action="store_true",
                    help="连已经出过图的条目也一起导出（默认只导未出图的）")
    sp.set_defaults(func=cmd_tasks)

    sp = sub.add_parser("import-images",
                        help="把外部生成的图片导回流水线（之后跑 qc,package）")
    sp.add_argument("--dir", required=True, help="放图片的目录")
    sp.add_argument("--no-recursive", action="store_true", help="不递归子目录")
    sp.add_argument("--no-glyph", action="store_true",
                    help="不自动叠加字形参与式的汉字")
    sp.add_argument("--move", action="store_true",
                    help="移动而非复制（省磁盘，但源目录会被清空）")
    sp.set_defaults(func=cmd_import_images)

    sp = sub.add_parser("merge", help="合并多路并发跑出来的结果")
    sp.add_argument("--dirs", nargs="+", required=True,
                    help="要合并的目录，可空格分隔或逗号分隔")
    sp.add_argument("--out", default="", help="合并后的输出目录（默认沿用配置）")
    sp.set_defaults(func=cmd_merge)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    ensure_dir(cfg.get("paths.output_dir"))
    setup_logging(cfg.get("paths.log_file"), verbose=args.verbose)
    LOG.info("配置：%s | 数据目录：%s | 输出目录：%s",
             cfg.config_path, cfg.data_dir, cfg.output_dir)
    return args.func(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
