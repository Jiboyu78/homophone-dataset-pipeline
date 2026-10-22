"""并发跑完之后，把多路结果合并成一份数据集。

为什么需要它：为了 4 路并发，每路必须用**独立的状态库**（共用一个库时，
worker0 会把不属于自己的词标成 rejected，其他路再看就已经不是候选了，切片失效）。
所以每路各自产出一份 `output/shardN/`，最后用本模块合并。

    pipeline.cli merge --dirs output/shard1,output/shard2 --out output

能合并两类产物，按目录里有什么自动判断：
  · `dataset.jsonl`（已出图、已打包）→ 合并记录、拷图、重做划分与全部导出格式
  · `image_tasks.jsonl`（只出描述模式，还没出图）→ 合并任务清单，重新编号
两类都有时优先按 dataset 合并。
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from . import export, review
from .config import Config
from .utils import LOG, resolve, write_jsonl

TRAIN_FILES = ("dataset.jsonl", "image_tasks.jsonl")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            LOG.warning("跳过无法解析的行：%s", path.name)
    return out


def _dedupe(rows: list[dict[str, Any]], key: str) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    dup = 0
    for r in rows:
        k = str(r.get(key) or "")
        if not k or k in seen:
            dup += 1
            continue
        seen.add(k)
        kept.append(r)
    return kept, dup


def merge_runs(cfg: Config, src_dirs: list[str | Path]) -> dict[str, Any]:
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    srcs = [resolve(str(d)) for d in src_dirs]
    missing = [str(s) for s in srcs if not s.is_dir()]
    if missing:
        raise FileNotFoundError("这些目录不存在：" + "、".join(missing))

    # 先把各路的 items/plans 并进主状态库：不并的话，合并后的目录是"空壳"——
    # 之后导回图片时查不到方案，字形参与式的叠字和打包都会失败。
    db_stats = _merge_databases(cfg, srcs)

    records: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    for s in srcs:
        records += _read_jsonl(s / "dataset.jsonl")
        tasks += _read_jsonl(s / "image_tasks.jsonl")

    if records:
        res = _merge_records(cfg, records, srcs, out_dir)
    elif tasks:
        res = _merge_tasks(cfg, tasks, out_dir)
    else:
        LOG.error("这些目录里既没有 dataset.jsonl 也没有 image_tasks.jsonl：%s",
                  "、".join(str(s) for s in srcs))
        res = {"mode": "empty"}
    res["dirs"] = [str(s) for s in srcs]
    res.update(db_stats)
    return res


def _merge_databases(cfg: Config, srcs: list[Path]) -> dict[str, Any]:
    """把各路独立状态库里的 items 与 plans 并进主库。

    两个坑（都实测踩过，都会"静默失效"）：
      · 目标库必须先用 State 建表并跑迁移——只 executescript(SCHEMA) 建出来的
        plans 表没有 degraded 列，SQL 里带这个列名就会 no such column。
      · 读附加库的列名必须写 `PRAGMA <schema>.table_info(<table>)`；
        写成 `PRAGMA table_info(<schema>.<table>)` 是**非法语法**、返回空，
        于是列清单为空 → 插入整段被跳过 → 合并后主库 0 条。
    """
    import sqlite3

    from .state import State

    target = cfg.db_path
    target.parent.mkdir(parents=True, exist_ok=True)
    State(target).close()          # 建表 + 迁移，保证目标库列齐全

    conn = sqlite3.connect(str(target))
    conn.execute("PRAGMA busy_timeout=60000")
    merged = {"items": 0, "plans": 0, "images": 0, "qc": 0, "dbs": []}

    def _q(sql: str, args: tuple = ()) -> list:
        """显式游标并及时关闭：否则未释放的语句会让 DETACH 失败，
        进而让后续库的 ATTACH 报 "database s is already in use" 被跳过
        （实测只并进了第一路，items/plans 看起来"部分成功"，很难发现）。"""
        cur = conn.cursor()
        try:
            cur.execute(sql, args)
            return cur.fetchall()
        finally:
            cur.close()

    try:
        for s in srcs:
            dbs = sorted(s.glob("*.db"))
            if not dbs:
                continue
            src_db = dbs[0]
            try:
                conn.execute("ATTACH DATABASE ? AS s", (str(src_db),))
            except sqlite3.Error as exc:
                LOG.warning("附加状态库失败 %s：%s", src_db, exc)
                continue
            merged["dbs"].append(str(src_db))
            for table in ("items", "plans", "images", "qc"):
                try:
                    tcols = [r[1] for r in _q(f"PRAGMA table_info({table})")]
                    scols = [r[1] for r in _q(f"PRAGMA s.table_info({table})")]
                    if not scols:
                        continue
                    cols = [c for c in tcols if c in scols]
                    if not cols:
                        continue
                    collist = ",".join(cols)
                    before = conn.total_changes
                    _q(f"INSERT OR IGNORE INTO {table} ({collist}) "
                       f"SELECT {collist} FROM s.{table}")
                    merged[table] += conn.total_changes - before
                except sqlite3.Error as exc:
                    LOG.warning("合并表 %s 失败：%s", table, exc)
            conn.commit()
            try:
                conn.execute("DETACH DATABASE s")
            except sqlite3.Error as exc:
                LOG.warning("分离状态库失败（后续库可能被跳过）：%s", exc)
        conn.commit()
    finally:
        conn.close()
    LOG.info("合并状态库：items +%d，plans +%d（来自 %d 个库）",
             merged["items"], merged["plans"], len(merged["dbs"]))
    return {"db_items": merged["items"], "db_plans": merged["plans"]}


def _merge_records(cfg: Config, records: list[dict[str, Any]],
                   srcs: list[Path], out_dir: Path) -> dict[str, Any]:
    records, dup = _dedupe(records, "id")
    LOG.info("合并记录：%d 条（去重丢弃 %d 条）", len(records), dup)

    # 图片按记录里的相对路径从各分片目录搬过来
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for s in srcs:
        src_img = s / "images"
        if not src_img.is_dir():
            continue
        for p in src_img.glob("*.png"):
            dest = images_dir / p.name
            if not dest.exists():
                shutil.copy2(p, dest)
                copied += 1
    LOG.info("合并图片：新拷入 %d 张 -> %s", copied, images_dir)

    write_jsonl(out_dir / "dataset.jsonl", records)
    (out_dir / "dataset.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")

    # 重新划分（各分片自己的 split 已无意义）+ 重做全部导出
    import random
    splits = cfg.get("package.splits") or {"train": 0.8, "val": 0.1, "test": 0.1}
    rnd = random.Random(int(cfg.get("package.seed", 42)))
    idx = list(range(len(records)))
    rnd.shuffle(idx)
    n = len(records)
    n_train = int(n * float(splits.get("train", 0.8)))
    n_val = int(n * float(splits.get("val", 0.1)))
    for pos, i in enumerate(idx):
        records[i]["split"] = ("train" if pos < n_train
                               else "val" if pos < n_train + n_val else "test")

    write_jsonl(out_dir / "dataset.jsonl", records)
    for split in ("train", "val", "test"):
        rows = [r for r in records if r.get("split") == split]
        write_jsonl(out_dir / f"{split}.jsonl", rows)

    # manifest.csv（与 package 阶段保持一致的列）
    with (out_dir / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "答案", "拼音", "类型", "分类", "构图范式", "难度", "split",
                     "降级", "降级原因", "质检分", "可解", "文字泄漏", "元素覆盖率",
                     "图片", "提示词"])
        for r in records:
            qc = r.get("qc") or {}
            wr.writerow([r["id"], r["answer"], r.get("pinyin", ""), r.get("type", ""),
                         r.get("category", ""), r.get("composition_zh",
                                                     r.get("composition", "")),
                         r.get("difficulty", ""), r.get("split", ""),
                         "是" if r.get("degraded") else "",
                         r.get("degraded_reason", ""), qc.get("score"),
                         qc.get("solvable"), qc.get("text_free"),
                         qc.get("element_coverage"), r.get("image", ""),
                         r.get("prompt_en", "")])

    exported = export.export_training_formats(cfg, records)
    text = export.write_data_card(cfg, records, exported)
    if records:
        review.write_review(cfg, records)
        if bool(cfg.get("package.contact_sheets", True)):
            review.write_contact_sheets(cfg, records)
    LOG.info("合并完成：%d 条 -> %s（数据卡 %s）", len(records), out_dir, text.name)
    return {"mode": "records", "records": len(records), "duplicates": dup,
            "images_copied": copied, "exported": exported,
            "dirs": [str(s) for s in srcs]}


def _merge_tasks(cfg: Config, tasks: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    """只出描述模式：合并各路的待生图清单，重新连续编号。"""
    tasks, dup = _dedupe(tasks, "item_id")
    tasks.sort(key=lambda r: (-float(r.get("difficulty") or 0), str(r.get("answer"))))
    for i, r in enumerate(tasks, 1):
        r["index"] = i
        fname = str(r.get("target_filename") or "")
        # 文件名里的序号要跟着重排，否则合起来看序号是断的
        parts = fname.split("_", 1)
        tail = parts[1] if len(parts) == 2 else f"{r.get('item_id')}.png"
        r["target_filename"] = f"{i:03d}_{tail}"

    write_jsonl(out_dir / "image_tasks.jsonl", tasks)
    with (out_dir / "image_tasks.csv").open("w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["序号", "文件名", "答案", "类型", "构图", "降级", "尺寸",
                     "提示词", "负向词", "需叠加的字", "必须出现的元素"])
        for r in tasks:
            wr.writerow([r["index"], r["target_filename"], r["answer"], r["type"],
                         r["composition"], "是" if r.get("degraded") else "",
                         r.get("size", ""), r.get("prompt", ""),
                         r.get("negative_prompt", ""), r.get("glyph", ""),
                         " / ".join(r.get("must_have") or [])])
    prompt_dir = out_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for r in tasks:
        (prompt_dir / f"{Path(str(r['target_filename'])).stem}.txt").write_text(
            str(r.get("prompt", "")), encoding="utf-8")

    LOG.info("合并待生图清单：%d 条（去重丢弃 %d 条）-> %s",
             len(tasks), dup, out_dir / "image_tasks.csv")
    return {"mode": "tasks", "tasks": len(tasks), "duplicates": dup}
