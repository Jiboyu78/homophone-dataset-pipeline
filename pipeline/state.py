"""SQLite 状态库：断点续跑、幂等、缓存、阶段隔离。

每个阶段只处理"尚未完成该阶段"的条目，因此在任意阶段中断后重跑都安全。
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from .schema import Item, Plan, QCResult
from .utils import LOG, resolve

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id            TEXT PRIMARY KEY,
    text          TEXT NOT NULL,
    type          TEXT NOT NULL,
    pinyin        TEXT DEFAULT '',
    meaning       TEXT DEFAULT '',
    source        TEXT DEFAULT '',
    category      TEXT DEFAULT '',
    raw_json      TEXT DEFAULT '{}',
    drawable_score REAL DEFAULT 0,
    coverage      REAL DEFAULT 0,
    composition   TEXT DEFAULT '',
    segments_json TEXT DEFAULT '[]',
    select_reason TEXT DEFAULT '',
    select_status TEXT DEFAULT 'pending',
    updated_at    REAL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_items_sel ON items(select_status);
CREATE INDEX IF NOT EXISTS idx_items_text ON items(text);

CREATE TABLE IF NOT EXISTS plans (
    item_id    TEXT PRIMARY KEY,
    plan_json  TEXT NOT NULL,
    planner    TEXT DEFAULT '',
    prompt_hash TEXT DEFAULT '',
    updated_at REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS images (
    id          TEXT PRIMARY KEY,
    item_id     TEXT NOT NULL,
    path        TEXT NOT NULL,
    sha1        TEXT DEFAULT '',
    attempt     INTEGER DEFAULT 1,
    provider    TEXT DEFAULT '',
    prompt_hash TEXT DEFAULT '',
    seed        INTEGER DEFAULT -1,
    selected    INTEGER DEFAULT 0,
    created_at  REAL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_images_item ON images(item_id);

CREATE TABLE IF NOT EXISTS qc (
    image_id   TEXT PRIMARY KEY,
    item_id    TEXT NOT NULL,
    result_json TEXT NOT NULL,
    verdict    TEXT DEFAULT 'pending',
    score      REAL DEFAULT 0,
    attempt    INTEGER DEFAULT 1,
    created_at REAL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_qc_item ON qc(item_id);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


class State:
    def __init__(self, db_path: str | Path):
        self.path = resolve(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=60)
        self.conn.row_factory = sqlite3.Row
        # WAL + busy_timeout：并发场景下的保险。虽然设计上"每路一个独立状态库"，
        # 但万一有人共用（或未来加了原子认领），也不至于立刻 database is locked。
        try:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA busy_timeout=60000")
            self.conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.Error as exc:  # 只读介质等情况下不致命
            LOG.debug("设置 WAL 失败（不影响使用）：%s", exc)
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """轻量迁移：老库缺列时补上，避免用户重跑时丢数据。"""
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(items)").fetchall()}
        if "category" not in cols:
            self.conn.execute("ALTER TABLE items ADD COLUMN category TEXT DEFAULT ''")
            LOG.info("数据库迁移：items 表新增 category 列")
        # plans.degraded：降级标记（元素只是并排铺开、没做语义合成）。
        # 方案本体存在 plan_json 里，这份冗余列只为方便直接 SQL 统计与过滤。
        pcols = {r["name"] for r in self.conn.execute("PRAGMA table_info(plans)").fetchall()}
        if pcols and "degraded" not in pcols:
            self.conn.execute("ALTER TABLE plans ADD COLUMN degraded INTEGER DEFAULT 0")
            LOG.info("数据库迁移：plans 表新增 degraded 列")

    def close(self) -> None:
        try:
            self.conn.commit()
            self.conn.close()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self) -> "State":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # meta
    # ------------------------------------------------------------------ #
    def set_meta(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value, ensure_ascii=False)),
        )
        self.conn.commit()

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    # ------------------------------------------------------------------ #
    # items
    # ------------------------------------------------------------------ #
    def upsert_items(self, items: Iterable[Item]) -> int:
        now = time.time()
        rows = [
            (it.id, it.text, it.type, it.pinyin, it.meaning, it.source, it.category,
             json.dumps(it.raw, ensure_ascii=False), it.drawable_score, it.coverage,
             it.composition, json.dumps(it.segments, ensure_ascii=False),
             it.select_reason, it.select_status, now)
            for it in items
        ]
        self.conn.executemany(
            """INSERT INTO items(id,text,type,pinyin,meaning,source,category,raw_json,
                                 drawable_score,coverage,composition,segments_json,
                                 select_reason,select_status,updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 text=excluded.text, type=excluded.type, pinyin=excluded.pinyin,
                 meaning=CASE WHEN excluded.meaning<>'' THEN excluded.meaning ELSE items.meaning END,
                 source=excluded.source, category=excluded.category,
                 raw_json=excluded.raw_json, updated_at=excluded.updated_at""",
            rows,
        )
        self.conn.commit()
        return len(rows)

    def update_select(self, item: Item) -> None:
        self.conn.execute(
            """UPDATE items SET drawable_score=?, coverage=?, composition=?,
               segments_json=?, select_reason=?, select_status=?, updated_at=?
               WHERE id=?""",
            (item.drawable_score, item.coverage, item.composition,
             json.dumps(item.segments, ensure_ascii=False), item.select_reason,
             item.select_status, time.time(), item.id),
        )
        self.conn.commit()

    def update_select_bulk(self, items: Iterable[Item]) -> int:
        """批量写入 select 结果（单事务，22k 条量级下比逐条快两个数量级）。"""
        now = time.time()
        rows = [
            (it.drawable_score, it.coverage, it.composition,
             json.dumps(it.segments, ensure_ascii=False), it.select_reason,
             it.select_status, it.category, now, it.id)
            for it in items
        ]
        with self.conn:
            self.conn.executemany(
                """UPDATE items SET drawable_score=?, coverage=?, composition=?,
                   segments_json=?, select_reason=?, select_status=?, category=?,
                   updated_at=? WHERE id=?""",
                rows,
            )
        return len(rows)

    def load_items(self, status: Sequence[str] | None = None, limit: int = 0,
                   order_by: str = "drawable_score DESC, text") -> list[Item]:
        sql = "SELECT * FROM items"
        args: list[Any] = []
        if status:
            sql += " WHERE select_status IN (%s)" % ",".join("?" * len(status))
            args.extend(status)
        sql += f" ORDER BY {order_by}"
        if limit and limit > 0:
            sql += f" LIMIT {int(limit)}"
        out: list[Item] = []
        for r in self.conn.execute(sql, args).fetchall():
            out.append(self._row_to_item(r))
        return out

    def get_item(self, item_id: str) -> Item | None:
        r = self.conn.execute("SELECT * FROM items WHERE id=?", (item_id,)).fetchone()
        return self._row_to_item(r) if r else None

    def counts(self) -> dict[str, int]:
        c: dict[str, int] = {}
        c["items"] = self.conn.execute("SELECT COUNT(*) n FROM items").fetchone()["n"]
        for row in self.conn.execute(
                "SELECT select_status s, COUNT(*) n FROM items GROUP BY select_status"):
            c[f"select:{row['s']}"] = row["n"]
        c["plans"] = self.conn.execute("SELECT COUNT(*) n FROM plans").fetchone()["n"]
        c["images"] = self.conn.execute("SELECT COUNT(*) n FROM images").fetchone()["n"]
        for row in self.conn.execute("SELECT verdict v, COUNT(*) n FROM qc GROUP BY verdict"):
            c[f"qc:{row['v']}"] = row["n"]
        return c

    # ------------------------------------------------------------------ #
    # plans
    # ------------------------------------------------------------------ #
    def save_plan(self, plan: Plan, prompt_hash: str = "") -> None:
        self.conn.execute(
            """INSERT INTO plans(item_id,plan_json,planner,prompt_hash,updated_at,degraded)
               VALUES(?,?,?,?,?,?)
               ON CONFLICT(item_id) DO UPDATE SET plan_json=excluded.plan_json,
                 planner=excluded.planner, prompt_hash=excluded.prompt_hash,
                 updated_at=excluded.updated_at, degraded=excluded.degraded""",
            (plan.item_id, json.dumps(plan.to_dict(), ensure_ascii=False),
             plan.planner, prompt_hash, time.time(),
             1 if getattr(plan, "degraded", False) else 0),
        )
        self.conn.commit()

    def get_plan(self, item_id: str) -> Plan | None:
        r = self.conn.execute("SELECT plan_json FROM plans WHERE item_id=?", (item_id,)).fetchone()
        return Plan.from_dict(json.loads(r["plan_json"])) if r else None

    def items_without_plan(self, limit: int = 0) -> list[Item]:
        sql = """SELECT i.* FROM items i LEFT JOIN plans p ON p.item_id=i.id
                 WHERE i.select_status='accepted' AND p.item_id IS NULL
                 ORDER BY i.drawable_score DESC, i.text"""
        if limit and limit > 0:
            sql += f" LIMIT {int(limit)}"
        rows = self.conn.execute(sql).fetchall()
        return [self._row_to_item(r) for r in rows]

    # ------------------------------------------------------------------ #
    # images
    # ------------------------------------------------------------------ #
    def add_image(self, image_id: str, item_id: str, path: str, sha1: str,
                  attempt: int, provider: str, prompt_hash: str, seed: int) -> None:
        self.conn.execute(
            """INSERT INTO images(id,item_id,path,sha1,attempt,provider,prompt_hash,seed,created_at)
               VALUES(?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET path=excluded.path, sha1=excluded.sha1,
                 attempt=excluded.attempt, provider=excluded.provider,
                 prompt_hash=excluded.prompt_hash, seed=excluded.seed""",
            (image_id, item_id, path, sha1, attempt, provider, prompt_hash, seed, time.time()),
        )
        self.conn.commit()

    def mark_image_selected(self, image_id: str, selected: bool = True) -> None:
        self.conn.execute("UPDATE images SET selected=? WHERE id=?",
                          (1 if selected else 0, image_id))
        self.conn.commit()

    def list_images(self, item_id: str | None = None, attempt: int | None = None) -> list[dict]:
        sql = "SELECT * FROM images"
        args: list[Any] = []
        conds = []
        if item_id:
            conds.append("item_id=?")
            args.append(item_id)
        if attempt is not None:
            conds.append("attempt=?")
            args.append(attempt)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY attempt, created_at"
        return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def items_needing_image(self, max_attempt: int, limit: int = 0,
                            balanced: bool = False,
                            max_glyph_share: float = 0.30) -> list[tuple[Item, Plan, int]]:
        """返回 (item, plan, 下一次尝试轮次)。

        只有满足下面条件的条目才会再次出图：
          · 还没有任何图片；或者
          · 最新一次质检结论是 retry（reject 是硬失败，重出图没意义；
            accept 已选中；needs_human 已转人工）

        balanced=True 时按"构图范式 + 出题手法"轮询取样，避免数据集被"分数最高"的
        单一范式（通常是字面直译式的简单复合词）淹没，也保证占比很小但很特别的
        字形参与式（举手之劳那种）能进入每一批。
        """
        rows = self.conn.execute(
            """SELECT i.*, p.plan_json,
                      (SELECT COUNT(*) FROM images im WHERE im.item_id=i.id) AS n_img,
                      (SELECT COUNT(*) FROM images im WHERE im.item_id=i.id AND im.selected=1) AS n_sel
               FROM items i JOIN plans p ON p.item_id=i.id
               WHERE i.select_status='accepted' AND p.plan_json IS NOT NULL"""
        ).fetchall()
        out: list[tuple[Item, Plan, int]] = []
        for r in rows:
            if r["n_sel"] > 0:
                continue
            attempt = int(r["n_img"]) + 1
            if attempt > max_attempt:
                continue
            if attempt > 1:
                last = self.conn.execute(
                    """SELECT q.verdict FROM qc q JOIN images im ON im.id=q.image_id
                       WHERE q.item_id=? ORDER BY im.attempt DESC, q.created_at DESC LIMIT 1""",
                    (r["id"],)).fetchone()
                if last is None or last["verdict"] != "retry":
                    continue
            out.append((self._row_to_item(r), Plan.from_dict(json.loads(r["plan_json"])), attempt))
        out.sort(key=lambda t: -t[0].drawable_score)
        if balanced and out:
            groups: dict[str, list[tuple[Item, Plan, int]]] = {}
            for t in out:
                # 手法也参与分桶：字形参与式只占 7%，按纯分数排会被整批挤掉
                key = f"{t[1].composition}/{t[1].technique}"
                groups.setdefault(key, []).append(t)
            mixed: list[tuple[Item, Plan, int]] = []
            keys = sorted(groups)
            while keys:
                for k in list(keys):
                    bucket = groups.get(k) or []
                    if bucket:
                        mixed.append(bucket.pop(0))
                    if not bucket:
                        keys.remove(k)
            out = mixed
        return out[:limit] if limit and limit > 0 else out

    def items_needing_qc(self, limit: int = 0) -> list[dict]:
        """已出图但还没有质检记录的图片。"""
        sql = """SELECT im.* FROM images im LEFT JOIN qc ON qc.image_id=im.id
                 WHERE qc.image_id IS NULL ORDER BY im.attempt, im.created_at"""
        if limit and limit > 0:
            sql += f" LIMIT {int(limit)}"
        return [dict(r) for r in self.conn.execute(sql).fetchall()]

    def latest_image(self, item_id: str) -> dict | None:
        rows = self.conn.execute(
            "SELECT * FROM images WHERE item_id=? ORDER BY attempt DESC, created_at DESC LIMIT 1",
            (item_id,)).fetchall()
        return dict(rows[0]) if rows else None

    def latest_qc_for_item(self, item_id: str) -> QCResult | None:
        rows = self.conn.execute(
            """SELECT q.result_json FROM qc q JOIN images im ON im.id=q.image_id
               WHERE q.item_id=? ORDER BY im.attempt DESC, q.created_at DESC LIMIT 1""",
            (item_id,)).fetchall()
        if not rows:
            return None
        d = json.loads(rows[0]["result_json"])
        known = set(QCResult.__dataclass_fields__)  # type: ignore[attr-defined]
        return QCResult(**{k: v for k, v in d.items() if k in known})

    def find_image_by_prompt_hash(self, item_id: str, prompt_hash: str) -> dict | None:
        rows = self.conn.execute(
            "SELECT * FROM images WHERE item_id=? AND prompt_hash=? ORDER BY attempt LIMIT 1",
            (item_id, prompt_hash)).fetchall()
        return dict(rows[0]) if rows else None

    # ------------------------------------------------------------------ #
    # qc
    # ------------------------------------------------------------------ #
    def save_qc(self, image_id: str, item_id: str, res: QCResult, attempt: int = 1) -> None:
        self.conn.execute(
            """INSERT INTO qc(image_id,item_id,result_json,verdict,score,attempt,created_at)
               VALUES(?,?,?,?,?,?,?)
               ON CONFLICT(image_id) DO UPDATE SET result_json=excluded.result_json,
                 verdict=excluded.verdict, score=excluded.score, attempt=excluded.attempt""",
            (image_id, item_id, json.dumps(res.to_dict(), ensure_ascii=False),
             res.verdict, res.score, attempt, time.time()),
        )
        self.conn.commit()

    def get_qc(self, image_id: str) -> QCResult | None:
        r = self.conn.execute("SELECT result_json FROM qc WHERE image_id=?", (image_id,)).fetchone()
        if not r:
            return None
        d = json.loads(r["result_json"])
        known = set(QCResult.__dataclass_fields__)  # type: ignore[attr-defined]
        return QCResult(**{k: v for k, v in d.items() if k in known})

    def ids_with_artifacts(self) -> set[str]:
        """返回已经有"下游产物"的条目 id（已出过题或已出过图）。

        用途：select 阶段的配额截断不能把这些条目踢掉——那是已经花过钱的结果，
        踢掉后既浪费已生成的内容，也让"分批扩产"无法累加。
        """
        ids: set[str] = set()
        for sql in ("SELECT DISTINCT item_id FROM plans",
                    "SELECT DISTINCT item_id FROM images"):
            try:
                ids |= {r[0] for r in self.conn.execute(sql).fetchall()}
            except Exception:  # noqa: BLE001
                continue
        return ids

    def accepted_pairs(self, min_score: float, only_accepted: bool = True) -> list[dict]:
        """选出每个条目得分最高且被判接受的图片。"""
        sql = """SELECT q.item_id, q.image_id, q.score, q.verdict, q.result_json,
                        im.path, im.sha1, im.attempt
                 FROM qc q JOIN images im ON im.id=q.image_id
                 WHERE q.score >= ?"""
        args: list[Any] = [min_score]
        if only_accepted:
            sql += " AND q.verdict='accept'"
        sql += " ORDER BY q.item_id, q.score DESC, im.attempt ASC"
        best: dict[str, dict] = {}
        for r in self.conn.execute(sql, args).fetchall():
            d = dict(r)
            if d["item_id"] not in best:
                best[d["item_id"]] = d
        return list(best.values())

    def all_qc(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM qc ORDER BY item_id").fetchall()]

    # ------------------------------------------------------------------ #
    # 维护
    # ------------------------------------------------------------------ #
    def dedupe_by_text(self) -> int:
        """清理历史遗留的"同词不同 id"重复行（例如早期版本用 hash() 生成 id）。

        保留优先级：成语类型 > 有释义 > 有类型字段 > id 字典序。
        """
        rows = self.conn.execute(
            "SELECT text, GROUP_CONCAT(id) ids FROM items GROUP BY text HAVING COUNT(*)>1"
        ).fetchall()
        if not rows:
            return 0
        removed = 0
        for r in rows:
            ids = r["ids"].split(",")
            loaded = [self.get_item(i) for i in ids]
            loaded = [x for x in loaded if x is not None]
            loaded.sort(key=lambda it: (
                0 if it.type == "idiom" else 1,
                0 if it.meaning else 1,
                0 if it.category else 1,
                -it.drawable_score,
                it.id,
            ))
            drop = [it.id for it in loaded[1:]]
            if not drop:
                continue
            qs = ",".join("?" * len(drop))
            self.conn.execute(f"DELETE FROM plans WHERE item_id IN ({qs})", drop)
            self.conn.execute(f"DELETE FROM images WHERE item_id IN ({qs})", drop)
            self.conn.execute(f"DELETE FROM qc WHERE item_id IN ({qs})", drop)
            self.conn.execute(f"DELETE FROM items WHERE id IN ({qs})", drop)
            removed += len(drop)
        self.conn.commit()
        if removed:
            LOG.info("去重：清理同词重复行 %d 条", removed)
        return removed

    def reset_stage(self, stage: str) -> None:
        if stage == "select":
            self.conn.execute("UPDATE items SET select_status='pending', drawable_score=0, coverage=0")
        elif stage == "plan":
            self.conn.execute("DELETE FROM plans")
        elif stage == "generate":
            self.conn.execute("DELETE FROM images")
            self.conn.execute("DELETE FROM qc")
        elif stage == "qc":
            self.conn.execute("DELETE FROM qc")
        elif stage == "all":
            for t in ("items", "plans", "images", "qc"):
                self.conn.execute(f"DELETE FROM {t}")
        else:
            raise ValueError(f"未知阶段: {stage}")
        self.conn.commit()
        LOG.info("已重置阶段: %s", stage)

    @staticmethod
    def _row_to_item(r: sqlite3.Row) -> Item:
        keys = r.keys()
        return Item(
            id=r["id"], text=r["text"], type=r["type"], pinyin=r["pinyin"],
            meaning=r["meaning"], source=r["source"],
            category=(r["category"] if "category" in keys else "") or "",
            raw=json.loads(r["raw_json"] or "{}"),
            drawable_score=r["drawable_score"], coverage=r["coverage"],
            composition=r["composition"], segments=json.loads(r["segments_json"] or "[]"),
            select_reason=r["select_reason"], select_status=r["select_status"],
        )
