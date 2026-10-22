"""select 阶段：判断一条词条能否用"会意元素组合"画出来，并打分筛选。

评分构成（0-1）：
    coverage       可入画语素覆盖率（词典权重加权）
    concreteness   可画语素的平均"具体度"
    element_score  元素个数的形状分（2-4 个最理想：太少不成谜，太多画不乱）
    penalties      抽象字 / 词典外字 / 虚词占比 / 否定字 / 多义字风险
    factors        词表"类型"先验 × 词长先验

硬规则（直接拒绝）：
    - 命中 select.deny_words
    - 类型属于专名/抽象类（地名、明星、品牌、歌名…），无法用元素会意
    - 可入画元素少于 2 个（那只是"看图识物"，不是元素组合）
    - 覆盖率低于阈值
    - 复合词的两个语素同属一个语义类（米饭=米+饭 这类同义并列，画出来还是同一个东西）
"""
from __future__ import annotations

from collections import Counter

from .config import Config
from .lexicon import (LEX, NEGATION_CHARS, POLYSEMY_RISK)
from .schema import Item
from .state import State
from .utils import LOG, clean_text

# 词表"类型"先验：专名类无法用元素组合表意，直接压到阈值以下
CATEGORY_WEIGHTS: dict[str, float] = {
    "食物": 1.0, "动物": 1.0, "水果": 1.0, "蔬菜": 1.0, "饮品": 1.0,
    "植物": 1.0, "生活用品": 1.0, "家用电器": 1.0, "文具办公": 1.0,
    "服饰配饰": 1.0, "交通工具": 1.0, "兵器": 1.0, "材料": 0.95,
    "自然": 1.0, "天气": 1.0, "身体": 1.0, "建筑场所": 0.95,
    "美妆个护": 0.9, "运动体育": 0.85, "常用词语": 0.9, "称谓": 0.8,
    "职业": 0.8, "游戏": 0.6, "学科术语": 0.5, "祝福语": 0.6,
    "流行语": 0.55, "网络热梗": 0.5, "成语": 0.95,
    "中国地名": 0.15, "世界国家地区": 0.15, "景点地标": 0.2,
    "历史人物": 0.12, "明星": 0.1, "名人": 0.12, "体育明星": 0.1,
    "品牌": 0.2, "网络平台": 0.2, "影视动漫角色": 0.15, "文学角色": 0.15,
    "影视剧": 0.15, "神话人物": 0.25, "歌名": 0.3, "抽象概念": 0.3,
}

ELEMENT_SCORE = {0: 0.0, 1: 0.25, 2: 1.0, 3: 1.0, 4: 0.9, 5: 0.7}


def _category_factor(category: str) -> float:
    if not category:
        return 1.0
    if category in CATEGORY_WEIGHTS:
        return CATEGORY_WEIGHTS[category]
    for key, w in CATEGORY_WEIGHTS.items():
        if key in category or category in key:
            return w
    return 0.6


def _length_factor(text: str, itype: str) -> float:
    n = len(text)
    if itype == "idiom":
        return {4: 1.0, 3: 0.85, 5: 0.85, 6: 0.75}.get(n, 0.6)
    return {2: 1.0, 3: 0.92, 1: 0.45, 4: 0.7}.get(n, 0.5)


def score_item(item: Item, cfg: Config) -> Item:
    text = item.text
    segs = LEX.segment(text, apply_suffix=(item.type == "word"))
    item.segments = segs

    content = [s for s in segs if s["mode"] != "no"]
    drawable = [s for s in segs if s["drawable"]]
    n_content = max(1, len(content))
    k = len(drawable)

    coverage = min(1.0, sum(s["weight"] for s in drawable) / n_content)
    concreteness = (sum(s["weight"] for s in drawable) / k) if k else 0.0
    element_score = ELEMENT_SCORE.get(k, 0.5)

    n_abstract = sum(1 for s in segs if s["kind"] == "abstract")
    n_unknown = sum(1 for s in segs if s["kind"] == "unknown")
    n_fn = sum(1 for s in segs if s["mode"] == "no")
    fn_ratio = n_fn / max(1, len(segs))
    n_neg = sum(1 for c in set(text) if c in NEGATION_CHARS)
    n_poly = sum(1 for c in set(text) if c in POLYSEMY_RISK)

    base = 0.45 * coverage + 0.35 * concreteness + 0.20 * element_score
    penalties = (
        0.10 * n_abstract
        + 0.10 * n_unknown
        + 0.25 * max(0.0, fn_ratio - 0.30)
        + 0.05 * n_neg
        + min(0.12, 0.03 * n_poly)
    )
    cat_factor = _category_factor(item.category)
    len_factor = _length_factor(text, item.type)
    score = max(0.0, base - penalties) * (0.60 + 0.40 * cat_factor) * (0.85 + 0.15 * len_factor)

    # ---------------- 硬规则 ---------------- #
    reasons: list[str] = []
    rejected = False

    for w in cfg.get("select.deny_words", []):
        if w and w in text:
            rejected = True
            reasons.append(f"含禁用词素「{w}」")
            break

    if cat_factor <= 0.25:
        rejected = True
        reasons.append(f"类型「{item.category}」为专名/抽象类，不适合会意化")

    if k < 2:
        rejected = True
        reasons.append(f"可入画元素仅 {k} 个，只是看图识物而非元素组合")

    # 复合词同义并列：两个语素语义类相同 -> 画出来还是同一个东西
    if item.type == "word" and k >= 2:
        kinds = [s["kind"] for s in drawable]
        if len(set(kinds)) == 1:
            rejected = True
            reasons.append(f"语素同属「{kinds[0]}」类，属同义并列复合词，字面直译无信息量")

    if coverage < float(cfg.get("select.min_coverage", 0.6)) and not rejected:
        rejected = True
        reasons.append(f"可画语素覆盖率 {coverage:.2f} 低于阈值")

    comp = LEX.composition_hint(text, segs)
    if item.type == "word" and k >= 2 and comp in ("juxtapose", "metaphor"):
        comp = "literal"     # 复合词逐字具象
    if comp == "homophone" and not cfg.get("select.allow_homophone", True):
        rejected = True
        reasons.append("谐音借代式已关闭")

    item.coverage = round(coverage, 4)
    item.drawable_score = round(score, 4)
    item.composition = comp
    item.select_status = "accepted" if (not rejected and score >= float(
        cfg.get("select.min_score", 0.55))) else "rejected"
    if not rejected and item.select_status == "rejected":
        reasons.append(f"综合分 {score:.2f} 低于阈值 {cfg.get('select.min_score')}")
    if not rejected and item.select_status == "accepted":
        flags = []
        if n_neg:
            flags.append(f"含否定字{'/'.join(c for c in set(text) if c in NEGATION_CHARS)}")
        if n_poly >= 2:
            flags.append("多义字风险高")
        if n_unknown:
            flags.append(f"含词典外字 {n_unknown} 个")
        item.select_reason = "通过" + ("（" + "；".join(flags) + "）" if flags else "")
    else:
        item.select_reason = "；".join(reasons)
    return item


def run(cfg: Config, state: State, limit: int = 0, rescore: bool = False) -> dict[str, int]:
    from .lexicon import lexicon_hash
    lh = lexicon_hash()
    if state.get_meta("select_lexicon_hash") != lh:
        LOG.info("检测到词典已更新，重新为全部词条打分")
        rescore = True
    # 分片并发时**必须忽略 limit**：limit 是"从词表头部取多少条来打分"，
    # 若照用，4 路会各自只打分同一小段前缀，再从中切走一片——合起来只覆盖
    # 前缀那几条，并发反而让总量变少（实测 4 路 × 5 条只覆盖到 5 个词）。
    # 正确做法：全量打分，再由 shard 切片决定归属，最后由 max_items 卡住本路额度。
    shard_count_cfg = int(cfg.get("select.shard_count", 1) or 1)
    if shard_count_cfg > 1:
        if limit:
            LOG.info("分片模式下忽略 limit=%d，改为全量打分后再切片", limit)
        limit = 0

    items = state.load_items(limit=limit, order_by="id")
    if not rescore:
        items = [it for it in items if it.select_status == "pending"]
    if not items:
        LOG.info("select：没有待处理词条")
        state.set_meta("select_lexicon_hash", lh)
        return state.counts()

    for it in items:
        it.category = clean_text(it.raw.get("类型") or it.raw.get("category") or "")

    scored = [score_item(it, cfg) for it in items]
    state.update_select_bulk(scored)
    state.set_meta("select_lexicon_hash", lh)

    c = Counter(it.select_status for it in scored)
    comps = Counter(it.composition for it in scored if it.select_status == "accepted")
    LOG.info("select 完成：接受 %d / 拒绝 %d（共 %d）",
             c.get("accepted", 0), c.get("rejected", 0), len(scored))
    LOG.info("接受条目的构图范式分布：%s", dict(comps.most_common()))
    if c.get("accepted", 0):
        top = sorted([i for i in scored if i.select_status == "accepted"],
                     key=lambda x: -x.drawable_score)[:12]
        LOG.info("样例（分数最高）：%s", "、".join(
            f"{t.text}({t.drawable_score:.2f}/{t.composition})" for t in top))

    # 分片：多进程并发时每个 worker 只处理自己那一份，避免 4 个进程挑同一批词。
    # 用"每隔 N 个取一个"而不是连续切块——候选是按分数排序的，连续切块会让
    # 1 号 worker 拿走全部高分词，各路线质量不均。
    # 注意：分片必须配合"每路独立状态库"。共用一个库时，worker0 会把不属于它的
    # 词标记为 rejected，worker1 再来看就已经不是 accepted 了，切片直接失效。
    shard_count = int(cfg.get("select.shard_count", 1) or 1)
    shard_index = int(cfg.get("select.shard_index", 0) or 0)
    if shard_count > 1:
        shard_index = max(0, min(shard_index, shard_count - 1))
        all_accepted = state.load_items(status=["accepted"],
                                        order_by="drawable_score DESC, text")
        mine = {it.id for it in all_accepted[shard_index::shard_count]}
        invested = state.ids_with_artifacts()
        outsiders = [it for it in all_accepted
                     if it.id not in mine and it.id not in invested]
        for it in outsiders:
            it.select_status = "rejected"
            it.select_reason = (f"分片 {shard_index + 1}/{shard_count}："
                                f"留给其他并发进程处理")
        state.update_select_bulk(outsiders)
        LOG.info("分片 %d/%d：本路保留 %d 条，让给其他路 %d 条",
                 shard_index + 1, shard_count, len(mine), len(outsiders))

    max_items = int(cfg.get("select.max_items", 0) or 0)
    if max_items > 0:
        accepted = state.load_items(status=["accepted"], order_by="drawable_score DESC, text")
        # 已经出过题/出过图的词不能被配额踢掉：那是已经花过钱的结果。
        # 不保护它们时，反复用同一个小 --limit 跑，产出会卡在 max_items 不再增长
        # （"分批扩产"直接失效）。保护之后，每轮都能在旧成果之上再新增 max_items 个名额。
        invested = state.ids_with_artifacts()
        dropped = [it for it in accepted[max_items:] if it.id not in invested]
        protected = sum(1 for it in accepted[max_items:] if it.id in invested)
        for it in dropped:
            it.select_status = "rejected"
            it.select_reason = f"超出 select.max_items={max_items} 限制"
        state.update_select_bulk(dropped)
        LOG.info("按 max_items=%d 截断，额外拒绝 %d 条（保护已有产物的 %d 条）",
                 max_items, len(dropped), protected)

    return state.counts()
