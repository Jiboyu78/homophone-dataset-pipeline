"""统一数据模型。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ItemType = Literal["idiom", "word"]

# 会意（元素组合）构图范式
COMPOSITION_TYPES = [
    "juxtapose",   # 并列式：两个以上元素同框相加
    "spatial",     # 方位式：元素位置关系本身表意
    "quantity",    # 数量式：数量对比/累加
    "contrast",    # 反义并置式：对立元素同框
    "action",      # 动作式：主体+动作+对象
    "inner",       # 表里式：同一主体的"外表"与"内心/影子"并置（人面兽心）
    "metaphor",    # 具象替代式：以具体物替代抽象概念
    "literal",     # 字面直译式：复合词逐字具象（火车=火+车）
    "homophone",   # 谐音借代式（可选，非严格会意）
]

COMPOSITION_ZH = {
    "juxtapose": "并列式",
    "spatial": "方位式",
    "quantity": "数量式",
    "contrast": "反义并置式",
    "action": "动作式",
    "inner": "表里式",
    "metaphor": "具象替代式",
    "literal": "字面直译式",
    "homophone": "谐音借代式",
}

# 题目手法（对应参考题标注里的"谐音"一栏，这里按机制细分）
TECHNIQUE_ZH = {
    "ideographic": "会意图解",   # 纯图形组合表意
    "glyph": "字形参与",         # 画面中出现汉字，字形本身是谜面的一部分
    "homophone": "谐音借代",
}


@dataclass
class Item:
    """一条待出题词条。"""
    id: str
    text: str
    type: ItemType = "idiom"
    pinyin: str = ""
    meaning: str = ""
    source: str = ""
    category: str = ""             # 原始数据里的"类型"字段（如 食物/动物）
    raw: dict[str, Any] = field(default_factory=dict)
    # select 阶段回填
    drawable_score: float = 0.0
    coverage: float = 0.0
    composition: str = ""
    segments: list[dict[str, Any]] = field(default_factory=list)
    select_reason: str = ""
    select_status: str = "pending"   # pending | accepted | rejected

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Item":
        known = set(Item.__dataclass_fields__)  # type: ignore[attr-defined]
        return Item(**{k: v for k, v in d.items() if k in known})


@dataclass
class Element:
    """构图中的一个视觉元素。"""
    morpheme: str = ""
    zh: str = ""
    en: str = ""
    count: int = 1
    role: str = "object"        # subject | object | prop | background | modifier
    position: str = ""
    scale: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Element":
        known = set(Element.__dataclass_fields__)  # type: ignore[attr-defined]
        return Element(**{k: v for k, v in d.items() if k in known})


@dataclass
class Plan:
    """一个词条的结构化谜面方案。"""
    item_id: str
    answer: str
    composition: str = "juxtapose"
    elements: list[Any] = field(default_factory=list)
    layout: str = ""
    prompt_zh: str = ""
    prompt_en: str = ""
    negative_prompt: str = ""
    must_have: list[str] = field(default_factory=list)
    must_not: list[str] = field(default_factory=list)
    difficulty: int = 3
    distractors: list[str] = field(default_factory=list)
    rationale: str = ""
    planner: str = "rule"
    glyph: str = ""                # 需要"写进画面"的汉字（字形参与式，如 举手之劳 的"劳"）
    technique: str = "ideographic"  # ideographic | glyph | homophone
    panels: int = 1                # 画面数：1=单幅
    relation: str = ""             # 表里式等特殊构图的元素关系说明
    # 降级标记：非成语类词条有时"想不出怎么把元素合成一个画面"，此时允许不合成，
    # 直接把元素并排铺开（如 刀鱼 -> 一把刀 + 一条鱼）。这类样本语义关联弱，
    # 交付时必须标出来，方便训练时过滤或降权。
    degraded: bool = False
    degraded_reason: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["elements"] = [e.to_dict() if isinstance(e, Element) else e for e in self.elements]
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Plan":
        d = dict(d)
        d["elements"] = [Element.from_dict(e) if isinstance(e, dict) else e
                         for e in d.get("elements", [])]
        known = set(Plan.__dataclass_fields__)  # type: ignore[attr-defined]
        extra = dict(d.pop("extra", {}) or {})
        extra.update({k: v for k, v in d.items() if k not in known})
        kwargs = {k: v for k, v in d.items() if k in known}
        kwargs["extra"] = extra
        return Plan(**kwargs)


@dataclass
class QCResult:
    """一条记录的质检结果。"""
    item_id: str
    verdict: str = "pending"        # accept | retry | reject | needs_human
    score: float = 0.0
    text_free: bool | None = None
    text_found: list[str] = field(default_factory=list)
    element_coverage: float | None = None
    missing_elements: list[str] = field(default_factory=list)
    solvable: bool | None = None
    guessed_answer: str = ""
    technical: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    feedback_for_retry: str = ""
    checker: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Record:
    """最终数据集记录。"""
    id: str
    answer: str
    type: ItemType
    pinyin: str
    meaning: str
    composition: str
    difficulty: int
    image: str
    image_sha1: str
    prompt_en: str
    prompt_zh: str
    negative_prompt: str
    elements: list[dict[str, Any]]
    must_have: list[str]
    distractors: list[str]
    qc: dict[str, Any]
    planner: str
    glyph: str = ""
    technique: str = "ideographic"
    panels: int = 1
    split: str = ""
    source: str = ""
    # 降级标记（见 Plan.degraded）：true = 元素只是并排铺开，没有做语义合成
    degraded: bool = False
    degraded_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
