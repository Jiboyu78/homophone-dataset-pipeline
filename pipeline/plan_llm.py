"""plan 阶段（LLM 版，可选增强）：用本地/云端 LLM 生成高质量构图方案。

相比规则版，LLM 能处理三件规则版做不到的事：
    1) 多义字消歧（"人云亦云"的云不是云彩）
    2) 判断某条词根本不适合会意化，直接建议剔除（composition = "reject"）
    3) 写出更自然、更可控的英文提示词与更准确的难度

离线可用：Ollama / LM Studio / vLLM 皆可（OpenAI 兼容）。
调用失败会自动退回规则版，不会中断流水线。
"""
from __future__ import annotations

import json
from typing import Any

from .config import Config
from .lexicon import LEX
from .plan_rule import build_plan
from . import plan_rule
from .providers.openai_compat import (OpenAICompatError, build_llm_client, extract_json)
from .schema import COMPOSITION_TYPES, TECHNIQUE_ZH, Element, Item, Plan
from .state import State
from .utils import LOG

SYSTEM = (
    "你是中文谜语数据集的设计专家，专门设计『会意式单图猜成语/词语』的谜面："
    "用若干具体可见的元素及其排布关系，恰好表达一个成语或常用词语。"
    "你只输出 JSON 数组，不要任何解释文字，不要 markdown 代码块。"
)

# 这份规则与 skills/zh-rebus-design/SKILL.md 同源：把出题最容易犯的错直接写进提示词，
# 让大模型一步到位，而不是靠事后校验反复打回重做。
RULES = """每条记录输出一个对象，字段如下：
{
  "answer": "原词条，必须与输入完全一致",
  "composition": "literal|juxtapose|spatial|quantity|contrast|action|inner|metaphor|reject",
  "technique": "ideographic|glyph",
  "glyph": "字形参与式要写进画面的那一个汉字，否则空字符串",
  "elements": [{"morpheme":"原词中的字/词","zh":"中文","en":"英文名词短语","count":1,
                "role":"subject|object|action|modifier|background","position":"画面位置"}],
  "layout": "中文构图说明：元素怎么摆、什么关系是谜眼",
  "prompt_en": "英文生图提示词",
  "must_have": ["质检时用来核对的关键元素英文短语，与 elements 的 en 保持一致"],
  "difficulty": 1-5,
  "rationale": "中文说明：为什么这样构图能被猜出来",
  "distractors": ["3 个干扰答案，与答案同字数优先"],
  "degraded": false,
  "degraded_reason": "仅当 degraded=true 时填，说明为什么放弃语义合成"
}

【构图范式怎么选】
  literal   复合词逐字具象成一个物体：火车=火+车、河马=河里的马、冰箱=冰做的箱
  juxtapose 几个独立事物同框并列
  spatial   含 上/下/中/里/间/前/后 → 必须写出上下或内外分区
  quantity  含 一/二/三…/半/双/满 → 必须写出数量与排列方式（如 3 by 3 grid），
            并指明那个"单独的一个"在画面哪一侧
  contrast  含 大小/长短/有无/进退 等反义对 → 左右对照、镜像
  action    含动词（举/守/待/画/添/浇…）→ 主体+动作+对象，动作在画面中心且醒目
  inner     含 面/心、表/里、名/实 → 同一个人的影子或剖开处是另一副样子
  metaphor  全抽象、画不出具体元素 → 用一个具体物象征
  若这个词条无论如何都没法用具体元素表达（纯抽象概念、专名、地名、人名），
  只输出 {"answer":"原词条","composition":"reject","reason":"简短理由"}。

【元素表的硬规则——违反会被机械校验打回】
  1. 元素必须是答案里真实出现的字/词，且是能画出来的具体东西。
  2. 禁止把虚词（之/是/而/其/亦）当元素。
  3. 禁止把数词（九/一）、方位词（上/中）、时间词单列成元素——它们是修饰语：
     数量落到元素的 count 上，方位写进 layout 与 prompt_en。
  4. en 必须是"图像模型能直接画出来的具体名词短语"：
     a green coiled snake ✓ ／ snake ✗ ／ the concept of toil ✗
  5. 重复出现的同类元素只写一条，用 count 表示数量。
  6. 多义字必须按整词意思取义项，不要按字面直译：
     「人面兽心」的面是"脸"不是"面条"；「人云亦云」的云是"说"不是"云彩"；
     「马首是瞻」的"是"不入画；「胸有成竹」恰恰要画成胸口有竹子（谜面即字面）。

【字形参与式（technique="glyph"）什么时候用】
  只有词里有"承载语义、却画不出来"的实义字时才用，例：举手之劳的「劳」、
  见仁见智的「智」。虚词不算。此时 glyph 填那一个字，prompt_en 里写
  "one single large brush-written chinese character <字>"，
  并且不要要求整幅画面禁止汉字（该字由字体后期叠加，不靠模型写）。

【想不出怎么把元素合成一个画面时 —— 用降级策略，别硬编】
  非成语类词语（尤其是"修饰语+主体"的普通名词）常常没法合成出一个合理的画面。
  例：「刀鱼」是一种鱼，你没法让「刀」和「鱼」合成一个有意义的东西；
  「冰枕」可以合成（冰做的枕头），「刀鱼」不行。
  这种情况下**允许降级**：不合成，直接把元素并排铺开——画面左边一把刀、右边一条鱼。
  观众看到「刀 + 鱼」两个并列物体，配上语境也能读出答案。

  降级时：
    - composition 用 "juxtapose"
    - degraded 设为 true
    - degraded_reason 写清为什么合成不了（如「刀与鱼无语义关联，只能并列」）
    - prompt_en 里要明确写出并排关系（side by side / next to each other），
      并让两个物体大小相当、间距清楚，不要试图让它们发生互动
    - rationale 里说明这是降级处理，语义关联弱

  能合成时 degraded 必须为 false。**优先尝试合成**：只有真的想不出合理联系时才降级。
  判断标准：如果合成后的画面会让人困惑或与答案无关，就降级；
  如果合成后画面自然可读（火车=火+车、河马=河里的马），就不要降级。

【提示词写法】
  %s
  非字形模式必须体现"画面不得出现文字"。结构性构图（数量/方位/表里/对比/直译）
  必须在提示词里显式写出版面，否则图像模型会画错：
  数量写 "3 by 3 grid"，方位写 "upper half / lower half"，
  表里写 "his shadow is ..."，直译写 "made entirely of"。
"""

TAG_STYLE = ("目标是 SD1.5 类底模：prompt_en 必须是不超过 40 个词的逗号分隔短标签，"
             "顺序为 主体+数量+版面 → 风格 → 禁止文字。"
             "例：9 oxen arranged in a 3 by 3 grid filling the left side, one single "
             "hair alone on the far right side, flat cartoon illustration, bold black "
             "outlines, warm cream paper texture background, no text, no letters, "
             "single image")
PROSE_STYLE = ("目标是 SDXL / 云端模型：prompt_en 可以写完整句子，"
               "但必须是一个画面、一个焦点；风格描述放句尾，"
               "并以 no text, no letters, no watermark 结尾。")

SCHEMA_HINT = RULES


def _item_payload(item: Item) -> dict[str, Any]:
    return {
        "answer": item.text,
        "type": "成语" if item.type == "idiom" else "词语",
        "pinyin": item.pinyin,
        "category": item.category,
        "meaning": item.meaning,
        "参考可画语素": [s["morpheme"] for s in item.segments if s["drawable"]],
    }


def _build_messages(items: list[Item], cfg: Config | None = None) -> list[dict[str, Any]]:
    style = str((cfg.get("plan.prompt_style") if cfg else "prose") or "prose").lower()
    style_hint = TAG_STYLE if style == "tags" else PROSE_STYLE
    user = (f"请为下面 {len(items)} 个中文词条各设计一个『会意式单图』谜面。\n\n"
            + SCHEMA_HINT % style_hint
            + "\n\n待设计词条（JSON 数组）：\n"
            + json.dumps([_item_payload(i) for i in items], ensure_ascii=False, indent=1))
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}]


# 结构性构图：版面机制（3×3 网格、影子剪影、上下分区…）是规则版显式编码的，
# 8B 模型经常把它丢掉（实测：九牛一毛被写成"九头牛和一根毛并排"，
# 人面兽心被写成"人脸和兽脸并排"）。所以这几类以规则版场景为准。
STRUCTURAL_COMPOSITIONS = {"quantity", "inner", "spatial", "contrast"}
# 不该被当成"画面元素"的语素类型：数字/方位/度量是修饰语，虚词和抽象字画不出来
NON_ELEMENT_KINDS = {"number", "direction", "measure", "time", "sound",
                     "function", "abstract", "unknown"}


def _sanitize_elements(elems: list[Element], item: Item, segs: list[dict[str, Any]],
                       fallback: Plan) -> tuple[list[Element], list[str]]:
    """把大模型给的元素表按词典校验一遍，返回 (清洗后的元素, 被丢弃的说明)。

    实测 Qwen3-8B 会把虚词和数词也当成元素：马首是瞻 里冒出「是=is」，
    九牛一毛 里把「九」「一」列成独立元素。这类必须在入库前拦掉。
    """
    drawable_morphemes = {s["morpheme"] for s in segs if s["drawable"]}
    seg_kind = {s["morpheme"]: s["kind"] for s in segs}
    dropped: list[str] = []
    kept: list[Element] = []
    for e in elems:
        m = (e.morpheme or "").strip()
        if not m:
            dropped.append("(空元素)")
            continue
        if m not in item.text:
            dropped.append(f"{m}(不在词条中)")
            continue
        kind = seg_kind.get(m, "")
        if kind in NON_ELEMENT_KINDS:
            dropped.append(f"{m}({kind}，属修饰语/虚词)")
            continue
        if m not in drawable_morphemes:
            dropped.append(f"{m}(词典判为不可画)")
            continue
        kept.append(e)
    if len(kept) < 2:
        dropped.append("清洗后元素不足 2 个，整条回退到规则版")
        return list(fallback.elements), dropped
    return kept, dropped


def _to_plan(raw: dict[str, Any], item: Item, cfg: Config, planner: str) -> Plan | None:
    """把 LLM 返回的一条记录转成 Plan；交给规则版兜底不合适的字段。"""
    fallback = build_plan(item, cfg)
    comp = str(raw.get("composition", "")).strip()
    if comp == "reject":
        return None
    if comp not in COMPOSITION_TYPES:
        comp = fallback.composition

    elems: list[Element] = []
    for e in (raw.get("elements") or []):
        if isinstance(e, dict):
            elems.append(Element(
                morpheme=str(e.get("morpheme", "")),
                zh=str(e.get("zh", "")),
                en=str(e.get("en", "")),
                count=int(e.get("count", 1) or 1),
                role=str(e.get("role", "object")),
                position=str(e.get("position", "")),
                scale=str(e.get("scale", "")),
            ))
        elif isinstance(e, str):
            elems.append(Element(morpheme=e, zh=e, en=e))
    segs_now = LEX.segment(item.text, apply_suffix=(item.type == "word"))
    elems, dropped = _sanitize_elements(elems or list(fallback.elements),
                                       item, segs_now, fallback)
    if dropped:
        LOG.debug("「%s」清洗掉大模型元素：%s", item.text, "、".join(dropped))

    scene = str(raw.get("prompt_en") or "").strip()
    avoid = ("absolutely no text, no letters, no chinese characters, no numbers, "
             "no captions, no labels, no watermark, no signature")
    style_en = cfg.get("plan.style_en", "")

    # 结构性构图以规则版为准：大模型给的"九头牛并排"这类描述会丢掉版面机制
    prefer_rule = [str(x) for x in (cfg.get("plan.llm.prefer_rule_compositions")
                                    or list(STRUCTURAL_COMPOSITIONS))]
    scene_source = "llm"
    if comp in prefer_rule and fallback.extra.get("scene_en"):
        scene = fallback.extra["scene_en"]
        elems = list(fallback.elements)      # 规则版的 count/position 更准
        scene_source = "rule"
        LOG.debug("「%s」属结构性构图 %s，场景采用规则版", item.text, comp)

    if scene:
        prompt_en = f"{scene.rstrip('.')}. {style_en}. {avoid}."
    else:
        scene = fallback.extra.get("scene_en", "")
        scene_source = "rule"
        prompt_en = fallback.prompt_en

    difficulty = raw.get("difficulty", fallback.difficulty)
    try:
        difficulty = max(1, min(5, int(difficulty)))
    except (TypeError, ValueError):
        difficulty = fallback.difficulty

    # 字形参与式：LLM 可以自己声明，但**必须通过规则版的判据校验**。
    # Qwen3-8B 这类 8B 模型很容易给"仓鼠/光剑"这种词也塞一个 glyph，直接信它会出错。
    glyph = str(raw.get("glyph") or "").strip()
    if len(glyph) > 1:
        glyph = glyph[0]
    allowed = plan_rule.glyph_candidates(
        item, LEX.segment(item.text, apply_suffix=(item.type == "word")))
    if glyph and glyph not in allowed:
        extra_note = f"（LLM 建议的字形「{glyph}」未通过校验，已忽略）"
        LOG.info("忽略 LLM 声明的字形：%s -> %s %s", item.text, glyph, extra_note)
        glyph = ""
    if not glyph:
        glyph = fallback.glyph
    technique = str(raw.get("technique") or "").strip()
    if technique not in ("ideographic", "glyph", "homophone"):
        technique = "glyph" if glyph else fallback.technique
    if glyph and technique == "ideographic":
        technique = "glyph"
    if not glyph and technique == "glyph":
        technique = "ideographic"
    try:
        panels = max(1, int(raw.get("panels", fallback.panels)))
    except (TypeError, ValueError):
        panels = fallback.panels

    prompt_en = fallback.prompt_en if glyph and fallback.glyph == glyph else prompt_en
    # scene_en 必须跟真正用于出图的那段文字一致，否则数据集里的描述会与实际提示词对不上
    extra = {**fallback.extra, "llm_reason": raw.get("reason", ""),
             "scene_en": scene or fallback.extra.get("scene_en", ""),
             "scene_source": scene_source,
             "llm_prompt_en": str(raw.get("prompt_en", "")).strip(),
             "dropped_elements": dropped}
    if glyph:
        extra["glyph"] = fallback.extra.get("glyph") or {
            "char": glyph, "mode": "composite", "position": "center-left"}
        extra["technique_zh"] = TECHNIQUE_ZH.get(technique, technique)

    # 降级标记：以大模型自报为准；它没填时，对"非成语 + 并列式"做保守推定。
    # 词语类往往确实没有可合成的语义联系（刀鱼 = 刀 + 鱼），并列铺开就是降级；
    # 成语用并列式常常是本来就这么设计的（杯弓蛇影），所以只对词语推定。
    degraded = bool(raw.get("degraded", False))
    degraded_reason = str(raw.get("degraded_reason", "") or "").strip()
    if not degraded and item.type == "word" and comp == "juxtapose":
        degraded = True
        degraded_reason = degraded_reason or "（自动判定）词语类且为并列式，未做语义合成"
    if degraded and not degraded_reason:
        degraded_reason = "元素无语义关联，并列铺开"

    return Plan(
        item_id=item.id,
        answer=item.text,
        composition=comp,
        elements=elems,
        layout=str(raw.get("layout", "")) or fallback.layout,
        prompt_zh=fallback.prompt_zh,
        prompt_en=prompt_en,
        negative_prompt=cfg.get("plan.negative_prompt", ""),
        must_have=[str(x) for x in (raw.get("must_have") or [e.en for e in elems if e.en])],
        must_not=["其他文字", "字母", "数字", "水印", "签名", "多格/拼贴"],
        difficulty=difficulty,
        distractors=[],
        rationale=str(raw.get("rationale", "")) or fallback.rationale,
        planner=planner,
        glyph=glyph,
        technique=technique,
        panels=panels,
        degraded=degraded,
        degraded_reason=degraded_reason,
        extra=extra,
    )


def _hard_subset(cfg: Config, state: State, max_n: int = 200,
                 only: set[str] | None = None) -> list[Item]:
    """挑出"规则版容易出错、值得花大模型算力"的难点词。

    判据全部来自 select/plan 阶段已经算好的信号，不需要额外推理：
      · 多义字风险（两个以上高频多义字）—— 规则版最典型的失败点
      · 含否定字（不/无/非…）—— 画面难表达"没有"
      · 需要字形参与（字形承载语义）
      · 构图落到"具象隐喻"（纯抽象，只靠词典画不出来）
      · 含词典外字（不在元素词典里，规则版只能丢）
    """
    items = state.load_items(status=["accepted"], order_by="drawable_score DESC, text")
    if only:
        items = [it for it in items if it.text in only]
    scored: list[tuple[float, Item]] = []
    for it in items:
        plan = state.get_plan(it.id)
        if plan is None:
            continue
        flags = " ".join(plan.extra.get("flags", []) or [])
        n_unknown = sum(1 for s in it.segments if s["kind"] == "unknown")
        n_abstract = sum(1 for s in it.segments if s["kind"] == "abstract")
        score = 0.0
        if "polysemy_risk" in flags:
            score += 3.0
        if "negation:" in flags:
            score += 2.0
        if plan.technique == "glyph":
            score += 1.5
        if plan.composition == "metaphor":
            score += 2.5
        score += 0.8 * n_unknown + 0.5 * n_abstract
        if len(it.text) >= 5:
            score += 0.5
        if score > 0:
            scored.append((score, it))
    scored.sort(key=lambda x: (-x[0], x[1].text))
    picked = [it for _s, it in scored[:max_n]]
    LOG.info("难点词筛选：%d 条候选（满分样本：%s）", len(picked),
             "、".join(it.text for _s, it in scored[:6]))
    return picked


class LLMUnavailable(RuntimeError):
    """描述模型不可用，且配置要求不得回退到规则出题器。"""


def run(cfg: Config, state: State, limit: int = 0, redo: bool = False,
        only: set[str] | None = None) -> int:
    provider = str(cfg.get("plan.provider", "rule")).lower()
    if provider not in ("llm", "hybrid"):
        LOG.info("plan.provider=%s，跳过 LLM 出题", provider)
        return 0
    if redo:
        state.reset_stage("plan")

    client = build_llm_client(cfg)
    ok, msg = client.health()
    if not ok:
        # 一键云端流程里，静默回退到规则出题器会毁掉整个价值（用户买的就是大模型
        # 会读语义、能避多义字坑）。所以提供 plan.fallback_to_rule=false：
        # 大模型不可用时直接失败退出，而不是生成一批规则版糊弄过去。
        if not bool(cfg.get("plan.fallback_to_rule", True)):
            raise LLMUnavailable(
                f"描述模型不可用：{msg}\n"
                f"请检查 plan.llm.base_url / api_key / model 是否正确、"
                f"以及网络能否访问该地址；\n"
                f"（若你就是想用本地规则出题器，把 plan.provider 改成 rule）")
        LOG.error("LLM 不可用（%s），回退到规则出题器", msg)
        from . import plan_rule
        return plan_rule.run(cfg, state, limit=limit)

    if provider == "hybrid":
        # 混合模式：先全量规则出题（快、免费），再挑难点词交给大模型重做
        n_rule = plan_rule.run(cfg, state, limit=limit, redo=redo, only=only)
        max_n = int(cfg.get("plan.llm.hard_max", 200))
        items = _hard_subset(cfg, state, max_n=max_n, only=only)
        LOG.info("混合出题：规则版 %d 条，其中 %d 条难点词改由大模型重做",
                 n_rule, len(items))
    else:
        items = state.items_without_plan(limit=0)
        if only:
            items = [it for it in items if it.text in only]
        if limit and limit > 0:
            items = items[:limit]
    if not items:
        LOG.info("plan(llm)：没有待生成的词条")
        return 0

    batch = max(1, int(cfg.get("plan.llm.batch", 8)))
    done = 0
    rejected = 0
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        try:
            raw = client.chat(_build_messages(chunk, cfg), json_mode=True, max_tokens=8192)
            data = extract_json(raw)
        except (OpenAICompatError, ValueError) as exc:
            LOG.warning("LLM 出题失败（第 %d 批），退回规则版：%s", i // batch + 1, exc)
            data = []
        if isinstance(data, dict):
            data = list(data.values())[0] if len(data) == 1 else data.get("plans", [])
        if not isinstance(data, list):
            data = []
        by_answer = {str(d.get("answer", "")).strip(): d for d in data if isinstance(d, dict)}

        for item in chunk:
            raw_one = by_answer.get(item.text)
            plan = _to_plan(raw_one, item, cfg, "llm") if raw_one else None
            if plan is None:
                if raw_one and str(raw_one.get("composition")) == "reject":
                    item.select_status = "rejected"
                    item.select_reason = f"LLM 判定不适合会意化：{raw_one.get('reason', '')[:80]}"
                    state.update_select_bulk([item])
                    rejected += 1
                    continue
                plan = build_plan(item, cfg)   # 兜底
            state.save_plan(plan)
            done += 1
        LOG.info("plan(llm) 进度 %d/%d（本批命中 %d）", min(i + batch, len(items)),
                 len(items), len(by_answer))

    LOG.info("plan(llm) 完成：生成 %d 个方案，LLM 建议剔除 %d 条", done, rejected)
    return done
