"""端到端自测：不需要 GPU、不需要真实大模型、不联网。

运行：
    python scripts/selftest.py              # 全部
    python scripts/selftest.py -v           # 显示每条用例
    python scripts/selftest.py QcTests      # 只跑某个类

覆盖范围：
    ingest 字段识别 / 去重 / 稳定 ID
    lexicon 分词与多义字消歧
    select 硬规则
    plan   模板不变量（冠词、重复词、必备片段）与 LLM 出题链路
    qc     技术质检 / OCR 缺失降级 / VLM 重试闭环 / 反馈生成
    generate 反馈回灌与提示词缓存
    ComfyUI 工作流注入
    package  字段完整性 / 划分 / 干扰项
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# 放在工作区内，避免受系统临时目录权限限制；跑完自动删除
TMP = ROOT / ".selftest-tmp"
if TMP.exists():
    shutil.rmtree(TMP, ignore_errors=True)
TMP.mkdir(parents=True, exist_ok=True)

# 覆盖全部构图范式的小词表
WORDS = [
    "画蛇添足", "守株待兔", "三长两短", "火上浇油", "有眼无珠", "杯弓蛇影",
    "瓜田李下", "一石二鸟", "亡羊补牢", "井蛙之见", "对牛弹琴", "画龙点睛",
    "笼中之鸟", "指鹿为马", "铁案如山", "亭台楼阁", "米饭", "香醋", "绿豆",
    "河马", "雨伞", "冰箱", "鸡腿", "雪糕", "火车", "菜包",
]

WORDS_JSON = TMP / "words.json"
DB_PATH = TMP / "state.db"
OUT_DIR = TMP / "out"


def _write_fixture() -> None:
    rows = [{"id": f"{i:05d}", "答案词": w, "拼音": "mock pin1 yin1",
             "类型": "常用词语" if len(w) <= 3 else "成语"}
            for i, w in enumerate(WORDS)]
    WORDS_JSON.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


_write_fixture()

CFG_DICT = {
    "paths": {"data_dir": str(TMP), "output_dir": str(OUT_DIR),
              "images_dir": str(OUT_DIR / "images"),
              "log_file": str(OUT_DIR / "t.log"), "db": str(DB_PATH)},
    "ingest": {"files": [str(WORDS_JSON)]},
    "select": {"min_score": 0.5, "min_coverage": 0.5},
    "plan": {"provider": "rule", "width": 256, "height": 256},
    "image": {"provider": "mock", "mock": {"width": 256, "height": 256,
                                           "draw_labels": True}},
    "qc": {"ocr": {"enabled": False}, "vlm": {"provider": "none"},
           "min_accept_score": 0.75, "max_retry": 2,
           "technical": {"min_side": 64}},
    "package": {"splits": {"train": 0.5, "val": 0.25, "test": 0.25}, "seed": 1,
                "contact_sheets": False},
}

from pipeline.config import Config, build_config  # noqa: E402
from pipeline import generate, ingest, package, plan_llm, plan_rule, qc, select  # noqa: E402
from pipeline.lexicon import LEX, lexicon_hash  # noqa: E402
from pipeline.providers.image_base import ImageProvider, ImageResult  # noqa: E402
from pipeline.providers.image_comfyui import ComfyUIProvider  # noqa: E402
from pipeline.providers.openai_compat import extract_json  # noqa: E402
from pipeline.schema import QCResult  # noqa: E402
from pipeline.state import State  # noqa: E402
from pipeline.utils import pinyin_tone_marks, stable_id  # noqa: E402

from PIL import Image  # noqa: E402


_SEQ = [0]


def make_cfg(**overrides) -> Config:
    """每个用例拿到独立的 DB / 输出目录，避免 Windows 上 SQLite 文件被占用。"""
    _SEQ[0] += 1
    raw = json.loads(json.dumps(CFG_DICT))
    raw["paths"]["db"] = str(TMP / f"state_{_SEQ[0]}.db")
    raw["paths"]["output_dir"] = str(TMP / f"out_{_SEQ[0]}")
    raw["paths"]["images_dir"] = str(TMP / f"out_{_SEQ[0]}" / "images")
    raw["paths"]["log_file"] = str(TMP / f"out_{_SEQ[0]}" / "t.log")
    for dotted, value in overrides.items():
        cur = raw
        parts = dotted.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value
    # 走 build_config 而不是自己 merge：否则画风预设不会被应用，
    # 测试环境和真实运行环境就会不一致。
    return build_config(raw, "<selftest>")


def fresh_state(cfg: Config) -> State:
    """新建状态库（路径在 make_cfg 里已保证唯一）。"""
    return State(cfg.db_path)


class RecordingProvider(ImageProvider):
    """记录每次调用收到的提示词，用来验证"质检反馈是否回灌"。"""

    name = "recording"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.calls: list[dict] = []

    def generate(self, item, plan, out_path, seed=-1, feedback=""):
        self.calls.append({"answer": item.text, "feedback": feedback,
                           "prompt": self.compose_prompt(plan, feedback)})
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (128, 128), (200, 210, 220)).save(out_path)
        return ImageResult(path=out_path, seed=1, provider=self.name)


class StubVLM:
    """可编程的假 VLM：用来精确验证 retry -> accept 的判定与反馈。

    text_leak_calls: 前 N 次核验都报"画面有文字"（模拟最难缠的失败模式）
    guess:           盲猜时给出的答案（模拟"猜对了"或"猜不出来"）
    """

    def __init__(self, text_leak_calls: int = 0, guess: str = "", guess_ok: bool = True):
        self.enabled = True
        self.check_solvability = True
        self.n = 0
        self.text_leak_calls = text_leak_calls
        self.guess = guess
        self.guess_ok = guess_ok

    def health(self):
        return True, "stub"

    def verify(self, image_path, answer, elements, composition, allowed_text=None):
        self.n += 1
        leak = self.n <= self.text_leak_calls
        return {"text_present": leak, "text_content": "文字" if leak else "",
                "elements_present": list(elements),
                "elements_missing": [] if not leak else ["missing thing"],
                "matches_answer": True, "problems": [], "confidence": 0.9}

    def solve(self, image_path):
        return {"guess": self.guess if self.guess_ok else "完全不搭界的答案",
                "objects": [], "layout": "", "confidence": 0.5}


# --------------------------------------------------------------------------- #
class IngestTests(unittest.TestCase):
    def test_field_detection_and_counts(self):
        cfg = make_cfg()
        st = fresh_state(cfg)
        n1 = ingest.run(cfg, st)
        self.assertEqual(n1, len(WORDS))
        counts = st.counts()
        self.assertEqual(counts["items"], len(WORDS))
        # 再跑一次必须幂等（历史上用过 hash() 生成 id 会导致重复）
        n2 = ingest.run(cfg, st)
        self.assertEqual(st.counts()["items"], len(WORDS), "ingest 不幂等，出现重复条目")
        self.assertEqual(n2, len(WORDS))
        row = st.load_items(order_by="id")[0]
        self.assertEqual(row.id, stable_id(row.type[:3], row.text, 12))
        st.close()

    def test_detects_english_and_chinese_keys(self):
        f = TMP / "mixed.json"
        f.write_text(json.dumps([
            {"word": "画蛇添足", "pinyin": "hua4 she2", "meaning": "比喻多此一举"},
            {"成语": "守株待兔", "拼音": "shou3 zhu1", "释义": "比喻死守经验"},
        ], ensure_ascii=False), encoding="utf-8")
        rows = ingest.load_json_lenient(f)
        cols = ingest.detect_columns(rows)
        self.assertTrue(cols["text"])
        self.assertIn(cols["text"], ("word", "成语"))


class LexiconTests(unittest.TestCase):
    def test_segmentation(self):
        segs = LEX.segment("画蛇添足")
        self.assertEqual([s["morpheme"] for s in segs], ["画", "蛇", "添", "足"])
        self.assertTrue(all(s["drawable"] for s in segs))

    def test_polysemy_sense(self):
        # 香 的主义项是"香烛"（object），但在"香醋"里应取"芳香的"
        self.assertEqual(LEX.entries["香"]["kind"], "object")
        alt = LEX.sense_in("香", {"property", "color", "dimension"})
        self.assertIsNotNone(alt)
        self.assertEqual(alt["en"], "fragrant")
        self.assertIn("color", LEX.senses_of("红"))

    def test_function_chars_excluded_from_content(self):
        segs = LEX.segment("无中生有")
        fn = [s for s in segs if s["mode"] == "no"]
        self.assertTrue(fn, "虚词应被标记为不可画")
        self.assertFalse(any(s["drawable"] for s in fn))

    def test_lexicon_hash_stable(self):
        self.assertEqual(lexicon_hash(), lexicon_hash())
        self.assertEqual(len(lexicon_hash()), 12)


class SelectTests(unittest.TestCase):
    def setUp(self):
        self.cfg = make_cfg()
        self.st = fresh_state(self.cfg)
        ingest.run(self.cfg, self.st)
        select.run(self.cfg, self.st)
        self.by_text = {i.text: i for i in self.st.load_items()}

    def tearDown(self):
        self.st.close()

    def test_same_kind_compound_rejected(self):
        it = self.by_text["米饭"]
        self.assertEqual(it.select_status, "rejected")
        self.assertIn("同义并列", it.select_reason)

    def test_attribute_pair_accepted_as_quantity(self):
        it = self.by_text["三长两短"]
        self.assertEqual(it.select_status, "accepted")
        self.assertEqual(it.composition, "quantity")

    def test_literal_compound(self):
        self.assertEqual(self.by_text["河马"].composition, "literal")
        self.assertEqual(self.by_text["火车"].composition, "literal")

    def test_contrast_detected(self):
        self.assertEqual(self.by_text["有眼无珠"].composition, "contrast")

    def test_spatial_detected(self):
        self.assertIn(self.by_text["火上浇油"].composition, ("spatial", "quantity"))

    def test_rescore_triggered_by_lexicon_change(self):
        self.st.set_meta("select_lexicon_hash", "stale")
        before = self.st.counts()["select:accepted"]
        select.run(self.cfg, self.st)       # 不带 rescore 也应因指纹变化而重算
        self.assertEqual(self.st.get_meta("select_lexicon_hash"), lexicon_hash())
        self.assertEqual(self.st.counts()["select:accepted"], before)


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = make_cfg()
        cls.st = fresh_state(cls.cfg)
        ingest.run(cls.cfg, cls.st)
        select.run(cls.cfg, cls.st)
        plan_rule.run(cls.cfg, cls.st)
        cls.items = cls.st.load_items(status=["accepted"])

    @classmethod
    def tearDownClass(cls):
        cls.st.close()

    def test_every_accepted_has_plan(self):
        for it in self.items:
            self.assertIsNotNone(self.st.get_plan(it.id), f"{it.text} 缺方案")

    def test_prompt_invariants(self):
        bad = []
        for it in self.items:
            p = self.st.get_plan(it.id)
            low = p.prompt_en.lower()
            self.assertIn("no text", low)
            if "identical identical" in low:
                bad.append((it.text, "identical 重复"))
            if " a a " in low or low.count("an a ") or " a an " in low:
                bad.append((it.text, "冠词重复"))
            if "  " in p.prompt_en:
                bad.append((it.text, "多余空格"))
            if not p.must_have:
                bad.append((it.text, "must_have 为空"))
            names = [e.morpheme for e in p.elements]
            if len(names) != len(set(names)):
                bad.append((it.text, f"元素重复 {names}"))
            if not (1 <= p.difficulty <= 5):
                bad.append((it.text, "难度越界"))
        self.assertEqual(bad, [], f"提示词/元素不变量被破坏：{bad[:6]}")

    def test_literal_templates_read_naturally(self):
        want = {
            "河马": "in the middle of a river",
            "绿豆": "green beans",
            "冰箱": "made entirely of ice",
            "鸡腿": "the leg of",
        }
        for text, frag in want.items():
            it = next((i for i in self.items if i.text == text), None)
            if it is None:
                continue
            p = self.st.get_plan(it.id)
            self.assertIn(frag, p.extra["scene_en"], f"{text} 场景模板不符合预期")

    def test_llm_plan_path_with_mock_server(self):
        from mock_llm_server import start_server
        httpd, base = start_server(0)
        try:
            cfg = make_cfg(**{"plan.provider": "llm"})
            cfg.set("plan.llm.base_url", base)
            cfg.set("plan.llm.model", "mock-llm")
            cfg.set("plan.llm.batch", 4)
            st = fresh_state(cfg)
            ingest.run(cfg, st)
            select.run(cfg, st)
            n = plan_llm.run(cfg, st)
            self.assertGreater(n, 0, "LLM 出题链路没有产出任何方案")
            plans = [st.get_plan(i.id) for i in st.load_items(status=["accepted"])]
            plans = [p for p in plans if p]
            self.assertTrue(plans)
            self.assertTrue(any(p.planner == "llm" for p in plans),
                            "没有任何方案标记为 llm 出题，说明请求/解析链路断了")
            llm_plan = next(p for p in plans if p.planner == "llm")
            self.assertTrue(llm_plan.elements)
            self.assertIn("no text", llm_plan.prompt_en.lower())
            st.close()
        finally:
            httpd.shutdown()

    def test_llm_falls_back_when_server_down(self):
        cfg = make_cfg(**{"plan.provider": "llm"})
        cfg.set("plan.llm.base_url", "http://127.0.0.1:1/v1")   # 必然连不上
        cfg.set("plan.llm.timeout_s", 2)
        st = fresh_state(cfg)
        ingest.run(cfg, st)
        select.run(cfg, st)
        n = plan_llm.run(cfg, st)
        self.assertGreater(n, 0, "LLM 不可用时没有回退到规则版")
        plan = next(p for p in (st.get_plan(i.id) for i in st.load_items(status=["accepted"])) if p)
        self.assertEqual(plan.planner, "rule")
        st.close()


class QcTests(unittest.TestCase):
    def setUp(self):
        self.cfg = make_cfg()
        self.st = fresh_state(self.cfg)
        ingest.run(self.cfg, self.st)
        select.run(self.cfg, self.st)
        plan_rule.run(self.cfg, self.st)

    def tearDown(self):
        self.st.close()

    def _images(self, n=6):
        generate.run(self.cfg, self.st, limit=n)
        return self.st.items_needing_qc()

    def test_blank_image_rejected(self):
        from pipeline.providers.ocr import OCRChecker
        from pipeline.providers.vlm_openai import VLMChecker
        img = TMP / "blank.png"
        Image.new("RGB", (256, 256), (255, 255, 255)).save(img)
        row = {"id": "x", "item_id": "y", "path": str(img), "attempt": 1, "provider": "test"}
        res = qc.check_image(self.cfg, row, ["a cat"], "画蛇添足", "action",
                             1, 3, OCRChecker(self.cfg), VLMChecker(self.cfg), {})
        self.assertEqual(res.verdict, "reject")
        self.assertEqual(res.score, 0.0)
        self.assertTrue(any("空白" in i or "纯色" in i for i in res.issues))

    def test_mock_images_accepted_without_vlm(self):
        rows = self._images(6)
        self.assertTrue(rows)
        res = qc.run(self.cfg, self.st, limit=6)
        self.assertGreater(res["checked"], 0)
        stats = self.st.counts()
        self.assertGreater(stats.get("qc:accept", 0), 0, "无 VLM 时技术质检通过的图应被接受")
        acc = self.st.get_qc(rows[0]["id"])
        self.assertEqual(acc.checker, "technical")
        self.assertIn("未启用 VLM", " ".join(acc.issues))

    def test_vlm_stub_retry_then_accept_and_feedback(self):
        from pipeline.providers.ocr import OCRChecker
        rows = self._images(2)
        row = rows[0]
        item = self.st.get_item(row["item_id"])
        plan = self.st.get_plan(row["item_id"])
        seen: dict = {}

        stub = StubVLM(text_leak_calls=1, guess=item.text)
        r1 = qc.check_image(self.cfg, row, [e.en for e in plan.elements], item.text,
                            plan.composition, 1, 3, OCRChecker(self.cfg), stub, seen)
        self.assertEqual(r1.verdict, "retry", f"首次报文字应触发重试，实际 {r1.verdict}")
        self.assertFalse(r1.text_free)
        self.assertTrue(r1.feedback_for_retry)
        self.assertIn("text", r1.feedback_for_retry.lower())

        # 第二轮：假 VLM 不再报文字，并且"猜对了"
        stub2 = StubVLM(text_leak_calls=0, guess=item.text)
        r2 = qc.check_image(self.cfg, row, [e.en for e in plan.elements], item.text,
                            plan.composition, 2, 3, OCRChecker(self.cfg), stub2, seen)
        self.assertEqual(r2.verdict, "accept")
        self.assertTrue(r2.text_free)
        self.assertIs(r2.solvable, True)
        self.assertGreaterEqual(r2.score, self.cfg.get("qc.min_accept_score"))

    def test_text_leak_at_last_attempt_is_rejected(self):
        """文字泄漏可以靠重试救，但重试次数用完后仍带文字就必须淘汰。"""
        from pipeline.providers.ocr import OCRChecker
        rows = self._images(2)
        row = rows[0]
        item = self.st.get_item(row["item_id"])
        plan = self.st.get_plan(row["item_id"])
        stub = StubVLM(text_leak_calls=99, guess=item.text)
        args = (self.cfg, row, [e.en for e in plan.elements], item.text,
                plan.composition)
        r1 = qc.check_image(*args, 1, 3, OCRChecker(self.cfg), stub, {})
        self.assertEqual(r1.verdict, "retry", "第 1 轮应给重试机会")
        r3 = qc.check_image(*args, 3, 3, OCRChecker(self.cfg), stub, {})
        self.assertEqual(r3.verdict, "reject", "到顶仍带文字应淘汰而不是转人工")

    def test_vlm_path_against_mock_server(self):
        from pipeline.providers.vlm_openai import VLMChecker
        from mock_llm_server import start_server
        httpd, base = start_server(0)
        try:
            cfg = make_cfg(**{"qc.vlm.provider": "openai"})
            cfg.set("qc.vlm.base_url", base)
            cfg.set("qc.vlm.model", "mock-vlm")
            st = fresh_state(cfg)
            ingest.run(cfg, st)
            select.run(cfg, st)
            plan_rule.run(cfg, st)
            generate.run(cfg, st, limit=8)
            res = qc.run(cfg, st, limit=8)
            self.assertGreater(res["checked"], 0)
            row = st.all_qc()[0]
            detail = json.loads(row["result_json"])
            self.assertIn("vlm_verify", detail["technical"],
                          "VLM 核验结果没有落库，说明视觉请求链路断了")
            self.assertIn("vlm_solve", detail["technical"], "可解性测试没有执行")
            self.assertIn("vlm", detail["checker"])
            st.close()
        finally:
            httpd.shutdown()


class GenerateTests(unittest.TestCase):
    def test_feedback_roundtrip_and_prompt_hash(self):
        cfg = make_cfg()
        st = fresh_state(cfg)
        ingest.run(cfg, st)
        select.run(cfg, st)
        plan_rule.run(cfg, st)
        prov = RecordingProvider(cfg)
        todo = st.items_needing_image(max_attempt=3, limit=1, balanced=True)
        item, plan, attempt = todo[0]
        generate._gen_one(cfg, st, prov, item, plan, attempt)   # noqa: SLF001
        self.assertEqual(prov.calls[0]["feedback"], "")
        self.assertNotIn("CORRECTION", prov.calls[0]["prompt"])

        # 写入一条"重试 + 反馈"，模拟质检结论
        img = st.list_images(item.id)[0]
        st.save_qc(img["id"], item.id,
                   QCResult(item_id=item.id, verdict="retry", score=0.4,
                            feedback_for_retry="remove every trace of text"),
                   1)
        todo2 = st.items_needing_image(max_attempt=3, limit=5, balanced=True)
        match = [t for t in todo2 if t[0].id == item.id]
        self.assertTrue(match, "retry 结论的条目应重新进入出图队列")
        _item, plan2, attempt2 = match[0]
        self.assertEqual(attempt2, 2)
        generate._gen_one(cfg, st, prov, _item, plan2, attempt2)  # noqa: SLF001
        self.assertIn("remove every trace of text", prov.calls[1]["feedback"])
        self.assertIn("CORRECTION", prov.calls[1]["prompt"])
        self.assertNotEqual(st.list_images(item.id)[0]["prompt_hash"],
                            st.list_images(item.id)[1]["prompt_hash"])
        st.close()

    def test_reject_does_not_requeue(self):
        cfg = make_cfg()
        st = fresh_state(cfg)
        ingest.run(cfg, st)
        select.run(cfg, st)
        plan_rule.run(cfg, st)
        prov = RecordingProvider(cfg)
        item, plan, attempt = st.items_needing_image(max_attempt=3, limit=1, balanced=True)[0]
        generate._gen_one(cfg, st, prov, item, plan, attempt)   # noqa: SLF001
        img = st.list_images(item.id)[0]
        st.save_qc(img["id"], item.id,
                   QCResult(item_id=item.id, verdict="reject", score=0.0), 1)
        requeue = [t for t in st.items_needing_image(max_attempt=3, balanced=True)
                   if t[0].id == item.id]
        self.assertEqual(requeue, [], "硬失败(reject)的条目不该再消耗算力重出图")
        st.close()


class ComfyUITests(unittest.TestCase):
    def test_workflow_patch(self):
        cfg = make_cfg()
        cfg.set("image.comfyui.workflow", str(ROOT / "workflows" / "sd15_txt2img_api.json"))
        cfg.set("image.comfyui.width", 640)
        cfg.set("image.comfyui.height", 480)
        cfg.set("image.comfyui.steps", 22)
        cfg.set("image.comfyui.cfg", 7.5)
        prov = ComfyUIProvider(cfg)
        item = None
        from pipeline.schema import Element, Item, Plan
        item = Item(id="t1", text="画蛇添足", type="idiom")
        plan = Plan(item_id="t1", answer="画蛇添足", composition="action",
                    elements=[Element(morpheme="蛇", en="a snake")],
                    prompt_en="a snake", negative_prompt="text, watermark")
        wf = prov.patch_workflow(plan, seed=1234)

        self.assertEqual(wf["5"]["inputs"]["width"], 640)
        self.assertEqual(wf["5"]["inputs"]["height"], 480)
        self.assertEqual(wf["3"]["inputs"]["seed"], 1234)
        self.assertEqual(wf["3"]["inputs"]["steps"], 22)
        self.assertEqual(wf["3"]["inputs"]["cfg"], 7.5)
        self.assertIn("a snake", wf["6"]["inputs"]["text"])
        self.assertIn("no text", wf["6"]["inputs"]["text"].lower())
        self.assertIn("chinese characters", wf["7"]["inputs"]["text"])
        self.assertNotEqual(wf["6"]["inputs"]["text"], wf["7"]["inputs"]["text"])

    def test_ui_format_workflow_rejected(self):
        bad = TMP / "ui_format.json"
        bad.write_text(json.dumps({"nodes": [], "links": []}), encoding="utf-8")
        cfg = make_cfg()
        cfg.set("image.comfyui.workflow", str(bad))
        prov = ComfyUIProvider(cfg)
        from pipeline.schema import Plan
        with self.assertRaises(Exception):
            prov.patch_workflow(Plan(item_id="x", answer="x"), seed=1)


class PackageTests(unittest.TestCase):
    def test_dataset_fields(self):
        cfg = make_cfg()
        st = fresh_state(cfg)
        ingest.run(cfg, st)
        select.run(cfg, st)
        plan_rule.run(cfg, st)
        generate.run(cfg, st, limit=12)
        qc.run(cfg, st, limit=12)
        res = package.run(cfg, st)
        self.assertGreater(res["records"], 0)

        out_dir = cfg.output_dir
        lines = (out_dir / "dataset.jsonl").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), res["records"])
        required = {"id", "answer", "pinyin", "type", "composition", "composition_zh",
                    "difficulty", "image", "elements", "must_have", "distractors",
                    "qc", "prompt_en", "split"}
        for line in lines:
            rec = json.loads(line)
            self.assertTrue(required <= set(rec), f"缺字段：{required - set(rec)}")
            self.assertTrue((out_dir / rec["image"]).exists(), f"图片缺失 {rec['image']}")
            self.assertEqual(len(rec["distractors"]), 3)
            self.assertNotIn(rec["answer"], rec["distractors"])
            self.assertTrue(rec["pinyin"])
        self.assertTrue((out_dir / "stats.md").exists())
        self.assertTrue((out_dir / "manifest.csv").exists())
        splits = {json.loads(x)["split"] for x in lines}
        self.assertTrue(splits)
        st.close()


class OcrTests(unittest.TestCase):
    """OCR 文字泄漏检测。未安装 OCR 引擎时自动跳过（不视为失败）。"""

    @classmethod
    def setUpClass(cls):
        from pipeline.providers.ocr import OCRChecker
        cls.cfg = make_cfg(**{"qc.ocr.enabled": True,
                              "qc.ocr.skip_providers": []})
        cls.checker = OCRChecker(cls.cfg)
        cls.checker._load()  # noqa: SLF001
        if not cls.checker.available:
            raise unittest.SkipTest("未安装 OCR 引擎（pip install --target .deps rapidocr-onnxruntime）")
        cls.dir = TMP / "ocr"
        cls.dir.mkdir(parents=True, exist_ok=True)

    def _render(self, name: str, draw_text: bool, text: str = "画蛇添足") -> Path:
        from PIL import ImageDraw, ImageFont
        p = self.dir / f"{name}.png"
        img = Image.new("RGB", (512, 512), (245, 244, 240))
        d = ImageDraw.Draw(img)
        d.ellipse([70, 90, 230, 250], fill=(200, 170, 160), outline=(90, 90, 100), width=3)
        d.rectangle([270, 110, 430, 270], fill=(170, 200, 180), outline=(90, 90, 100), width=3)
        if draw_text:
            try:
                font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 56)
            except OSError:
                font = ImageFont.load_default()
            d.text((150, 330), text, font=font, fill=(40, 40, 50))
        img.save(p)
        return p

    def test_clean_shapes_are_not_flagged(self):
        """几何图形被误读成单个字母（圆->O）不应判为文字泄漏。"""
        out = self.checker.detect(self._render("clean", draw_text=False))
        self.assertIs(out["has_text"], False, f"误报：{out.get('texts')}")

    def test_chinese_text_is_flagged(self):
        out = self.checker.detect(self._render("zh", draw_text=True))
        self.assertIs(out["has_text"], True, f"漏检：{out.get('texts')}")
        joined = "".join(t for t, _ in out["texts"])
        self.assertIn("画蛇", joined.replace(" ", ""))

    def test_weak_evidence_scoring(self):
        from pipeline.providers.ocr import score_text_evidence
        self.assertFalse(score_text_evidence([("O", 0.9)])["effective_chars"] > 0.5)
        self.assertTrue(score_text_evidence([("CAT", 0.9)])["effective_chars"] > 0.5)
        self.assertTrue(score_text_evidence([("画蛇添足", 0.9)])["effective_chars"] > 0.5)
        self.assertTrue(score_text_evidence([("O", 0.9), ("X", 0.9), ("0", 0.9)])
                        ["effective_chars"] > 0.5)


class GlyphTests(unittest.TestCase):
    """字形参与式（举手之劳的"劳"）：叠加要成功，质检要认可，别的文字仍要拦。"""

    @classmethod
    def setUpClass(cls):
        from pipeline.providers.ocr import OCRChecker
        cls.cfg = make_cfg(**{"qc.ocr.enabled": True, "qc.ocr.skip_providers": [],
                              "image.glyph_overlay.enabled": True,
                              "image.glyph_overlay.mode": "composite",
                              "image.mock.draw_labels": False,
                              "image.mock.width": 512, "image.mock.height": 512,
                              "plan.width": 512, "plan.height": 512})
        cls.ocr = OCRChecker(cls.cfg)
        cls.ocr._load()  # noqa: SLF001
        if not cls.ocr.available:
            raise unittest.SkipTest("未安装 OCR 引擎，无法验证字形叠加结果")

    def test_glyph_overlay_and_qc_acceptance(self):
        from pipeline.providers.image_mock import MockProvider
        from pipeline.providers.vlm_openai import VLMChecker
        from pipeline.schema import Item

        item = Item(id="glyphtest", text="举手之劳", type="idiom")
        plan = plan_rule.build_plan(item, self.cfg)
        self.assertEqual(plan.glyph, "劳", "举手之劳 应判为字形参与式")
        self.assertEqual(plan.technique, "glyph")
        self.assertIn("clean empty area", plan.prompt_en.lower(),
                      "合成模式下提示词应预留空白区域")

        out = TMP / "glyph_case.png"
        MockProvider(self.cfg).generate(item, plan, out)
        self.assertTrue(generate.apply_glyph_overlay(self.cfg, plan, out), "字形叠加失败")

        det = self.ocr.detect(out)
        joined = "".join(t for t, _ in det["texts"])
        self.assertIn("劳", joined, f"叠加后 OCR 应能读到「劳」，实际 {det['texts']}")

        row = {"id": "glyphtest_a1", "item_id": "glyphtest", "path": str(out),
               "attempt": 1, "provider": "test"}
        elems = [e.en for e in plan.elements]

        # 1) 告知质检"这个字是计划内的" -> 通过
        ok = qc.check_image(self.cfg, dict(row), elems, item.text, plan.composition,
                            1, 3, self.ocr, VLMChecker(self.cfg), {},
                            expected_text=["劳"])
        self.assertTrue(ok.text_free, f"计划内的字形被误判为文字泄漏：{ok.text_found}")
        self.assertEqual(ok.technical.get("glyph_detected"), ["劳"])

        # 2) 不告知 -> 应判为文字泄漏（说明这套豁免是精确的，不是把检测关掉）
        leaked = qc.check_image(self.cfg, dict(row), elems, item.text, plan.composition,
                                1, 3, self.ocr, VLMChecker(self.cfg), {})
        self.assertFalse(leaked.text_free, "没声明是计划内文字时应当报泄漏")


class PinyinTests(unittest.TestCase):
    """拼音规范化：标调位置、音节表、粘连拆分、错误检出。"""

    def test_tone_mark_placement_iu_ui(self):
        from pipeline.pinyin import mark_syllable
        # 这几个是标调规则的生死线：iu 标在 u 上、ui 标在 i 上
        cases = [("jiu", 3, "jiǔ"), ("gui", 4, "guì"), ("liu", 2, "liú"),
                 ("shui", 3, "shuǐ"), ("hua", 4, "huà"), ("xue", 2, "xué"),
                 ("lv", 4, "lǜ"), ("nv", 3, "nǚ"), ("er", 2, "ér"),
                 ("zhong", 1, "zhōng"), ("qun", 2, "qún"), ("juan", 4, "juàn")]
        for base, tone, want in cases:
            self.assertEqual(mark_syllable(base, tone), want,
                             f"{base}{tone} 标调位置错误")

    def test_tone_marks_end_to_end(self):
        from pipeline.utils import pinyin_tone_marks
        self.assertEqual(pinyin_tone_marks("hua4 she2 tian1 zu2"), "huà shé tiān zú")
        self.assertEqual(pinyin_tone_marks("jiu3 niu2 yi1 mao2"), "jiǔ niú yī máo")
        self.assertEqual(pinyin_tone_marks("mi3 fan4"), "mǐ fàn")
        # 符号声调输入也要能解析
        self.assertEqual(pinyin_tone_marks("huà shé"), "huà shé")

    def test_syllable_table_self_check(self):
        from pipeline.pinyin import MISSING_SYLLABLES, SYLLABLES
        self.assertEqual(MISSING_SYLLABLES, [], "音节表生成有误，合法音节被判为非法")
        self.assertGreater(len(SYLLABLES), 380)
        # 曾经被误判为非法的那批，必须都在表内
        for s in ("ju", "qu", "xu", "jun", "quan", "xun", "yuan", "yue",
                  "jue", "nv", "lv", "lue", "nue", "zhi", "ri", "zi", "si"):
            self.assertIn(s, SYLLABLES, f"合法音节 {s} 被判为非法")

    def test_u_umlaut_spelling_after_jqx(self):
        from pipeline.pinyin import normalize
        # jüe 是错误写法（j/q/x 后的 ü 写作 u），normalize 要能纠正成 jue
        res = normalize("mai4 guan1 yu4 jüe2")
        self.assertIn("jue2", res.numbered, f"ü 拼写未纠正：{res.numbered}")

    def test_merged_syllable_split(self):
        from pipeline.pinyin import normalize
        res = normalize("bu4 xiang2 zhizhao4", expect_chars=4)
        self.assertEqual(len(res.syllables), 4, f"粘连未拆分：{res.numbered}")
        self.assertIn("zhi", res.syllables)
        self.assertIn("zhao", res.syllables)
        self.assertTrue(any("拆分" in i for i in res.issues))
        # 不传字数时不做拆分（没有依据）
        self.assertEqual(len(normalize("bu4 xiang2 zhizhao4").syllables), 3)

    def test_count_mismatch_detected(self):
        from pipeline.pinyin import check_item
        self.assertTrue(check_item("一路平安", "yi1 lu4 ping2"), "截断的拼音应被检出")
        self.assertEqual(check_item("画蛇添足", "hua4 she2 tian1 zu2"), [])

    def test_real_data_issues_are_flagged(self):
        from pipeline.pinyin import check_item
        # 这些都出自用户词表里的真实错误
        for text, py, frag in [
            ("不贪为宝", "bu4 tan1 we2 bao3", "we"),
            ("人多势众", "ren2 duo1 shi4 zhon4", "zhon"),
            ("周游列国", "zhou1 yuo2 lie4 guo2", "yuo"),
            ("作恶多端", "zuo4", "不匹配"),
        ]:
            problems = check_item(text, py)
            self.assertTrue(any(frag in p for p in problems),
                            f"{text}「{py}」的问题没被检出：{problems}")


class LLMGuardrailTests(unittest.TestCase):
    """大模型出题的护栏：元素校验、结构性构图回退规则版。这些不需要真的起服务。"""

    def setUp(self):
        self.cfg = make_cfg(**{"plan.provider": "llm"})

    def _item(self, text="马首是瞻"):
        from pipeline.schema import Item
        return Item(id="t", text=text, type="idiom")

    def test_drops_function_words_and_numbers(self):
        from pipeline.plan_llm import _to_plan
        from pipeline.schema import Item
        # 马首是瞻：Qwen3-8B 真的会把虚词"是"当成画面元素
        raw = {"answer": "马首是瞻", "composition": "action",
               "elements": [{"morpheme": "是", "en": "is"},
                            {"morpheme": "瞻", "en": "looking"},
                            {"morpheme": "马首", "en": "the head of a horse"}],
               "prompt_en": "a horse head and someone looking"}
        plan = _to_plan(raw, self._item(), self.cfg, "llm")
        names = [e.morpheme for e in plan.elements]
        self.assertNotIn("是", names, "虚词「是」不该成为画面元素")
        self.assertIn("瞻", names)

        raw2 = {"answer": "九牛一毛", "composition": "quantity",
                "elements": [{"morpheme": "九", "en": "nine"},
                             {"morpheme": "一", "en": "one"},
                             {"morpheme": "牛", "en": "ox"},
                             {"morpheme": "毛", "en": "hair"}],
                "prompt_en": "nine cows and one hair side by side"}
        plan2 = _to_plan(raw2, Item(id="t2", text="九牛一毛", type="idiom"),
                         self.cfg, "llm")
        names2 = [e.morpheme for e in plan2.elements]
        self.assertNotIn("九", names2, "数词应作为数量修饰而不是独立元素")
        self.assertNotIn("一", names2)
        self.assertIn("牛", names2)

    def test_structural_composition_falls_back_to_rule_scene(self):
        from pipeline.plan_llm import _to_plan
        from pipeline.schema import Item
        raw = {"answer": "九牛一毛", "composition": "quantity",
               "elements": [{"morpheme": "牛", "en": "ox"},
                            {"morpheme": "毛", "en": "hair"}],
               "prompt_en": "nine cows with one hair among them"}
        plan = _to_plan(raw, Item(id="t", text="九牛一毛", type="idiom"),
                        self.cfg, "llm")
        self.assertEqual(plan.extra.get("scene_source"), "rule")
        # 版面机制必须保留：3×3 网格，这是规则版显式编码的
        self.assertIn("3 by 3 grid", plan.extra["scene_en"])
        self.assertIn("3 by 3 grid", plan.prompt_en)
        # 大模型的原始描述仍然留档，便于对比
        self.assertIn("nine cows", plan.extra.get("llm_prompt_en", ""))

    def test_glyph_from_llm_is_validated(self):
        from pipeline.plan_llm import _to_plan
        from pipeline.schema import Item
        # 仓鼠 全可画，不该有字形；LLM 乱报必须被忽略
        raw = {"answer": "仓鼠", "composition": "literal", "glyph": "鼠",
               "elements": [{"morpheme": "仓", "en": "a granary"},
                            {"morpheme": "鼠", "en": "a mouse"}],
               "prompt_en": "a box next to a mouse"}
        plan = _to_plan(raw, Item(id="t", text="仓鼠", type="word"), self.cfg, "llm")
        self.assertEqual(plan.glyph, "", "全可画的词条不该被授予字形")
        # 举手之劳 该有字形，且必须保留
        raw2 = {"answer": "举手之劳", "composition": "action", "glyph": "劳",
                "elements": [{"morpheme": "举", "en": "a raised hand"},
                             {"morpheme": "手", "en": "a hand"}],
                "prompt_en": "a raised hand"}
        plan2 = _to_plan(raw2, Item(id="t2", text="举手之劳", type="idiom"),
                         self.cfg, "llm")
        self.assertEqual(plan2.glyph, "劳")
        self.assertEqual(plan2.technique, "glyph")


class UtilTests(unittest.TestCase):
    def test_extract_json_variants(self):
        self.assertEqual(extract_json('{"a":1}'), {"a": 1})
        self.assertEqual(extract_json('```json\n{"a":1}\n```'), {"a": 1})
        self.assertEqual(extract_json('好的，结果如下：\n[{"a":1}]\n希望有帮助'),
                         [{"a": 1}])
        with self.assertRaises(ValueError):
            extract_json("完全没有 JSON")

    def test_pinyin_tone_marks(self):
        self.assertEqual(pinyin_tone_marks("hua4 she2 tian1 zu2"), "huà shé tiān zú")
        self.assertEqual(pinyin_tone_marks("mi3 fan4"), "mǐ fàn")
        self.assertEqual(pinyin_tone_marks(""), "")
        self.assertEqual(pinyin_tone_marks("a1"), "ā")

    def test_stable_id(self):
        self.assertEqual(stable_id("idi", "画蛇添足", 12), stable_id("idi", "画蛇添足", 12))
        self.assertNotEqual(stable_id("idi", "画蛇添足", 12), stable_id("idi", "守株待兔", 12))


def main() -> int:
    argv = sys.argv[1:]
    print(f"临时工作目录：{TMP}")
    # 自测时压低日志：故意触发的"后端不可用→回退"等警告会污染报告
    import logging
    level = logging.DEBUG if "-v" in argv else logging.CRITICAL
    logging.getLogger("hv").setLevel(level)
    logging.getLogger("hv").handlers.clear()
    logging.getLogger("hv").addHandler(logging.NullHandler())

    loader = unittest.TestLoader()
    if argv and not argv[0].startswith("-"):
        suite = loader.loadTestsFromName(argv[0])
        argv = argv[1:]
    else:
        suite = loader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2 if "-v" in argv else 1).run(suite)
    if result.wasSuccessful():
        print(f"\n全部自测通过 ✅（{result.testsRun} 条用例）")
        shutil.rmtree(TMP, ignore_errors=True)
        return 0
    print(f"\n自测失败 ❌（临时目录保留：{TMP}）")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
