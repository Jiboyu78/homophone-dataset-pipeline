"""配置加载：config.json + 环境变量覆盖 + 深层合并默认值。"""
from __future__ import annotations

import copy
import os
from typing import Any

from .utils import LOG, read_json, resolve

# --------------------------------------------------------------------------- #
# 画风预设：默认的 rebus_cartoon 对齐你给的参考题（单幅横向、米黄纸纹、粗描边卡通）
# --------------------------------------------------------------------------- #
STYLE_PRESETS: dict[str, dict[str, Any]] = {
    "rebus_cartoon": {
        "style_en": (
            "single-panel cartoon illustration, bold clean dark outlines, flat warm "
            "coloring with simple cel shading, warm cream paper texture background with "
            "subtle vintage mottling and slightly darkened edges, expressive simplified "
            "figures with big heads, humorous symbolic scene, wide landscape composition"
        ),
        "style_zh": (
            "单幅卡通插画，粗黑描边，暖色平涂加简单赛璐璐阴影，米黄做旧纸纹背景、"
            "四边略深，形象夸张简化（大头娃娃风），画面诙谐，横向宽幅构图"
        ),
        "width": 1152,
        "height": 768,
    },
    "clean_vector": {
        "style_en": (
            "clean minimal flat vector illustration, single unified composition, "
            "centered, generous negative space, soft muted palette, subtle cel shading, "
            "harmonious symbolic still life, no text of any kind"
        ),
        "style_zh": (
            "简约扁平矢量插画风，构图居中，留白充足，色调柔和统一，纯象征静物组合"
        ),
        "width": 768,
        "height": 768,
    },
}

DEFAULTS: dict[str, Any] = {
    "paths": {
        "data_dir": "data",
        "output_dir": "output",
        "images_dir": "output/images",
        "log_file": "output/pipeline.log",
        "db": "output/pipeline.db",
    },
    "ingest": {
        # 留空表示自动扫描 data_dir 下所有 .json
        "files": [],
        "max_len_idiom": 8,
        "max_len_word": 6,
        "min_len": 2,
    },
    "select": {
        "min_score": 0.55,          # 可图化总分阈值
        "min_coverage": 0.6,        # 可画语素覆盖率阈值
        "max_items": 0,             # 0 = 不限制；>0 时按分数取前 N
        "allow_homophone": True,    # 是否允许谐音借代式
        "deny_words": [
            "主义", "思想", "精神", "道德", "阶级", "政治", "革命", "哲学",
            "经济", "文化", "社会", "制度", "理论", "概念", "作风", "品质",
        ],
        "deny_chars": ["之", "乎", "者", "也", "矣", "焉", "哉"],
        "prefer_types": {"idiom": 1.0, "word": 0.96},
    },
    "plan": {
        "provider": "rule",         # rule | llm | hybrid（难点词才用大模型）
        # llm/hybrid 模式下大模型不可用时，是否退回规则出题器。
        # 一键云端流程会设成 false：宁可失败报错，也不要悄悄给出一批规则版结果。
        "fallback_to_rule": True,
        "style_preset": "rebus_cartoon",   # rebus_cartoon | clean_vector
        # prose=长句描述（吃得住的是 SDXL / 云模型）；tags=短标签串（SD1.5 必须用这个）
        "prompt_style": "prose",
        "style_tags_sd15": "",      # 留空则用内置的扁平卡通标签串
        "style_en": "",             # 留空则由 style_preset 填充
        "style_zh": "",
        "negative_prompt": (
            "text, words, letters, chinese characters, calligraphy, caption, subtitle, "
            "watermark, signature, logo, stamp, seal, number, speech bubble, ui, "
            "frame, border, blurry, low quality, jpeg artifacts, deformed, extra limbs, "
            "multiple panels, collage, split screen, photorealistic human face"
        ),
        "width": 0,
        "height": 0,
        "self_consistency_n": 1,    # 同一词条生成几张候选（>1 时由质检挑选）
        "llm": {
            "base_url": "http://127.0.0.1:8080/v1",   # 本机 llama.cpp（OpenAI 兼容）
            "api_key_env": "HV_LLM_API_KEY",
            "api_key": "",
            "model": "Qwen3-8B-Q5_K_M.gguf",
            "temperature": 0.6,
            "timeout_s": 300,
            "max_retry": 2,
            "batch": 4,             # 每次请求让模型出多少条
            "disable_thinking": True,   # Qwen3 关掉思维链：实测 18.1s -> 3.6s
            # 这几类构图的版面机制由规则版显式编码，小模型的描述会丢掉它，
            # 所以场景/元素以规则版为准，只借用大模型的难度与理由
            "prefer_rule_compositions": ["quantity", "inner", "spatial", "contrast"],
            "hard_max": 200,        # hybrid 模式下最多把多少条难点词交给大模型
        },
    },
    "image": {
        "provider": "mock",         # comfyui | sdwebui | mock
        "comfyui": {
            "base_url": "http://127.0.0.1:8188",
            "workflow": "workflows/sd15_txt2img_api.json",
            "width": 768,
            "height": 768,
            "steps": 28,
            "cfg": 6.5,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "ckpt_name": "",        # 留空则用工作流里写死的 checkpoint
            "batch_size": 1,
            "timeout_s": 600,
            "poll_interval_s": 2.0,
            "filename_prefix": "hv",
            "overrides": {},        # 额外按节点 id 覆盖输入，例如 {"3": {"cfg": 7.0}}
        },
        "sdwebui": {
            "base_url": "http://127.0.0.1:7860",
            "width": 768,
            "height": 768,
            "steps": 28,
            "cfg_scale": 6.5,
            "sampler_name": "DPM++ 2M",
            "scheduler": "Karras",
            "timeout_s": 600,
        },
        "api": {
            "mode": "openai",       # openai（/images/generations）| dashscope（通义万相异步）
            "base_url": "",         # 例 https://api.openai.com/v1 / https://dashscope.aliyuncs.com
            "api_key": "",          # 也可以只设环境变量 HV_IMAGE_API_KEY
            "api_key_env": "HV_IMAGE_API_KEY",
            "model": "",            # 例 gpt-image-1 / doubao-seedream-3-0-t2i / wanx2.1-t2i-turbo
            "size": "1024x1024",    # dashscope 会自动转成 1024*1024
            "images_path": "",      # 留空按 mode 自动推断
            "timeout_s": 300,
            "poll_interval_s": 3.0,
            "max_poll_s": 600,
            "extra": {},            # 额外塞进请求体的字段，例如 {"watermark": false}
        },
        "mock": {
            "width": 0,             # 0 = 跟随 plan.width/height
            "height": 0,
            "draw_labels": True,    # 在占位图上画出元素框与标签，便于审阅构图
        },
        # 字形参与式（举手之劳的"劳"）：默认用字体确定性叠加，比让扩散模型写字可靠得多
        "glyph_overlay": {
            "enabled": True,
            "mode": "composite",    # composite=出图后用字体叠加 | prompt=让模型自己写
            "position": "center-left",
            "size_ratio": 0.55,     # 字号 = 画面高度 × 该比例
            "font": "",             # 留空则自动在系统字体里找行楷/楷体
            "ink_color": [26, 26, 30],
            "halo_color": [250, 248, 240],
        },
        "max_retry": 2,
        "concurrency": 1,           # 单卡建议 1
        # score=纯按可图化分排序；balanced=按构图范式轮询，数据集更多样
        "selection_strategy": "balanced",
        # 字形参与式在平衡取样里的占比上限（参考题经验值：4 张里约 1 张）
        "max_glyph_share": 0.30,
        # 出图后统一缩放到交付尺寸（小分辨率生成、Lanczos 放大，省显存）
        "postprocess": {
            "enabled": False,
            "target_width": 1152,
            "target_height": 768,
            "method": "lanczos",
        },
    },
    "qc": {
        "technical": {
            "min_side": 256,
            "max_aspect": 3.0,
            "min_std": 12.0,        # 灰度标准差下限，低于判为接近纯色
            "min_edge_density": 0.005,
            "phash_hamming_dup": 6, # 感知哈希汉明距离 <= 此值判为重复
            "dedup_scope": "cross_item",   # cross_item | all | off
            "skip_dedup_providers": ["mock"],
        },
        "ocr": {
            "enabled": True,
            "engine": "auto",       # auto | rapidocr | paddleocr | easyocr | pytesseract | none
            "min_conf": 0.6,
            "max_text_chars": 0,    # 允许出现的文字字符数上限（会意图应为 0）
            # Mock 后端画的是给人看的"构图示意卡"，其标签不计为生成缺陷
            "skip_providers": ["mock"],
        },
        "vlm": {
            "provider": "none",     # openai | none
            "base_url": "http://127.0.0.1:8080/v1",
            "api_key_env": "HV_VLM_API_KEY",
            "api_key": "",
            "model": "",            # 需指向一个视觉模型（如 Qwen2.5-VL 的 mmproj GGUF）
            "timeout_s": 300,
            "max_retry": 2,
            "disable_thinking": True,
            "check_solvability": True,
        },
        "min_accept_score": 0.75,
        "require_solvable": False,  # true = VLM 盲猜不出的图直接不入库（更严，产量更低）
        "max_retry": 2,             # 生成->质检->反馈重生成 的最大轮数
    },
    "package": {
        "splits": {"train": 0.8, "val": 0.1, "test": 0.1},
        "seed": 42,
        "only_accepted": True,
        "contact_sheets": True,
        # 训练可直接用的导出格式（raw 始终写出）
        "export_formats": ["raw", "sft_chat", "multiple_choice", "llava", "classification"],
        "question": "这张图表达的是哪个成语？",
        "question_idiom": "这张图表达的是哪个成语？",
        "question_word": "这张图表达的是哪个词语？",
        # 降级样本（元素只是并排铺开、没做语义合成，如 刀鱼 -> 一把刀 + 一条鱼）
        # 默认保留并在字段里标记；设 true 则整体排除，不进入任何导出格式。
        "exclude_degraded": False,
        "dataset_name": "huivision-rebus-zh",
        "version": "0.1.0",
        "license_note": "生成图像，供研究与教学使用",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Config:
    def __init__(self, data: dict[str, Any], config_path: str = "config.json"):
        self._d = data
        self.config_path = config_path

    def __getitem__(self, key: str) -> Any:
        return self._d[key]

    def get(self, path: str, default: Any = None) -> Any:
        """点号路径取值，例如 cfg.get('image.comfyui.width')。"""
        cur: Any = self._d
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def set(self, path: str, value: Any) -> None:
        parts = path.split(".")
        cur = self._d
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._d)

    # -- 常用路径 -- #
    @property
    def data_dir(self):
        return resolve(self.get("paths.data_dir"))

    @property
    def output_dir(self):
        d = resolve(self.get("paths.output_dir"))
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def images_dir(self):
        d = resolve(self.get("paths.images_dir"))
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def db_path(self):
        return resolve(self.get("paths.db"))


def build_config(raw: dict[str, Any] | None = None,
                 config_path: str = "<dict>") -> Config:
    """由原始配置字典构造 Config：默认值 -> 画风预设 -> 用户值 -> 环境变量。

    所有入口（CLI、脚本、自测）都应该走这里，否则会静默丢掉画风预设。
    """
    raw = raw or {}
    merged = _deep_merge(DEFAULTS, {})
    preset_name = ((raw.get("plan") or {}).get("style_preset")
                   or merged["plan"].get("style_preset") or "")
    preset = STYLE_PRESETS.get(str(preset_name))
    if preset:
        merged["plan"].update({"style_en": preset["style_en"],
                               "style_zh": preset["style_zh"],
                               "width": preset.get("width", 768),
                               "height": preset.get("height", 768)})
    else:
        LOG.warning("未知的画风预设 %r，可选：%s", preset_name, list(STYLE_PRESETS))

    merged = _deep_merge(merged, raw)
    if not merged["plan"].get("style_en"):
        fb = STYLE_PRESETS["clean_vector"]
        merged["plan"]["style_en"] = fb["style_en"]
        merged["plan"]["style_zh"] = fb["style_zh"]
    if not merged["plan"].get("width") or not merged["plan"].get("height"):
        fb = STYLE_PRESETS.get(str(merged["plan"].get("style_preset")),
                               STYLE_PRESETS["clean_vector"])
        merged["plan"]["width"] = fb.get("width", 768)
        merged["plan"]["height"] = fb.get("height", 768)

    _apply_env(merged)
    cfg = Config(merged, config_path)
    LOG.debug("配置就绪: %s（画风预设 %s，%dx%d）", config_path,
              merged["plan"].get("style_preset"),
              merged["plan"]["width"], merged["plan"]["height"])
    return cfg


def load_config(path: str | None = None) -> Config:
    cfg_path = path or os.environ.get("HV_CONFIG", "config.json")
    raw = read_json(cfg_path, default={}) or {}
    return build_config(raw, cfg_path)


def _apply_env(cfg: dict[str, Any]) -> None:
    """环境变量覆盖：HV_IMAGE_PROVIDER / HV_LLM_BASE_URL / HV_LLM_MODEL ..."""
    mapping = {
        "HV_IMAGE_PROVIDER": "image.provider",
        "HV_IMAGE_MODE": "image.api.mode",
        "HV_IMAGE_BASE_URL": "image.api.base_url",
        "HV_IMAGE_MODEL": "image.api.model",
        "HV_IMAGE_SIZE": "image.api.size",
        "HV_PLAN_PROVIDER": "plan.provider",
        "HV_VLM_PROVIDER": "qc.vlm.provider",
        "HV_COMFYUI_URL": "image.comfyui.base_url",
        "HV_COMFYUI_WORKFLOW": "image.comfyui.workflow",
        "HV_SDWEBUI_URL": "image.sdwebui.base_url",
        "HV_LLM_BASE_URL": "plan.llm.base_url",
        "HV_LLM_MODEL": "plan.llm.model",
        "HV_VLM_BASE_URL": "qc.vlm.base_url",
        "HV_VLM_MODEL": "qc.vlm.model",
    }
    for env, dotted in mapping.items():
        val = os.environ.get(env)
        if val:
            parts = dotted.split(".")
            cur = cfg
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = val
