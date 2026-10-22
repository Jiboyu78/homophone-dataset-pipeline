"""ComfyUI 生图后端（HTTP API，无需装任何 ComfyUI 端插件）。

工作方式：
  1) 读入一份 ComfyUI「API 格式」工作流 JSON（workflows/*.json）；
  2) 按节点类型自动注入提示词 / 尺寸 / 种子 / 步数 / CFG；
  3) POST /prompt 提交，轮询 /history/{id}，再从 /view 下载图片落盘。

节点注入规则（按 class_type + _meta.title 识别，所以你自己换工作流也不用改代码）：
  CLIPTextEncode      -> text（title 含 NEGATIVE 的当负向，否则当正向）
  EmptyLatentImage    -> width / height / batch_size
  KSampler            -> seed / steps / cfg / sampler_name / scheduler
  KSamplerAdvanced    -> noise_seed / steps / cfg / sampler_name / scheduler
  CheckpointLoaderSimple -> ckpt_name（仅当配置里显式给了）
  SaveImage           -> filename_prefix
另外支持 image.comfyui.overrides = {"3": {"cfg": 7.0}} 做精确覆盖。
"""
from __future__ import annotations

import copy
import json
import mimetypes
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from ..schema import Item, Plan
from ..utils import LOG, read_json, resolve
from .image_base import ImageProvider, ImageResult


class ComfyUIError(RuntimeError):
    pass


def _http_json(url: str, payload: dict | None = None, timeout: float = 30.0) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_bytes(url: str, timeout: float = 60.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


class ComfyUIProvider(ImageProvider):
    name = "comfyui"

    def __init__(self, cfg: Any):
        super().__init__(cfg)
        self.base = str(cfg.get("image.comfyui.base_url", "http://127.0.0.1:8188")).rstrip("/")
        self.workflow_path = resolve(cfg.get("image.comfyui.workflow",
                                             "workflows/sd15_txt2img_api.json"))
        self._workflow_cache: dict[str, Any] | None = None
        self.client_id = str(uuid.uuid4())

    # ------------------------------------------------------------------ #
    def health(self) -> tuple[bool, str]:
        try:
            info = _http_json(f"{self.base}/system_stats", timeout=5)
        except Exception as exc:  # noqa: BLE001
            return False, f"无法连接 ComfyUI {self.base}：{exc}"
        dev = ""
        try:
            devs = info.get("devices") or []
            if devs:
                dev = f"{devs[0].get('name','?')} (vram {round(devs[0].get('vram_total',0)/2**30,1)}GB)"
        except Exception:  # noqa: BLE001
            pass
        if not self.workflow_path.exists():
            return False, f"工作流文件不存在：{self.workflow_path}"
        return True, f"ComfyUI 在线 {self.base} {dev}；工作流 {self.workflow_path.name}"

    def list_checkpoints(self) -> list[str]:
        try:
            info = _http_json(f"{self.base}/object_info/CheckpointLoaderSimple", timeout=10)
            opts = info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
            return list(opts)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("读取 checkpoint 列表失败: %s", exc)
            return []

    # ------------------------------------------------------------------ #
    def load_workflow(self) -> dict[str, Any]:
        if self._workflow_cache is None:
            wf = read_json(self.workflow_path)
            if not isinstance(wf, dict) or not wf:
                raise ComfyUIError(f"工作流必须是 ComfyUI API 格式的非空 JSON 对象：{self.workflow_path}")
            if "nodes" in wf and "links" in wf:
                raise ComfyUIError(
                    "这看起来是 ComfyUI 的『界面格式』工作流。请用 "
                    "ComfyUI 菜单 Settings → Enable Dev mode Options，"
                    "然后『Save (API Format)』导出 API 格式。")
            self._workflow_cache = wf
        return copy.deepcopy(self._workflow_cache)

    def patch_workflow(self, plan: Plan, seed: int, feedback: str = "") -> dict[str, Any]:
        wf = self.load_workflow()
        positive = self.compose_prompt(plan, feedback)
        negative = self.compose_negative(plan, self.cfg)

        width = int(self.cfg.get("image.comfyui.width") or self.cfg.get("plan.width") or 768)
        height = int(self.cfg.get("image.comfyui.height") or self.cfg.get("plan.height") or 768)
        steps = int(self.cfg.get("image.comfyui.steps", 28))
        cfg_scale = float(self.cfg.get("image.comfyui.cfg", 6.5))
        sampler = self.cfg.get("image.comfyui.sampler_name", "dpmpp_2m")
        scheduler = self.cfg.get("image.comfyui.scheduler", "karras")
        ckpt = self.cfg.get("image.comfyui.ckpt_name", "")
        prefix = self.cfg.get("image.comfyui.filename_prefix", "hv")

        text_nodes: list[tuple[str, dict]] = []
        for nid, node in wf.items():
            ct = node.get("class_type", "")
            ins = node.setdefault("inputs", {})
            title = str(node.get("_meta", {}).get("title", "")).lower()
            if ct == "CLIPTextEncode":
                text_nodes.append((nid, node))
                continue
            if ct == "EmptyLatentImage":
                ins["width"], ins["height"] = width, height
                ins["batch_size"] = int(self.cfg.get("image.comfyui.batch_size", 1))
            elif ct == "KSampler":
                ins["seed"] = seed
                ins["steps"] = steps
                ins["cfg"] = cfg_scale
                ins["sampler_name"] = sampler
                ins["scheduler"] = scheduler
            elif ct == "KSamplerAdvanced":
                ins["noise_seed"] = seed
                ins["steps"] = steps
                ins["cfg"] = cfg_scale
                ins["sampler_name"] = sampler
                ins["scheduler"] = scheduler
            elif ct == "CheckpointLoaderSimple" and ckpt:
                ins["ckpt_name"] = ckpt
            elif ct == "SaveImage":
                ins["filename_prefix"] = prefix

        # 正向/负向文本节点：优先按标题判断，其次按"是否被 KSampler 的 negative 引用"
        neg_ids = set()
        for node in wf.values():
            if node.get("class_type") in ("KSampler", "KSamplerAdvanced"):
                link = node.get("inputs", {}).get("negative")
                if isinstance(link, list) and link:
                    neg_ids.add(str(link[0]))
        pos_set = False
        for nid, node in text_nodes:
            title = str(node.get("_meta", {}).get("title", "")).lower()
            is_neg = ("negative" in title or "neg" == title.strip()) or str(nid) in neg_ids
            node["inputs"]["text"] = negative if is_neg else positive
            if not is_neg:
                pos_set = True
        if text_nodes and not pos_set:
            text_nodes[0][1]["inputs"]["text"] = positive

        for nid, patch in (self.cfg.get("image.comfyui.overrides", {}) or {}).items():
            if str(nid) in wf:
                wf[str(nid)].setdefault("inputs", {}).update(patch)
            else:
                LOG.warning("overrides 里的节点 id %s 不在工作流中", nid)
        return wf

    # ------------------------------------------------------------------ #
    def generate(self, item: Item, plan: Plan, out_path: Path, seed: int = -1,
                 feedback: str = "") -> ImageResult:
        if seed < 0:
            seed = int.from_bytes(uuid.uuid4().bytes[:4], "big") & 0x7FFFFFFF
        wf = self.patch_workflow(plan, seed, feedback)
        timeout = float(self.cfg.get("image.comfyui.timeout_s", 600))
        interval = float(self.cfg.get("image.comfyui.poll_interval_s", 2.0))

        body = {"prompt": wf, "client_id": self.client_id}
        try:
            resp = _http_json(f"{self.base}/prompt", body, timeout=60)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:800]
            raise ComfyUIError(f"提交任务失败 HTTP {exc.code}: {detail}") from exc
        if "error" in resp:
            raise ComfyUIError(f"ComfyUI 拒绝任务: {resp['error']}")
        node_errors = resp.get("node_errors") or {}
        if node_errors:
            raise ComfyUIError(f"节点参数错误: {json.dumps(node_errors, ensure_ascii=False)[:800]}")
        prompt_id = resp.get("prompt_id")
        if not prompt_id:
            raise ComfyUIError(f"未返回 prompt_id: {resp}")

        images = self._wait_images(prompt_id, timeout, interval)
        if not images:
            raise ComfyUIError("任务完成但没有产出图像")

        # 需要平面化时也兼容
        first = images[0]
        raw = _http_bytes(self._view_url(first))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(raw)
        LOG.debug("comfyui 出图: %s (seed=%s)", out_path.name, seed)
        return ImageResult(path=out_path, seed=seed, provider=self.name,
                           meta={"prompt_id": prompt_id, "n_images": len(images),
                                 "workflow": self.workflow_path.name})

    def _wait_images(self, prompt_id: str, timeout: float, interval: float) -> list[dict]:
        deadline = time.time() + timeout
        last_status = ""
        while time.time() < deadline:
            hist = _http_json(f"{self.base}/history/{prompt_id}", timeout=30)
            entry = hist.get(prompt_id)
            if entry:
                status = entry.get("status", {})
                last_status = status.get("status_str", "")
                if last_status == "error":
                    msgs = json.dumps(status.get("messages", []), ensure_ascii=False)[:600]
                    raise ComfyUIError(f"ComfyUI 执行出错: {msgs}")
                imgs: list[dict] = []
                for out in (entry.get("outputs") or {}).values():
                    imgs.extend(out.get("images") or [])
                if imgs:
                    return imgs
            time.sleep(interval)
        raise ComfyUIError(f"等待出图超时（{timeout:.0f}s，最后状态 {last_status or 'unknown'}）")

    def _view_url(self, img: dict) -> str:
        q = urllib.parse.urlencode({
            "filename": img.get("filename", ""),
            "subfolder": img.get("subfolder", ""),
            "type": img.get("type", "output"),
        })
        return f"{self.base}/view?{q}"

    # ------------------------------------------------------------------ #
    def interrupt(self) -> None:
        try:
            _http_json(f"{self.base}/interrupt", {}, timeout=10)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("interrupt 失败: %s", exc)
