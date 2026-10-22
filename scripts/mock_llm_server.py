"""OpenAI 兼容的本地 Mock 服务：用来在没有真实大模型时验证 LLM/VLM 代码路径。

它实现两个端点：
    GET  /v1/models
    POST /v1/chat/completions

根据 model 名与提示词内容区分四种请求，并返回**符合 pipeline 预期 schema** 的 JSON：
    mock-llm  -> plan_llm.py 的批量出题请求（返回构图方案数组）
    mock-vlm  -> vlm_openai.py 的两种请求：
                 含"标准答案/text_present" 的是核验 -> 返回元素齐备性
                 含"猜出它想表达" 的是盲猜     -> 返回猜测答案

独立运行（给别的工具/手动 curl 用）：
    python scripts/mock_llm_server.py --port 11500
然后：
    set HV_LLM_BASE_URL=http://127.0.0.1:11500/v1
    set HV_VLM_BASE_URL=http://127.0.0.1:11500/v1
    set HV_PLAN_PROVIDER=llm
    set HV_VLM_PROVIDER=openai
    python -m pipeline.cli run --mock --limit 20

注意：它只是"接线测试替身"，不做任何真实推理。
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def _mock_png(seed_text: str) -> bytes:
    """造一张"象样的"占位图：按提示词哈希决定配色与图形数量。

    用于仿真出图接口——让整条流水线（含质检与打包）能在没有真实 key 的情况下跑通。
    """
    try:
        from PIL import Image, ImageDraw
    except Exception:  # noqa: BLE001
        return base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/"
            "q842iQAAAABJRU5ErkJggg==")
    h = hashlib.sha1(seed_text.encode("utf-8")).hexdigest()
    n = 2 + int(h[:2], 16) % 4
    img = Image.new("RGB", (1024, 1024), (247, 240, 226))
    d = ImageDraw.Draw(img)
    palette = [(214, 168, 140), (166, 196, 172), (176, 186, 214), (222, 200, 150),
               (196, 166, 196), (160, 200, 208)]
    for i in range(n):
        c = palette[int(h[i * 2:i * 2 + 2], 16) % len(palette)]
        cx = 180 + (int(h[i * 3:i * 3 + 3], 16) % 660)
        cy = 180 + (int(h[i * 3 + 3:i * 3 + 6], 16) % 660)
        r = 90 + int(h[i * 2 + 4:i * 2 + 6], 16) % 120
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c, outline=(70, 70, 80), width=4)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()

# --------------------------------------------------------------------------- #
# 从请求里还原"图片内容 + 目标词条"，用于生成自洽的假回复
# --------------------------------------------------------------------------- #
DATA_URI_RE = re.compile(r"data:image/[a-z]+;base64,([A-Za-z0-9+/=]+)")


def _extract_image_b64(payload: dict[str, Any]) -> str:
    for msg in payload.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    m = DATA_URI_RE.match(url or "")
                    if m:
                        return m.group(1)
    return ""


def _full_text(payload: dict[str, Any]) -> str:
    out = []
    for msg in payload.get("messages", []):
        content = msg.get("content")
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    out.append(str(part.get("text", "")))
    return "\n".join(out)


def _answer_from_prompt(text: str) -> str:
    m = re.search(r"标准答案（不要在判断中假定它一定对）：([^\n]+)", text)
    return m.group(1).strip() if m else ""


def _elements_from_prompt(text: str) -> list[str]:
    m = re.search(r"关键视觉元素：([^\n]+)", text)
    if not m:
        return []
    raw = m.group(1).strip()
    if raw.startswith("（"):
        return []
    return [x.strip() for x in raw.split("、") if x.strip()]


def _items_from_prompt(text: str) -> list[dict[str, Any]]:
    """从出题提示词里抠出待设计的词条数组。"""
    i = text.find("待设计词条（JSON 数组）：")
    if i < 0:
        return []
    blob = text[i:]
    start = blob.find("[")
    try:
        return json.loads(blob[start:])
    except json.JSONDecodeError:
        return []


# --------------------------------------------------------------------------- #
# 假回复生成
# --------------------------------------------------------------------------- #
def _llm_reply(payload: dict[str, Any]) -> str:
    items = _items_from_prompt(_full_text(payload))
    plans = []
    for it in items:
        ans = str(it.get("answer", ""))
        segs = it.get("参考可画语素") or list(ans)
        # 每两个语素合成一个元素，保证 schema 合法且能被 pipeline 接受
        elems = []
        for k, ch in enumerate(segs[:4]):
            elems.append({
                "morpheme": ch,
                "zh": ch,
                "en": f"a mock object for {ch}",
                "count": 1,
                "role": "action" if k == 0 else "object",
                "position": "left" if k % 2 == 0 else "right",
            })
        if not elems:
            elems = [{"morpheme": ans, "zh": ans, "en": f"a mock object for {ans}",
                      "count": 1, "role": "object"}]
        plans.append({
            "answer": ans,
            "composition": "juxtapose",
            "elements": elems,
            "layout": f"【mock】把「{ans}」的元素左右并列排布",
            "prompt_en": (f"a mock scene arranging {len(elems)} objects side by side "
                          f"for the concept of a rebus"),
            "must_have": [e["en"] for e in elems],
            "difficulty": 3,
            "rationale": "【mock】仅用于验证 LLM 出题链路，不代表真实质量",
        })
    return json.dumps(plans, ensure_ascii=False)


def _vlm_reply(payload: dict[str, Any]) -> str:
    text = _full_text(payload)
    b64 = _extract_image_b64(payload)
    digest = hashlib.sha1(b64.encode("ascii", "ignore")).hexdigest() if b64 else "0" * 40
    bucket = int(digest[:4], 16) % 100

    if "text_present" in text:                      # 一致性核验
        answer = _answer_from_prompt(text)
        elems = _elements_from_prompt(text)
        # 用图片哈希决定"是否报了文字"，从而两种分支都能被跑到；
        # 约 1/3 的图会被判"有文字"，进而触发重出图闭环。
        text_present = bucket < 33
        missing = elems[1:2] if bucket >= 80 else []   # 少量图故意缺一个元素
        return json.dumps({
            "text_present": text_present,
            "text_content": "MOCK" if text_present else "",
            "elements_present": [e for e in elems if e not in missing],
            "elements_missing": missing,
            "matches_answer": bucket % 7 != 0,
            "problems": (["mock：画面像是多格拼贴"] if bucket >= 95 else []),
            "confidence": 0.9,
            "_mock": {"answer": answer, "bucket": bucket},
        }, ensure_ascii=False)

    if "猜出它想表达" in text:                       # 可解性盲猜
        # 约 2/3 的图"能猜出来"；猜对时回显一个占位答案（真实校验由 pipeline 比对）
        guess = "MOCK_GUESS"
        return json.dumps({
            "objects": ["mock object"],
            "layout": "mock layout",
            "guess": guess,
            "alternatives": [],
            "confidence": 0.5,
            "_mock": {"bucket": bucket},
        }, ensure_ascii=False)

    return json.dumps({"error": "mock server: 无法识别该请求"}, ensure_ascii=False)


def _handle_completion(payload: dict[str, Any]) -> dict[str, Any]:
    model = str(payload.get("model", ""))
    if "vlm" in model.lower() or "vl" in model.lower():
        content = _vlm_reply(payload)
    else:
        content = _llm_reply(payload)
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# --------------------------------------------------------------------------- #
class _Handler(BaseHTTPRequestHandler):
    server_version = "HV-MockLLM/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # 静音
        if getattr(self.server, "verbose", False):
            print(f"[mock] {fmt % args}")

    def _send(self, code: int, obj: Any) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, code: int, body: bytes, ctype: str = "image/png") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0].rstrip("/")
        if path.endswith("/models"):
            self._send(200, {"object": "list", "data": [
                {"id": "mock-llm", "object": "model"},
                {"id": "mock-vlm", "object": "model"},
                {"id": "mock-image", "object": "model"}]})
            return
        if "/api/v1/tasks/" in path:            # dashscope 轮询
            port = self.server.server_address[1]
            self._send(200, {"output": {
                "task_id": "mock-task", "task_status": "SUCCEEDED",
                "results": [{"url": f"http://127.0.0.1:{port}/mock-image.png"}]}})
            return
        if path.endswith("/mock-image.png"):     # dashscope 结果图下载
            self._send_bytes(200, _mock_png("mock-dashscope"))
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0].rstrip("/")
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            self._send(400, {"error": f"bad json: {exc}"})
            return

        if path.endswith("/chat/completions"):
            self._send(200, _handle_completion(payload))
            return
        # 出图：OpenAI 形态（同步返回 b64）
        if path.endswith("/images/generations"):
            prompt = str(payload.get("prompt", ""))
            b64 = base64.b64encode(_mock_png(prompt)).decode("ascii")
            self._send(200, {"created": int(time.time()),
                             "data": [{"b64_json": b64, "revised_prompt": prompt}]})
            return
        # 出图：DashScope 异步形态（先提交、再轮询）
        if "text2image/image-synthesis" in path:
            self._send(200, {"output": {"task_id": "mock-task",
                                        "task_status": "PENDING"}})
            return
        self._send(404, {"error": f"mock server 不认识这个路径: {path}"})


def start_server(port: int = 0, verbose: bool = False) -> tuple[ThreadingHTTPServer, str]:
    """在后台线程启动 mock 服务，返回 (server, base_url)。port=0 时自动选空闲端口。"""
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    httpd.verbose = verbose  # type: ignore[attr-defined]
    base_url = f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, base_url


def main() -> int:
    ap = argparse.ArgumentParser(description="OpenAI 兼容的本地 Mock LLM/VLM 服务")
    ap.add_argument("--port", type=int, default=11500)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    httpd, base_url = start_server(args.port, args.verbose)
    print(f"Mock LLM/VLM 服务已启动：{base_url}")
    print("可用的假模型名：mock-llm（文本出题）、mock-vlm（视觉质检）")
    print("按 Ctrl+C 停止")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
