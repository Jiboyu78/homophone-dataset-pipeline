"""读 SD WebUI 的启动日志，判断状态（给自动化用，避免反复手敲）。"""
from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

LOG = Path(r"G:\Language-and-Vision\.sdlog.txt")
API = "http://127.0.0.1:7860"


def api_ready() -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(f"{API}/sdapi/v1/options", timeout=10) as r:
            import json
            d = json.loads(r.read().decode("utf-8"))
        return True, str(d.get("sd_model_checkpoint", "?"))
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    wait = float(sys.argv[1]) if len(sys.argv) > 1 else 0
    deadline = time.time() + wait
    while True:
        ok, info = api_ready()
        if ok:
            print(f"✅ SD WebUI API 就绪，当前模型：{info}")
            return 0
        if time.time() >= deadline:
            break
        time.sleep(10)
    print(f"⏳ 尚未就绪（{info}）")
    if LOG.exists():
        text = LOG.read_text(encoding="utf-8", errors="replace")
        tail = [l for l in text.splitlines() if l.strip()][-12:]
        print(f"--- 日志尾部（{LOG.stat().st_size} 字节）---")
        for l in tail:
            print("  " + l[:160])
        for kw in ("Installing requirements", "Traceback", "Error", "OutOfMemory",
                   "no kernel image", "Running on local URL"):
            if kw in text:
                print(f"  ⚠ 日志中出现关键词：{kw}")
    else:
        print("（还没有日志文件）")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
