"""直接调用 SD WebUI API 出一张图，测速度与显存是否吃得消。"""
from __future__ import annotations

import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

API = "http://127.0.0.1:7860"
OUT = Path(r"G:\Language-and-Vision\output_sdtest")
OUT.mkdir(parents=True, exist_ok=True)

PROMPT = ("a simple human figure who paints with a brush a green coiled snake, and then "
          "adds an extra bare foot, the gesture is unmistakable and occupies the centre "
          "of the frame. single-panel cartoon illustration, bold clean dark outlines, "
          "flat warm coloring with simple cel shading, warm cream paper texture "
          "background, humorous symbolic scene. absolutely no text, no letters, "
          "no chinese characters, no numbers, no captions, no labels, no watermark")
NEGATIVE = ("text, words, letters, chinese characters, calligraphy, caption, subtitle, "
            "watermark, signature, logo, stamp, seal, number, speech bubble, ui, frame, "
            "border, blurry, low quality, jpeg artifacts, deformed, extra limbs, "
            "multiple panels, collage, split screen")


def txt2img(w: int, h: int, steps: int, seed: int) -> tuple[float, Path]:
    payload = {
        "prompt": PROMPT, "negative_prompt": NEGATIVE,
        "width": w, "height": h, "steps": steps, "cfg_scale": 7,
        "sampler_name": "DPM++ 2M", "scheduler": "Karras",
        "seed": seed, "batch_size": 1, "save_images": False,
    }
    req = urllib.request.Request(f"{API}/sdapi/v1/txt2img",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.loads(r.read().decode("utf-8"))
    dt = time.time() - t0
    img = base64.b64decode(out["images"][0].split(",", 1)[-1])
    path = OUT / f"testsd_{w}x{h}_{steps}steps.png"
    path.write_bytes(img)
    return dt, path


def main() -> int:
    w, h, steps = (int(x) for x in (sys.argv[1:4] if len(sys.argv) >= 4 else (768, 512, 20)))
    print(f"生成 {w}x{h} / {steps} 步（P100 fp32）...")
    try:
        dt, path = txt2img(w, h, steps, 12345)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 失败：{type(exc).__name__}: {exc}")
        return 1
    size_mb = path.stat().st_size / 2**20
    print(f"✅ {dt:.1f}s  单张（{dt/steps:.2f}s/步）  {size_mb:.2f}MB  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
