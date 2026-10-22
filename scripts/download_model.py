"""底模下载器：走 hf-mirror（国内可达），带断点续传、进度显示与 safetensors 完整性校验。

为什么不用 curl：这个环境里 curl.exe 连不上外网（HTTP 000），但 Python 的 urllib 可以。
为什么不用 huggingface_hub：直连 huggingface.co 会超时，必须走 hf-mirror。

用法：
    python scripts/download_model.py --list Lykon/dreamshaper-8
    python scripts/download_model.py --repo Lykon/dreamshaper-8 --file dreamshaper_8.safetensors
    python scripts/download_model.py --repo Lykon/dreamshaper-8            # 自动挑一个
    python scripts/download_model.py --verify-only <路径.safetensors>
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
UA = {"User-Agent": "hv-model-downloader/1.0"}

DEFAULT_DEST = Path(r"G:\sd-webui-aki\sd-webui-aki-v4.11.1-cu128\models\Stable-diffusion")
# 挑文件时的偏好顺序：优先 fp16/pruned 单文件 safetensors，避开 fp32 与分片
PREFER = ["pruned", "fp16", "emaonly", "safetensors"]


def api(path: str) -> dict:
    url = f"{MIRROR}{path}"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def repo_files(repo: str) -> list[tuple[str, int]]:
    # blobs=true 才会带文件大小；不带的话 size 全是 0，没法判断该下哪个
    data = api(f"/api/models/{repo}?blobs=true")
    out: list[tuple[str, int]] = []
    for s in data.get("siblings", []):
        name = s.get("rfilename", "")
        if not name.lower().endswith((".safetensors", ".ckpt")):
            continue
        # 排除 diffusers 分片组件（unet/ vae/ text_encoder/ 之类），它们不是整模
        if "/" in name:
            continue
        size = int(s.get("size") or 0)
        out.append((name, size))
    return out


def pick_file(files: list[tuple[str, int]]) -> str:
    if not files:
        raise SystemExit("该仓库没有 safetensors/ckpt 文件")
    scored = []
    for name, size in files:
        low = name.lower()
        if "fp32" in low or "full" in low:
            continue
        score = sum(1 for i, kw in enumerate(PREFER) if kw in low)
        # 1.5~7GB 之间最像可用的单文件 SD1.5/SDXL 权重
        if 1.0 * 2**30 < size < 7.5 * 2**30:
            score += 3
        scored.append((score, -abs(size - 2.2 * 2**30), name, size))
    scored.sort(reverse=True)
    return scored[0][2]


def human(n: float) -> str:
    return f"{n/2**30:.2f} GB" if n > 2**30 else f"{n/2**20:.1f} MB"


def download(repo: str, filename: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(filename).name
    url = f"{MIRROR}/{repo}/resolve/main/{urllib.parse.quote(filename)}"

    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = dict(UA)
    if have:
        headers["Range"] = f"bytes={have}-"
        print(f"检测到未完成的下载，从 {human(have)} 处续传")

    req = urllib.request.Request(url, headers=headers)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            total = int(r.headers.get("Content-Length", 0)) + (have if have else 0)
            mode = "ab" if (have and r.status == 206) else "wb"
            if mode == "wb":
                have = 0
            done = have
            last = 0.0
            with part.open(mode) as f:
                while True:
                    chunk = r.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    now = time.time()
                    if now - last > 1.0:
                        last = now
                        sp = done / max(1e-6, now - t0)
                        pct = f"{done*100/total:.1f}%" if total else "?"
                        print(f"\r  下载中 {pct}  {human(done)}"
                              f"{' / ' + human(total) if total else ''}"
                              f"  {sp/2**20:.1f} MB/s", end="", flush=True)
        print()
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"下载失败 HTTP {exc.code}：{url}") from exc

    # 完整性校验：safetensors 头 + 数据段长度是否与文件大小自洽
    ok, msg = verify_safetensors(part)
    print(f"  校验：{msg}")
    if not ok:
        raise SystemExit("文件不完整，已保留 .part 供重试续传；请重跑本命令")
    if dest.exists():
        dest.unlink()
    part.rename(dest)
    print(f"  已保存：{dest}（{human(dest.stat().st_size)}）")
    return dest


def verify_safetensors(p: Path) -> tuple[bool, str]:
    if not p.exists():
        return False, "文件不存在"
    try:
        with p.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n <= 0 or n > 200 * 1024 * 1024:
                return False, f"头部长度异常({n})，非 safetensors 或已损坏"
            head = json.loads(f.read(n).decode("utf-8"))
        head.pop("__metadata__", None)
        declared = max(v["data_offsets"][1] for v in head.values())
        expect = 8 + n + declared
        diff = abs(expect - p.stat().st_size)
        if diff > 2 * 1024 * 1024:
            return False, (f"数据段 {human(declared)}，但文件 {human(p.stat().st_size)}，"
                           f"差 {diff/2**20:.1f}MB → 下载不完整")
        return True, f"张量 {len(head)} 个，数据段 {human(declared)}，结构自洽 ✓"
    except Exception as exc:  # noqa: BLE001
        return False, f"解析失败：{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Lykon/dreamshaper-8")
    ap.add_argument("--file", default="")
    ap.add_argument("--dest", default=str(DEFAULT_DEST))
    ap.add_argument("--list", metavar="REPO", default="")
    ap.add_argument("--verify-only", default="")
    args = ap.parse_args()

    if args.verify_only:
        ok, msg = verify_safetensors(Path(args.verify_only))
        print(f"{args.verify_only}\n  {msg}")
        return 0 if ok else 1

    if args.list:
        files = repo_files(args.list)
        print(f"{args.list} 里的权重文件（来自 {MIRROR}）：")
        for name, size in sorted(files, key=lambda x: -x[1]):
            print(f"  {human(size):>10}  {name}")
        if files:
            print(f"\n自动选择会挑：{pick_file(files)}")
        return 0

    files = repo_files(args.repo)
    name = args.file or pick_file(files)
    size = next((s for f, s in files if f == name), 0)
    print(f"仓库  ：{args.repo}")
    print(f"镜像  ：{MIRROR}")
    print(f"文件  ：{name}（{human(size)}）")
    print(f"目标  ：{args.dest}")
    download(args.repo, name, Path(args.dest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
