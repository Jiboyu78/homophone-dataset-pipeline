"""交付前环境体检：决定该下哪个底模、以及 P100 能不能跑。

只读检查，不改动任何东西。
"""
from __future__ import annotations

import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

SD = Path(r"G:\sd-webui-aki\sd-webui-aki-v4.11.1-cu128")
REPORT: list[str] = []


def line(s: str = "") -> None:
    print(s)
    REPORT.append(s)


def check_safetensors(p: Path) -> str:
    """safetensors 头部是 uint64 长度 + JSON；能解析出来就说明文件结构正常。"""
    if not p.exists():
        return "文件不存在"
    size_gb = p.stat().st_size / 2**30
    try:
        with p.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n <= 0 or n > 200 * 1024 * 1024:
                return f"{size_gb:.2f}GB 头部长度异常({n})，疑似损坏或非 safetensors"
            head = json.loads(f.read(n).decode("utf-8"))
        if "__metadata__" in head:
            head.pop("__metadata__")
        n_tensors = len(head)
        # data_offsets 是"数据段内的累计偏移"，所以要取最大值（数据段总长），
        # 累加它们会得到一个荒谬的大数。
        declared = max((v.get("data_offsets", [0, 0])[1] for v in head.values()),
                       default=0)
        expect = 8 + n + declared
        diff = abs(expect - p.stat().st_size)
        ok = diff < 1024 * 1024
        return (f"{size_gb:.2f}GB 张量 {n_tensors} 个，数据段 {declared/2**30:.2f}GB，"
                f"{'结构自洽 ✓' if ok else f'与文件实际大小不符（差 {diff/2**20:.1f}MB）→ 下载不完整 ✗'}")
    except Exception as exc:  # noqa: BLE001
        return f"{size_gb:.2f}GB 解析失败：{type(exc).__name__}: {exc}"


def check_torch_arch() -> None:
    """P100 是 sm_60。torch 的 cu128 构建如果没编 sm_60，SD 根本跑不起来。"""
    cands = [SD / "python" / "python.exe", SD / "venv" / "Scripts" / "python.exe"]
    py = next((c for c in cands if c.exists()), None)
    line(f"  内置 Python: {py if py else '未找到'}")
    if not py:
        return
    code = (
        "import json,torch;print(json.dumps({'ver':torch.__version__,"
        "'cuda':torch.version.cuda,'avail':torch.cuda.is_available(),"
        "'arch':torch.cuda.get_arch_list()}))"
    )
    try:
        out = subprocess.run([str(py), "-c", code], capture_output=True, text=True,
                             timeout=180, errors="replace")
        info = None
        for raw in out.stdout.splitlines():
            raw = raw.strip()
            if raw.startswith("{"):
                try:
                    info = json.loads(raw)
                except json.JSONDecodeError:
                    pass
        if not info:
            line(f"  读取 torch 信息失败：{(out.stdout or out.stderr)[-300:]}")
            return
        arch = info.get("arch") or []
        line(f"  torch {info['ver']} / CUDA {info['cuda']} / 可用={info['avail']}")
        line(f"  编译架构: {arch}")
        has60 = any(a in ("sm_60", "sm_61", "sm_62", "compute_60") for a in arch)
        if has60:
            line("  ✅ 含 sm_60，可以直接在 P100 上跑")
        else:
            line("  ❌ 不含 sm_60 —— 这个 torch 构建在 P100 上会报 "
                 "'no kernel image is available'，换底模也跑不了，必须先换 torch")
    except Exception as exc:  # noqa: BLE001
        line(f"  执行失败：{type(exc).__name__}: {exc}")


def main() -> int:
    line("=" * 78)
    line("1) 现有底模完整性")
    line("=" * 78)
    for p in sorted((SD / "models" / "Stable-diffusion").rglob("*.safetensors")):
        line(f"  {p.relative_to(SD)}")
        line(f"    -> {check_safetensors(p)}")
    if not list((SD / "models" / "Stable-diffusion").rglob("*.safetensors")):
        line("  （没有任何底模）")

    line("")
    line("=" * 78)
    line("2) torch 是否支持 P100 (sm_60) —— 这决定要不要先换 torch")
    line("=" * 78)
    check_torch_arch()

    line("")
    line("=" * 78)
    line("3) 磁盘空间")
    line("=" * 78)
    for drive in ("C:\\", "G:\\", "D:\\"):
        try:
            u = shutil.disk_usage(drive)
            line(f"  {drive} 可用 {u.free/2**30:.1f} GB / 共 {u.total/2**30:.1f} GB")
        except Exception as exc:  # noqa: BLE001
            line(f"  {drive} 读取失败 {exc}")

    line("")
    line("=" * 78)
    line("4) 下载工具")
    line("=" * 78)
    for tool in ("curl.exe", "aria2c.exe"):
        p = shutil.which(tool)
        line(f"  {tool}: {p or '未找到'}")
    try:
        import huggingface_hub  # noqa: F401
        line("  huggingface_hub: 已安装")
    except Exception:  # noqa: BLE001
        line("  huggingface_hub: 未安装（可用 curl 直接下）")

    (Path(__file__).resolve().parent.parent / "output" / "env_report.txt").write_text(
        "\n".join(REPORT), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
