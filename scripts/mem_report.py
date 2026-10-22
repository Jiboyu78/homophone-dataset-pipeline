"""出图前的内存体检：判断"现在能不能安全起 SD"，避免把机器拖进交换。

    python scripts/mem_report.py
    python scripts/mem_report.py --want-width 768 --want-height 512 --also-llm

只读检查，不做任何改动。判定依据是实测出来的经验值：
    · SD1.5 fp16 权重约 2.0GB，--no-half(fp32) 会翻倍到约 4GB，加载峰值再高一截
    · SD1.5 512x512 推理激活约 1.5~2.5GB（768x512 约 2~3.5GB，1152x768 会明显更高）
    · llama.cpp Qwen3-8B Q5_K_M：显存 6.9GB / 主机内存约 6.1GB
"""
from __future__ import annotations

import argparse
import ctypes
import shutil
import subprocess


class _MemStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


GB = 1024 ** 3


def host_memory() -> tuple[float, float, float, float]:
    m = _MemStatus()
    m.dwLength = ctypes.sizeof(_MemStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return (m.ullTotalPhys / GB, m.ullAvailPhys / GB,
            m.ullTotalPageFile / GB, m.ullAvailPageFile / GB)


def gpu_memory() -> tuple[float, float, str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, errors="replace")
        line = (out.stdout or "").strip().splitlines()[0]
        name, used, total = [x.strip() for x in line.split(",")]
        return float(used) / 1024, float(total) / 1024, name
    except Exception as exc:  # noqa: BLE001
        return -1.0, -1.0, f"读取失败：{exc}"


def estimate_sd(width: int, height: int, half_precision: bool) -> tuple[float, float]:
    """粗略估算 SD1.5 的 (显存, 主机内存) 需求（GB）。"""
    weights_vram = 2.0 if half_precision else 4.0
    mpixels = (width * height) / (512 * 512)
    act = (1.5 if half_precision else 2.5) * mpixels
    vram = weights_vram + act + 0.8          # +VAE/编码器/碎片
    load_peak_ram = (2.0 if half_precision else 5.0) + 1.5   # 权重 + 转换/临时缓冲
    return vram, load_peak_ram


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--want-width", type=int, default=512)
    ap.add_argument("--want-height", type=int, default=512)
    ap.add_argument("--half", action="store_true",
                    help="SD 用 fp16 权重（--upcast-sampling）；不加则按 --no-half(fp32) 估算")
    ap.add_argument("--also-llm", action="store_true", help="是否同时还要跑 llama.cpp")
    args = ap.parse_args()

    total_ram, avail_ram, total_commit, avail_commit = host_memory()
    used_vram, total_vram, gpu_name = gpu_memory()
    free_vram = total_vram - used_vram
    vram_need, ram_need = estimate_sd(args.want_width, args.want_height, args.half)

    print("=" * 74)
    print("主机内存")
    print("=" * 74)
    print(f"  物理内存 : 总 {total_ram:.1f} GB，可用 {avail_ram:.1f} GB"
          f"（已用 {total_ram - avail_ram:.1f} GB）")
    print(f"  提交内存 : 上限 {total_commit:.1f} GB，可用 {avail_commit:.1f} GB")
    print("=" * 74)
    print("显存")
    print("=" * 74)
    print(f"  {gpu_name}")
    print(f"  已用 {used_vram:.2f} GB / 总 {total_vram:.2f} GB，空闲 {free_vram:.2f} GB")

    print("=" * 74)
    print(f"本次计划：SD {args.want_width}x{args.want_height}"
          f"{'（fp16 权重 + upcast-sampling）' if args.half else '（--no-half，fp32 权重）'}"
          f"{'，且同时运行 llama.cpp' if args.also_llm else '，llama.cpp 已停'}")
    print("=" * 74)
    llm_vram, llm_ram = (6.9, 6.1) if args.also_llm else (0.0, 0.0)
    need_vram = vram_need + llm_vram
    print(f"  预计需要显存 : {vram_need:.1f} GB（SD）"
          f"{f' + {llm_vram:.1f} GB（llama.cpp）' if args.also_llm else ''}"
          f" = {need_vram:.1f} GB")
    print(f"  预计需要内存 : {ram_need:.1f} GB（加载峰值）"
          f"{f' + {llm_ram:.1f} GB（llama.cpp 常驻）' if args.also_llm else ''}")
    print()

    vram_ok = need_vram < free_vram * 0.90
    ram_ok = (ram_need + llm_ram) < avail_ram * 0.85

    print(f"  显存检查 : {'✅ 安全' if vram_ok else '❌ 会爆'}"
          f"（需要 {need_vram:.1f} GB，空闲 {free_vram:.1f} GB，留 10% 余量）")
    print(f"  内存检查 : {'✅ 安全' if ram_ok else '❌ 会爆'}"
          f"（需要 {ram_need + llm_ram:.1f} GB，可用 {avail_ram:.1f} GB，留 15% 余量）")
    print()
    if not vram_ok or not ram_ok:
        print("建议：")
        if args.also_llm:
            print("  · 把 llama.cpp 停掉再出图（两个大模型不要同时驻留）")
        if not vram_ok:
            print("  · 分辨率降到 512x512（SD1.5 的原生分辨率，质量也最好）")
            print("  · 或用 --upcast-sampling 代替 --no-half（权重 fp16，省一半显存）")
        if not ram_ok:
            print("  · 内存不足时优先关掉其他占内存的程序；Windows 会走页面文件，"
                  "表现为卡死而不是报错")
        print("  · 加 --medvram 让 A1111 分块搬运权重")
    else:
        print("结论：按这个配置启动是安全的。")
    return 0 if (vram_ok and ram_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
