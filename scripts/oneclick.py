"""一键生成「会意式单图猜成语/词语」训练数据集 —— 跨平台入口。

    python3 scripts/oneclick.py                    # 交互式问你要 key 和地址
    python3 scripts/oneclick.py --limit 200        # 生成 200 条
    python3 scripts/oneclick.py --reuse            # 复用上次保存的配置，不再问
    python3 scripts/oneclick.py --stages ingest,select   # 只筛选不出题（不花钱）
    python3 scripts/oneclick.py --plan-rule        # 用本地规则出题器（不需要文本大模型）

Windows 上可以双击 scripts\\生成数据集.bat（内部调用 oneclick.ps1）；
macOS / Linux 上用 bash scripts/oneclick.sh，或 Finder 里双击 生成数据集.command。

为什么入口用 Python 而不是 shell 脚本：同一份逻辑在 macOS / Linux / Windows 都能跑，
不需要用户处理可执行位与换行符，也避免了在两个平台上各写一份必然漂移的问题。

你只需要准备两个模型的 API Key 和请求地址：
    1) 描述模型：挑词、设计谜面、写图片描述（文本大模型，OpenAI 兼容）
    2) 出图模型：把图片描述变成图（文生图）
其余全自动：筛选可入画的词 -> 出题 -> 出图 -> OCR 质检 -> 打包训练格式。
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from make_run_config import build, validate  # noqa: E402
from pipeline.utils import read_json, write_json  # noqa: E402

USER_CFG = ROOT / "config.user.json"
RUN_CFG = ROOT / "config.run.json"


def say(msg: str = "", color: str = "") -> None:
    codes = {"red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
             "cyan": "\033[36m", "gray": "\033[90m"}
    if color and sys.stdout.isatty() and platform.system() != "Windows":
        print(f"{codes.get(color, '')}{msg}\033[0m")
    elif color and sys.stdout.isatty() and os.environ.get("WT_SESSION"):
        print(f"{codes.get(color, '')}{msg}\033[0m")
    else:
        print(msg)


def title(msg: str) -> None:
    say("")
    say("=" * 74, "cyan")
    say(msg, "cyan")
    say("=" * 74, "cyan")


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        val = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        val = ""
    return val or default


def ask_secret(prompt: str) -> str:
    """密钥尽量不回显；非交互环境（管道/EOF）退回普通输入，保证可脚本化。"""
    if sys.stdin.isatty():
        try:
            return getpass.getpass(f"{prompt}: ").strip()
        except (EOFError, getpass.GetpassWarning):
            return ""
    try:
        return input(f"{prompt}: ").strip()
    except EOFError:
        return ""


def collect(need_image: bool = True) -> dict:
    say("")
    if need_image:
        say("请填写两个模型的信息（直接回车用方括号里的默认值）", "cyan")
    else:
        say("请填写描述模型的信息（直接回车用方括号里的默认值）", "cyan")
    say("  描述模型 = 挑词/设计谜面/写图片描述的大模型（OpenAI 兼容 chat 接口）", "gray")
    say("     例 https://api.deepseek.com/v1  模型 deepseek-chat", "gray")
    say("     例 https://dashscope.aliyuncs.com/compatible-mode/v1  模型 qwen-max", "gray")
    say("")
    llm_base = ask("描述模型 请求地址", "https://api.deepseek.com/v1")
    llm_key = ask_secret("描述模型 API Key")
    llm_model = ask("描述模型 模型名", "deepseek-chat")

    if need_image:
        say("")
        say("  出图模型 = 文生图（openai=通用 /images/generations；dashscope=通义万相异步）", "gray")
        image_mode = ask("出图接口形态 openai/dashscope", "openai")
        image_base = ask("出图模型 请求地址", "https://api.openai.com/v1")
        image_key = ask_secret("出图模型 API Key")
        image_model = ask("出图模型 模型名", "gpt-image-1")
        image_size = ask("图片尺寸", "1024x1024")
    else:
        # 只出描述模式：这次运行完全不碰出图模型，问了也是白问，
        # 而且强行要求会逼用户填占位值（实测有人填了 0 然后被格式校验拒掉）。
        say("")
        say("  本次为「只出图片描述」模式，不需要出图模型 —— 已跳过。", "cyan")
        say("  以后要出图时，删掉 config.user.json 重新运行即可补填这两项。", "gray")
        image_mode, image_base, image_key = "openai", "", ""
        image_model, image_size = "", "1024x1024"

    return {
        "llm_base_url": llm_base, "llm_api_key": llm_key, "llm_model": llm_model,
        "image_mode": image_mode, "image_base_url": image_base,
        "image_api_key": image_key, "image_model": image_model,
        "image_size": image_size, "prompt_style": "prose",
    }


def save_user(user: dict) -> None:
    write_json(USER_CFG, user)
    try:
        os.chmod(USER_CFG, 0o600)       # 里面有 API Key，收一下权限
    except OSError:
        pass
    say(f"配置已保存到 {USER_CFG}（内含 API Key，别提交到 git）", "yellow")


def check_runtime() -> None:
    if sys.version_info < (3, 9):
        say(f"需要 Python 3.9+，当前是 {platform.python_version()}", "red")
        sys.exit(1)
    # Windows 上如果用的是 32 位 Python，OCR（onnxruntime）根本没有 32 位 wheel，
    # 文字泄漏检测就永远装不上——早点说清楚，别让用户以为是别的问题。
    import struct
    if sys.platform == "win32" and struct.calcsize("P") == 4:
        say("提示：当前是 32 位 Python。", "yellow")
        say("      Pillow（字形叠加/联系表）可以装，但 OCR 用的 onnxruntime 只有 64 位版本，")
        say("      因此「文字泄漏检测」在 32 位解释器下无法启用（质检会自动降级并注明）。")
        say("      想用完整质检，请改用 64 位 Python，或用 Windows 入口脚本：")
        say(r"        .\scripts\oneclick.ps1          它会自动优先选用 64 位解释器")
        say("")
    try:
        import PIL  # noqa: F401
    except ImportError:
        say("提示：未安装 Pillow，字形参与式（如「举手之劳」的「劳」）与联系表将不可用。", "yellow")
        say(f"      安装：{Path(sys.executable).name} -m pip install --user pillow", "yellow")
        say("")
    try:
        import sqlite3  # noqa: F401
    except ImportError:
        say("错误：这个 Python 缺 sqlite3 模块，无法建立状态库。", "red")
        sys.exit(1)


def _import_and_finish(args) -> int:
    """把外部生成的图片导回流水线，然后接着质检与打包。

    这条路不需要 API Key（出图已在别处完成），所以不要求先填配置。
    """
    run_cfg = read_json(RUN_CFG, default=None)
    if not isinstance(run_cfg, dict):
        run_cfg = {
            "paths": {"data_dir": "data", "output_dir": args.out_dir,
                      "images_dir": f"{args.out_dir}/images",
                      "log_file": f"{args.out_dir}/pipeline.log",
                      "db": f"{args.out_dir}/pipeline.db"},
            "ingest": {"files": ["成语大全_整理版(1).json",
                                 "homophone_answer_vocab.json"]},
            "select": {"min_score": 0.55, "min_coverage": 0.6},
            "package": {"splits": {"train": 0.8, "val": 0.1, "test": 0.1}, "seed": 42,
                        "only_accepted": True, "contact_sheets": True,
                        "export_formats": ["raw", "sft_chat", "multiple_choice",
                                           "llava", "classification"],
                        "exclude_degraded": bool(args.exclude_degraded),
                        "question_idiom": "这张图表达的是哪个成语？",
                        "question_word": "这张图表达的是哪个词语？"},
            "qc": {"ocr": {"enabled": True}, "vlm": {"provider": "none"},
                   "max_retry": 2},
        }
        say("未找到 config.run.json，已生成一份最小运行配置"
            "（导回外图不需要 API Key）", "gray")
    run_cfg.setdefault("paths", {})
    run_cfg["paths"]["output_dir"] = args.out_dir
    run_cfg["paths"]["images_dir"] = f"{args.out_dir}/images"
    run_cfg["paths"]["db"] = f"{args.out_dir}/pipeline.db"
    run_cfg["paths"]["log_file"] = f"{args.out_dir}/pipeline.log"
    run_cfg.setdefault("package", {})
    if args.exclude_degraded:
        run_cfg["package"]["exclude_degraded"] = True
    write_json(RUN_CFG, run_cfg)

    py = sys.executable or "python3"
    title("导回外部生成的图片")
    rc = subprocess.call([py, "-m", "pipeline.cli", "--config", "config.run.json",
                          "import-images", "--dir", args.import_images], cwd=str(ROOT))
    if rc != 0:
        say("导回失败或没有匹配到任何图片。"
            "请确认文件名里带了稳定 id（如 idi_xxxxxxxxxxxx）或答案文字。", "red")
        return rc

    title("质检 + 打包")
    rc = subprocess.call([py, "-m", "pipeline.cli", "--config", "config.run.json",
                          "run", "--stages", "qc,package"], cwd=str(ROOT))
    if rc == 4:
        say("质检后没有产出任何记录：可能图片都没通过质检。"
            f"看 {args.out_dir}/stats.md 与 pipeline.log 了解原因。", "red")
        return rc
    if rc != 0:
        say(f"退出码 {rc}，请查看 {args.out_dir}/pipeline.log", "red")
        return rc

    title("完成")
    out_full = ROOT / args.out_dir
    for f in ("data_card.md", "dataset.jsonl", "manifest.csv", "review.html", "stats.md"):
        p = out_full / f
        if p.exists():
            say(f"  {f:<34}{p.stat().st_size / 1024:>10.1f} KB", "green")
    img_dir = out_full / "images"
    if img_dir.is_dir():
        say(f"  {'images/(PNG)':<34}{len(list(img_dir.glob('*.png'))):>7} 张", "green")
    say("")
    say(f"数据集目录：{out_full}", "cyan")
    say("打开 review.html 可以人工复核；确认无误后即可用于训练。", "cyan")
    return 0


def _run_workers(args, py: str, stage_list: str, per: int, workers: int,
                 force: bool, no_images: bool) -> int:
    """并发跑 N 路，每路独立输出目录与状态库，跑完自动合并。

    为什么必须"每路独立"：
      · 共用一个状态库 → 4 个进程互相抢锁，且分片会互相把对方的候选踢掉；
      · 每路独立后，各自只处理自己那一片词，互不干扰，最后合并。
    日志各写各的（<out>/shardN/console.log），否则 4 路输出会糊在一起没法看。
    """
    import json as _json

    base = read_json(RUN_CFG, default={}) or {}
    shard_dirs: list[str] = []
    procs: list = []

    for i in range(1, workers + 1):
        d = f"{args.out_dir}/shard{i}"
        (ROOT / d).mkdir(parents=True, exist_ok=True)
        shard_dirs.append(d)
        cfg_i = _json.loads(_json.dumps(base))          # 深拷贝，避免相互污染
        cfg_i.setdefault("paths", {})
        cfg_i["paths"].update({"output_dir": d, "images_dir": f"{d}/images",
                               "log_file": f"{d}/pipeline.log", "db": f"{d}/pipeline.db"})
        cfg_i.setdefault("select", {})
        cfg_i["select"]["shard_index"] = i - 1
        cfg_i["select"]["shard_count"] = workers
        # 每路的额度用 max_items 精确控制，**不能**用 --limit-words：
        # limit-words 会把 ingest 也限制住，导致每路只读入前 N 个词
        # （实测 4 路 × 5 条 → 各自只 ingest 5 个词，合起来还是那 5 个）。
        cfg_i["select"]["max_items"] = max(1, per)
        cfg_path = ROOT / f"config.run.shard{i}.json"
        write_json(cfg_path, cfg_i)

        cmd = [py, "-m", "pipeline.cli", "--config", cfg_path.name, "run",
               "--stages", stage_list, "--limit", str(per), "--rounds", "3"]
        if force:
            cmd.append("--force")
        log_path = ROOT / d / "console.log"
        logf = log_path.open("w", encoding="utf-8")
        say(f"  [{i}/{workers}] 启动 -> {d}（日志 {d}/console.log）", "gray")
        procs.append((i, subprocess.Popen(cmd, cwd=str(ROOT), stdout=logf,
                                         stderr=subprocess.STDOUT), d, logf))

    # 轮询等待，顺便报心跳，免得长时间没输出让人以为卡死
    started = time.time()
    pending = {i for i, _p, _d, _f in procs}
    while pending:
        time.sleep(5)
        for i, p, _d, _f in procs:
            if i in pending and p.poll() is not None:
                pending.discard(i)
                say(f"  [{i}/{workers}] 结束，退出码 {p.returncode}", "gray")
        if pending:
            say(f"  … 还在跑：{len(pending)} 路，已用 {int(time.time() - started)} 秒", "gray")

    codes: list[tuple[int, int]] = []
    for i, p, _d, logf in procs:
        codes.append((i, p.returncode))
        logf.close()
    failed = [(i, rc) for i, rc in codes if rc != 0]
    if failed:
        say(f"  有 {len(failed)} 路非零退出：{failed}（仍会合并成功的部分）", "yellow")
        say(f"  对应日志：{args.out_dir}/shardN/console.log", "gray")

    say("")
    say("  合并各路结果…", "cyan")
    merge_cmd = [py, "-m", "pipeline.cli", "--config", "config.run.json", "merge",
                 "--dirs", ",".join(shard_dirs), "--out", args.out_dir]
    rc = subprocess.call(merge_cmd, cwd=str(ROOT))
    if rc != 0:
        say("  合并失败。可以手工重试：", "red")
        say(f"    python -m pipeline.cli merge --dirs {','.join(shard_dirs)} "
            f"--out {args.out_dir}", "gray")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(
        description="一键生成会意猜谜数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=50, help="生成多少条（默认 50）")
    ap.add_argument("--stages", default="all",
                    help="all 或逗号分隔的阶段，如 ingest,select")
    ap.add_argument("--reuse", action="store_true", help="复用 config.user.json，不再提问")
    ap.add_argument("--prompt-style", default="", choices=["", "prose", "tags"],
                    help="prose=完整句子（SDXL/云模型）｜tags=短标签（SD1.5）")
    ap.add_argument("--image-mode", default="", choices=["", "openai", "dashscope"])
    ap.add_argument("--size", default="", help="如 1024x1024")
    ap.add_argument("--llm-batch", type=int, default=4,
                    help="每批送给描述模型的词条数。网关 60 秒超时（504）时调小，如 1")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--skip-check", action="store_true", help="跳过第 0 步自检")
    ap.add_argument("--plan-rule", action="store_true",
                    help="用本地规则出题器（不需要文本大模型，谜面质量会下降）")
    ap.add_argument("--redo-plan", action="store_true",
                    help="重新为所有候选词出题（会再次调用大模型）")
    ap.add_argument("--exclude-degraded", action="store_true",
                    help="整体排除降级样本（元素仅并排铺开、未做语义合成）")
    ap.add_argument("--no-images", action="store_true",
                    help="只出图片描述不出图：跑完出题后导出待生图清单，"
                         "图你拿去别处生成（省钱路线）")
    ap.add_argument("--workers", type=int, default=1, metavar="N",
                    help="并发路数（默认 1）。N>1 时每路独立输出目录与状态库，"
                         "自动切片互不重复，跑完自动合并到 --out-dir")
    ap.add_argument("--import-images", default="", metavar="DIR",
                    help="把外部生成的图片导回流水线，然后继续质检与打包")
    ap.add_argument("--print-config", action="store_true",
                    help="只生成 config.run.json 后退出，不跑流水线")
    args = ap.parse_args()

    check_runtime()

    # ------------------------------------------------------------ 导入外图模式
    # 只做"导回 + 质检 + 打包"，不需要 key（出图在别处完成），所以放在最前面，
    # 免得用户为了导图还要重新填一遍配置。
    if args.import_images:
        return _import_and_finish(args)

    # ------------------------------------------------------------ 采集配置
    need_image = not args.no_images
    if args.reuse and USER_CFG.exists():
        user = read_json(USER_CFG, default={}) or {}
        say(f"已复用 {USER_CFG}")
        # 上次可能是用 --no-images 保存的，出图模型那几项是空的；
        # 这次要出图时给出针对性的提示，而不是笼统地让人"去掉 --reuse"。
        if need_image and not str(user.get("image_base_url") or "").strip():
            say("")
            say("出图模型信息是空的（上次是用 --no-images 模式保存的配置）。", "red")
            say("本次要真正出图，请先删掉 config.user.json 再重新运行以补填：", "yellow")
            say(f"  Remove-Item {USER_CFG.name}      # 或 macOS/Linux： rm {USER_CFG.name}", "yellow")
            return 2
    else:
        user = collect(need_image=need_image)
        bad = validate(user, require_image=need_image)
        if bad:
            say("")
            say("这些信息不完整或有误，配置没有保存：", "red")
            for b in bad:
                say(f"  - {b}", "red")
            say("")
            say("正确填法：请求地址形如 https://api.deepseek.com/v1", "yellow")
            say("          API Key  形如 sk-xxxxxxxx（一串密钥，不是网址）", "yellow")
            say("请重新运行，注意不要把这两项填反。", "yellow")
            return 2
        save_user(user)

    if not USER_CFG.exists():
        say(f"配置里缺少模型地址（{USER_CFG}）。", "red")
        say("请去掉 --reuse 重新运行一次，把两项地址和 key 填进去。", "yellow")
        return 2

    # ------------------------------------------------------------ 运行配置
    ns = argparse.Namespace(
        out_dir=args.out_dir, limit=args.limit, llm_batch=args.llm_batch,
        llm_timeout=300, concurrency=1,
        prompt_style=args.prompt_style or str(user.get("prompt_style") or "prose"),
        image_mode=args.image_mode or str(user.get("image_mode") or "openai"),
        image_size=args.size or str(user.get("image_size") or "1024x1024"),
        plan_provider="rule" if args.plan_rule else "llm",
        exclude_degraded=args.exclude_degraded,
        no_images=args.no_images,
    )
    bad = validate(user, require_image=need_image)
    if bad:
        say("配置有问题，未生成运行配置：", "red")
        for b in bad:
            say(f"  - {b}", "red")
        say(f"\n删掉 {USER_CFG.name} 后重新运行即可重新填写。", "yellow")
        return 2
    cfg = build(user, ns)
    write_json(RUN_CFG, cfg)
    say(f"运行配置已生成：{RUN_CFG.name}", "gray")
    say(f"  目标条数 {args.limit}，送去出题 {cfg['select']['max_items']} 条，"
        f"出题方式 {cfg['plan']['provider']}，每批 {args.llm_batch} 条", "gray")
    if args.print_config:
        return 0

    py = sys.executable or "python3"

    # ------------------------------------------------------------ 自检
    if not args.skip_check:
        title("第 0 步：自检（确认 key 与地址齐全、词表可读、OCR 可用）")
        subprocess.call([py, "-m", "pipeline.cli", "--config", "config.run.json", "health"],
                        cwd=str(ROOT))

    # ------------------------------------------------------------ 跑流水线
    if args.stages != "all":
        stage_list = args.stages
    elif args.no_images:
        # 只出描述不出图：跑完出题后导出待生图清单
        stage_list = "ingest,select,plan,tasks"
    else:
        stage_list = "ingest,select,plan,tasks,generate,qc,package"
    max_words = max(40, args.limit * 3)
    workers = max(1, int(args.workers))
    title("开始生成" + ("（只出图片描述，不出图）" if args.no_images else ""))
    say(f"  阶段：{stage_list}   目标：{args.limit} 条")
    if workers > 1:
        per = -(-args.limit // workers)          # 向上取整
        per_words = max(40, per * 3)
        say(f"  并发 {workers} 路：每路约 {per} 条（各自约 {per_words} 条送去出题），")
        say("  每路一个独立输出目录与独立状态库，跑完自动合并成一份数据集。")
    else:
        say(f"  词表全量读取并打分（本地计算，不花钱），其中约 {max_words} 条送去大模型出题，")
    if args.no_images:
        say("  只调用描述模型；不调用出图模型，因此不产生出图费用。")
    else:
        say(f"  出题通过的才调用出图模型（约 {args.limit} 条，另加未通过时的少量重试）。")
        say(f"  出图未通过质检的条目会自动带修正建议重试，重试不占 {args.limit} 额度。")
    say("")

    if workers > 1:
        started = time.time()
        code = _run_workers(args, py, stage_list, -(-args.limit // workers),
                            workers, args.redo_plan, args.no_images)
        elapsed = time.time() - started
        if code == 0:
            code = 0
        else:
            return code
    else:
        cmd = [py, "-m", "pipeline.cli", "--config", "config.run.json", "run",
               "--stages", stage_list, "--limit", str(args.limit),
               # 同时约束"非出图阶段"（select/plan/tasks）的处理上限，
               # 否则"只出描述"模式会把整个候选池都导出成任务，与 --limit 不符
               "--limit-words", str(args.limit), "--rounds", "3"]
        if args.redo_plan:
            say("  --redo-plan：将重新为所有候选词出题（会再次调用大模型）", "yellow")
            cmd.append("--force")
        started = time.time()
        code = subprocess.call(cmd, cwd=str(ROOT))
        elapsed = time.time() - started

    if code == 3:
        say("")
        say("失败：描述模型不可用（连不上 / key 不对 / 网关超时 / 域名被拦截）。", "red")
        say("先用分层诊断看清卡在哪一层：", "yellow")
        say(f"  {Path(sys.executable).name} scripts/check_endpoint.py "
            f"<你的请求地址> --key <你的key> --model <模型名>", "yellow")
        say("")
        say("对照结论处理：")
        say("  · TLS 握手就失败        → 中间有代理/拦截，或地址写错")
        say("  · SSL: UNEXPECTED_EOF   → TLS 能握手但请求一发出就被切断，")
        say("                            该域名在你当前网络下被拦截。换服务商地址")
        say("                            （本机验证可直连：api.deepseek.com、")
        say("                             dashscope.aliyuncs.com/compatible-mode/v1）")
        say("  · HTTP 401/403         → 网络没问题，是 key 或模型名不对")
        say("  · 60 秒后 504          → 网关超时，把每批条数调小：--llm-batch 1")
        say("")
        say("只想用本地规则出题器（不调用大模型，谜面质量会下降）时，加 --plan-rule 重跑。",
            "gray")
        return 3
    if code == 4:
        say("")
        say("失败：一张图都没生成出来（数据集体积为 0）。", "red")
        say("最常见原因是出图模型的 base_url / api_key / model 不对，"
            "或该服务不支持所选接口形态。", "yellow")
        say("建议先跑：python3 scripts/oneclick.py --limit 2  观察第一张图的报错。", "gray")
        return 4
    if code != 0:
        say(f"流水线退出码 {code}，请查看 {args.out_dir}/pipeline.log", "red")
        return code

    # ------------------------------------------------------------ 只出描述模式
    if args.no_images:
        title("图片描述已全部生成（未出图）")
        out_full = ROOT / args.out_dir
        for f in ("image_tasks.csv", "image_tasks.jsonl", "README_image_tasks.md"):
            p = out_full / f
            if p.exists():
                say(f"  {f:<34}{p.stat().st_size / 1024:>10.1f} KB", "green")
        pd = out_full / "prompts"
        if pd.is_dir():
            n = len(list(pd.glob("*.txt")))
            say(f"  {'prompts/*.txt（每条一个）':<34}{n:>7} 个", "green")
        say("")
        say(f"清单目录：{out_full}", "cyan")
        say("把 image_tasks.csv 交给你的生图工具（桌面端会员那种），", "cyan")
        say("按「文件名」列命名输出文件，然后导回本流水线：", "cyan")
        say("")
        say(f"  python3 scripts/oneclick.py --import-images <图片目录>", "yellow")
        say("")
        say("导回会自动完成：格式统一成 PNG、按文件名匹配答案、"
            "给字形参与式的图叠字、", "gray")
        say("登记入库，然后立刻跑质检与打包，产出完整训练数据集。", "gray")
        return 0

    # ------------------------------------------------------------ 交付摘要
    title(f"完成，用时 {int(elapsed // 60)} 分 {int(elapsed % 60)} 秒")
    out_full = ROOT / args.out_dir
    files = ["data_card.md", "dataset.jsonl", "train.jsonl", "val.jsonl", "test.jsonl",
             "sft_chat_train.jsonl", "multiple_choice_train.jsonl", "llava_train.jsonl",
             "classification_train.jsonl", "labels.json", "manifest.csv",
             "review.html", "stats.md"]
    for f in files:
        p = out_full / f
        if p.exists():
            say(f"  {f:<34}{p.stat().st_size / 1024:>10.1f} KB", "green")
    img_dir = out_full / "images"
    if img_dir.is_dir():
        n = len(list(img_dir.glob("*.png")))
        say(f"  {'images/(PNG)':<34}{n:>7} 张", "green")
    say("")
    say(f"数据集目录：{out_full}", "cyan")
    say("先看 data_card.md 了解字段与分布；review.html 可直接用浏览器打开做人工复核。", "cyan")
    say("训练格式各有 train/val/test 三份，按你的框架取用即可。", "cyan")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        raise SystemExit(130)
