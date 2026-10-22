#!/usr/bin/env bash
# macOS / Linux 入口：把参数原样透传给跨平台的 oneclick.py。
#
# 用法：
#     bash scripts/oneclick.sh                       # 交互式问你要 key 和地址
#     bash scripts/oneclick.sh --limit 200            # 生成 200 条
#     bash scripts/oneclick.sh --reuse                # 复用上次保存的配置
#     bash scripts/oneclick.sh --stages ingest,select # 只筛选不出题（不花钱）
#     bash scripts/oneclick.sh --plan-rule            # 用本地规则出题器
#     bash scripts/oneclick.sh --help                 # 全部参数
#
# 首次使用如果想在 Finder 里双击运行，用同目录的「生成数据集.command」，
# 并先给它可执行位：chmod +x 生成数据集.command
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

PY=""
for cand in python3 python; do
  if command -v "$cand" >/dev/null 2>&1; then
    if "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,9) else 1)' 2>/dev/null; then
      PY="$cand"
      break
    fi
  fi
done
if [ -z "$PY" ]; then
  echo "找不到 Python 3.9+。" >&2
  echo "  macOS：brew install python   或到 python.org 下载安装包" >&2
  echo "  Linux：sudo apt install python3 python3-pip" >&2
  exit 1
fi

cd "$ROOT"
exec "$PY" scripts/oneclick.py "$@"
