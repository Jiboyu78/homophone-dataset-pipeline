#!/bin/bash
# macOS 上可双击运行的入口（Finder 里双击本文件即可）。
#
# 首次使用若提示"无法打开，因为它来自身份不明的开发者"：
#     右键 → 打开 → 仍要打开
# 若提示没有执行权限，在终端里执行一次：
#     chmod +x "生成数据集.command"
cd "$(dirname "$0")" || exit 1
bash ./oneclick.sh "$@"
echo ""
read -r -p "按回车关闭窗口..." _ || true
