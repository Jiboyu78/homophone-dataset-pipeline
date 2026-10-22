# =============================================================================
# 一键生成「会意式单图猜成语/词语」训练数据集 —— Windows 入口
#
# 这只是个壳：真正的逻辑在跨平台的 scripts\oneclick.py 里，
# 三个平台（Windows / macOS / Linux）共用同一份实现，避免行为漂移。
#
# 用法（两种写法都认，随便挑）：
#     .\scripts\oneclick.ps1 -Limit 100
#     .\scripts\oneclick.ps1 --limit 100
#     .\scripts\oneclick.ps1 -NoImages -Workers 4 -Limit 4000
#     .\scripts\oneclick.ps1 --no-images --workers 4 --limit 4000
#     .\scripts\oneclick.ps1 -Reuse             # 复用上次保存的配置
#     .\scripts\oneclick.ps1 -LlmBatch 1        # 网关 60 秒超时时调小
#     .\scripts\oneclick.ps1 -ImportImages <目录>
#     .\scripts\oneclick.ps1 -Help              # 全部参数
#
# 注意：这是 PowerShell 脚本，别用 python 去执行它
#      （python 会把它当 Python 代码解析并报 SyntaxError）。
#      要用 Python 入口请执行：python scripts\oneclick.py --limit 100
#
# 为什么参数要自己解析：PowerShell 只按字面匹配参数名，
#   `-NoImages` 能绑上，而 `--no-images` 会被当成多余的位置参数**静默丢弃**——
#   既不报错也不生效（实测踩过：用户以为开了只出描述模式，其实没开）。
#   这里统一归一化两种写法，遇到不认识的标志直接报错退出，不再静默忽略。
# =============================================================================

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# ------------------------------------------------------------------ 归一化参数
# 键 = 去掉所有连字符并转小写后的名字，值 = 传给 oneclick.py 的正式标志
$flagMap = @{
    'limit'            = '--limit'
    'stages'           = '--stages'
    'reuse'            = '--reuse'
    'promptstyle'      = '--prompt-style'
    'imagemode'        = '--image-mode'
    'size'             = '--size'
    'llmbatch'         = '--llm-batch'
    'outdir'           = '--out-dir'
    'skipcheck'        = '--skip-check'
    'planproviderrule' = '--plan-rule'
    'planrule'         = '--plan-rule'
    'redoplan'         = '--redo-plan'
    'excludedegraded'  = '--exclude-degraded'
    'noimages'         = '--no-images'
    'workers'          = '--workers'
    'importimages'     = '--import-images'
    'printconfig'      = '--print-config'
    'help'             = '--help'
    'h'                = '--help'
}

$fwd = New-Object System.Collections.Generic.List[string]
$bad = New-Object System.Collections.Generic.List[string]

$i = 0
while ($i -lt $Rest.Count) {
    $tok = $Rest[$i]
    if ([string]::IsNullOrWhiteSpace($tok)) { $i++; continue }

    # 去掉前导横线、去掉内部连字符、转小写，这样 -NoImages 与 --no-images 归一成同一个键
    $key = $tok.TrimStart('-').Replace('-', '').ToLower()
    if (-not $flagMap.ContainsKey($key)) {
        $bad.Add($tok)
        $i++
        continue
    }
    $fwd.Add($flagMap[$key])

    # 带值的参数：把后面的值一并转交（开关型参数没有值）
    $takesValue = @('--limit', '--stages', '--prompt-style', '--image-mode', '--size',
                    '--llm-batch', '--out-dir', '--workers', '--import-images')
    if (($takesValue -contains $flagMap[$key]) -and ($i + 1 -lt $Rest.Count)) {
        $fwd.Add($Rest[$i + 1])
        $i += 2
        continue
    }
    $i++
}

if ($bad.Count -gt 0) {
    Write-Host ""
    Write-Host ("无法识别的参数：" + ($bad -join "、")) -ForegroundColor Red
    Write-Host "可用参数（-Xxx 与 --xxx 两种写法都行）：" -ForegroundColor Yellow
    Write-Host "  -Limit N          -Stages a,b        -Reuse            -OutDir DIR"
    Write-Host "  -NoImages         -Workers N         -ImportImages DIR"
    Write-Host "  -LlmBatch N       -PromptStyle X     -ImageMode X      -Size WxH"
    Write-Host "  -SkipCheck        -PlanProviderRule  -RedoPlan         -ExcludeDegraded"
    Write-Host "  -PrintConfig      -Help"
    exit 64
}

# ------------------------------------------------------------------ 找 Python
$cands = @(
    'D:\DSH\resources\runtime\primary-runtime\dependencies\python\python.exe',
    (Get-Command python  -ErrorAction SilentlyContinue).Source,
    (Get-Command python3 -ErrorAction SilentlyContinue).Source,
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
)
# Windows 上 python3 常常是微软商店的 0 字节别名存根，执行它没有任何反应。
# 所以优先选真正的 python.exe，并实际跑一次确认版本。
$py = $null
foreach ($c in $cands) {
    if (-not $c -or -not (Test-Path $c)) { continue }
    if ($c -like '*WindowsApps*') { continue }
    $ver = & $c -c "import sys;print(sys.version_info[0])" 2>$null
    if ($LASTEXITCODE -eq 0 -and "$ver" -eq "3") { $py = $c; break }
}
if (-not $py) {
    Write-Host "找不到可用的 Python 3。" -ForegroundColor Red
    Write-Host "请到 https://www.python.org/downloads/ 安装（勾选 Add to PATH）。" -ForegroundColor Yellow
    Write-Host "提示：Windows 上 python3 常是商店别名存根，本脚本会跳过它。" -ForegroundColor DarkGray
    exit 1
}

# ------------------------------------------------------------------ 执行
& $py (Join-Path $root 'scripts\oneclick.py') @fwd
exit $LASTEXITCODE
