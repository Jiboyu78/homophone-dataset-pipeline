# 会意猜谜数据集 pipeline 启动脚本（Windows PowerShell）
# 用法：
#   .\scripts\run.ps1 health
#   .\scripts\run.ps1 run --mock --limit 30
#   .\scripts\run.ps1 run --stages generate,qc --limit 200 --rounds 3
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# 中文输出不乱码
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# 优先用 DSH 自带的 64 位 Python（含 Pillow），否则退回系统 python
$candidates = @(
    'D:\DSH\resources\runtime\primary-runtime\dependencies\python\python.exe',
    (Get-Command python -ErrorAction SilentlyContinue).Source
) | Where-Object { $_ -and (Test-Path $_) }

if (-not $candidates) {
    Write-Error '找不到可用的 Python，请修改 scripts\run.ps1 里的路径。'
}
$py = $candidates[0]

if (-not $Args -or $Args.Count -eq 0) {
    Write-Host "用法: .\scripts\run.ps1 <子命令> [参数]" -ForegroundColor Cyan
    Write-Host "  health | run | inspect <词> | stats | prompts | review | reset | apply-decisions"
    Write-Host ""
    Write-Host "示例："
    Write-Host "  .\scripts\run.ps1 health"
    Write-Host "  .\scripts\run.ps1 run --mock --limit 30"
    exit 0
}

Write-Host "Python: $py" -ForegroundColor DarkGray
& $py -m pipeline.cli @Args
exit $LASTEXITCODE
