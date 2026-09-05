$ErrorActionPreference = "Stop"

$pythonPath = Join-Path $env:USERPROFILE "anaconda3\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "未找到 Anaconda Python：$pythonPath"
}

Set-Location -LiteralPath $PSScriptRoot
Write-Host "正在启动 Agent Evaluation 实验结果前端..."
Write-Host "访问：http://127.0.0.1:8081/frontend/"
& $pythonPath -m http.server 8081 --bind 127.0.0.1
