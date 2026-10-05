param([int]$Port = 8765, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$DavePython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $DavePython)) { throw '缺少项目 Python 环境 .venv，请先安装现有项目依赖。' }
$DaveArgs = @('-m', 'davework', '--port', "$Port")
if ($NoBrowser) { $DaveArgs += '--no-browser' }
& $DavePython @DaveArgs
if ($LASTEXITCODE -ne 0) { throw "Dave Work 退出码：$LASTEXITCODE" }
