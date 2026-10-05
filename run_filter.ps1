param([string]$Text)
$ErrorActionPreference = "Stop"
$taskRoot = $PSScriptRoot
$taskPython = Join-Path $taskRoot ".venv\Scripts\python.exe"
Push-Location -LiteralPath $taskRoot
try {
    if ($PSBoundParameters.ContainsKey("Text")) {
        & $taskPython -m filter10 predict $Text
    } else {
        & $taskPython -m filter10 interactive
    }
    if ($LASTEXITCODE -ne 0) { throw "filter1.0 exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
