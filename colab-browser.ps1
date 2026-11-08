[CmdletBinding()]
param(
    [ValidateSet('Login', 'Run')]
    [string]$Action = 'Run',
    [ValidatePattern('^[A-Za-z0-9_-]{1,100}$')]
    [string]$ProjectId,
    [ValidateSet('auto', 'chrome', 'edge')]
    [string]$Browser = 'auto',
    [string]$ApiUrl = 'http://127.0.0.1:8765'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonPath = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw 'AutoResearch virtual environment was not found. Run start.cmd once.'
}
Set-Location -LiteralPath $ProjectRoot
if ($Action -eq 'Login') {
    Write-Host 'Opening ordinary Chrome/Edge with the dedicated Colab profile. Complete Google login, then close that browser window before Run.'
    & $PythonPath -m autoresearch.colab_browser_cli --login-only --browser $Browser --root $ProjectRoot
    if ($LASTEXITCODE -ne 0) { throw "Colab manual login browser exited with code $LASTEXITCODE." }
    exit 0
}
if ([string]::IsNullOrWhiteSpace($ProjectId)) {
    throw 'ProjectId is required for Action Run.'
}
Write-Host 'Opening a dedicated visible Colab browser. Complete Google login and Drive permissions manually if requested.'
Write-Host 'AutoResearch stores retry times and checkpoints; no activity is sent to prolong Colab sessions.'
& $PythonPath -m autoresearch.colab_browser_cli --project $ProjectId --browser $Browser --api-url $ApiUrl --root $ProjectRoot
if ($LASTEXITCODE -ne 0) {
    throw "Colab browser worker exited with code $LASTEXITCODE."
}
