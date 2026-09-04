$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectRoot

$PythonCommand = Get-Command python -ErrorAction Stop
$VenvPython = ".venv\Scripts\python.exe"
$NeedsInstall = $false

if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    & $PythonCommand.Source -m venv .venv
    $NeedsInstall = $true
}

if (-not $NeedsInstall) {
    & $VenvPython -c "import autoresearch, mcp" 2>$null
    $NeedsInstall = ($LASTEXITCODE -ne 0)
}

if ($NeedsInstall) {
    Write-Host "Installing AutoResearch dependencies. This may take a few minutes..."
    & $VenvPython -m pip install -e .
}

if (-not (Test-Path -LiteralPath ".env")) {
    Copy-Item -LiteralPath ".env.example" -Destination ".env"
    Write-Host "Created .env. Configure OpenRouter and AutoDL in the web settings."
}
& $VenvPython -m autoresearch
