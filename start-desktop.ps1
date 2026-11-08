param(
    [ValidatePattern('^[A-Za-z0-9_-]+$')]
    [string]$ProjectId = '5801a3a4eec4',
    [switch]$NoOpen
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$DesktopPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$DesktopData = Join-Path $ProjectRoot 'data'
$DesktopMutex = $null
$DesktopMutexOwned = $false

function Get-AutoResearchHealth([int]$Port) {
    try {
        $DesktopHealth = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 2
        if ($DesktopHealth.ok -eq $true -and $DesktopHealth.application_root -and
            [IO.Path]::GetFullPath($DesktopHealth.application_root) -eq $ProjectRoot) {
            return $DesktopHealth
        }
    } catch { }
    return $null
}

try {
    if (-not (Test-Path -LiteralPath $DesktopPython -PathType Leaf)) {
        throw 'Python environment is missing. Run start.cmd once to install AutoResearch.'
    }
    New-Item -ItemType Directory -Path $DesktopData -Force | Out-Null
    Set-Location -LiteralPath $ProjectRoot
    $DesktopRootHash = [Security.Cryptography.SHA256]::Create()
    try {
        $DesktopHash = [BitConverter]::ToString($DesktopRootHash.ComputeHash([Text.Encoding]::UTF8.GetBytes($ProjectRoot))).Replace('-', '')
    } finally { $DesktopRootHash.Dispose() }
    $DesktopMutex = [Threading.Mutex]::new($false, "Local\AutoResearchDesktop_$($DesktopHash.Substring(0, 16))")
    try { $DesktopMutexOwned = $DesktopMutex.WaitOne(45000) }
    catch [Threading.AbandonedMutexException] { $DesktopMutexOwned = $true }
    if (-not $DesktopMutexOwned) { throw 'Another AutoResearch launch is still in progress. Try again shortly.' }

    $DesktopPreferredPort = & $DesktopPython -c 'from autoresearch.config import Settings; print(Settings.load().port)'
    if ($LASTEXITCODE -ne 0) { throw 'AutoResearch dependencies are missing. Run start.cmd once.' }
    $DesktopPreferredPort = [int]$DesktopPreferredPort
    if ($DesktopPreferredPort -lt 1024 -or $DesktopPreferredPort -gt 65525) { throw 'Invalid AutoResearch port setting.' }
    $DesktopCandidates = @(@($DesktopPreferredPort, 8765, 8766, 8767) + @(($DesktopPreferredPort + 1)..($DesktopPreferredPort + 10)) | Select-Object -Unique)
    $DesktopListeningPorts = @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty LocalPort -Unique)
    $DesktopApiPort = $null
    foreach ($DesktopCandidate in $DesktopCandidates) {
        if (($DesktopListeningPorts -contains $DesktopCandidate) -and (Get-AutoResearchHealth $DesktopCandidate)) {
            $DesktopApiPort = $DesktopCandidate
            break
        }
    }
    if ($null -eq $DesktopApiPort) {
        foreach ($DesktopCandidate in $DesktopCandidates) {
            if ($DesktopListeningPorts -notcontains $DesktopCandidate) {
                $DesktopApiPort = $DesktopCandidate
                break
            }
        }
        if ($null -eq $DesktopApiPort) { throw 'No local port is available for AutoResearch.' }
        $env:AUTORESEARCH_HOST = '127.0.0.1'
        $env:AUTORESEARCH_PORT = [string]$DesktopApiPort
        $env:PYTHONUTF8 = '1'
        $env:PYTHONIOENCODING = 'utf-8'
        $DesktopAppProcess = Start-Process -FilePath $DesktopPython -ArgumentList @('-u', '-m', 'autoresearch', '--no-open') `
            -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $DesktopData "desktop-server-$DesktopApiPort.stdout.log") `
            -RedirectStandardError (Join-Path $DesktopData "desktop-server-$DesktopApiPort.stderr.log")
        $DesktopDeadline = [DateTime]::UtcNow.AddSeconds(40)
        while (-not (Get-AutoResearchHealth $DesktopApiPort)) {
            if ($DesktopAppProcess.HasExited -or [DateTime]::UtcNow -ge $DesktopDeadline) {
                throw "AutoResearch did not start. See data\desktop-server-$DesktopApiPort.stderr.log."
            }
            Start-Sleep -Milliseconds 250
        }
    }
    $DesktopUrl = "http://127.0.0.1:$DesktopApiPort/?project=$([Uri]::EscapeDataString($ProjectId))"
    Write-Output $DesktopUrl
    if (-not $NoOpen) { Start-Process -FilePath $DesktopUrl }
} catch {
    $DesktopError = "AutoResearch could not start: $($_.Exception.Message)"
    if ($NoOpen) { throw }
    Add-Type -AssemblyName PresentationFramework
    [Windows.MessageBox]::Show($DesktopError, 'AutoResearch') | Out-Null
    exit 1
} finally {
    if ($DesktopMutexOwned) { $DesktopMutex.ReleaseMutex() }
    if ($null -ne $DesktopMutex) { $DesktopMutex.Dispose() }
}
