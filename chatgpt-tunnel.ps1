[CmdletBinding()]
param(
    [ValidateSet("Setup", "Run", "Doctor", "Status")]
    [string]$Action = "Setup",
    [string]$TunnelId = "",
    [string]$Profile = "autoresearch",
    [string]$TunnelClientPath = "",
    [switch]$NoRun,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$LocalAppDataPath = [Environment]::GetFolderPath("LocalApplicationData")
if ([string]::IsNullOrWhiteSpace($LocalAppDataPath)) {
    $LocalAppDataPath = Join-Path $ProjectRoot ".local"
}
$StateDirectory = Join-Path $LocalAppDataPath "AutoResearch"
$ConfigPath = Join-Path $StateDirectory "tunnel.json"
$SecretPath = Join-Path $StateDirectory "tunnel-api-key.dpapi"
$RuntimePath = Join-Path $StateDirectory "tunnel-status.json"
$McpServerUrl = "http://127.0.0.1:8765/mcp"
$PlatformUrl = "https://platform.openai.com/settings/organization/tunnels"
$ChatGPTPluginsUrl = "https://chatgpt.com/plugins"

function Find-TunnelClient {
    param([string]$PreferredPath)

    if (-not [string]::IsNullOrWhiteSpace($PreferredPath)) {
        $resolved = Resolve-Path -LiteralPath $PreferredPath -ErrorAction SilentlyContinue
        if ($resolved) { return $resolved.Path }
    }

    $command = Get-Command "tunnel-client" -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }

    $bundled = Join-Path $ProjectRoot "tools\tunnel-client.exe"
    if (Test-Path -LiteralPath $bundled -PathType Leaf) { return $bundled }

    $downloads = Join-Path ([Environment]::GetFolderPath("UserProfile")) "Downloads"
    if (Test-Path -LiteralPath $downloads -PathType Container) {
        $downloaded = Get-ChildItem -LiteralPath $downloads -Filter "tunnel-client*.exe" -File -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if ($downloaded) { return $downloaded.FullName }
    }
    return $null
}

function Read-TunnelConfig {
    if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) { return $null }
    try {
        return Get-Content -Raw -LiteralPath $ConfigPath | ConvertFrom-Json
    }
    catch {
        throw "Tunnel configuration is invalid. Run setup-chatgpt.cmd again."
    }
}

function Convert-ToPlainText {
    param([Security.SecureString]$SecureValue)

    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecureValue)
    try {
        return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
}

function Invoke-TunnelClient {
    param(
        [string]$Executable,
        [Security.SecureString]$SecureApiKey,
        [string[]]$Arguments
    )

    $previousKey = [Environment]::GetEnvironmentVariable("CONTROL_PLANE_API_KEY", "Process")
    $plainKey = Convert-ToPlainText $SecureApiKey
    try {
        [Environment]::SetEnvironmentVariable("CONTROL_PLANE_API_KEY", $plainKey, "Process")
        & $Executable @Arguments
        $exitCode = $LASTEXITCODE
    }
    finally {
        [Environment]::SetEnvironmentVariable("CONTROL_PLANE_API_KEY", $previousKey, "Process")
        $plainKey = $null
    }
    if ($exitCode -ne 0) {
        throw "tunnel-client exited with code $exitCode."
    }
}

function Test-LocalMcp {
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:8765/api/health" -TimeoutSec 3
        return $health.ok -eq $true -and $health.chatgpt_bridge -eq "ready"
    }
    catch { return $false }
}

function Ensure-LocalMcp {
    if (Test-LocalMcp) { return }

    Write-Host "Starting AutoResearch in the background..."
    $startScript = Join-Path $ProjectRoot "start.ps1"
    $arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$startScript`""
    Start-Process -FilePath "powershell.exe" -ArgumentList $arguments -WorkingDirectory $ProjectRoot -WindowStyle Hidden
    for ($attempt = 0; $attempt -lt 40; $attempt++) {
        Start-Sleep -Milliseconds 500
        if (Test-LocalMcp) { return }
    }
    throw "AutoResearch did not become ready. Run start.cmd and inspect its error output."
}

function Read-SavedApiKey {
    if (-not (Test-Path -LiteralPath $SecretPath -PathType Leaf)) {
        throw "The encrypted runtime API key is missing. Run setup-chatgpt.cmd first."
    }
    try {
        return Get-Content -Raw -LiteralPath $SecretPath | ConvertTo-SecureString
    }
    catch {
        throw "The runtime API key cannot be decrypted by this Windows user. Run setup-chatgpt.cmd again."
    }
}

function Invoke-Doctor {
    param($Config, [Security.SecureString]$SecureApiKey)

    Ensure-LocalMcp
    Invoke-TunnelClient -Executable $Config.tunnel_client -SecureApiKey $SecureApiKey -Arguments @(
        "doctor", "--profile", $Config.profile, "--explain"
    )
}

function Invoke-Run {
    param($Config, [Security.SecureString]$SecureApiKey)

    Ensure-LocalMcp
    New-Item -ItemType Directory -Path $StateDirectory -Force | Out-Null
    @{
        pid = $PID
        profile = $Config.profile
        started_at = (Get-Date).ToUniversalTime().ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath $RuntimePath -Encoding UTF8

    Write-Host ""
    Write-Host "AutoResearch MCP: $McpServerUrl"
    Write-Host "Tunnel profile: $($Config.profile)"
    Write-Host "Keep this window open while using AutoResearch in ChatGPT."
    Write-Host "Press Ctrl+C to disconnect."
    Write-Host ""
    try {
        Invoke-TunnelClient -Executable $Config.tunnel_client -SecureApiKey $SecureApiKey -Arguments @(
            "run", "--profile", $Config.profile
        )
    }
    finally {
        if (Test-Path -LiteralPath $RuntimePath -PathType Leaf) {
            Remove-Item -LiteralPath $RuntimePath -Force
        }
    }
}

if ($Action -eq "Status") {
    $config = Read-TunnelConfig
    Write-Host "Local MCP ready: $(Test-LocalMcp)"
    Write-Host "Tunnel configured: $($null -ne $config -and (Test-Path -LiteralPath $SecretPath))"
    if ($config) {
        Write-Host "Profile: $($config.profile)"
        Write-Host "Tunnel client: $($config.tunnel_client)"
    }
    exit 0
}

if ($Action -eq "Setup") {
    Ensure-LocalMcp
    New-Item -ItemType Directory -Path $StateDirectory -Force | Out-Null

    $client = Find-TunnelClient $TunnelClientPath
    if (-not $client) {
        if (-not $NoBrowser) { Start-Process $PlatformUrl }
        Write-Host ""
        Write-Host "Download tunnel-client for Windows from OpenAI Platform tunnel settings."
        Write-Host "After downloading, place tunnel-client.exe in the project tools folder or Downloads folder."
        $enteredPath = Read-Host "Enter the downloaded tunnel-client.exe path (or press Enter to search again)"
        $client = Find-TunnelClient $enteredPath
    }
    if (-not $client) {
        throw "tunnel-client.exe was not found. Download it from $PlatformUrl and run setup-chatgpt.cmd again."
    }

    if ([string]::IsNullOrWhiteSpace($TunnelId)) {
        if (-not $NoBrowser) { Start-Process $PlatformUrl }
        $TunnelId = Read-Host "Enter the tunnel_id created in OpenAI Platform"
    }
    if ($TunnelId -notmatch '^tunnel_[A-Za-z0-9_-]+$') {
        throw "Tunnel ID must start with tunnel_."
    }
    if ([string]::IsNullOrWhiteSpace($Profile) -or $Profile -notmatch '^[A-Za-z0-9._-]+$') {
        throw "Profile may contain only letters, numbers, dot, underscore, and dash."
    }

    $secureApiKey = Read-Host "Enter the OpenAI tunnel runtime API key (input is hidden)" -AsSecureString
    $plainCheck = Convert-ToPlainText $secureApiKey
    if ([string]::IsNullOrWhiteSpace($plainCheck)) {
        throw "Runtime API key cannot be empty."
    }
    $plainCheck = $null

    Invoke-TunnelClient -Executable $client -SecureApiKey $secureApiKey -Arguments @(
        "init",
        "--profile", $Profile,
        "--tunnel-id", $TunnelId,
        "--mcp-server-url", $McpServerUrl
    )

    $secureApiKey | ConvertFrom-SecureString | Set-Content -LiteralPath $SecretPath -Encoding UTF8
    @{
        profile = $Profile
        tunnel_id = $TunnelId
        tunnel_client = $client
        mcp_server_url = $McpServerUrl
        configured_at = (Get-Date).ToUniversalTime().ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath $ConfigPath -Encoding UTF8

    $config = Read-TunnelConfig
    Invoke-Doctor -Config $config -SecureApiKey $secureApiKey

    Write-Host ""
    Write-Host "Local tunnel setup passed."
    Write-Host "In ChatGPT: Settings > Security and login > enable Developer mode."
    Write-Host "Then create a developer-mode app at $ChatGPTPluginsUrl and select Tunnel."
    if (-not $NoBrowser) { Start-Process $ChatGPTPluginsUrl }

    if (-not $NoRun) {
        $answer = Read-Host "Start the tunnel now? [Y/n]"
        if ([string]::IsNullOrWhiteSpace($answer) -or $answer -match '^[Yy]') {
            Invoke-Run -Config $config -SecureApiKey $secureApiKey
        }
    }
    exit 0
}

$savedConfig = Read-TunnelConfig
if (-not $savedConfig) {
    throw "Tunnel is not configured. Run setup-chatgpt.cmd first."
}
if (-not (Test-Path -LiteralPath $savedConfig.tunnel_client -PathType Leaf)) {
    throw "Configured tunnel-client was not found. Run setup-chatgpt.cmd again."
}
$savedApiKey = Read-SavedApiKey

if ($Action -eq "Doctor") {
    Invoke-Doctor -Config $savedConfig -SecureApiKey $savedApiKey
    exit 0
}

Invoke-Run -Config $savedConfig -SecureApiKey $savedApiKey
