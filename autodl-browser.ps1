[CmdletBinding()]
param(
    [ValidateSet("Setup", "Login")]
    [string]$Action = "Login",
    [ValidateSet("auto", "chrome", "edge")]
    [string]$Browser = "auto",
    [ValidateSet("console", "market", "none")]
    [string]$OpenPage = "console",
    [switch]$KeepOpen
)

$ErrorActionPreference = "Stop"
# Load and check encryption before asking the user for credentials.
try {
    Import-Module "$PSHOME\Modules\Microsoft.PowerShell.Security\Microsoft.PowerShell.Security.psd1" -ErrorAction Stop
    $probe = ConvertTo-SecureString "AutoResearch encryption check" -AsPlainText -Force
    $protectedProbe = ConvertFrom-SecureString $probe
    $restoredProbe = ConvertTo-SecureString $protectedProbe
    if ($restoredProbe.Length -ne $probe.Length) { throw "Encryption check failed" }
    $probe.Dispose()
    $restoredProbe.Dispose()
    $protectedProbe = $null
}
catch {
    throw "Windows encryption initialization failed before credential input: $($_.Exception.Message)"
}
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$LocalAppDataPath = [Environment]::GetFolderPath("LocalApplicationData")
if ([string]::IsNullOrWhiteSpace($LocalAppDataPath)) {
    $LocalAppDataPath = Join-Path $ProjectRoot ".local"
}
$StateDirectory = Join-Path $LocalAppDataPath "AutoResearch"
$PhonePath = Join-Path $StateDirectory "autodl-phone.dpapi"
$SecretPath = Join-Path $StateDirectory "autodl-password.dpapi"
$PythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

function Write-Utf8NoBom {
    param([string]$Path, [string]$Value)
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    [IO.File]::WriteAllText($Path, $Value, $utf8)
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

if ($Action -eq "Setup") {
    New-Item -ItemType Directory -Path $StateDirectory -Force | Out-Null
    $phone = (Read-Host "AutoDL phone number").Trim()
    if ($phone -notmatch '^\+?[0-9]{6,20}$') {
        throw "Enter a valid phone number using digits and an optional leading plus sign."
    }
    $password = Read-Host "AutoDL password (input is hidden)" -AsSecureString
    $plainCheck = Convert-ToPlainText $password
    try {
        if ([string]::IsNullOrWhiteSpace($plainCheck)) {
            throw "The AutoDL password cannot be empty."
        }
    }
    finally {
        $plainCheck = $null
    }

    $securePhone = ConvertTo-SecureString -String $phone -AsPlainText -Force
    Write-Utf8NoBom -Path $PhonePath -Value ($securePhone | ConvertFrom-SecureString)
    Write-Utf8NoBom -Path $SecretPath -Value ($password | ConvertFrom-SecureString)
    Write-Host "AutoDL login details were saved for this Windows user only."
    Write-Host "Return to AutoResearch and click 'Auto Login to AutoDL'."
    Read-Host "Press Enter to close"
    exit 0
}

if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "AutoResearch virtual environment was not found. Run start.cmd once."
}
if (
    -not (Test-Path -LiteralPath $PhonePath -PathType Leaf) -or
    -not (Test-Path -LiteralPath $SecretPath -PathType Leaf)
) {
    throw "AutoDL login is not configured. Click 'Configure AutoDL Login' first."
}

try {
    $protectedPhone = (Get-Content -Raw -LiteralPath $PhonePath).Trim()
    $protectedPassword = (Get-Content -Raw -LiteralPath $SecretPath).Trim()
    $securePhone = $protectedPhone | ConvertTo-SecureString
    $securePassword = $protectedPassword | ConvertTo-SecureString
}
catch {
    throw "The saved AutoDL login cannot be decrypted by this Windows user. Configure it again."
}

$plainPhone = Convert-ToPlainText $securePhone
$plainPassword = Convert-ToPlainText $securePassword
$previousPhone = [Environment]::GetEnvironmentVariable("AUTODL_BROWSER_PHONE", "Process")
$previousPassword = [Environment]::GetEnvironmentVariable("AUTODL_BROWSER_PASSWORD", "Process")
try {
    [Environment]::SetEnvironmentVariable("AUTODL_BROWSER_PHONE", $plainPhone, "Process")
    [Environment]::SetEnvironmentVariable("AUTODL_BROWSER_PASSWORD", $plainPassword, "Process")
    $arguments = @(
        "-m", "autoresearch", "autodl-browser-login",
        "--sms",
        "--browser", $Browser,
        "--open", $OpenPage,
        "--manual-captcha"
    )
    if ($KeepOpen) { $arguments += "--keep-open" }
    & $PythonPath @arguments
    $exitCode = $LASTEXITCODE
}
finally {
    [Environment]::SetEnvironmentVariable("AUTODL_BROWSER_PHONE", $previousPhone, "Process")
    [Environment]::SetEnvironmentVariable("AUTODL_BROWSER_PASSWORD", $previousPassword, "Process")
    $plainPhone = $null
    $plainPassword = $null
}

if ($exitCode -ne 0) {
    throw "AutoDL browser login exited with code $exitCode."
}
