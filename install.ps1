<#
.SYNOPSIS
    One-command installer for Gemini Verbatim Transcripts (Windows).
.DESCRIPTION
    Usage:
        irm https://raw.githubusercontent.com/Biexy/Gemini-Verbatim-Transcripts/main/install.ps1 | iex

    Optional environment overrides:
        $env:GVT_INSTALL_DIR  - Custom install directory (default: $HOME\.gemini-transcriber)
        $env:GVT_NO_LAUNCH    - Set to "1" to skip launching the Web UI after install
        $env:GVT_NO_SHORTCUT  - Set to "1" to skip creating a Desktop shortcut
        $env:GVT_NO_PATH      - Set to "1" to skip adding the CLI shim to User PATH
#>

$ErrorActionPreference = "Stop"

$RepoOwner   = "Biexy"
$RepoName    = "Gemini-Verbatim-Transcripts"
$RepoBranch  = "main"
$RepoGitUrl  = "https://github.com/$RepoOwner/$RepoName.git"
$RepoZipUrl  = "https://github.com/$RepoOwner/$RepoName/archive/refs/heads/$RepoBranch.zip"

$InstallDir  = if ($env:GVT_INSTALL_DIR) { $env:GVT_INSTALL_DIR } else { Join-Path $HOME ".gemini-transcriber" }
$BinDir      = Join-Path $HOME ".local\bin"
$VenvDir     = Join-Path $InstallDir ".venv"
$NoLaunch    = ($env:GVT_NO_LAUNCH -eq "1")
$NoShortcut  = ($env:GVT_NO_SHORTCUT -eq "1")
$NoPath      = ($env:GVT_NO_PATH -eq "1")

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "  [OK] $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "  [WARN] $msg" -ForegroundColor Yellow }
function Write-Err($msg)  { Write-Host "  [ERROR] $msg" -ForegroundColor Red }

function Refresh-SessionPath {
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath    = [Environment]::GetEnvironmentVariable("Path", "User")
    $combined    = @($machinePath, $userPath) -join ";"
    $env:Path    = $combined
}

function Find-Python310Plus {
    $candidates = @(
        @{ Cmd = "python"; Args = @() },
        @{ Cmd = "py";     Args = @("-3") },
        @{ Cmd = "python3"; Args = @() }
    )
    foreach ($c in $candidates) {
        $exe = Get-Command $c.Cmd -ErrorAction SilentlyContinue
        if (-not $exe) { continue }
        try {
            $verArgs = @($c.Args) + @("-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}'); sys.exit(0 if sys.version_info >= (3, 10) else 1)")
            $out = & $c.Cmd @verArgs 2>$null
            if ($LASTEXITCODE -eq 0 -and $out) {
                return @{ Cmd = $c.Cmd; Args = $c.Args; Version = ($out | Select-Object -First 1).Trim() }
            }
        } catch {
            continue
        }
    }
    return $null
}

function Ensure-Python {
    Write-Step "Checking Python 3.10+..."
    $py = Find-Python310Plus
    if ($py) {
        Write-Ok "Found Python $($py.Version) ($($py.Cmd))"
        return $py
    }

    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Warn "Python 3.10+ not found. Installing Python 3.12 via winget..."
        winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements --silent
        Refresh-SessionPath
        $py = Find-Python310Plus
        if ($py) {
            Write-Ok "Installed Python $($py.Version)"
            return $py
        }
    }

    Write-Err "Python 3.10+ is required. Please install it from https://www.python.org/downloads/ (check 'Add Python to PATH') and re-run."
    exit 1
}

function Ensure-FFmpeg {
    Write-Step "Checking FFmpeg & FFprobe..."
    $ff  = Get-Command ffmpeg  -ErrorAction SilentlyContinue
    $ffp = Get-Command ffprobe -ErrorAction SilentlyContinue
    if ($ff -and $ffp) {
        Write-Ok "FFmpeg ready ($($ff.Source))"
        return
    }

    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Warn "FFmpeg not found. Installing Gyan.FFmpeg via winget..."
        winget install -e --id Gyan.FFmpeg --accept-package-agreements --accept-source-agreements --silent
        Refresh-SessionPath
    } elseif (Get-Command choco -ErrorAction SilentlyContinue) {
        Write-Warn "FFmpeg not found. Installing ffmpeg via Chocolatey..."
        choco install ffmpeg -y --no-progress
        Refresh-SessionPath
    }

    if ((Get-Command ffmpeg -ErrorAction SilentlyContinue) -and (Get-Command ffprobe -ErrorAction SilentlyContinue)) {
        Write-Ok "FFmpeg installed successfully"
    } else {
        Write-Warn "FFmpeg not detected in current shell PATH yet. Restart your terminal or install via: winget install Gyan.FFmpeg"
    }
}

function Sync-Repository {
    Write-Step "Downloading $RepoName into $InstallDir..."
    $gitCmd = Get-Command git -ErrorAction SilentlyContinue

    if ($gitCmd -and (Test-Path (Join-Path $InstallDir ".git"))) {
        Push-Location $InstallDir
        try {
            git fetch origin $RepoBranch --quiet
            git pull --ff-only origin $RepoBranch --quiet
            Write-Ok "Updated existing installation via git"
            return
        } catch {
            Write-Warn "git pull skipped ($($_.Exception.Message)); refreshing archive..."
        } finally {
            Pop-Location
        }
    } elseif ($gitCmd -and -not (Test-Path $InstallDir)) {
        git clone --depth 1 --branch $RepoBranch $RepoGitUrl $InstallDir --quiet
        Write-Ok "Cloned repository via git"
        return
    }

    # Pure PowerShell fallback (works with zero Git dependency)
    $tmpZip = Join-Path $env:TEMP "gvt_install_$(Get-Random).zip"
    $tmpExt = Join-Path $env:TEMP "gvt_extract_$(Get-Random)"
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $RepoZipUrl -OutFile $tmpZip -UseBasicParsing
        Expand-Archive -Path $tmpZip -DestinationPath $tmpExt -Force

        $inner = Get-ChildItem -Path $tmpExt -Directory | Select-Object -First 1
        if (-not $inner) { throw "Archive extraction produced no directory." }

        if (-not (Test-Path $InstallDir)) {
            New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
        }

        # Copy source files while preserving existing runtime data/ and .venv/
        Get-ChildItem -Path $inner.FullName -Force | ForEach-Object {
            $target = Join-Path $InstallDir $_.Name
            if ($_.PSIsContainer) {
                Copy-Item -Path $_.FullName -Destination $InstallDir -Recurse -Force
            } else {
                Copy-Item -Path $_.FullName -Destination $target -Force
            }
        }
        Write-Ok "Installed source archive cleanly (no Git required)"
    } finally {
        if (Test-Path $tmpZip) { Remove-Item $tmpZip -Force -ErrorAction SilentlyContinue }
        if (Test-Path $tmpExt) { Remove-Item $tmpExt -Recurse -Force -ErrorAction SilentlyContinue }
    }
}

function Setup-VirtualEnv($py) {
    Write-Step "Setting up isolated Python virtual environment (.venv)..."
    $venvPython = Join-Path $VenvDir "Scripts\python.exe"
    if (-not (Test-Path $venvPython)) {
        $venvArgs = @($py.Args) + @("-m", "venv", $VenvDir)
        & $py.Cmd @venvArgs
        if ($LASTEXITCODE -ne 0) { throw "Failed to create virtual environment." }
    }

    Write-Step "Installing minimal dependencies (yt-dlp, google-genai, requests)..."
    & $venvPython -m pip install --disable-pip-version-check --prefer-binary --retries 10 --timeout 60 -r (Join-Path $InstallDir "requirements.txt")
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }

    Write-Ok "Environment & dependencies ready"
    return $venvPython
}

function Install-Launchers($venvPython) {
    Write-Step "Creating command shims & Windows launcher..."
    if (-not (Test-Path $BinDir)) {
        New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
    }

    $shimCmd = Join-Path $BinDir "gemini-transcribe.cmd"
    $shimBody = "@echo off`r`nsetlocal`r`nset `"PYTHONIOENCODING=utf-8`"`r`nset `"PYTHONUTF8=1`"`r`n`"$venvPython`" `"$InstallDir\run.py`" %*`r`nendlocal`r`n"
    [System.IO.File]::WriteAllText($shimCmd, $shimBody, [System.Text.Encoding]::ASCII)

    $localBat = Join-Path $InstallDir "Start_App.bat"
    [System.IO.File]::WriteAllText($localBat, $shimBody, [System.Text.Encoding]::ASCII)

    if (-not $NoPath) {
        $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
        $entries  = @()
        if ($userPath) { $entries = $userPath -split ";" | Where-Object { $_ } }
        if ($entries -notcontains $BinDir) {
            $newPath = (@($entries) + @($BinDir)) -join ";"
            [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
            $env:Path = "$env:Path;$BinDir"
            Write-Ok "Added $BinDir to User PATH"
        } else {
            Write-Ok "$BinDir already in User PATH"
        }
    }

    if (-not $NoShortcut) {
        try {
            $desktop = [Environment]::GetFolderPath("Desktop")
            if ($desktop -and (Test-Path $desktop)) {
                $lnkPath  = Join-Path $desktop "Gemini Verbatim Transcripts.lnk"
                $wsh      = New-Object -ComObject WScript.Shell
                $shortcut = $wsh.CreateShortcut($lnkPath)
                $shortcut.TargetPath       = $localBat
                $shortcut.WorkingDirectory = $InstallDir
                $shortcut.Description      = "Gemini Verbatim Transcripts - YouTube & Google Drive Lecture Transcriber"
                $shortcut.Save()
                Write-Ok "Created Desktop shortcut: $lnkPath"
            }
        } catch {
            Write-Warn "Could not create Desktop shortcut ($($_.Exception.Message))"
        }
    }
}

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Magenta
Write-Host "   Gemini Verbatim Transcripts - One-Command Installer    " -ForegroundColor Magenta
Write-Host "==========================================================" -ForegroundColor Magenta
Write-Host ""

$pyInfo     = Ensure-Python
Ensure-FFmpeg
Sync-Repository
$venvPython = Setup-VirtualEnv $pyInfo
Install-Launchers $venvPython

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  Installation Complete!" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  Install folder : $InstallDir"
Write-Host "  Run Web UI     : gemini-transcribe   (or Desktop shortcut)"
Write-Host "  Run CLI        : gemini-transcribe --cli --course `"MyCourse`" --urls `"<URL>`""
Write-Host "  Update later   : Re-run this exact same one-line command"
Write-Host ""

if (-not $NoLaunch) {
    Write-Step "Launching Gemini Verbatim Transcripts Web UI..."
    & $venvPython (Join-Path $InstallDir "run.py")
}
