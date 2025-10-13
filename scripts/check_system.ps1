# TripoSR System Information Collection Script (PowerShell)
# Run this from Windows PowerShell to check system compatibility

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "TripoSR System Information Collection" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

function Print-Section {
    param([string]$Title)
    Write-Host "==================== $Title ====================" -ForegroundColor Blue
}

# 1. Windows Information
Print-Section "Windows System Information"
try {
    $os = Get-CimInstance Win32_OperatingSystem
    Write-Host "OS: " -NoNewline -ForegroundColor Green
    Write-Host "$($os.Caption) (Build $($os.BuildNumber))"
    Write-Host "Version: " -NoNewline -ForegroundColor Green
    Write-Host $os.Version
    Write-Host ""
} catch {
    Write-Host "Failed to get Windows information" -ForegroundColor Red
}

# 2. Hardware Information
Print-Section "Hardware Information"
try {
    $cs = Get-CimInstance Win32_ComputerSystem
    Write-Host "Manufacturer: " -NoNewline -ForegroundColor Green
    Write-Host $cs.Manufacturer
    Write-Host "Model: " -NoNewline -ForegroundColor Green
    Write-Host $cs.Model
    $totalRAM = [math]::Round($cs.TotalPhysicalMemory / 1GB, 2)
    Write-Host "Total RAM: " -NoNewline -ForegroundColor Green
    Write-Host "$totalRAM GB"
    Write-Host ""

    $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
    Write-Host "Processor: " -NoNewline -ForegroundColor Green
    Write-Host $cpu.Name
    Write-Host "Cores: " -NoNewline -ForegroundColor Green
    Write-Host "$($cpu.NumberOfCores) physical, $($cpu.NumberOfLogicalProcessors) logical"
    Write-Host ""
} catch {
    Write-Host "Failed to get hardware information" -ForegroundColor Red
}

# 3. GPU Information
Print-Section "GPU Information"
try {
    $gpus = Get-CimInstance Win32_VideoController
    foreach ($gpu in $gpus) {
        Write-Host "GPU: " -NoNewline -ForegroundColor Green
        Write-Host $gpu.Name
        if ($gpu.AdapterRAM) {
            $vram = [math]::Round($gpu.AdapterRAM / 1GB, 2)
            Write-Host "VRAM: " -NoNewline -ForegroundColor Green
            Write-Host "$vram GB"
        }
        Write-Host "Driver Version: " -NoNewline -ForegroundColor Green
        Write-Host $gpu.DriverVersion
        Write-Host ""
    }
} catch {
    Write-Host "Failed to get GPU information" -ForegroundColor Red
}

# 4. NVIDIA Specific Information
Print-Section "NVIDIA GPU Details"
$nvidiaSmiPath = "C:\Windows\System32\nvidia-smi.exe"
if (Test-Path $nvidiaSmiPath) {
    Write-Host "NVIDIA Driver Found!" -ForegroundColor Green
    Write-Host ""
    & $nvidiaSmiPath
    Write-Host ""

    Write-Host "CUDA-capable GPUs:" -ForegroundColor Green
    & $nvidiaSmiPath --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
    Write-Host ""
} else {
    Write-Host "nvidia-smi not found. NVIDIA drivers may not be installed." -ForegroundColor Yellow
    Write-Host ""
}

# 5. WSL Information
Print-Section "WSL (Windows Subsystem for Linux)"
try {
    $wslVersion = wsl --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host $wslVersion
        Write-Host ""

        Write-Host "Installed WSL Distributions:" -ForegroundColor Green
        wsl --list --verbose
        Write-Host ""
    } else {
        Write-Host "WSL not installed or not configured" -ForegroundColor Yellow
        Write-Host ""
    }
} catch {
    Write-Host "WSL not available" -ForegroundColor Yellow
    Write-Host ""
}

# 6. Python Information (Windows)
Print-Section "Python on Windows"
$pythonVersions = @("python3.11", "python3.10", "python3", "python")
$foundPython = $false
foreach ($pyCmd in $pythonVersions) {
    try {
        $version = & $pyCmd --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "$pyCmd : " -NoNewline -ForegroundColor Green
            Write-Host $version
            $foundPython = $true
        }
    } catch {
        # Command not found, skip
    }
}
if (-not $foundPython) {
    Write-Host "Python not found in PATH" -ForegroundColor Yellow
}
Write-Host ""

# 7. Check for uv
Print-Section "Package Managers"
try {
    $uvVersion = uv --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "uv: " -NoNewline -ForegroundColor Green
        Write-Host $uvVersion
    }
} catch {
    Write-Host "uv not installed" -ForegroundColor Yellow
    Write-Host "To install: " -NoNewline
    Write-Host "powershell -c `"irm https://astral.sh/uv/install.ps1 | iex`"" -ForegroundColor Cyan
}
Write-Host ""

# 8. Disk Space
Print-Section "Disk Space"
try {
    $drive = Get-PSDrive -Name C
    $freeGB = [math]::Round($drive.Free / 1GB, 2)
    $usedGB = [math]::Round($drive.Used / 1GB, 2)
    $totalGB = [math]::Round(($drive.Free + $drive.Used) / 1GB, 2)
    Write-Host "C: Drive - Total: $totalGB GB, Used: $usedGB GB, Free: $freeGB GB" -ForegroundColor Green
} catch {
    Write-Host "Failed to get disk space information" -ForegroundColor Red
}
Write-Host ""

# 9. Summary and Recommendations
Print-Section "Summary and Recommendations"

# Check GPU
$hasNvidiaGPU = Test-Path $nvidiaSmiPath
if ($hasNvidiaGPU) {
    Write-Host "[OK] NVIDIA GPU with drivers detected" -ForegroundColor Green

    # Get CUDA version
    $cudaVersion = & $nvidiaSmiPath | Select-String "CUDA Version: ([0-9]+\.[0-9]+)" | ForEach-Object { $_.Matches.Groups[1].Value }
    if ($cudaVersion) {
        Write-Host "     CUDA Version: $cudaVersion" -ForegroundColor Green
        if ($cudaVersion -like "12.*") {
            Write-Host "     Recommended: Install PyTorch with CUDA 12.x support" -ForegroundColor Cyan
        } elseif ($cudaVersion -like "11.*") {
            Write-Host "     Recommended: Install PyTorch with CUDA 11.x support" -ForegroundColor Cyan
        }
    }
} else {
    Write-Host "[!!] NVIDIA GPU not detected or drivers not installed" -ForegroundColor Yellow
    Write-Host "     TripoSR will run on CPU (much slower)" -ForegroundColor Yellow
}
Write-Host ""

# Check Memory
if ($totalRAM -ge 16) {
    Write-Host "[OK] Sufficient RAM: $totalRAM GB" -ForegroundColor Green
} elseif ($totalRAM -ge 8) {
    Write-Host "[OK] Adequate RAM: $totalRAM GB (minimum for TripoSR)" -ForegroundColor Green
} else {
    Write-Host "[!!] Low RAM: $totalRAM GB (TripoSR may have issues)" -ForegroundColor Yellow
}
Write-Host ""

# Check WSL
try {
    wsl --version 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] WSL is installed" -ForegroundColor Green
        Write-Host "     You can run TripoSR in WSL for better Linux compatibility" -ForegroundColor Cyan
    }
} catch {
    Write-Host "[!!] WSL not installed" -ForegroundColor Yellow
    Write-Host "     Consider installing WSL2 for better development experience" -ForegroundColor Yellow
    Write-Host "     Run: wsl --install" -ForegroundColor Cyan
}
Write-Host ""

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "System information collection complete!" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
