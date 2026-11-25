# Qinyan Zeng: native PowerShell counterpart to run_benchmark.sh.
param(
    [string]$OutputDir = "",
    [ValidateSet('float', 'double')][string]$ScalePrecision = 'float',
    [string]$BuildDir = "",
    [string]$Python = 'python',
    [switch]$Quick
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $OutputDir) {
    $OutputDir = Join-Path $projectRoot "results/$(Get-Date -Format yyyy-MM-dd)_windows-$ScalePrecision"
}
if (-not $BuildDir) { $BuildDir = Join-Path $projectRoot 'build/release' }
$pythonPath = (Get-Command $Python -ErrorAction Stop).Source

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program exited with status $LASTEXITCODE" }
}

Invoke-Checked 'cmake' @('-S', $projectRoot, '-B', $BuildDir, '-DCMAKE_BUILD_TYPE=Release',
    "-DPython3_EXECUTABLE=$pythonPath")
Invoke-Checked 'cmake' @('--build', $BuildDir, '--config', 'Release')
Invoke-Checked 'ctest' @('--test-dir', $BuildDir, '-C', 'Release', '--output-on-failure')
$benchmark = @((Join-Path $BuildDir 'int4_bench.exe'),
               (Join-Path $BuildDir 'Release/int4_bench.exe')) |
    Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $benchmark) { throw "int4_bench.exe not found under $BuildDir" }
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$environmentInfo = @(
    "date_utc=$([DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ'))",
    "os=$([System.Environment]::OSVersion)",
    "cpu=$((Get-CimInstance Win32_Processor | Select-Object -First 1).Name)",
    "logical_cpus=$([System.Environment]::ProcessorCount)",
    "memory_bytes=$((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory)",
    "scale_precision=$ScalePrecision"
)
$environmentInfo += Get-Content -LiteralPath (Join-Path $BuildDir 'CMakeCache.txt') |
    Where-Object { $_ -match '^CMAKE_(CXX_COMPILER|CXX_FLAGS|CXX_FLAGS_RELEASE|BUILD_TYPE|GENERATOR)(:[A-Z]+)?=' }
$environmentInfo += & cmake --version | Select-Object -First 1
$environmentInfo += & $pythonPath --version
$environmentInfo += & $pythonPath -c 'import numpy,matplotlib; print("numpy="+numpy.__version__); print("matplotlib="+matplotlib.__version__)'
if ($LASTEXITCODE -ne 0) { throw 'NumPy and Matplotlib are required for the benchmark pipeline' }
$environmentInfo | Set-Content -LiteralPath (Join-Path $OutputDir 'environment.txt') -Encoding utf8
$benchmarkArgs = @('--csv', (Join-Path $OutputDir 'bench.csv'),
    '--dump-dir', (Join-Path $OutputDir 'dumps'), '--tuning-tiles', '16,64',
    '--scale-precision', $ScalePrecision)
if ($Quick) { $benchmarkArgs += '--quick' }
Invoke-Checked $benchmark $benchmarkArgs
Invoke-Checked $pythonPath @((Join-Path $PSScriptRoot 'reference_check.py'),
    (Join-Path $OutputDir 'bench.csv'), (Join-Path $OutputDir 'dumps'),
    '--out', (Join-Path $OutputDir 'bench_with_reference.csv'))
foreach ($plotScript in @('plot_results.py', 'plot_rotation_error.py')) {
    Invoke-Checked $pythonPath @((Join-Path $PSScriptRoot $plotScript),
        (Join-Path $OutputDir 'bench_with_reference.csv'), '--out-dir', $OutputDir)
}
Write-Output "run_benchmark: results in $OutputDir"
