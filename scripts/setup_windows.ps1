# CPU environment only; NVIDIA JAX runs through Linux/WSL. No global settings changed.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Push-Location -LiteralPath $projectRoot
try {
    New-Item -ItemType Directory -Force -Path '.tools','.cache' | Out-Null
    $env:UV_CACHE_DIR = Join-Path $projectRoot '.cache\uv'
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $projectRoot '.tools\python'
    $uvPath = Join-Path $projectRoot '.tools\uv.exe'
    if (-not (Test-Path -LiteralPath $uvPath)) {
        $archive = Join-Path $projectRoot '.tools\uv.zip'
        Invoke-WebRequest 'https://github.com/astral-sh/uv/releases/download/0.13.5/uv-x86_64-pc-windows-msvc.zip' -OutFile $archive
        Expand-Archive -LiteralPath $archive -DestinationPath '.tools' -Force
    }
    & $uvPath python install 3.12.15
    if ($LASTEXITCODE -ne 0) { throw 'Python install failed.' }
    & $uvPath venv --python 3.12.15 .venv
    if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed.' }
    & $uvPath pip sync --python '.venv\Scripts\python.exe' 'requirements-windows.lock.txt'
    if ($LASTEXITCODE -ne 0) { throw 'Dependency install failed.' }
    & $uvPath pip install --python '.venv\Scripts\python.exe' --no-deps -e .
    if ($LASTEXITCODE -ne 0) { throw 'Package install failed.' }
    & '.\.venv\Scripts\python.exe' -m jax_transformer_lab.cli doctor --backend cpu --output results/environment_cpu.json
    if ($LASTEXITCODE -ne 0) { throw 'JAX validation failed.' }
} finally { Pop-Location }

