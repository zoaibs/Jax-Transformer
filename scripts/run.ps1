param(
    [ValidateSet('test','smoke','pilot','main','workload','refine','confirm','profile','full','report','doctor')]
    [string]$Action = 'smoke',
    [ValidateSet('cpu','gpu')][string]$Backend = 'cpu'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Push-Location -LiteralPath $projectRoot
try {
    $pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Run scripts/setup_windows.ps1 first.' }
    function Invoke-Lab {
        param([string[]]$LabArgs)
        & $pythonPath -m jax_transformer_lab.cli @LabArgs
        if ($LASTEXITCODE -ne 0) { throw "jax-lab failed: $LabArgs" }
    }
    $actions = if ($Action -eq 'full') { @('doctor','pilot','main','workload','refine','confirm','profile','report') } else { @($Action) }
    foreach ($step in $actions) {
        switch ($step) {
            'test' {
                & $pythonPath -m pytest -q
                if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }
            }
            'doctor' { Invoke-Lab -LabArgs @('doctor','--backend',$Backend,'--output',"results/environment_${Backend}.json") }
            'report' { Invoke-Lab -LabArgs @('report','--results','results','--output','reports') }
            'profile' {
                foreach ($variant in @('whole','packed','components')) {
                    Invoke-Lab -LabArgs @('profile','--backend',$Backend,'--variant',$variant,
                                          '--output',"results/profile_${Backend}_${variant}")
                }
            }
            default {
                $outputName = if ($step -eq 'smoke') { 'smoke' } else { "${step}_${Backend}" }
                if ($step -eq 'refine') { $outputName = "workload_refine_${Backend}" }
                Invoke-Lab -LabArgs @('benchmark','--config',"configs/$step.json",'--backend',$Backend,
                                      '--output',"results/$outputName")
            }
        }
    }
} finally { Pop-Location }

