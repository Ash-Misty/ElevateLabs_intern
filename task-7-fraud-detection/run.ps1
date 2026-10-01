<#
.SYNOPSIS
    One-command setup and launch for FraudGuard.

.EXAMPLE
    .\run.ps1 train          Train every model and write the artifacts
    .\run.ps1 web            Train if needed, then serve the Flask app on :5000
    .\run.ps1 streamlit      Serve the Streamlit dashboard on :8501
    .\run.ps1 notebook       Train if needed, then open JupyterLab
    .\run.ps1 test           Run the smoke tests
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('setup', 'train', 'web', 'streamlit', 'notebook', 'test')]
    [string]$Command = 'web'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $ProjectRoot '..\venv\Scripts\python.exe'
$workdir = Join-Path $ProjectRoot 'src'

function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $venvPython @Arguments
    if ($LASTEXITCODE -ne 0) { throw "python $($Arguments -join ' ') failed with exit code $LASTEXITCODE" }
}

function Test-Bundle {
    Test-Path (Join-Path $ProjectRoot 'artifacts\models\fraud_pipeline.joblib')
}

Push-Location $ProjectRoot
try {
    if (-not (Test-Path $venvPython)) {
        throw "Virtualenv not found at $venvPython. Create it from the repo root with: python -m venv venv"
    }

    switch ($Command) {
        'setup' {
            Write-Host "Installing dependencies..." -ForegroundColor Cyan
            Invoke-Python -m pip install --upgrade pip
            Invoke-Python -m pip install -r requirements.txt
            Write-Host "Done. Next: .\run.ps1 train" -ForegroundColor Green
        }

        'train' {
            Write-Host "Training the fraud detection pipeline..." -ForegroundColor Cyan
            $env:PYTHONPATH = $workdir
            Invoke-Python -m fraudguard.train
        }

        'web' {
            if (-not (Test-Bundle)) {
                Write-Host "No trained model found, training first..." -ForegroundColor Yellow
                $env:PYTHONPATH = $workdir
                Invoke-Python -m fraudguard.train
            }
            Write-Host "Flask app on http://127.0.0.1:5000" -ForegroundColor Green
            Invoke-Python app/flask_app.py 5000
        }

        'streamlit' {
            if (-not (Test-Bundle)) {
                Write-Host "No trained model found, training first..." -ForegroundColor Yellow
                $env:PYTHONPATH = $workdir
                Invoke-Python -m fraudguard.train
            }
            Write-Host "Streamlit dashboard on http://127.0.0.1:8501" -ForegroundColor Green
            Invoke-Python -m streamlit run app/streamlit_app.py --server.port 8501
        }

        'notebook' {
            if (-not (Test-Bundle)) {
                Write-Host "No trained model found, training first..." -ForegroundColor Yellow
                $env:PYTHONPATH = $workdir
                Invoke-Python -m fraudguard.train
            }
            Invoke-Python -m jupyter lab --notebook-dir notebooks
        }

        'test' {
            $env:PYTHONPATH = $workdir
            Invoke-Python -m unittest discover -s tests -v
        }
    }
}
finally {
    Pop-Location
}
