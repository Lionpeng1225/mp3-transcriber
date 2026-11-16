$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:HF_XET_CACHE = Join-Path $PSScriptRoot 'models\.xet'
$env:HF_HUB_DISABLE_XET = '1'
try {
    if (-not (Test-Path '.venv\Scripts\python.exe')) {
        $pythonCommand = Get-Command py -ErrorAction SilentlyContinue
        if (-not $pythonCommand) { throw 'Install Python 3.11 or 3.12 from python.org with the Python launcher, then retry.' }
        & py -3 -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed. Install Python 3.11 or 3.12 and retry.' }
    }
    & '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Package installation failed. Check your internet connection and retry.' }
    & '.\.venv\Scripts\python.exe' -m streamlit run app.py --server.address 127.0.0.1 --server.maxUploadSize 1024 --browser.gatherUsageStats false
    if ($LASTEXITCODE -ne 0) { throw 'Application startup failed. See the error above.' }
} catch {
    Write-Host $_ -ForegroundColor Red
    Read-Host 'Press Enter to close'
    exit 1
}
