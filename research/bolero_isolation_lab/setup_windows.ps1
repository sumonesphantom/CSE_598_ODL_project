$ErrorActionPreference = "Stop"

Write-Host "Bolero Isolation Lab setup"
Write-Host "Checking the selected Python interpreter..."
python --version
if ($LASTEXITCODE -ne 0) {
    throw "Python was not found on PATH. Install CPython 3.11.9 or edit this script to use the full path to python.exe."
}

python -m venv .venv
if ($LASTEXITCODE -ne 0) { throw "Could not create .venv. Check that the selected Python is a working CPython installation." }

$VenvPython = Join-Path (Get-Location) ".venv\Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
& $VenvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed. See README.md for troubleshooting and CUDA setup." }
& $VenvPython -m ipykernel install --user --name bolero-isolation --display-name "Python 3.11.9 (Bolero Isolation)"
if ($LASTEXITCODE -ne 0) { throw "Could not register the Jupyter kernel." }

Write-Host "Setup completed. Open notebooks\01_depth_subject_isolation.ipynb and select the Python 3.11.9 (Bolero Isolation) kernel."
