# PowerShell helper to create a venv and install Streamlit runtime dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements-streamlit.txt
Write-Host "Environment setup complete. Run: .\.venv\Scripts\Activate.ps1 and then 'streamlit run app.py'" -ForegroundColor Green
