"""Simple script to verify imports for Streamlit demo"""
import importlib
pkgs = ['streamlit', 'shap', 'joblib', 'matplotlib', 'seaborn']

for p in pkgs:
    try:
        m = importlib.import_module(p)
        print(f"{p}: {getattr(m, '__version__', 'version unknown')}")
    except Exception as e:
        print(f"{p}: import failed: {e}")
