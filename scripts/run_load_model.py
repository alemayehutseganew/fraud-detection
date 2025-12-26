import sys, pathlib
# ensure project root is importable when running scripts from inside scripts/
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src import api
m = None
try:
    m = api.load_model()
    print('Loaded model type:', type(m))
except Exception as e:
    print('load_model raised:', repr(e))
    import traceback
    traceback.print_exc()
