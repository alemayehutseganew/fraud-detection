import sys
import pathlib
# ensure project root is on sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
import os

if __name__ == '__main__':
    port = int(os.getenv('UVICORN_PORT', '8000'))
    uvicorn.run('src.api:app', host='127.0.0.1', port=port, log_level='info')
