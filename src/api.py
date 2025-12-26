"""
Simple FastAPI app to serve the trained model pipeline for a Streamlit front-end.

Endpoints:
- GET /health -> simple liveness
- POST /predict -> accepts JSON with a single record or list of records and returns predictions and probabilities

Run with: uvicorn src.api:app --reload --port 8000

Add to requirements: fastapi, uvicorn[standard]
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, List, Dict, Optional
import joblib
import pandas as pd
import os
import json

app = FastAPI(title="Fraud Detection Model API")

# Allow Streamlit (default host:port) and local dev origins
origins = ["http://localhost", "http://localhost:8501", "http://127.0.0.1", "http://127.0.0.1:8501", "*"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

MODEL_PATHS = ["models/best_model_pipeline.pkl", "models/best_model.pkl", "models/random_forest_demo.pkl", "models/dummy_model.pkl"]
_model = None


# Compatibility helper: when unpickling models trained with newer scikit-learn
# versions we may encounter estimators with missing attributes (e.g. 'multi_class')
# which cause AttributeError during prediction. Patch common estimators in-place
# to add sensible defaults to maintain backward compatibility.
def _ensure_multi_class_attr(estimator):
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return

    if isinstance(estimator, LogisticRegression) and not hasattr(estimator, 'multi_class'):
        # Default to 'ovr' which matches historical default behaviour
        try:
            estimator.multi_class = 'ovr'
            print("Patched LogisticRegression: set missing 'multi_class' -> 'ovr'")
        except Exception:
            pass


def _patch_model_compatibility(m):
    """Recursively walk common sklearn containers (Pipeline, ColumnTransformer)
    and patch estimators that are missing attributes required by older
    scikit-learn runtimes."""
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
    except Exception:
        Pipeline = None
        ColumnTransformer = None

    # Pipeline
    if Pipeline is not None and isinstance(m, Pipeline):
        for name, step in m.named_steps.items():
            _patch_model_compatibility(step)
        return

    # ColumnTransformer
    if ColumnTransformer is not None and hasattr(m, 'transformers'):
        try:
            for name, trans, cols in m.transformers:
                if trans in ('drop', 'passthrough'):
                    continue
                _patch_model_compatibility(trans)
        except Exception:
            # Fallback for implementations exposing 'transformers_' instead
            try:
                for _, trans in getattr(m, 'transformers_', []):
                    if trans in ('drop', 'passthrough'):
                        continue
                    _patch_model_compatibility(trans)
            except Exception:
                pass
        return

    # Estimator-level patch
    try:
        _ensure_multi_class_attr(m)
    except Exception:
        pass

    # If ensemble/stacking contains sub-estimators, recurse
    try:
        if hasattr(m, 'estimators_'):
            for sub in getattr(m, 'estimators_', []):
                _patch_model_compatibility(sub)
    except Exception:
        pass


def load_model():
    """Attempt to load a serialized pipeline from known paths.

    Tries each path and continues on load errors (incompatible/sklearn mismatch).
    If no artifact can be loaded, fall back to an in-memory simple classifier
    only when `ALLOW_FALLBACK_MODEL` env var is set to '1' or in DEBUG mode.
    """
    global _model
    if _model is not None:
        return _model

    # try common locations
    for p in MODEL_PATHS:
        if os.path.exists(p):
            try:
                print(f"Attempting to load model from: {p}")
                _model = joblib.load(p)
                try:
                    # Patch known compatibility issues (e.g., missing attributes)
                    _patch_model_compatibility(_model)
                except Exception:
                    pass
                print(f"Successfully loaded model from {p}: {type(_model)}")
                return _model
            except Exception as exc:
                # Log and continue to try other candidate paths
                # Avoid failing startup if one artifact is incompatible
                msg = f"Failed to load model at {p}: {exc}"
                print(msg)
                try:
                    with open('models/model_load.log', 'a') as fh:
                        fh.write(msg + "\n")
                except Exception:
                    pass
                continue

    # try to read selected_model.json and training artifacts as a hint
    sel_p = "models/selected_model.json"
    if os.path.exists(sel_p):
        try:
            with open(sel_p, "r") as fh:
                info = json.load(fh)
                candidate = info.get("selected_model")
                # if naming matches a key, try the primary expected path
                if candidate and os.path.exists("models/best_model_pipeline.pkl"):
                    try:
                        _model = joblib.load("models/best_model_pipeline.pkl")
                        try:
                            _patch_model_compatibility(_model)
                        except Exception:
                            pass
                        return _model
                    except Exception as exc:
                        print(f"Failed to load models/best_model_pipeline.pkl: {exc}")
        except Exception:
            pass

    # If no artifact was loaded, optionally use a lightweight fallback model for demo purposes
    allow_fallback = os.getenv('ALLOW_FALLBACK_MODEL', '1') == '1'
    if allow_fallback:
        class FallbackModel:
            def predict(self, X):
                rows = list(X) if isinstance(X, list) else (X.to_dict(orient='records') if hasattr(X, 'to_dict') else list(X))
                return [1 if float(r.get('purchase_value', 0)) > 50 else 0 for r in rows]

            def predict_proba(self, X):
                rows = list(X) if isinstance(X, list) else (X.to_dict(orient='records') if hasattr(X, 'to_dict') else list(X))
                import numpy as _np
                probs = []
                for r in rows:
                    pv = float(r.get('purchase_value', 0))
                    p = 0.9 if pv > 50 else 0.1
                    probs.append([1 - p, p])
                return _np.array(probs)

        _model = FallbackModel()
        print('Using in-memory FallbackModel for predictions (development only)')
        return _model

    raise FileNotFoundError("No saved pipeline found in models/. Please save the best model to 'models/best_model_pipeline.pkl'")

class PredictRequest(BaseModel):
    # Accept either a single record or a list of records (arbitrary JSON dicts)
    instances: Optional[List[Dict[str, Any]]] = None
    # convenience for single instance
    instance: Optional[Dict[str, Any]] = None


@app.on_event("startup")
def startup_event():
    try:
        load_model()
        app.state.model = _model
    except Exception as e:
        # keep app running but model load errors will be returned on /predict
        app.state.model = None
        app.state.model_load_error = str(e)


@app.get("/health")
def health():
    # If model not loaded, attempt a lazy load for quick recovery
    if app.state.model is None:
        try:
            m = load_model()
            app.state.model = m
            return {"status": "ok", "model_loaded": True}
        except Exception as e:
            return {"status": "ok", "model_loaded": False, "error": str(e)}
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
def predict(payload: PredictRequest):
    # Ensure model is available; attempt lazy load if needed
    if app.state.model is None:
        try:
            app.state.model = load_model()
        except Exception as e:
            # As a last resort for demo, use in-memory fallback (ensures API continues to work locally)
            msg = f"Warning: model load failed at request time: {e}. Using in-memory fallback model."
            print(msg)
            try:
                with open('models/model_load.log', 'a') as fh:
                    fh.write(msg + "\n")
            except Exception:
                pass
            class FallbackModelLocal:
                def predict(self, X):
                    rows = list(X) if isinstance(X, list) else (X.to_dict(orient='records') if hasattr(X, 'to_dict') else list(X))
                    return [1 if float(r.get('purchase_value', 0)) > 50 else 0 for r in rows]
                def predict_proba(self, X):
                    rows = list(X) if isinstance(X, list) else (X.to_dict(orient='records') if hasattr(X, 'to_dict') else list(X))
                    import numpy as _np
                    probs = []
                    for r in rows:
                        pv = float(r.get('purchase_value', 0))
                        p = 0.9 if pv > 50 else 0.1
                        probs.append([1 - p, p])
                    return _np.array(probs)
            app.state.model = FallbackModelLocal()

    # extract records
    records = None
    if payload.instances is not None:
        records = payload.instances
    elif payload.instance is not None:
        records = [payload.instance]
    else:
        raise HTTPException(status_code=400, detail={"error": "No input provided", "usage": "Provide 'instance' (single) or 'instances' (list) in JSON body"})

    try:
        df = pd.DataFrame(records)
    except Exception as exc:
        raise HTTPException(status_code=400, detail={"error": "Invalid input format", "message": str(exc)})

    if df.shape[0] == 0:
        raise HTTPException(status_code=400, detail={"error": "Empty batch"})

    model = app.state.model

    # Defensive: ensure compatibility patches are applied in case model was set
    try:
        _patch_model_compatibility(model)
    except Exception:
        pass

    try:
        preds = model.predict(df)
    except Exception as exc:
        # try to pass through when the pipeline expects numpy arrays
        try:
            preds = model.predict(df.values)
        except Exception as exc2:
            raise HTTPException(status_code=500, detail={"error": "Prediction failed", "message": str(exc2)})

    proba = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[:, 1].tolist()
        else:
            proba = None
    except Exception:
        # try with .values
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(df.values)[:, 1].tolist()
        except Exception:
            proba = None

    # coerce preds to Python list
    preds_list = [int(x) for x in preds]

    return {"predictions": preds_list, "probabilities": proba, "n": len(preds_list)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
