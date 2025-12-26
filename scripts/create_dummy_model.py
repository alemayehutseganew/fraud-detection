"""Create and save a dummy pipeline to models/best_model_pipeline.pkl
"""
import os
import joblib
import numpy as np

class DummyPipeline:
    def predict(self, X):
        # Accept DataFrame-like (dict-list or pd.DataFrame)
        try:
            rows = list(X) if isinstance(X, list) else (X.to_dict(orient='records') if hasattr(X, 'to_dict') else list(X))
            return [1 if float(r.get('purchase_value', 0)) > 50 else 0 for r in rows]
        except Exception:
            return [0 for _ in (rows if 'rows' in locals() else [0])]

    def predict_proba(self, X):
        rows = list(X) if isinstance(X, list) else (X.to_dict(orient='records') if hasattr(X, 'to_dict') else list(X))
        probs = []
        for r in rows:
            pv = float(r.get('purchase_value', 0))
            p = 0.9 if pv > 50 else 0.1
            probs.append([1-p, p])
        return np.array(probs)

os.makedirs('models', exist_ok=True)
joblib.dump(DummyPipeline(), 'models/best_model_pipeline.pkl')
print('Saved dummy pipeline to models/best_model_pipeline.pkl')
