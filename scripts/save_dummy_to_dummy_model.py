import joblib
import numpy as np

class DummyPipeline:
    def predict(self, X):
        rows = list(X) if isinstance(X, list) else (X.to_dict(orient='records') if hasattr(X, 'to_dict') else list(X))
        return [1 if float(r.get('purchase_value', 0)) > 50 else 0 for r in rows]
    def predict_proba(self, X):
        rows = list(X) if isinstance(X, list) else (X.to_dict(orient='records') if hasattr(X, 'to_dict') else list(X))
        probs = []
        for r in rows:
            pv = float(r.get('purchase_value', 0))
            p = 0.9 if pv > 50 else 0.1
            probs.append([1-p, p])
        return np.array(probs)

joblib.dump(DummyPipeline(), 'models/dummy_model.pkl')
print('Saved dummy_model.pkl')
