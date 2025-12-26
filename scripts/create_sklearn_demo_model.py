from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import numpy as np

# Create simple dataset
X = np.array([[10, 20], [100, 5], [20, 50], [200, 1], [5,3]])
y = np.array([0, 1, 0, 1, 0])

# We'll map purchase_value to feature 0 for demo; pipeline expects DataFrame with 'purchase_value'
# But simpler: create a pipeline that accepts DataFrame with 'purchase_value' via a small wrapper

class SklearnWrapper:
    def __init__(self, clf):
        self.clf = clf
    def predict(self, X):
        # X may be DataFrame or list of dicts
        try:
            rows = X.to_dict(orient='records') if hasattr(X, 'to_dict') else list(X)
            arr = [[float(r.get('purchase_value', 0))] for r in rows]
        except Exception:
            import numpy as _np
            arr = X
        return self.clf.predict(arr)
    def predict_proba(self, X):
        try:
            rows = X.to_dict(orient='records') if hasattr(X, 'to_dict') else list(X)
            arr = [[float(r.get('purchase_value', 0))] for r in rows]
        except Exception:
            arr = X
        return self.clf.predict_proba(arr)

clf = LogisticRegression()
# Train on purchase_value only
X_train = np.array([[10.],[100.],[20.],[200.],[5.]])
clf.fit(X_train, y)
wrapped = SklearnWrapper(clf)
joblib.dump(wrapped, 'models/best_model_pipeline.pkl')
print('Saved sklearn-wrapped model to models/best_model_pipeline.pkl')
