from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np

# Train a simple classifier on a single feature
X = np.array([[10.],[100.],[20.],[200.],[5.],[80.],[30.]])
y = np.array([0,1,0,1,0,1,0])
clf = LogisticRegression()
clf.fit(X, y)
joblib.dump(clf, 'models/best_model_pipeline.pkl')
print('Saved sklearn LogisticRegression to models/best_model_pipeline.pkl')
