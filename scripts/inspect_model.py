import joblib
m = joblib.load('models/best_model_pipeline.pkl')
print('Loaded type:', type(m))
try:
    print('predict sample:', m.predict([[200]]))
    print('predict_proba sample:', m.predict_proba([[200]]))
except Exception as e:
    print('Error calling model methods:', e)
