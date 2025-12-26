import joblib, traceback
p='models/best_model_pipeline.pkl'
try:
    m=joblib.load(p)
    print('Loaded OK:', type(m))
except Exception as e:
    print('Load failed:', e)
    traceback.print_exc()