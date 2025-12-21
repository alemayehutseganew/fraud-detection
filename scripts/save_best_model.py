"""Copy demo model to best_model_pipeline.pkl if best model doesn't exist."""
import os
import joblib

os.makedirs('models', exist_ok=True)

best_path = os.path.join('models', 'best_model_pipeline.pkl')
if os.path.exists(best_path):
    print(f"Best pipeline already exists at {best_path}")
else:
    demo_path = os.path.join('models', 'random_forest_demo.pkl')
    if os.path.exists(demo_path):
        demo = joblib.load(demo_path)
        joblib.dump(demo, best_path)
        print(f"Copied demo model to {best_path}")
    else:
        print('No demo model found to copy.')
