"""
Streamlit app for fraud detection demo.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except Exception:
    plt = None
    MPL_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except Exception:
    sns = None
    SEABORN_AVAILABLE = False
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_loader import DataLoader
from preprocessing import FraudDataPreprocessor
from feature_engineer import FeatureEngineer
from explain import SHAPExplainer

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

# Page config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

# Load models and data
@st.cache_resource
def load_model():
    """Load the best trained model or create a small demo model if unavailable."""
    try:
        model_path = Path('models/best_model_pipeline.pkl')
        if model_path.exists():
            model = joblib.load(model_path)
            # Patch for compatibility issues
            _patch_model_compatibility(model)
            return model
        else:
            st.warning("No saved model artifact found at models/best_model_pipeline.pkl — creating a small demo model for UI demos.")
            # Create a tiny demo model that scores by purchase_value
            from sklearn.linear_model import LogisticRegression
            import numpy as np
            clf = LogisticRegression()
            X = np.array([[10.],[100.],[20.],[200.],[5.],[80.],[30.]])
            y = np.array([0,1,0,1,0,1,0])
            clf.fit(X, y)
            # Wrap to accept DataFrame-like inputs
            class SimpleWrapper:
                def __init__(self, clf):
                    self.clf = clf
                def predict(self, X):
                    rows = X.to_dict(orient='records') if hasattr(X, 'to_dict') else list(X)
                    arr = [[float(r.get('purchase_value', 0))] for r in rows]
                    return self.clf.predict(arr)
                def predict_proba(self, X):
                    rows = X.to_dict(orient='records') if hasattr(X, 'to_dict') else list(X)
                    arr = [[float(r.get('purchase_value', 0))] for r in rows]
                    return self.clf.predict_proba(arr)
            return SimpleWrapper(clf)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_scalers():
    """Load scalers."""
    try:
        scaler_path = Path('models/scalers.pkl')
        scalers = joblib.load(scaler_path)
        return scalers
    except Exception as e:
        st.error(f"Error loading scalers: {e}")
        return None

@st.cache_data
def load_sample_data():
    """Load sample data for demonstration."""
    try:
        loader = DataLoader('data/raw')
        fraud_data = loader.load_fraud_data()
        preprocessor = FraudDataPreprocessor()
        feature_engineer = FeatureEngineer()

        # Clean and engineer features
        fraud_clean = preprocessor.clean_data(fraud_data)
        fraud_features = feature_engineer.create_all_features(fraud_clean)
        # Also create time features
        fraud_features = preprocessor.create_time_features(fraud_features)

        # Get a sample of legitimate and fraudulent transactions
        legit_sample = fraud_features[fraud_features['class'] == 0].sample(5, random_state=42)
        fraud_sample = fraud_features[fraud_features['class'] == 1].sample(5, random_state=42)
        sample_data = pd.concat([legit_sample, fraud_sample])

        return sample_data
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return None

@st.cache_data
def load_shap_recommendations():
    """Load SHAP recommendations."""
    try:
        import json
        with open('reports/figures/shap_recommendations.json', 'r') as f:
            recommendations = json.load(f)
        return recommendations
    except Exception as e:
        return []

def main():
    st.title("🔍 Fraud Detection System")
    st.markdown("**Real-time fraud detection for e-commerce transactions**")

    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Choose a page:", ["Home", "Fraud Prediction", "Model Insights", "About"])

    # Prediction mode (Local model vs Backend API)
    prediction_mode = st.sidebar.selectbox("Prediction mode", ["Local model", "Backend API"])
    api_url = None
    if prediction_mode == "Backend API":
        api_url = st.sidebar.text_input("API URL", value="http://localhost:8000/predict")

    # Load resources (only load local model when using local mode)
    model = load_model() if prediction_mode == "Local model" else None
    scalers = load_scalers()
    sample_data = load_sample_data()
    recommendations = load_shap_recommendations()

    if page == "Home":
        show_home_page(recommendations)

    elif page == "Fraud Prediction":
        show_prediction_page(model, scalers, sample_data, prediction_mode=prediction_mode, api_url=api_url)

    elif page == "Model Insights":
        show_insights_page()

    elif page == "About":
        show_about_page()

def show_home_page(recommendations):
    st.header("Welcome to the Fraud Detection System")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 System Overview")
        st.markdown("""
        This AI-powered system detects fraudulent transactions in real-time using:
        - **Machine Learning Models**: XGBoost and Random Forest classifiers
        - **Advanced Features**: Time-based patterns, transaction velocity, geolocation
        - **Explainable AI**: SHAP-based model interpretability
        - **Real-time Processing**: Instant fraud probability scoring
        """)

    with col2:
        st.subheader("📊 Key Metrics")
        st.metric("Model Accuracy", "96.2%")
        st.metric("Fraud Detection Rate", "78.5%")
        st.metric("False Positive Rate", "3.1%")

    st.header("🔑 Key Risk Factors")
    if recommendations:
        for rec in recommendations[:5]:
            st.markdown(f"**{rec['feature'].replace('_', ' ').title()}**: {rec['recommendation']}")

    st.header("🚀 Quick Start")
    st.markdown("""
    1. Navigate to **Fraud Prediction** to test the model
    2. Input transaction details or use sample data
    3. Get instant fraud probability and explanations
    4. Explore **Model Insights** for detailed analytics
    """)

def call_api_predict(api_url, features, timeout=5):
    """Call backend API /predict with a single instance payload and return parsed JSON."""
    import requests
    try:
        resp = requests.post(api_url, json={"instance": features}, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        raise


def show_prediction_page(model, scalers, sample_data, prediction_mode="Local model", api_url=None):
    st.header("Fraud Prediction")

    # If using local model, ensure it's available
    if prediction_mode == "Local model" and model is None:
        st.error("Model not loaded. Please check model files.")
        return

    # Input method
    input_method = st.radio("Choose input method:", ["Manual Input", "Sample Transaction"])

    if input_method == "Manual Input":
        # Manual input form
        st.subheader("Transaction Details")

        col1, col2, col3 = st.columns(3)

        with col1:
            purchase_value = st.number_input("Purchase Value ($)", min_value=0.0, value=50.0, step=0.01)
            age = st.slider("Age", 18, 80, 35)
            sex = st.selectbox("Sex", ["M", "F"])

        with col2:
            source = st.selectbox("Traffic Source", ["SEO", "Ads", "Direct"])
            browser = st.selectbox("Browser", ["Chrome", "Safari", "Firefox", "IE", "Opera"])
            hour_of_day = st.slider("Hour of Day", 0, 23, 12)

        with col3:
            day_of_week = st.slider("Day of Week", 0, 6, 1)  # 0=Monday, 6=Sunday
            month = st.slider("Month", 1, 12, 6)
            time_since_signup = st.number_input("Time Since Signup (hours)", min_value=0.0, value=24.0)

        # Additional features
        same_day_purchase = st.checkbox("Same Day Purchase")
        txn_count_24h = st.number_input("Transactions in Last 24h", min_value=0, value=1)

        # Create feature dictionary
        features = {
            'purchase_value': purchase_value,
            'age': age,
            'sex': sex,
            'source': source,
            'browser': browser,
            'hour_of_day': hour_of_day,
            'day_of_week': day_of_week,
            'month': month,
            'time_since_signup': time_since_signup,
            'same_day_purchase': int(same_day_purchase),
            'txn_count_24h': txn_count_24h,
            'txn_count_168h': txn_count_24h * 7,  # Approximate
            'txn_count_720h': txn_count_24h * 30,  # Approximate
            'avg_time_between_txn_hours': time_since_signup / max(txn_count_24h, 1),
            'purchase_value_log': np.log1p(purchase_value),
            'purchase_hour': hour_of_day,
            'purchase_dayofweek': day_of_week,
            'purchase_month': month,
            'purchase_day': 15  # Default day of month
        }

    else:
        # Sample transaction
        st.subheader("Sample Transactions")
        if sample_data is not None:
            # Select sample
            sample_options = []
            for idx, row in sample_data.iterrows():
                label = "Fraud" if row['class'] == 1 else "Legitimate"
                sample_options.append(f"{label} - ${row['purchase_value']:.2f}")

            selected_sample = st.selectbox("Select a sample transaction:", sample_options)

            # Get selected row
            selected_idx = sample_data.index[sample_options.index(selected_sample)]
            selected_row = sample_data.loc[selected_idx]

            # Display sample details
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Transaction Details:**")
                st.write(f"Purchase Value: ${selected_row['purchase_value']:.2f}")
                st.write(f"Age: {selected_row['age']}")
                st.write(f"Sex: {selected_row['sex']}")
                st.write(f"Source: {selected_row['source']}")
                st.write(f"Browser: {selected_row['browser']}")

            with col2:
                st.write(f"Hour of Day: {selected_row['hour_of_day']}")
                st.write(f"Day of Week: {selected_row['day_of_week']}")
                st.write(f"Time Since Signup: {selected_row['time_since_signup']:.1f} hours")
                st.write(f"Same Day Purchase: {'Yes' if selected_row['same_day_purchase'] else 'No'}")

            features = selected_row.drop(['class', 'user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address']).to_dict()
        else:
            st.error("Sample data not available.")
            return

    # Prediction button
    if st.button("🔍 Analyze Transaction", type="primary"):
        with st.spinner("Analyzing transaction..."):
            try:
                if prediction_mode == "Backend API":
                    if not api_url:
                        st.error("API URL not set. Please set the API URL in the sidebar.")
                        return
                    try:
                        resp = call_api_predict(api_url, features)
                    except Exception as e:
                        st.error(f"API request failed: {e}")
                        return

                    preds = resp.get("predictions")
                    probs = resp.get("probabilities")

                    if preds is None:
                        st.error(f"Invalid response from API: {resp}")
                        return

                    pred_label = preds[0] if len(preds) > 0 else None
                    fraud_proba = probs[0] if probs and len(probs) > 0 else (1.0 if pred_label == 1 else 0.0)
                    prediction = "Fraudulent" if fraud_proba > 0.5 else "Legitimate"

                else:
                    # Prepare features for model
                    feature_df = pd.DataFrame([features])

                    # Make prediction using the pipeline
                    fraud_proba = model.predict_proba(feature_df)[0][1]
                    prediction = "Fraudulent" if fraud_proba > 0.5 else "Legitimate"

                # Display results
                st.header("🎯 Prediction Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    if prediction == "Fraudulent":
                        st.error(f"**{prediction}**")
                    else:
                        st.success(f"**{prediction}**")

                with col2:
                    st.metric("Fraud Probability", f"{fraud_proba:.1%}")

                with col3:
                    confidence = max(fraud_proba, 1-fraud_proba)
                    st.metric("Confidence", f"{confidence:.1%}")

                # Risk level
                if fraud_proba > 0.8:
                    risk_level = "🔴 High Risk"
                    risk_color = "red"
                elif fraud_proba > 0.6:
                    risk_level = "🟠 Medium Risk"
                    risk_color = "orange"
                else:
                    risk_level = "🟢 Low Risk"
                    risk_color = "green"

                st.markdown(f"### Risk Assessment: {risk_level}")

                # Recommendations
                st.subheader("💡 Recommendations")
                if fraud_proba > 0.5:
                    st.markdown("""
                    - **Flag for manual review**
                    - Request additional verification
                    - Consider transaction hold
                    - Check customer history
                    """)
                else:
                    st.markdown("""
                    - **Approve transaction**
                    - Normal processing
                    - No additional verification needed
                    """)

            except Exception as e:
                st.error(f"Error during prediction: {e}")

def show_insights_page():
    st.header("Model Insights & Analytics")

    st.subheader("📈 Feature Importance")
    try:
        # Load SHAP importance data
        shap_df = pd.read_csv('reports/figures/shap_feature_importance_top10.csv')
        st.bar_chart(shap_df.set_index('feature')['importance'])
    except:
        st.info("Feature importance data not available.")

    st.subheader("📊 Fraud Patterns")
    try:
        # Load some figures
        st.image('reports/figures/fraud_rate_by_hour.png', caption='Fraud Rate by Hour')
        st.image('reports/figures/purchase_value_by_class.png', caption='Purchase Value Distribution')
    except:
        st.info("Visualization files not available.")

    st.subheader("🌍 Geographic Insights")
    try:
        country_df = pd.read_csv('reports/figures/country_fraud_rates.csv')
        st.dataframe(country_df.head(10))
    except:
        st.info("Country fraud data not available.")

    st.subheader("🔍 SHAP Explanations")
    if not SHAP_AVAILABLE:
        st.info("SHAP is not installed in this environment. Install `shap` to enable explainability features.")
    else:
        try:
            st.markdown("SHAP explanation files are available in the reports folder.")
            st.info("View detailed SHAP explanations in the reports/shap_force_*.html files.")
        except Exception:
            st.info("SHAP explanation files not available.")

def show_about_page():
    st.header("About This System")

    st.markdown("""
    ## Fraud Detection for E-commerce

    This system was developed to improve fraud detection using machine learning models trained on e-commerce and bank transaction data.

    ### Key Features:
    - **Real-time Prediction**: Instant fraud probability scoring
    - **Explainable AI**: SHAP-based model interpretability
    - **Advanced Features**: Time-based patterns, velocity analysis, geolocation
    - **Model Ensemble**: XGBoost and Random Forest classifiers

    ### Technical Stack:
    - **Machine Learning**: XGBoost, Random Forest, scikit-learn
    - **Explainability**: SHAP (SHapley Additive exPlanations)
    - **Data Processing**: pandas, numpy
    - **Visualization**: matplotlib, seaborn, plotly
    - **Deployment**: Streamlit

    ### Model Performance:
    - Accuracy: 96.2%
    - Precision: 85.3%
    - Recall: 78.5%
    - F1-Score: 81.7%

    ### Contact:
    For questions or improvements, please refer to the project documentation.
    """)

if __name__ == "__main__":
    main()