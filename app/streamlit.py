"""
Streamlit app for Risk Assessment - Consolidated single-file deployment.
Run with: streamlit run app/streamlit.py
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_config
from src.decisioning.engine import DecisionEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Risk Assessment Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .risk-high {
        color: #d32f2f;
        font-weight: bold;
    }
    .risk-medium {
        color: #f57c00;
        font-weight: bold;
    }
    .risk-low {
        color: #388e3c;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CACHING & MODEL LOADING
# ============================================================================

@st.cache_resource
def load_models_and_config():
    """Load models and configuration (cached)."""
    try:
        project_root = Path(__file__).parent.parent
        
        # Load config
        config_path = project_root / "configs" / "config.yaml"
        config = load_config(str(config_path))
        logger.info(f"✓ Config loaded from {config_path}")
        
        # Load preprocessor
        preprocessor_path = f"{project_root} / models / preprocessor.pkl"
        if preprocessor_path.exists():
            preprocessor = joblib.load(preprocessor_path)
            logger.info(f"✓ Preprocessor loaded")
        else:
            logger.warning(f"✗ Preprocessor not found at {preprocessor_path}")
            preprocessor = None
        
        # Load model
        model_path = f"{project_root} / models / baseline_model.pkl"
        if model_path.exists():
            model = joblib.load(model_path)
            logger.info(f"✓ Model loaded")
        else:
            logger.error(f"✗ Model not found at {model_path}")
            # List available files
            models_dir = project_root / "models"
            if models_dir.exists():
                pkl_files = list(models_dir.glob("*.pkl"))
                logger.error(f"Available files: {[f.name for f in pkl_files]}")
            model = None
        
        # Initialize decision engine
        reports_dir = project_root / "reports"
        decision_engine = DecisionEngine(output_dir=str(reports_dir))
        
        return {
            "model": model,
            "preprocessor": preprocessor,
            "decision_engine": decision_engine,
            "config": config
        }
    
    except Exception as e:
        logger.error(f"Error loading models: {e}", exc_info=True)
        return {
            "model": None,
            "preprocessor": None,
            "decision_engine": None,
            "config": None
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_risk_color(risk_score: float) -> str:
    """Get color based on risk score."""
    if risk_score < 0.33:
        return "green"
    elif risk_score < 0.67:
        return "orange"
    else:
        return "red"


def get_risk_level(risk_score: float) -> str:
    """Get risk level text."""
    if risk_score < 0.33:
        return "✅ Low Risk"
    elif risk_score < 0.67:
        return "⚠️ Medium Risk"
    else:
        return "🔴 High Risk"


def make_prediction(features: Dict[str, Any], state: Dict) -> Dict[str, Any]:
    """
    Make a single prediction.
    
    Args:
        features: Feature dictionary
        state: Model state dict
        
    Returns:
        Prediction result dict
    """
    if state["model"] is None:
        return {
            "error": "Model not loaded. Please run 'python train.py' first.",
            "success": False
        }
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Preprocess
        if state["preprocessor"]:
            df_processed = state["preprocessor"].transform(df)
        else:
            df_processed = df
        
        # Predict
        risk_score = float(state["model"].predict_proba(df_processed)[:, 1][0])
        
        # Apply decision policy
        config = state["config"]
        approve_threshold = config.get("approve_threshold", 0.30)
        reject_threshold = config.get("reject_threshold", 0.70)
        
        if risk_score < approve_threshold:
            decision = "✅ APPROVE"
            decision_type = "APPROVE"
            explanation = f"Risk score {risk_score:.1%} is below approval threshold ({approve_threshold:.0%})"
        elif risk_score > reject_threshold:
            decision = "❌ REJECT"
            decision_type = "REJECT"
            explanation = f"Risk score {risk_score:.1%} exceeds rejection threshold ({reject_threshold:.0%})"
        else:
            decision = "⏸️ REVIEW"
            decision_type = "REVIEW"
            explanation = f"Risk score {risk_score:.1%} requires manual review (between {approve_threshold:.0%}-{reject_threshold:.0%})"
        
        return {
            "success": True,
            "risk_score": risk_score,
            "decision": decision,
            "decision_type": decision_type,
            "explanation": explanation,
            "approve_threshold": approve_threshold,
            "reject_threshold": reject_threshold
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return {
            "error": f"Prediction failed: {str(e)}",
            "success": False
        }


def create_risk_gauge(risk_score: float, thresholds: Dict) -> go.Figure:
    """Create risk gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score * 100,
        title={'text': "Default Risk Probability"},
        domain={'x': [0, 1], 'y': [0, 1]},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': get_risk_color(risk_score)},
            'steps': [
                {'range': [0, 33], 'color': "lightgreen"},
                {'range': [33, 67], 'color': "lightyellow"},
                {'range': [67, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    return fig


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main app."""
    st.title("🏦 Loan Risk Assessment Dashboard")
    st.markdown("---")
    
    # Load models
    state = load_models_and_config()
    
    # Sidebar navigation
    with st.sidebar:
        st.header("🗂️ Navigation")
        page = st.radio(
            "Select Page",
            ["Single Prediction", "Batch Prediction", "Configuration", "About"]
        )
    
    # Check model status
    if state["model"] is None:
        st.error(
            "⚠️ **Model Not Loaded**\n\n"
            "The model file (models/xgboost_model.pkl) is not found. "
            "Please run `python train.py` first to train and save the model."
        )
    else:
        st.success("✅ Model and Configuration Loaded")
    
    # ========================================================================
    # SINGLE PREDICTION PAGE
    # ========================================================================
    
    if page == "Single Prediction":
        st.header("Individual Applicant Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Personal Information")
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
            city = st.text_input("City", value="New York")
        
        with col2:
            st.subheader("💰 Financial Information")
            income = st.number_input("Annual Income ($)", min_value=0, max_value=1000000, value=75000)
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=720)
            loan_amount = st.number_input("Loan Amount ($)", min_value=0, max_value=500000, value=15000)
            years_experience = st.number_input("Years of Work Experience", min_value=0, max_value=60, value=5)
            employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
        
        if st.button("📊 Assess Risk", use_container_width=True, type="primary"):
            features = {
                "Age": age,
                "Income": income,
                "CreditScore": credit_score,
                "LoanAmount": loan_amount,
                "YearsExperience": years_experience,
                "Gender": gender,
                "Education": education,
                "City": city,
                "EmploymentType": employment_type
            }
            
            result = make_prediction(features, state)
            
            if result["success"]:
                st.markdown("---")
                st.subheader("📈 Risk Assessment Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Risk Score",
                        f"{result['risk_score']:.1%}",
                        delta=None,
                        delta_color="inverse"
                    )
                
                with col2:
                    st.metric(
                        "Risk Level",
                        get_risk_level(result['risk_score']),
                        delta=None
                    )
                
                with col3:
                    st.metric(
                        "Decision",
                        result['decision'],
                        delta=None
                    )
                
                # Explanation
                st.info(result['explanation'])
                
                # Risk gauge
                fig = create_risk_gauge(
                    result['risk_score'],
                    {
                        "approve": result['approve_threshold'],
                        "reject": result['reject_threshold']
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Applicant details
                st.subheader("📋 Submitted Information")
                details_df = pd.DataFrame({
                    "Field": list(features.keys()),
                    "Value": list(features.values())
                })
                st.dataframe(details_df, use_container_width=True, hide_index=True)
            
            else:
                st.error(f"❌ {result.get('error', 'Unknown error')}")
    
    # ========================================================================
    # BATCH PREDICTION PAGE
    # ========================================================================
    
    elif page == "Batch Prediction":
        st.header("📊 Batch Applicant Assessment")
        
        st.info("Upload a CSV file with applicant data for batch processing")
        
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            if st.button("🚀 Process Batch", use_container_width=True, type="primary"):
                try:
                    predictions = []
                    progress_bar = st.progress(0)
                    
                    for idx, row in df.iterrows():
                        features = row.to_dict()
                        result = make_prediction(features, state)
                        
                        if result["success"]:
                            predictions.append({
                                "Risk Score": result['risk_score'],
                                "Decision": result['decision_type'],
                                "Explanation": result['explanation']
                            })
                        else:
                            predictions.append({
                                "Risk Score": None,
                                "Decision": "ERROR",
                                "Explanation": result.get("error", "Unknown error")
                            })
                        
                        progress_bar.progress((idx + 1) / len(df))
                    
                    st.markdown("---")
                    st.subheader("✅ Batch Results")
                    
                    results_df = pd.DataFrame(predictions)
                    
                    # Calculate statistics
                    risk_scores = [p["Risk Score"] for p in predictions if p["Risk Score"] is not None]
                    
                    if risk_scores:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Applicants", len(predictions))
                        with col2:
                            st.metric("Mean Risk Score", f"{np.mean(risk_scores):.1%}")
                        with col3:
                            st.metric("Median Risk Score", f"{np.median(risk_scores):.1%}")
                        with col4:
                            approvals = sum(1 for p in predictions if p["Decision"] == "APPROVE")
                            st.metric("Approval Rate", f"{approvals / len(predictions):.1%}")
                        
                        # Decision distribution
                        st.subheader("Decision Distribution")
                        decision_counts = results_df["Decision"].value_counts()
                        
                        fig = px.bar(
                            x=decision_counts.index,
                            y=decision_counts.values,
                            labels={"x": "Decision", "y": "Count"},
                            color=decision_counts.index,
                            color_discrete_map={
                                "APPROVE": "green",
                                "REVIEW": "orange",
                                "REJECT": "red",
                                "ERROR": "gray"
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Risk distribution histogram
                        st.subheader("Risk Score Distribution")
                        fig = px.histogram(
                            x=risk_scores,
                            nbins=20,
                            labels={"x": "Risk Score", "count": "Frequency"},
                            title="Distribution of Risk Scores"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed results
                    st.subheader("Detailed Results")
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name="batch_predictions.csv",
                        mime="text/csv"
                    )
                
                except Exception as e:
                    st.error(f"❌ Batch processing failed: {str(e)}")
    
    # ========================================================================
    # CONFIGURATION PAGE
    # ========================================================================
    
    elif page == "Configuration":
        st.header("⚙️ System Configuration")
        
        if state["config"]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Decision Thresholds")
                approve_th = state["config"].get("approve_threshold", 0.30)
                reject_th = state["config"].get("reject_threshold", 0.70)
                
                st.metric("Approval Threshold", f"{approve_th:.0%}")
                st.metric("Rejection Threshold", f"{reject_th:.0%}")
            
            with col2:
                st.subheader("Model Information")
                st.metric("Model Type", state["config"].get("model_type", "XGBoost"))
                st.metric("Features", state["config"].get("feature_count", "Unknown"))
            
            st.info(
                f"**Decision Logic:**\n\n"
                f"- Risk Score < {approve_th:.0%} → **✅ APPROVE**\n"
                f"- Risk Score {approve_th:.0%} to {reject_th:.0%} → **⏸️ REVIEW**\n"
                f"- Risk Score > {reject_th:.0%} → **❌ REJECT**"
            )
        else:
            st.error("Configuration not loaded")
    
    # ========================================================================
    # ABOUT PAGE
    # ========================================================================
    
    elif page == "About":
        st.header("ℹ️ About Risk Assessment System")
        
        st.markdown("""
        ### 🎯 Overview
        
        The **Risk Assessment Manager** is a production-grade machine learning system that predicts 
        loan default risk and automates lending decisions.
        
        ### ✨ Key Features
        
        - **🎯 Accurate Predictions:** 80%+ AUC using ensemble methods (XGBoost, LightGBM)
        - **📊 Explainable Decisions:** Rule-based decision policies with clear thresholds
        - **⚙️ Configurable Policies:** Tune approval/rejection thresholds based on business needs
        - **🔍 Real-time Scoring:** Instant risk assessment for individual or batch applicants
        - **📈 Interactive Dashboard:** Streamlit-based interface for easy exploration
        
        ### 🔄 How It Works
        
        1. **Feature Extraction:** Applicant data is validated and engineered
        2. **Risk Scoring:** ML model predicts probability of default (0-1)
        3. **Decision Policy:** Risk score mapped to decision (APPROVE/REVIEW/REJECT)
        4. **Notification:** Decision and explanation returned to user
        
        ### 🏗️ Architecture
        
        - **Frontend:** Streamlit (this app)
        - **Models:** XGBoost, LightGBM
        - **Preprocessing:** scikit-learn pipelines
        - **Deployment:** Streamlit Cloud compatible
        
        ### 📚 Required Features
        
        The model expects the following applicant information:
        
        **Numerical Features:**
        - Age (18-100)
        - Annual Income ($)
        - Credit Score (300-850)
        - Loan Amount ($)
        - Years of Experience (0-60)
        
        **Categorical Features:**
        - Gender (Male/Female/Other)
        - Education (High School/Bachelor/Master/PhD)
        - City (text)
        - Employment Type (Full-time/Part-time/Self-employed/Unemployed)
        
        ### 🚀 Deployment
        
        To deploy on Streamlit Cloud:
        
        1. Push code to GitHub
        2. Go to share.streamlit.io
        3. Connect your repo
        4. Set main file to `app/streamlit.py`
        
        ### 📖 Documentation
        
        For detailed information, see:
        - ARCHITECTURE.md - System design
        - MODEL_GUIDE.md - Model training details
        - DECISIONING_POLICY.md - Decision rules
        """)
        
        st.markdown("---")
        st.text("Risk Assessment Manager v1.0.0 | Streamlit Edition")


if __name__ == "__main__":
    main()
