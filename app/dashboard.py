"""Streamlit dashboard for Risk Assessment predictions."""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import requests
import json

# Configure page
st.set_page_config(
    page_title="Risk Assessment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API endpoint
API_URL = "http://localhost:8000"

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


def check_api_health():
    """Check if API is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def get_risk_color(risk_score):
    """Get color based on risk score."""
    if risk_score < 0.33:
        return "green"
    elif risk_score < 0.67:
        return "orange"
    else:
        return "red"


def get_risk_level(risk_score):
    """Get risk level text."""
    if risk_score < 0.33:
        return "Low Risk"
    elif risk_score < 0.67:
        return "Medium Risk"
    else:
        return "High Risk"


# Main layout
st.title("🏦 Loan Risk Assessment Dashboard")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Select Page",
        ["Single Prediction", "Batch Prediction", "Configuration", "About"]
    )

# Check API status
if not check_api_health():
    st.error(
        "⚠️ **API Not Running**\n\n"
        "The backend API is not available. Please start it with:\n"
        "```bash\npython -m uvicorn app.api:app --host 0.0.0.0 --port 8000\n```"
    )
    st.stop()

st.success("✅ API Connected")

# Single Prediction Page
if page == "Single Prediction":
    st.header("Individual Applicant Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
        city = st.text_input("City", value="New York")
    
    with col2:
        st.subheader("Financial Information")
        income = st.number_input("Annual Income ($)", min_value=0, max_value=500000, value=75000)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=720)
        loan_amount = st.number_input("Loan Amount ($)", min_value=0, max_value=500000, value=15000)
        years_experience = st.number_input("Years of Work Experience", min_value=0, max_value=60, value=5)
        employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
    
    if st.button("📊 Assess Risk", use_container_width=True):
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
        
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={"features": features}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Display results
                st.markdown("---")
                st.subheader("Risk Assessment Results")
                
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
                
                # Risk gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=result['risk_score'] * 100,
                    title={'text': "Default Probability"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': get_risk_color(result['risk_score'])},
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
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
        
        except Exception as e:
            st.error(f"Connection error: {str(e)}")


# Batch Prediction Page
elif page == "Batch Prediction":
    st.header("Batch Applicant Assessment")
    
    st.info("Upload a CSV file with applicant data for batch processing")
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.subheader("Data Preview")
        st.dataframe(df.head(10))
        
        if st.button("🚀 Process Batch", use_container_width=True):
            try:
                applicants = df.to_dict(orient='records')
                
                response = requests.post(
                    f"{API_URL}/predict_batch",
                    json={"applicants": applicants}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    stats = result['statistics']
                    predictions = result['predictions']
                    
                    st.markdown("---")
                    st.subheader("Batch Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Applicants", stats['total_applicants'])
                    with col2:
                        st.metric("Mean Risk Score", f"{stats['mean_risk_score']:.1%}")
                    with col3:
                        st.metric("Approval Rate", f"{stats['approval_rate']:.1%}")
                    with col4:
                        st.metric("Median Risk Score", f"{stats['median_risk_score']:.1%}")
                    
                    # Decision distribution
                    st.subheader("Decision Distribution")
                    decision_data = stats['decision_distribution']
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(decision_data.keys()),
                            y=list(decision_data.values()),
                            marker_color=['green', 'orange', 'red']
                        )
                    ])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed results
                    st.subheader("Detailed Results")
                    results_df = pd.DataFrame([
                        {
                            "Risk Score": p['risk_score'],
                            "Decision": p['decision'],
                            "Explanation": p['explanation']
                        }
                        for p in predictions
                    ])
                    st.dataframe(results_df, use_container_width=True)
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
            
            except Exception as e:
                st.error(f"Connection error: {str(e)}")


# Configuration Page
elif page == "Configuration":
    st.header("System Configuration")
    
    try:
        response = requests.get(f"{API_URL}/config")
        
        if response.status_code == 200:
            config = response.json()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Decision Thresholds")
                st.metric("Approval Threshold", f"{config['approve_threshold']:.0%}")
                st.metric("Rejection Threshold", f"{config['reject_threshold']:.0%}")
            
            with col2:
                st.subheader("Model Info")
                st.metric("Model Type", config.get('model_type', 'Unknown'))
                st.metric("Features", config.get('feature_count', 'Unknown'))
            
            st.info(
                "**Decision Logic:**\n\n"
                f"- Risk Score < {config['approve_threshold']:.0%} → **APPROVE**\n"
                f"- Risk Score {config['approve_threshold']:.0%} to {config['reject_threshold']:.0%} → **REVIEW**\n"
                f"- Risk Score > {config['reject_threshold']:.0%} → **REJECT**"
            )
        else:
            st.error("Could not fetch configuration")
    
    except Exception as e:
        st.error(f"Connection error: {str(e)}")


# About Page
elif page == "About":
    st.header("About Risk Assessment System")
    
    st.markdown("""
    ### Overview
    
    The **Risk Assessment Manager** is a production-grade machine learning system that predicts 
    loan default risk and automates lending decisions.
    
    ### Key Features
    
    - 🎯 **Accurate Predictions:** 80%+ AUC using ensemble methods
    - 📊 **Explainable Decisions:** SHAP-based feature importance
    - ⚙️ **Configurable Policies:** Customize approval thresholds
    - 📈 **Real-time API:** RESTful endpoints for integration
    - 🎨 **Interactive Dashboard:** Streamlit-based interface
    
    ### How It Works
    
    1. **Feature Extraction:** Applicant data is processed and engineered
    2. **Risk Scoring:** ML model predicts probability of default (0-1)
    3. **Decision Policy:** Risk score mapped to decision (APPROVE/REVIEW/REJECT)
    4. **Notification:** Decision and explanation returned to user
    
    ### Architecture
    
    - **Backend:** FastAPI with uvicorn
    - **Frontend:** Streamlit dashboard
    - **Models:** Scikit-learn, XGBoost, LightGBM
    - **Explainability:** SHAP
    
    ### Quick Links
    
    - [Full Documentation](../documentation/ARCHITECTURE.md)
    - [API Reference](../documentation/API_REFERENCE.md)
    - [Model Guide](../documentation/MODEL_GUIDE.md)
    """)
    
    st.markdown("---")
    st.text("Risk Assessment Manager v1.0.0")


if __name__ == "__main__":
    pass
