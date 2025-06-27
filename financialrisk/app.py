"""
Financial Risk Assessment Application with PRefLexOR
Streamlit-based interface for transparent credit analysis
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
from models.ollama_client import OllamaClient
from models.risk_assessor import FinancialRiskAssessor
from utils.data_validation import DataValidator
from utils.visualization import RiskVisualizer
from config.settings import (
    APP_SETTINGS, OLLAMA_CONFIG, get_model_config,
    RISK_THRESHOLDS, COMPLIANCE_RULES
)

# Page configuration
st.set_page_config(
    page_title=APP_SETTINGS["title"],
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .risk-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .risk-low {
        background-color: #d4edda;
        border-color: #28a745;
    }
    .risk-moderate {
        background-color: #fff3cd;
        border-color: #ffc107;
    }
    .risk-high {
        background-color: #f8d7da;
        border-color: #dc3545;
    }
    .thinking-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'ollama_client' not in st.session_state:
        st.session_state.ollama_client = None
    if 'risk_assessor' not in st.session_state:
        st.session_state.risk_assessor = None
    if 'assessment_results' not in st.session_state:
        st.session_state.assessment_results = []
    if 'current_model' not in st.session_state:
        st.session_state.current_model = OLLAMA_CONFIG["default_model"]

def setup_ollama_connection():
    """Setup and validate Ollama connection"""
    
    st.sidebar.header("🤖 Model Configuration")
    
    # Model selection
    available_models = OLLAMA_CONFIG["alternative_models"]
    selected_model = st.sidebar.selectbox(
        "Select Model",
        available_models,
        index=available_models.index(st.session_state.current_model),
        help="Choose the Ollama model for risk assessment"
    )
    
    # Update model if changed
    if selected_model != st.session_state.current_model:
        st.session_state.current_model = selected_model
        st.session_state.ollama_client = None  # Force reconnection
        st.session_state.risk_assessor = None
    
    # Initialize client if needed
    if st.session_state.ollama_client is None:
        with st.spinner("Connecting to Ollama..."):
            try:
                st.session_state.ollama_client = OllamaClient(
                    base_url=OLLAMA_CONFIG["base_url"],
                    model=selected_model
                )
                
                # Validate setup
                validation = st.session_state.ollama_client.validate_model_setup()
                
                # Debug information
                st.sidebar.write("**Debug Info:**")
                st.sidebar.write(f"Ollama running: {validation['ollama_running']}")
                st.sidebar.write(f"Model available: {validation['model_available']}")
                st.sidebar.write(f"Model functional: {validation['model_functional']}")
                
                if validation["model_functional"]:
                    st.sidebar.success("✅ Model ready")
                    st.session_state.risk_assessor = FinancialRiskAssessor(st.session_state.ollama_client)
                    return True
                else:
                    st.sidebar.error(f"❌ {validation['recommended_action']}")
                    if validation["available_models"]:
                        st.sidebar.info("Available models: " + ", ".join(validation["available_models"]))
                    else:
                        st.sidebar.info("No models found. Please pull a model first.")
                    
                    # Show detailed setup instructions
                    st.sidebar.markdown("**Setup Instructions:**")
                    st.sidebar.code("""
# 1. Install Ollama
# Download from https://ollama.ai

# 2. Start Ollama service
ollama serve

# 3. Pull recommended model
ollama pull llama3.1:8b-instruct-q4_K_M

# 4. Verify installation
ollama list
                    """)
                    return False
                    
            except Exception as e:
                st.sidebar.error(f"❌ Connection failed: {str(e)}")
                st.sidebar.write(f"**Error details:** {type(e).__name__}: {str(e)}")
                
                # Try to provide specific guidance based on error
                if "Connection refused" in str(e) or "ConnectionError" in str(e):
                    st.sidebar.warning("Ollama service not running. Start with: `ollama serve`")
                elif "timeout" in str(e).lower():
                    st.sidebar.warning("Connection timeout. Check if Ollama is responding.")
                
                return False
    
    # If we get here, connection should be established
    if st.session_state.risk_assessor is None and st.session_state.ollama_client is not None:
        st.session_state.risk_assessor = FinancialRiskAssessor(st.session_state.ollama_client)
    
    return st.session_state.risk_assessor is not None

def create_mortgage_form():
    """Create mortgage application form"""
    
    st.subheader("🏠 Mortgage Application")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Applicant Information**")
        applicant_name = st.text_input("Applicant Name", value="John Doe")
        email = st.text_input("Email", value="john.doe@example.com")
        phone = st.text_input("Phone", value="555-123-4567")
        
        st.markdown("**Employment Details**")
        employment_type = st.selectbox(
            "Employment Type",
            ["full_time", "part_time", "self_employed", "contract", "retired"]
        )
        employment_years = st.number_input("Years of Employment", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
        monthly_income = st.number_input("Monthly Income ($)", min_value=0, max_value=100000, value=7500, step=100)
        
    with col2:
        st.markdown("**Loan Details**")
        loan_amount = st.number_input("Loan Amount ($)", min_value=1000, max_value=10000000, value=400000, step=1000)
        property_value = st.number_input("Property Value ($)", min_value=1000, max_value=50000000, value=500000, step=1000)
        down_payment = st.number_input("Down Payment ($)", min_value=0, max_value=10000000, value=100000, step=1000)
        
        st.markdown("**Financial Information**")
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=720, step=1)
        monthly_debt_payments = st.number_input("Monthly Debt Payments ($)", min_value=0, max_value=50000, value=800, step=50)
        
        st.markdown("**Property Details**")
        property_type = st.selectbox("Property Type", ["single_family", "condo", "townhouse", "manufactured"])
        occupancy_type = st.selectbox("Occupancy", ["primary_residence", "second_home", "investment"])
    
    # Calculate proposed monthly payment (estimate)
    interest_rate = 7.0  # Default rate for calculation
    loan_term = 30
    monthly_rate = interest_rate / 12 / 100
    num_payments = loan_term * 12
    
    if monthly_rate > 0:
        proposed_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / \
                          ((1 + monthly_rate)**num_payments - 1)
    else:
        proposed_payment = loan_amount / num_payments
    
    return {
        "applicant_name": applicant_name,
        "email": email,
        "phone": phone,
        "employment_type": employment_type,
        "employment_years": employment_years,
        "monthly_income": monthly_income,
        "loan_amount": loan_amount,
        "property_value": property_value,
        "down_payment": down_payment,
        "credit_score": credit_score,
        "monthly_debt_payments": monthly_debt_payments,
        "property_type": property_type,
        "occupancy_type": occupancy_type,
        "proposed_monthly_payment": proposed_payment
    }

def create_personal_loan_form():
    """Create personal loan application form"""
    
    st.subheader("💳 Personal Loan Application")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Applicant Information**")
        applicant_name = st.text_input("Applicant Name", value="Jane Smith")
        email = st.text_input("Email", value="jane.smith@example.com") 
        phone = st.text_input("Phone", value="555-987-6543")
        
        st.markdown("**Employment Details**")
        employment_type = st.selectbox(
            "Employment Type", 
            ["full_time", "part_time", "self_employed", "contract", "retired"],
            key="pl_employment"
        )
        employment_years = st.number_input("Years of Employment", min_value=0.0, max_value=50.0, value=3.0, step=0.5, key="pl_emp_years")
        monthly_income = st.number_input("Monthly Income ($)", min_value=0, max_value=50000, value=4500, step=100, key="pl_income")
        
    with col2:
        st.markdown("**Loan Request**")
        loan_amount = st.number_input("Loan Amount ($)", min_value=1000, max_value=100000, value=15000, step=500, key="pl_amount")
        loan_purpose = st.selectbox("Loan Purpose", ["debt_consolidation", "home_improvement", "vacation", "medical", "other"])
        
        st.markdown("**Financial Information**")
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=680, step=1, key="pl_credit")
        monthly_debt_payments = st.number_input("Monthly Debt Payments ($)", min_value=0, max_value=10000, value=400, step=50, key="pl_debts")
        
        st.markdown("**Additional Information**")
        housing_payment = st.number_input("Monthly Housing Payment ($)", min_value=0, max_value=10000, value=1200, step=50)
        assets_savings = st.number_input("Savings Account Balance ($)", min_value=0, max_value=1000000, value=5000, step=100)
    
    # Estimate monthly payment
    loan_term = 5  # 5 years typical for personal loans
    interest_rate = 12.0  # Typical personal loan rate
    monthly_rate = interest_rate / 12 / 100
    num_payments = loan_term * 12
    
    if monthly_rate > 0:
        requested_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / \
                           ((1 + monthly_rate)**num_payments - 1)
    else:
        requested_payment = loan_amount / num_payments
    
    return {
        "applicant_name": applicant_name,
        "email": email,
        "phone": phone,
        "employment_type": employment_type,
        "employment_years": employment_years,
        "monthly_income": monthly_income,
        "loan_amount": loan_amount,
        "loan_purpose": loan_purpose,
        "credit_score": credit_score,
        "monthly_debt_payments": monthly_debt_payments,
        "housing_payment": housing_payment,
        "assets_savings": assets_savings,
        "requested_monthly_payment": requested_payment
    }

def create_business_credit_form():
    """Create business credit application form"""
    
    st.subheader("🏢 Business Credit Application")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Business Information**")
        business_name = st.text_input("Business Name", value="Tech Solutions LLC")
        business_age_years = st.number_input("Business Age (Years)", min_value=0.0, max_value=100.0, value=7.0, step=0.5)
        industry = st.selectbox(
            "Industry",
            ["technology", "healthcare", "retail", "manufacturing", "construction", 
             "professional_services", "food_service", "transportation", "real_estate", "finance", "other"]
        )
        
        st.markdown("**Financial Information**")
        annual_revenue = st.number_input("Annual Revenue ($)", min_value=0, max_value=100000000, value=2500000, step=10000)
        net_operating_income = st.number_input("Net Operating Income ($)", min_value=0, max_value=50000000, value=400000, step=5000)
        business_credit_score = st.number_input("Business Credit Score", min_value=0, max_value=100, value=75, step=1)
        
    with col2:
        st.markdown("**Credit Request**")
        requested_amount = st.number_input("Requested Amount ($)", min_value=1000, max_value=50000000, value=150000, step=1000)
        credit_purpose = st.selectbox("Credit Purpose", ["working_capital", "equipment", "expansion", "real_estate", "other"])
        
        st.markdown("**Balance Sheet Information**")
        current_assets = st.number_input("Current Assets ($)", min_value=0, max_value=100000000, value=300000, step=5000)
        current_liabilities = st.number_input("Current Liabilities ($)", min_value=0, max_value=100000000, value=150000, step=5000)
        total_debt = st.number_input("Total Debt ($)", min_value=0, max_value=100000000, value=200000, step=5000)
        total_equity = st.number_input("Total Equity ($)", min_value=-10000000, max_value=100000000, value=500000, step=5000)
        
        st.markdown("**Debt Service**")
        annual_debt_service = st.number_input("Annual Debt Service ($)", min_value=0, max_value=10000000, value=50000, step=1000)
    
    return {
        "business_name": business_name,
        "business_age_years": business_age_years,
        "industry": industry,
        "annual_revenue": annual_revenue,
        "net_operating_income": net_operating_income,
        "business_credit_score": business_credit_score,
        "requested_amount": requested_amount,
        "credit_purpose": credit_purpose,
        "current_assets": current_assets,
        "current_liabilities": current_liabilities,
        "total_debt": total_debt,
        "total_equity": total_equity,
        "annual_debt_service": annual_debt_service
    }

def display_assessment_result(result):
    """Display comprehensive assessment results"""
    
    st.markdown('<p class="sub-header">📊 Risk Assessment Results</p>', unsafe_allow_html=True)
    
    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Recommendation", result.recommendation)
    with col2:
        st.metric("Risk Level", result.risk_level)
    with col3:
        st.metric("Confidence", f"{result.confidence_score:.1%}")
    with col4:
        st.metric("Processing Time", f"{len(result.thinking_process.split())} tokens")
    
    # Risk level styling
    risk_class = f"risk-{result.risk_level.lower()}"
    st.markdown(f"""
    <div class="risk-box {risk_class}">
        <strong>{result.recommendation}</strong> - {result.risk_level} Risk Level
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Analytics", "🧠 AI Reasoning", "📋 Summary", "⚖️ Compliance", "📊 Data"])
    
    with tab1:
        st.markdown("### Financial Analytics")
        
        # Risk gauge
        col1, col2 = st.columns(2)
        
        with col1:
            gauge_fig = RiskVisualizer.create_risk_gauge(result.risk_level, result.confidence_score)
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        with col2:
            # Financial metrics chart
            if result.financial_metrics:
                metrics_fig = RiskVisualizer.create_financial_metrics_chart(
                    result.financial_metrics, result.assessment_type
                )
                st.plotly_chart(metrics_fig, use_container_width=True)
        
        # Risk factors analysis
        if result.risk_factors:
            risk_factors_fig = RiskVisualizer.create_risk_factors_chart(result.risk_factors)
            st.plotly_chart(risk_factors_fig, use_container_width=True)
            
            # Detailed risk factors table
            st.markdown("### Risk Factors Detail")
            risk_df = pd.DataFrame(result.risk_factors)
            st.dataframe(risk_df, use_container_width=True)
    
    with tab2:
        st.markdown("### AI Reasoning Process")
        
        if result.thinking_process:
            st.markdown(f"""
            <div class="thinking-box">
                <h4>🧠 Step-by-Step Analysis</h4>
                {result.thinking_process.replace('\n', '<br>')}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No detailed reasoning available for this assessment.")
        
        st.markdown("### Final Analysis")
        st.write(result.final_analysis)
        
        # Model information
        st.markdown("### Model Information")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Model Used:** {result.model_used}")
        with col2:
            st.info(f"**Timestamp:** {result.timestamp}")
    
    with tab3:
        st.markdown("### Assessment Summary")
        
        # Create summary table
        summary_fig = RiskVisualizer.create_assessment_summary_table(result)
        st.plotly_chart(summary_fig, use_container_width=True)
        
        # Key findings
        st.markdown("### Key Findings")
        
        # High-risk factors
        high_risk_factors = [f for f in result.risk_factors if f.get("risk_level") in ["high", "critical"]]
        if high_risk_factors:
            st.warning("**High Risk Factors:**")
            for factor in high_risk_factors:
                st.write(f"• {factor.get('description', 'Unknown factor')}")
        
        # Low-risk factors (strengths)
        low_risk_factors = [f for f in result.risk_factors if f.get("risk_level") == "low"]
        if low_risk_factors:
            st.success("**Strengths:**")
            for factor in low_risk_factors[:3]:  # Show top 3
                st.write(f"• {factor.get('description', 'Unknown factor')}")
    
    with tab4:
        st.markdown("### Regulatory Compliance")
        
        if result.compliance_status:
            for rule, status in result.compliance_status.items():
                if status:
                    st.success(f"✅ {rule.replace('_', ' ').title()}: Compliant")
                else:
                    st.error(f"❌ {rule.replace('_', ' ').title()}: Non-Compliant")
        else:
            st.info("No specific compliance checks performed for this assessment type.")
        
        # Compliance information
        if result.assessment_type == "mortgage":
            st.markdown("### Qualified Mortgage (QM) Requirements")
            st.info("""
            **Key QM Requirements:**
            - Debt-to-Income ratio ≤ 43%
            - Verified income documentation
            - Limited points and fees
            - No interest-only payments
            - No negative amortization
            """)
    
    with tab5:
        st.markdown("### Raw Assessment Data")
        
        # Export functionality
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📄 Download JSON Report"):
                assessment_json = json.dumps(st.session_state.risk_assessor.export_assessment(result), indent=2)
                st.download_button(
                    label="Download Assessment Report",
                    data=assessment_json,
                    file_name=f"assessment_{result.application_id}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📊 Download CSV Summary"):
                # Create summary DataFrame
                summary_data = {
                    "Metric": ["Application ID", "Recommendation", "Risk Level", "Confidence Score"] + 
                             list(result.financial_metrics.keys()),
                    "Value": [result.application_id, result.recommendation, result.risk_level, 
                             f"{result.confidence_score:.1%}"] + 
                             [f"{v:.4f}" if isinstance(v, float) else str(v) for v in result.financial_metrics.values()]
                }
                summary_df = pd.DataFrame(summary_data)
                
                csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV Summary",
                    data=csv,
                    file_name=f"summary_{result.application_id}.csv",
                    mime="text/csv"
                )
        
        # Display raw data
        st.json(st.session_state.risk_assessor.export_assessment(result))

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🏦 Financial Risk Assessment</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #7f8c8d; font-size: 1.2rem;">Transparent Credit Analysis with Explainable AI</p>', unsafe_allow_html=True)
    
    # Setup Ollama connection
    if not setup_ollama_connection():
        st.error("Please configure Ollama connection in the sidebar before proceeding.")
        return
    
    # Main content
    st.sidebar.header("📋 Assessment Type")
    assessment_type = st.sidebar.selectbox(
        "Select Assessment Type",
        ["mortgage", "personal_loan", "business_credit"],
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    # Application form based on type
    if assessment_type == "mortgage":
        application_data = create_mortgage_form()
    elif assessment_type == "personal_loan":
        application_data = create_personal_loan_form()
    else:  # business_credit
        application_data = create_business_credit_form()
    
    # Validation and assessment
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🔍 Perform Risk Assessment", type="primary", use_container_width=True):
            
            # Validate input data
            validation_result = DataValidator.validate_application_completeness(application_data, assessment_type)
            
            if not validation_result["is_valid"]:
                st.error("**Validation Errors:**")
                for error in validation_result["errors"]:
                    st.error(f"• {error}")
                return
            
            # Show warnings for missing optional fields
            if validation_result["missing_optional_fields"]:
                st.warning("**Optional Information Missing** (assessment quality may be reduced):")
                for field in validation_result["missing_optional_fields"][:3]:  # Show first 3
                    st.warning(f"• {field.replace('_', ' ').title()}")
            
            # Check if risk assessor is available
            if st.session_state.risk_assessor is None:
                st.error("❌ **Risk Assessor Not Available**")
                st.error("Please ensure Ollama is properly configured in the sidebar before performing assessments.")
                st.info("Check the sidebar for connection status and setup instructions.")
                return
            
            # Perform assessment
            with st.spinner("🤖 Analyzing application with AI..."):
                try:
                    if assessment_type == "mortgage":
                        result = st.session_state.risk_assessor.assess_mortgage_risk(application_data)
                    elif assessment_type == "personal_loan":
                        result = st.session_state.risk_assessor.assess_personal_loan_risk(application_data)
                    else:  # business_credit
                        result = st.session_state.risk_assessor.assess_business_credit_risk(application_data)
                    
                    # Validate result
                    if result is None:
                        st.error("Assessment returned no result. Please check Ollama connection.")
                        return
                    
                    # Store result
                    st.session_state.assessment_results.append(result)
                    
                    # Display results
                    display_assessment_result(result)
                    
                except Exception as e:
                    st.error(f"Assessment failed: {str(e)}")
                    st.error(f"Error type: {type(e).__name__}")
                    
                    # Provide specific guidance based on error type
                    if "Connection" in str(e) or "timeout" in str(e).lower():
                        st.info("This appears to be a connection issue. Please check that Ollama is running.")
                    elif "Model" in str(e):
                        st.info("This appears to be a model issue. Please verify the model is properly loaded in Ollama.")
                    
                    logger.error(f"Assessment error: {e}")
                    
                    # Show debug info
                    with st.expander("Debug Information"):
                        st.write(f"Ollama client status: {st.session_state.ollama_client is not None}")
                        st.write(f"Risk assessor status: {st.session_state.risk_assessor is not None}")
                        st.write(f"Selected model: {st.session_state.current_model}")
                        st.write(f"Assessment type: {assessment_type}")
    
    # Assessment history
    if st.session_state.assessment_results:
        st.sidebar.markdown("---")
        st.sidebar.header("📚 Assessment History")
        
        history_df = pd.DataFrame([
            {
                "ID": r.application_id[-8:],  # Last 8 chars
                "Type": r.assessment_type,
                "Recommendation": r.recommendation,
                "Risk": r.risk_level,
                "Time": r.timestamp[:16]
            }
            for r in st.session_state.assessment_results[-10:]  # Last 10
        ])
        
        st.sidebar.dataframe(history_df, hide_index=True, use_container_width=True)
        
        if st.sidebar.button("🗑️ Clear History"):
            st.session_state.assessment_results = []
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">
        <p>{APP_SETTINGS["title"]} v{APP_SETTINGS["version"]}</p>
        <p>Powered by PRefLexOR Framework | Contact: {APP_SETTINGS["contact"]["email"]}</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()