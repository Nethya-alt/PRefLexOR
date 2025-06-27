"""
Medical Diagnosis Support Application with PRefLexOR
Streamlit-based interface for transparent clinical decision support
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
import logging
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
from models.ollama_client import OllamaClient
from config.settings import (
    APP_SETTINGS, OLLAMA_CONFIG, MEDICAL_CONDITIONS, 
    LAB_REFERENCES, SEVERITY_CRITERIA, SEVERITY_COLORS,
    SYSTEM_PROMPTS, THINKING_TOKENS, get_lab_reference, assess_lab_abnormality
)

# Page configuration
st.set_page_config(
    page_title=APP_SETTINGS["title"],
    page_icon="🩺",
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
    .severity-critical {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .severity-urgent {
        background-color: #fff3cd;
        border-left: 5px solid #fd7e14;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .severity-routine {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .thinking-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .disclaimer {
        background-color: #e9ecef;
        border: 2px solid #6c757d;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'ollama_client' not in st.session_state:
        st.session_state.ollama_client = None
    if 'diagnosis_results' not in st.session_state:
        st.session_state.diagnosis_results = []
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
        help="Choose the Ollama model for medical analysis"
    )
    
    # Update model if changed
    if selected_model != st.session_state.current_model:
        st.session_state.current_model = selected_model
        st.session_state.ollama_client = None
    
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
                
                if validation["model_functional"]:
                    st.sidebar.success("✅ Model ready")
                    return True
                else:
                    st.sidebar.error(f"❌ {validation['recommended_action']}")
                    if validation["available_models"]:
                        st.sidebar.info("Available models: " + ", ".join(validation["available_models"]))
                    return False
                    
            except Exception as e:
                st.sidebar.error(f"❌ Connection failed: {str(e)}")
                return False
    
    return True

def create_patient_form():
    """Create patient case input form"""
    
    st.subheader("👤 Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Demographics**")
        patient_id = st.text_input("Patient ID", value="PT_001")
        age = st.number_input("Age", min_value=0, max_value=120, value=45, step=1)
        gender = st.selectbox("Gender", ["male", "female", "other"])
        
        st.markdown("**Chief Complaint & History**")
        chief_complaint = st.text_area("Chief Complaint", value="Chest pain and shortness of breath")
        duration = st.text_input("Duration of Symptoms", value="3 days")
        medical_history = st.text_area("Past Medical History", value="Hypertension, diabetes")
        
    with col2:
        st.markdown("**Current Symptoms**")
        symptoms = st.multiselect(
            "Select Symptoms",
            ["chest_pain", "shortness_of_breath", "fatigue", "nausea", "dizziness", 
             "fever", "cough", "headache", "joint_pain", "abdominal_pain"],
            default=["chest_pain", "shortness_of_breath"]
        )
        
        st.markdown("**Medications**")
        medications = st.text_area("Current Medications", value="Lisinopril 10mg daily, Metformin 1000mg BID")
        
        st.markdown("**Family History**")
        family_history = st.text_area("Family History", value="Father: MI at age 60, Mother: diabetes")
    
    return {
        "patient_id": patient_id,
        "age": age,
        "gender": gender,
        "chief_complaint": chief_complaint,
        "duration": duration,
        "symptoms": symptoms,
        "medical_history": medical_history,
        "medications": medications,
        "family_history": family_history
    }

def create_lab_results_form():
    """Create lab results input form"""
    
    st.subheader("🔬 Laboratory Results")
    
    lab_data = {}
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Complete Blood Count**")
        lab_data["hemoglobin"] = st.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=25.0, value=12.5, step=0.1)
        lab_data["hematocrit"] = st.number_input("Hematocrit (%)", min_value=0.0, max_value=70.0, value=38.0, step=0.1)
        lab_data["white_blood_cells"] = st.number_input("WBC (K/uL)", min_value=0.0, max_value=50.0, value=8.5, step=0.1)
        lab_data["platelets"] = st.number_input("Platelets (K/uL)", min_value=0, max_value=1000, value=280, step=1)
    
    with col2:
        st.markdown("**Basic Metabolic Panel**")
        lab_data["glucose"] = st.number_input("Glucose (mg/dL)", min_value=0, max_value=500, value=145, step=1)
        lab_data["sodium"] = st.number_input("Sodium (mEq/L)", min_value=100, max_value=160, value=140, step=1)
        lab_data["potassium"] = st.number_input("Potassium (mEq/L)", min_value=2.0, max_value=7.0, value=4.2, step=0.1)
        lab_data["creatinine"] = st.number_input("Creatinine (mg/dL)", min_value=0.0, max_value=15.0, value=1.1, step=0.1)
    
    with col3:
        st.markdown("**Inflammatory Markers**")
        lab_data["esr"] = st.number_input("ESR (mm/hr)", min_value=0, max_value=150, value=48, step=1)
        lab_data["crp"] = st.number_input("CRP (mg/L)", min_value=0.0, max_value=50.0, value=12.0, step=0.1)
        lab_data["rheumatoid_factor"] = st.number_input("RF (IU/mL)", min_value=0, max_value=200, value=25, step=1)
        
        st.markdown("**Additional Tests**")
        lab_data["troponin"] = st.number_input("Troponin (ng/mL)", min_value=0.0, max_value=50.0, value=0.15, step=0.01)
    
    return lab_data

def analyze_lab_results(lab_data, gender):
    """Analyze lab results against reference ranges"""
    
    analysis = {}
    
    for test_name, value in lab_data.items():
        if test_name in ["hemoglobin", "hematocrit", "creatinine"]:
            ref_range = get_lab_reference("complete_blood_count" if test_name in ["hemoglobin", "hematocrit"] else "basic_metabolic_panel", 
                                        test_name, gender)
        elif test_name in ["white_blood_cells", "platelets"]:
            ref_range = get_lab_reference("complete_blood_count", test_name, "normal")
        elif test_name in ["glucose", "sodium", "potassium"]:
            ref_range = get_lab_reference("basic_metabolic_panel", test_name, "normal")
        elif test_name in ["esr", "crp", "rheumatoid_factor"]:
            ref_range = get_lab_reference("inflammatory_markers", test_name, "normal")
        else:
            ref_range = (0, 0)  # Unknown test
        
        status = assess_lab_abnormality(value, ref_range)
        
        analysis[test_name] = {
            "value": value,
            "reference_range": ref_range,
            "status": status,
            "abnormal": status != "normal"
        }
    
    return analysis

def perform_diagnosis_analysis(patient_data, lab_data, analysis_type="differential_diagnosis"):
    """Perform AI-powered diagnosis analysis"""
    
    # Prepare comprehensive prompt
    analysis_prompt = f"""
    Perform a comprehensive {analysis_type.replace('_', ' ')} for this patient case. Use {THINKING_TOKENS['start']} to show your detailed clinical reasoning.

    PATIENT INFORMATION:
    {json.dumps(patient_data, indent=2)}

    LABORATORY RESULTS:
    {json.dumps(lab_data, indent=2)}

    Please provide a thorough analysis including:

    1. **Clinical Assessment**: Evaluate presenting symptoms and patient history
    2. **Laboratory Interpretation**: Analyze lab results in clinical context
    3. **Differential Diagnosis**: List most likely diagnoses with reasoning
    4. **Risk Stratification**: Assess severity and urgency level
    5. **Recommended Actions**: Suggest next steps and monitoring
    6. **Prognosis**: Discuss expected course and outcomes

    Use detailed step-by-step reasoning in your {THINKING_TOKENS['start']} section, then provide a clear clinical summary.

    Focus on:
    - Evidence-based clinical reasoning
    - Risk assessment and urgency determination
    - Clear explanation of diagnostic rationale
    - Specific recommendations for further evaluation
    - Always include appropriate medical disclaimers
    """
    
    # Get system prompt for analysis type
    system_prompt = SYSTEM_PROMPTS.get(analysis_type, SYSTEM_PROMPTS["differential_diagnosis"])
    
    # Generate LLM response
    if st.session_state.ollama_client:
        llm_response = st.session_state.ollama_client.generate_response(
            prompt=analysis_prompt,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=3000
        )
        
        if llm_response["success"]:
            return st.session_state.ollama_client.extract_thinking_section(llm_response["content"])
        else:
            return {"thinking": "Analysis failed", "answer": f"Error: {llm_response.get('error', 'Unknown error')}"}
    
    return {"thinking": "No connection", "answer": "Ollama connection not available"}

def create_lab_visualization(lab_analysis, gender):
    """Create visualization of lab results"""
    
    # Prepare data for visualization
    test_names = []
    values = []
    statuses = []
    colors = []
    
    for test_name, data in lab_analysis.items():
        test_names.append(test_name.replace('_', ' ').title())
        values.append(data["value"])
        statuses.append(data["status"])
        
        # Color coding based on status
        if data["status"] == "normal":
            colors.append("#28a745")  # Green
        elif data["status"] in ["low", "high"]:
            colors.append("#fd7e14")  # Orange
        else:
            colors.append("#6c757d")  # Gray for unknown
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=test_names,
            y=values,
            marker_color=colors,
            text=[f"{v:.1f}" for v in values],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Laboratory Results Analysis",
        xaxis_title="Laboratory Tests",
        yaxis_title="Values",
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis_tickangle=-45
    )
    
    return fig

def display_diagnosis_results(patient_data, lab_analysis, diagnosis_result):
    """Display comprehensive diagnosis results"""
    
    st.markdown("## 🔍 Clinical Analysis Results")
    
    # Main summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Patient Age", f"{patient_data['age']} years")
    with col2:
        st.metric("Abnormal Labs", f"{sum(1 for data in lab_analysis.values() if data['abnormal'])}/{len(lab_analysis)}")
    with col3:
        st.metric("Analysis Model", st.session_state.current_model.split(':')[0])
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["🧠 Clinical Reasoning", "📊 Lab Analysis", "📋 Summary", "⚠️ Disclaimers"])
    
    with tab1:
        st.markdown("### AI Clinical Reasoning Process")
        
        if diagnosis_result.get("thinking"):
            st.markdown(f"""
            <div class="thinking-box">
                <h4>🧠 Step-by-Step Clinical Analysis</h4>
                {diagnosis_result["thinking"].replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### Clinical Summary")
        st.write(diagnosis_result.get("answer", "No analysis available"))
    
    with tab2:
        st.markdown("### Laboratory Results Visualization")
        
        # Lab results chart
        lab_fig = create_lab_visualization(lab_analysis, patient_data["gender"])
        st.plotly_chart(lab_fig, use_container_width=True)
        
        # Detailed lab table
        st.markdown("### Detailed Laboratory Analysis")
        
        lab_df_data = []
        for test_name, data in lab_analysis.items():
            ref_range_str = f"{data['reference_range'][0]:.1f} - {data['reference_range'][1]:.1f}" if data['reference_range'] != (0, 0) else "Unknown"
            lab_df_data.append({
                "Test": test_name.replace('_', ' ').title(),
                "Value": f"{data['value']:.1f}",
                "Reference Range": ref_range_str,
                "Status": data['status'].title(),
                "Abnormal": "Yes" if data['abnormal'] else "No"
            })
        
        lab_df = pd.DataFrame(lab_df_data)
        st.dataframe(lab_df, use_container_width=True)
    
    with tab3:
        st.markdown("### Case Summary")
        
        # Patient summary
        st.markdown("**Patient Information:**")
        st.write(f"- **Age/Gender:** {patient_data['age']} year old {patient_data['gender']}")
        st.write(f"- **Chief Complaint:** {patient_data['chief_complaint']}")
        st.write(f"- **Duration:** {patient_data['duration']}")
        st.write(f"- **Symptoms:** {', '.join(patient_data['symptoms'])}")
        
        # Key findings
        abnormal_labs = [test for test, data in lab_analysis.items() if data['abnormal']]
        if abnormal_labs:
            st.markdown("**Key Laboratory Abnormalities:**")
            for lab in abnormal_labs:
                data = lab_analysis[lab]
                st.write(f"- **{lab.replace('_', ' ').title()}:** {data['value']:.1f} ({data['status']})")
        
        # Export functionality
        st.markdown("### Export Options")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📄 Download Case Report"):
                case_report = {
                    "patient_data": patient_data,
                    "lab_analysis": lab_analysis,
                    "diagnosis_analysis": diagnosis_result,
                    "timestamp": datetime.now().isoformat()
                }
                
                report_json = json.dumps(case_report, indent=2)
                st.download_button(
                    label="Download JSON Report",
                    data=report_json,
                    file_name=f"case_report_{patient_data['patient_id']}.json",
                    mime="application/json"
                )
    
    with tab4:
        st.markdown(f"""
        <div class="disclaimer">
            <h3>⚠️ Important Medical Disclaimer</h3>
            <p><strong>{APP_SETTINGS['disclaimer']}</strong></p>
            
            <h4>Limitations of AI-Assisted Diagnosis:</h4>
            <ul>
                <li>This tool provides decision support only, not definitive diagnoses</li>
                <li>All recommendations must be validated by qualified healthcare professionals</li>
                <li>Clinical judgment and patient-specific factors must always be considered</li>
                <li>Emergency situations require immediate medical attention regardless of AI analysis</li>
                <li>This system is not a substitute for proper medical education and training</li>
            </ul>
            
            <h4>For Healthcare Professionals:</h4>
            <ul>
                <li>Use this tool as a supplement to, not replacement for, clinical expertise</li>
                <li>Always correlate AI suggestions with clinical presentation and your judgment</li>
                <li>Ensure proper documentation and follow institutional protocols</li>
                <li>Consider patient consent for AI-assisted decision support tools</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🩺 Medical Diagnosis Support</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #7f8c8d; font-size: 1.2rem;">AI-Powered Clinical Decision Support with Transparent Reasoning</p>', unsafe_allow_html=True)
    
    # Medical disclaimer at top
    st.markdown(f"""
    <div class="disclaimer">
        <strong>⚠️ MEDICAL DISCLAIMER:</strong> {APP_SETTINGS['disclaimer']}
    </div>
    """, unsafe_allow_html=True)
    
    # Setup Ollama connection
    if not setup_ollama_connection():
        st.error("Please configure Ollama connection in the sidebar before proceeding.")
        return
    
    # Main content
    st.sidebar.header("📋 Analysis Type")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["differential_diagnosis", "lab_interpretation", "treatment_planning"],
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    # Patient and lab forms
    patient_data = create_patient_form()
    lab_data = create_lab_results_form()
    
    # Analyze lab results
    lab_analysis = analyze_lab_results(lab_data, patient_data["gender"])
    
    # Analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🔍 Perform Clinical Analysis", type="primary", use_container_width=True):
            
            # Check if client is available
            if st.session_state.ollama_client is None:
                st.error("❌ **Analysis Not Available**")
                st.error("Please ensure Ollama is properly configured in the sidebar.")
                return
            
            # Perform analysis
            with st.spinner("🤖 Analyzing patient case with AI..."):
                try:
                    diagnosis_result = perform_diagnosis_analysis(patient_data, lab_data, analysis_type)
                    
                    # Store result
                    result_data = {
                        "patient_data": patient_data,
                        "lab_analysis": lab_analysis,
                        "diagnosis_result": diagnosis_result,
                        "analysis_type": analysis_type,
                        "timestamp": datetime.now().isoformat()
                    }
                    st.session_state.diagnosis_results.append(result_data)
                    
                    # Display results
                    display_diagnosis_results(patient_data, lab_analysis, diagnosis_result)
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    logger.error(f"Analysis error: {e}")
    
    # Analysis history
    if st.session_state.diagnosis_results:
        st.sidebar.markdown("---")
        st.sidebar.header("📚 Analysis History")
        
        history_df = pd.DataFrame([
            {
                "Patient ID": r["patient_data"]["patient_id"],
                "Age": r["patient_data"]["age"],
                "Analysis": r["analysis_type"].replace('_', ' ').title(),
                "Time": r["timestamp"][:16]
            }
            for r in st.session_state.diagnosis_results[-10:]
        ])
        
        st.sidebar.dataframe(history_df, hide_index=True, use_container_width=True)
        
        if st.sidebar.button("🗑️ Clear History"):
            st.session_state.diagnosis_results = []
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">
        <p>{APP_SETTINGS["title"]} v{APP_SETTINGS["version"]}</p>
        <p>Powered by PRefLexOR Framework | For Educational and Decision Support Purposes Only</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()