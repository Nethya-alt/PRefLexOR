"""
Legal Document Analysis Application with PRefLexOR
Streamlit-based interface for transparent contract and legal document review
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Ollama client
from models.ollama_client import OllamaClient

# Page configuration
st.set_page_config(
    page_title="Legal Document Analysis - PRefLexOR",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
OLLAMA_CONFIG = {
    "base_url": "http://localhost:11434",
    "default_model": "llama3.1:8b-instruct-q4_K_M",
    "alternative_models": [
        "llama3.1:70b-instruct-q4_K_M",  # Best for legal analysis
        "llama3.1:8b-instruct-q4_K_M",   # Balanced
        "llama3.2:3b-instruct-q4_K_M"    # Lightweight
    ]
}

SYSTEM_PROMPT = """You are an experienced corporate attorney with expertise in contract law, compliance, and risk assessment. 
You have deep knowledge of commercial agreements, regulatory requirements, and legal risk factors.
Provide detailed, step-by-step analysis using <|thinking|> tags to show your legal reasoning process.
Focus on identifying risks, compliance issues, and providing actionable legal recommendations.
IMPORTANT: Always include disclaimers that this is for informational purposes and not legal advice."""

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .risk-medium {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .risk-low {
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
    .legal-disclaimer {
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
    if 'legal_analyses' not in st.session_state:
        st.session_state.legal_analyses = []
    if 'current_model' not in st.session_state:
        st.session_state.current_model = OLLAMA_CONFIG["default_model"]

def setup_ollama_connection():
    """Setup and validate Ollama connection"""
    
    st.sidebar.header("🤖 Model Configuration")
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        OLLAMA_CONFIG["alternative_models"],
        index=OLLAMA_CONFIG["alternative_models"].index(st.session_state.current_model),
        help="Choose the Ollama model for legal analysis"
    )
    
    if selected_model != st.session_state.current_model:
        st.session_state.current_model = selected_model
        st.session_state.ollama_client = None
    
    if st.session_state.ollama_client is None:
        with st.spinner("Connecting to Ollama..."):
            try:
                st.session_state.ollama_client = OllamaClient(
                    base_url=OLLAMA_CONFIG["base_url"],
                    model=selected_model
                )
                
                validation = st.session_state.ollama_client.validate_model_setup()
                
                if validation["model_functional"]:
                    st.sidebar.success("✅ Model ready")
                    return True
                else:
                    st.sidebar.error(f"❌ {validation['recommended_action']}")
                    return False
                    
            except Exception as e:
                st.sidebar.error(f"❌ Connection failed: {str(e)}")
                return False
    
    return True

def create_contract_form():
    """Create contract analysis form"""
    
    st.subheader("📋 Contract Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Contract Information**")
        contract_type = st.selectbox(
            "Contract Type",
            ["software_licensing", "service_agreement", "employment", "nda", "partnership", "supply_agreement"]
        )
        
        parties = st.text_input("Contracting Parties", value="Company A and Company B")
        contract_value = st.number_input("Contract Value ($)", min_value=0, value=100000, step=1000)
        duration = st.text_input("Contract Duration", value="2 years")
        
        st.markdown("**Key Terms**")
        payment_terms = st.text_area("Payment Terms", value="Net 30 days payment terms")
        termination_clause = st.text_area("Termination Clause", value="Either party may terminate with 30 days written notice")
        
    with col2:
        st.markdown("**Risk Areas**")
        liability_terms = st.text_area("Liability Terms", value="Liability limited to direct damages only, excluding consequential damages")
        ip_ownership = st.text_area("IP Ownership", value="Each party retains ownership of pre-existing IP")
        confidentiality = st.text_area("Confidentiality", value="Standard mutual confidentiality provisions")
        
        st.markdown("**Compliance Considerations**")
        jurisdiction = st.selectbox("Governing Law", ["delaware", "new_york", "california", "texas", "international"])
        regulatory_requirements = st.multiselect(
            "Applicable Regulations",
            ["gdpr", "ccpa", "hipaa", "sox", "pci_dss", "none"],
            default=["none"]
        )
    
    return {
        "contract_type": contract_type,
        "parties": parties,
        "contract_value": contract_value,
        "duration": duration,
        "payment_terms": payment_terms,
        "termination_clause": termination_clause,
        "liability_terms": liability_terms,
        "ip_ownership": ip_ownership,
        "confidentiality": confidentiality,
        "jurisdiction": jurisdiction,
        "regulatory_requirements": regulatory_requirements
    }

def perform_legal_analysis(contract_data):
    """Perform AI-powered legal analysis"""
    
    analysis_prompt = f"""
    Perform a comprehensive legal risk assessment for this contract. Use <|thinking|> to show your detailed legal reasoning.

    CONTRACT DETAILS:
    {json.dumps(contract_data, indent=2)}

    Please provide a thorough analysis including:

    1. **Risk Assessment**: Identify and categorize legal risks (high, medium, low)
    2. **Contract Structure Analysis**: Evaluate overall contract framework
    3. **Key Terms Review**: Analyze critical provisions for fairness and enforceability
    4. **Liability Analysis**: Assess liability allocation and limitation clauses
    5. **Compliance Review**: Check regulatory compliance requirements
    6. **Negotiation Recommendations**: Suggest improvements and amendments
    7. **Enforcement Considerations**: Evaluate contract enforceability

    Use detailed step-by-step reasoning in your <|thinking|> section, then provide clear legal recommendations.

    Focus on:
    - Specific legal risks and their business impact
    - Compliance with applicable laws and regulations
    - Contract enforceability and dispute resolution
    - Recommended contract amendments
    - Risk mitigation strategies
    
    IMPORTANT: Include appropriate legal disclaimers about this being informational analysis only.
    """
    
    if st.session_state.ollama_client:
        llm_response = st.session_state.ollama_client.generate_response(
            prompt=analysis_prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=3000
        )
        
        if llm_response["success"]:
            return st.session_state.ollama_client.extract_thinking_section(llm_response["content"])
        else:
            return {"thinking": "Analysis failed", "answer": f"Error: {llm_response.get('error', 'Unknown error')}"}
    
    return {"thinking": "No connection", "answer": "Ollama connection not available"}

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">⚖️ Legal Document Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #7f8c8d; font-size: 1.2rem;">AI-Powered Contract Review with Transparent Legal Reasoning</p>', unsafe_allow_html=True)
    
    # Legal disclaimer
    st.markdown("""
    <div class="legal-disclaimer">
        <strong>⚖️ LEGAL DISCLAIMER:</strong> This tool provides informational analysis only and does not constitute legal advice. 
        Always consult with qualified legal professionals for actual legal matters and contract decisions.
    </div>
    """, unsafe_allow_html=True)
    
    # Setup Ollama connection
    if not setup_ollama_connection():
        st.error("Please configure Ollama connection in the sidebar before proceeding.")
        return
    
    # Main content
    contract_data = create_contract_form()
    
    # Analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🔍 Perform Legal Analysis", type="primary", use_container_width=True):
            
            if st.session_state.ollama_client is None:
                st.error("❌ **Legal Analysis Not Available**")
                st.error("Please ensure Ollama is properly configured in the sidebar.")
                return
            
            with st.spinner("🤖 Analyzing contract with AI legal expert..."):
                try:
                    analysis_result = perform_legal_analysis(contract_data)
                    
                    # Store result
                    legal_analysis = {
                        "contract_data": contract_data,
                        "analysis_result": analysis_result,
                        "timestamp": datetime.now().isoformat()
                    }
                    st.session_state.legal_analyses.append(legal_analysis)
                    
                    # Display results
                    st.markdown("## 📊 Legal Analysis Results")
                    
                    # Key metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Contract Type", contract_data["contract_type"].replace('_', ' ').title())
                    with col2:
                        st.metric("Contract Value", f"${contract_data['contract_value']:,}")
                    with col3:
                        st.metric("Duration", contract_data["duration"])
                    
                    # Create tabs for results
                    tab1, tab2, tab3 = st.tabs(["🧠 Legal Reasoning", "📋 Risk Summary", "📄 Recommendations"])
                    
                    with tab1:
                        st.markdown("### AI Legal Analysis Process")
                        
                        if analysis_result.get("thinking"):
                            st.markdown(f"""
                            <div class="thinking-box">
                                <h4>🧠 Step-by-Step Legal Analysis</h4>
                                {analysis_result["thinking"].replace(chr(10), '<br>')}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("### Legal Summary")
                        st.write(analysis_result.get("answer", "No analysis available"))
                    
                    with tab2:
                        st.markdown("### Contract Risk Assessment")
                        
                        # Sample risk categorization
                        risk_areas = {
                            "Liability Limitations": "Medium",
                            "IP Ownership Clarity": "Low", 
                            "Termination Rights": "Low",
                            "Payment Terms": "Medium",
                            "Regulatory Compliance": "High" if "gdpr" in contract_data["regulatory_requirements"] else "Low"
                        }
                        
                        for risk_area, risk_level in risk_areas.items():
                            risk_class = f"risk-{risk_level.lower()}"
                            st.markdown(f"""
                            <div class="{risk_class}">
                                <strong>{risk_area}:</strong> {risk_level} Risk
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with tab3:
                        st.markdown("### Recommended Actions")
                        
                        st.markdown("**High Priority:**")
                        st.write("• Review liability limitation clauses for adequacy")
                        st.write("• Ensure compliance with applicable data protection regulations")
                        st.write("• Clarify intellectual property ownership and licensing terms")
                        
                        st.markdown("**Medium Priority:**")
                        st.write("• Negotiate more balanced termination provisions")
                        st.write("• Consider adding dispute resolution mechanisms")
                        st.write("• Review payment terms for cash flow optimization")
                        
                        # Export functionality
                        if st.button("📄 Download Legal Analysis Report"):
                            report_json = json.dumps(legal_analysis, indent=2)
                            st.download_button(
                                label="Download JSON Report",
                                data=report_json,
                                file_name=f"legal_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                    
                except Exception as e:
                    st.error(f"Legal analysis failed: {str(e)}")
                    logger.error(f"Analysis error: {e}")
    
    # Analysis history
    if st.session_state.legal_analyses:
        st.sidebar.markdown("---")
        st.sidebar.header("📚 Analysis History")
        
        history_df = pd.DataFrame([
            {
                "Contract": r["contract_data"]["contract_type"].replace('_', ' ').title(),
                "Value": f"${r['contract_data']['contract_value']/1000:.0f}K",
                "Parties": r["contract_data"]["parties"][:20] + "..." if len(r["contract_data"]["parties"]) > 20 else r["contract_data"]["parties"],
                "Time": r["timestamp"][:16]
            }
            for r in st.session_state.legal_analyses[-10:]
        ])
        
        st.sidebar.dataframe(history_df, hide_index=True, use_container_width=True)
        
        if st.sidebar.button("🗑️ Clear History"):
            st.session_state.legal_analyses = []
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">
        <p>Legal Document Analysis - PRefLexOR v1.0.0</p>
        <p>Powered by PRefLexOR Framework | For Informational Purposes Only - Not Legal Advice</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()