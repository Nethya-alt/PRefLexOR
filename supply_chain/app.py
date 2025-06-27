"""
Supply Chain Risk Management Application with PRefLexOR
Streamlit-based interface for transparent supply chain analysis
"""

import streamlit as st
import pandas as pd
import json
import numpy as np
from datetime import datetime, timedelta
import logging
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
from models.ollama_client import OllamaClient

# Page configuration
st.set_page_config(
    page_title="Supply Chain Risk Management - PRefLexOR",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
OLLAMA_CONFIG = {
    "base_url": "http://localhost:11434",
    "default_model": "llama3.1:8b-instruct-q4_K_M",
    "alternative_models": [
        "llama3.1:70b-instruct-q4_K_M",
        "llama3.2:3b-instruct-q4_K_M",
        "llama3.1:8b-instruct-q4_K_M"
    ],
    "timeout": 120,
    "temperature": 0.1,
    "max_tokens": 3000
}

# Risk categories and thresholds
RISK_CATEGORIES = {
    "supplier_concentration": {
        "low": 0.3,      # <30% from single supplier
        "moderate": 0.5, # 30-50%
        "high": 0.7,     # 50-70%
        "critical": 1.0  # >70%
    },
    "geographic_risk": {
        "low": 0.2,      # <20% from single region
        "moderate": 0.4, # 20-40%
        "high": 0.6,     # 40-60%
        "critical": 1.0  # >60%
    },
    "inventory_days": {
        "critical": 15,  # <15 days
        "high": 30,      # 15-30 days
        "moderate": 60,  # 30-60 days
        "low": 999       # >60 days
    },
    "financial_impact": {
        "low": 1000000,     # <$1M
        "moderate": 5000000, # $1-5M
        "high": 20000000,   # $5-20M
        "critical": 999999999 # >$20M
    }
}

RISK_COLORS = {
    "low": "#28a745",
    "moderate": "#ffc107", 
    "high": "#fd7e14",
    "critical": "#dc3545"
}

SYSTEM_PROMPT = """You are a senior supply chain risk management expert with 15+ years of experience in global logistics, procurement, and risk assessment. 
You have deep knowledge of geopolitical risks, supplier management, inventory optimization, and business continuity planning.
Provide detailed, step-by-step analysis using <|thinking|> tags to show your reasoning process.
Focus on risk assessment, mitigation strategies, and actionable recommendations."""

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-critical {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .risk-high {
        background-color: #fff3cd;
        border-left: 5px solid #fd7e14;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .risk-moderate {
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
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'ollama_client' not in st.session_state:
        st.session_state.ollama_client = None
    if 'risk_assessments' not in st.session_state:
        st.session_state.risk_assessments = []
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
        help="Choose the Ollama model for supply chain analysis"
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

def create_supply_chain_form():
    """Create supply chain risk assessment form"""
    
    st.subheader("🌐 Supply Chain Risk Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Disruption Event**")
        event_type = st.selectbox(
            "Event Type",
            ["geopolitical_tension", "natural_disaster", "pandemic", "cyberattack", "trade_dispute", "supplier_bankruptcy"]
        )
        event_description = st.text_area("Event Description", value="Geopolitical tensions affecting Asia-Pacific region")
        affected_regions = st.multiselect(
            "Affected Regions",
            ["north_america", "europe", "asia_pacific", "latin_america", "middle_east", "africa"],
            default=["asia_pacific"]
        )
        
        st.markdown("**Current Inventory Status**")
        inventory_data = {}
        components = ["semiconductors", "rare_earth_elements", "raw_materials", "finished_goods"]
        
        for component in components:
            inventory_data[component] = st.number_input(
                f"{component.replace('_', ' ').title()} (days of inventory)",
                min_value=0, max_value=365, value=45 if component == "semiconductors" else 30, step=1
            )
    
    with col2:
        st.markdown("**Supplier Information**")
        supplier_concentration = st.slider("Supplier Concentration (%)", 0, 100, 70, step=5,
                                         help="Percentage of critical suppliers in affected region")
        
        alternative_sources = st.selectbox("Alternative Source Availability", 
                                         ["none", "limited", "moderate", "abundant"])
        
        lead_time_increase = st.slider("Expected Lead Time Increase (%)", 0, 300, 150, step=25)
        
        st.markdown("**Financial Impact**")
        potential_revenue_loss = st.number_input("Potential Revenue Loss ($)", 
                                               min_value=0, max_value=1000000000, value=30000000, step=1000000)
        
        mitigation_cost = st.number_input("Estimated Mitigation Cost ($)", 
                                        min_value=0, max_value=100000000, value=5000000, step=100000)
        
        st.markdown("**Business Context**")
        demand_forecast = st.selectbox("Demand Forecast", ["decreasing", "stable", "increasing", "volatile"])
        seasonal_factor = st.selectbox("Seasonal Impact", ["none", "low", "moderate", "high"])
        
    return {
        "event_type": event_type,
        "event_description": event_description,
        "affected_regions": affected_regions,
        "inventory_data": inventory_data,
        "supplier_concentration": supplier_concentration / 100,
        "alternative_sources": alternative_sources,
        "lead_time_increase": lead_time_increase / 100,
        "potential_revenue_loss": potential_revenue_loss,
        "mitigation_cost": mitigation_cost,
        "demand_forecast": demand_forecast,
        "seasonal_factor": seasonal_factor
    }

def calculate_risk_metrics(supply_chain_data):
    """Calculate key risk metrics"""
    
    metrics = {}
    
    # Supplier concentration risk
    concentration = supply_chain_data["supplier_concentration"]
    if concentration < RISK_CATEGORIES["supplier_concentration"]["low"]:
        metrics["supplier_risk"] = "low"
    elif concentration < RISK_CATEGORIES["supplier_concentration"]["moderate"]:
        metrics["supplier_risk"] = "moderate"
    elif concentration < RISK_CATEGORIES["supplier_concentration"]["high"]:
        metrics["supplier_risk"] = "high"
    else:
        metrics["supplier_risk"] = "critical"
    
    # Inventory risk (based on minimum inventory days)
    min_inventory = min(supply_chain_data["inventory_data"].values())
    if min_inventory > RISK_CATEGORIES["inventory_days"]["low"]:
        metrics["inventory_risk"] = "low"
    elif min_inventory > RISK_CATEGORIES["inventory_days"]["moderate"]:
        metrics["inventory_risk"] = "moderate"
    elif min_inventory > RISK_CATEGORIES["inventory_days"]["high"]:
        metrics["inventory_risk"] = "high"
    else:
        metrics["inventory_risk"] = "critical"
    
    # Financial impact risk
    financial_impact = supply_chain_data["potential_revenue_loss"]
    if financial_impact < RISK_CATEGORIES["financial_impact"]["low"]:
        metrics["financial_risk"] = "low"
    elif financial_impact < RISK_CATEGORIES["financial_impact"]["moderate"]:
        metrics["financial_risk"] = "moderate"
    elif financial_impact < RISK_CATEGORIES["financial_impact"]["high"]:
        metrics["financial_risk"] = "high"
    else:
        metrics["financial_risk"] = "critical"
    
    # Overall risk score (weighted average)
    risk_weights = {"supplier_risk": 0.4, "inventory_risk": 0.35, "financial_risk": 0.25}
    risk_scores = {"low": 1, "moderate": 2, "high": 3, "critical": 4}
    
    weighted_score = sum(risk_scores[metrics[risk]] * weight for risk, weight in risk_weights.items())
    
    if weighted_score <= 1.5:
        metrics["overall_risk"] = "low"
    elif weighted_score <= 2.5:
        metrics["overall_risk"] = "moderate"
    elif weighted_score <= 3.5:
        metrics["overall_risk"] = "high"
    else:
        metrics["overall_risk"] = "critical"
    
    return metrics

def perform_supply_chain_analysis(supply_chain_data, risk_metrics):
    """Perform AI-powered supply chain risk analysis"""
    
    analysis_prompt = f"""
    Perform a comprehensive supply chain risk assessment for this disruption scenario. Use <|thinking|> to show your detailed analysis.

    DISRUPTION EVENT:
    {json.dumps({
        "event_type": supply_chain_data["event_type"],
        "description": supply_chain_data["event_description"],
        "affected_regions": supply_chain_data["affected_regions"]
    }, indent=2)}

    SUPPLY CHAIN STATUS:
    {json.dumps({
        "inventory_status": supply_chain_data["inventory_data"],
        "supplier_concentration": f"{supply_chain_data['supplier_concentration']:.1%}",
        "alternative_sources": supply_chain_data["alternative_sources"],
        "lead_time_increase": f"{supply_chain_data['lead_time_increase']:.0%}"
    }, indent=2)}

    BUSINESS IMPACT:
    {json.dumps({
        "potential_revenue_loss": f"${supply_chain_data['potential_revenue_loss']:,}",
        "mitigation_cost": f"${supply_chain_data['mitigation_cost']:,}",
        "demand_forecast": supply_chain_data["demand_forecast"],
        "seasonal_factor": supply_chain_data["seasonal_factor"]
    }, indent=2)}

    CALCULATED RISK METRICS:
    {json.dumps(risk_metrics, indent=2)}

    Please provide a thorough analysis including:

    1. **Risk Assessment Matrix**: Evaluate each risk dimension and overall exposure
    2. **Impact Analysis**: Assess short-term and long-term business impacts
    3. **Vulnerability Assessment**: Identify key weaknesses in the supply chain
    4. **Scenario Planning**: Consider best, worst, and most likely scenarios
    5. **Mitigation Strategies**: Recommend immediate and strategic actions
    6. **Contingency Planning**: Suggest backup plans and alternative approaches
    7. **Monitoring Recommendations**: Propose early warning indicators

    Use detailed step-by-step reasoning in your <|thinking|> section, then provide clear strategic recommendations.

    Focus on:
    - Quantitative risk assessment with time horizons
    - Specific actionable mitigation strategies
    - Cost-benefit analysis of interventions
    - Supply chain resilience improvements
    - Stakeholder communication strategies
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

def create_risk_dashboard(supply_chain_data, risk_metrics):
    """Create comprehensive risk visualization dashboard"""
    
    # Risk metrics gauge chart
    fig_gauge = go.Figure()
    
    risk_scores = {"low": 25, "moderate": 50, "high": 75, "critical": 100}
    overall_score = risk_scores[risk_metrics["overall_risk"]]
    
    fig_gauge.add_trace(go.Indicator(
        mode = "gauge+number+delta",
        value = overall_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Overall Risk Level: {risk_metrics['overall_risk'].title()}"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': RISK_COLORS[risk_metrics["overall_risk"]]},
            'steps': [
                {'range': [0, 25], 'color': RISK_COLORS["low"]},
                {'range': [25, 50], 'color': RISK_COLORS["moderate"]},
                {'range': [50, 75], 'color': RISK_COLORS["high"]},
                {'range': [75, 100], 'color': RISK_COLORS["critical"]}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    
    # Risk breakdown chart
    risk_categories = ["Supplier Risk", "Inventory Risk", "Financial Risk"]
    risk_levels = [risk_metrics["supplier_risk"], risk_metrics["inventory_risk"], risk_metrics["financial_risk"]]
    colors = [RISK_COLORS[level] for level in risk_levels]
    
    fig_breakdown = go.Figure(data=[
        go.Bar(
            x=risk_categories,
            y=[risk_scores[level] for level in risk_levels],
            marker_color=colors,
            text=[level.title() for level in risk_levels],
            textposition='auto'
        )
    ])
    
    fig_breakdown.update_layout(
        title="Risk Category Breakdown",
        yaxis_title="Risk Score",
        height=400
    )
    
    # Inventory timeline
    inventory_items = list(supply_chain_data["inventory_data"].keys())
    inventory_days = list(supply_chain_data["inventory_data"].values())
    
    fig_inventory = go.Figure()
    
    for i, (item, days) in enumerate(zip(inventory_items, inventory_days)):
        color = RISK_COLORS["critical"] if days < 30 else RISK_COLORS["high"] if days < 45 else RISK_COLORS["moderate"] if days < 60 else RISK_COLORS["low"]
        
        fig_inventory.add_trace(go.Bar(
            name=item.replace('_', ' ').title(),
            x=[item.replace('_', ' ').title()],
            y=[days],
            marker_color=color,
            text=f"{days} days",
            textposition='auto'
        ))
    
    fig_inventory.update_layout(
        title="Current Inventory Levels",
        yaxis_title="Days of Inventory",
        height=400,
        showlegend=False
    )
    
    # Add risk threshold lines
    fig_inventory.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
    fig_inventory.add_hline(y=60, line_dash="dash", line_color="orange", annotation_text="Safe Threshold")
    
    return fig_gauge, fig_breakdown, fig_inventory

def display_risk_assessment_results(supply_chain_data, risk_metrics, analysis_result):
    """Display comprehensive risk assessment results"""
    
    st.markdown("## 📊 Supply Chain Risk Assessment Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Overall Risk", risk_metrics["overall_risk"].title())
    with col2:
        st.metric("Supplier Risk", risk_metrics["supplier_risk"].title())
    with col3:
        st.metric("Inventory Risk", risk_metrics["inventory_risk"].title())
    with col4:
        st.metric("Financial Risk", risk_metrics["financial_risk"].title())
    
    # Risk level styling
    risk_class = f"risk-{risk_metrics['overall_risk']}"
    st.markdown(f"""
    <div class="{risk_class}">
        <strong>Risk Level: {risk_metrics['overall_risk'].title()}</strong> - Immediate action required: {risk_metrics['overall_risk'] in ['high', 'critical']}
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Risk Dashboard", "🧠 AI Analysis", "📋 Action Plan", "📊 Data Export"])
    
    with tab1:
        st.markdown("### Risk Visualization Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gauge_fig, breakdown_fig, inventory_fig = create_risk_dashboard(supply_chain_data, risk_metrics)
            st.plotly_chart(gauge_fig, use_container_width=True)
            st.plotly_chart(breakdown_fig, use_container_width=True)
        
        with col2:
            st.plotly_chart(inventory_fig, use_container_width=True)
            
            # Key metrics table
            st.markdown("### Key Risk Indicators")
            
            kri_data = {
                "Metric": ["Supplier Concentration", "Min Inventory Days", "Revenue at Risk", "Lead Time Increase"],
                "Value": [f"{supply_chain_data['supplier_concentration']:.1%}", 
                         f"{min(supply_chain_data['inventory_data'].values())} days",
                         f"${supply_chain_data['potential_revenue_loss']:,}",
                         f"{supply_chain_data['lead_time_increase']:.0%}"],
                "Risk Level": [risk_metrics["supplier_risk"].title(), 
                              risk_metrics["inventory_risk"].title(),
                              risk_metrics["financial_risk"].title(),
                              "High" if supply_chain_data['lead_time_increase'] > 1.0 else "Moderate"]
            }
            
            kri_df = pd.DataFrame(kri_data)
            st.dataframe(kri_df, use_container_width=True)
    
    with tab2:
        st.markdown("### AI Supply Chain Analysis")
        
        if analysis_result.get("thinking"):
            st.markdown(f"""
            <div class="thinking-box">
                <h4>🧠 Step-by-Step Risk Analysis</h4>
                {analysis_result["thinking"].replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### Strategic Recommendations")
        st.write(analysis_result.get("answer", "No analysis available"))
    
    with tab3:
        st.markdown("### Recommended Action Plan")
        
        # Generate action items based on risk levels
        if risk_metrics["overall_risk"] in ["critical", "high"]:
            st.error("🚨 **IMMEDIATE ACTION REQUIRED**")
            
            st.markdown("**Immediate Actions (0-24 hours):**")
            st.write("• Activate crisis management team")
            st.write("• Assess current inventory and prioritize critical items")
            st.write("• Contact alternative suppliers for emergency procurement")
            st.write("• Communicate with key customers about potential impacts")
            
            st.markdown("**Short-term Actions (1-7 days):**")
            st.write("• Implement alternative sourcing strategies")
            st.write("• Adjust production schedules based on available inventory")
            st.write("• Negotiate expedited shipping for critical components")
            st.write("• Update demand forecasting models")
            
            st.markdown("**Medium-term Actions (1-4 weeks):**")
            st.write("• Develop supplier diversification plan")
            st.write("• Increase safety stock for critical components")
            st.write("• Establish backup logistics routes")
            st.write("• Review and update supplier contracts")
        
        else:
            st.success("✅ **MONITORING MODE** - Situation manageable with standard procedures")
            
            st.markdown("**Recommended Monitoring Actions:**")
            st.write("• Increase supplier communication frequency")
            st.write("• Monitor geopolitical developments")
            st.write("• Review inventory levels weekly")
            st.write("• Prepare contingency plans for potential escalation")
    
    with tab4:
        st.markdown("### Data Export and Documentation")
        
        # Export functionality
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Download Risk Assessment"):
                assessment_data = {
                    "supply_chain_data": supply_chain_data,
                    "risk_metrics": risk_metrics,
                    "analysis_result": analysis_result,
                    "timestamp": datetime.now().isoformat(),
                    "model_used": st.session_state.current_model
                }
                
                assessment_json = json.dumps(assessment_data, indent=2)
                st.download_button(
                    label="Download JSON Report",
                    data=assessment_json,
                    file_name=f"supply_chain_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📊 Download CSV Summary"):
                summary_data = {
                    "Event Type": [supply_chain_data["event_type"]],
                    "Overall Risk": [risk_metrics["overall_risk"]],
                    "Supplier Risk": [risk_metrics["supplier_risk"]],
                    "Inventory Risk": [risk_metrics["inventory_risk"]],
                    "Financial Risk": [risk_metrics["financial_risk"]],
                    "Revenue at Risk": [supply_chain_data["potential_revenue_loss"]],
                    "Supplier Concentration": [f"{supply_chain_data['supplier_concentration']:.1%}"],
                    "Min Inventory Days": [min(supply_chain_data["inventory_data"].values())],
                    "Assessment Date": [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
                }
                
                summary_df = pd.DataFrame(summary_data)
                csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV Summary",
                    data=csv,
                    file_name=f"risk_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🚚 Supply Chain Risk Management</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #7f8c8d; font-size: 1.2rem;">Transparent Risk Assessment with AI-Powered Analysis</p>', unsafe_allow_html=True)
    
    # Setup Ollama connection
    if not setup_ollama_connection():
        st.error("Please configure Ollama connection in the sidebar before proceeding.")
        return
    
    # Main content
    supply_chain_data = create_supply_chain_form()
    
    # Calculate risk metrics
    risk_metrics = calculate_risk_metrics(supply_chain_data)
    
    # Analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🔍 Perform Risk Assessment", type="primary", use_container_width=True):
            
            if st.session_state.ollama_client is None:
                st.error("❌ **Risk Assessment Not Available**")
                st.error("Please ensure Ollama is properly configured in the sidebar.")
                return
            
            with st.spinner("🤖 Analyzing supply chain risks with AI..."):
                try:
                    analysis_result = perform_supply_chain_analysis(supply_chain_data, risk_metrics)
                    
                    # Store result
                    assessment_data = {
                        "supply_chain_data": supply_chain_data,
                        "risk_metrics": risk_metrics,
                        "analysis_result": analysis_result,
                        "timestamp": datetime.now().isoformat()
                    }
                    st.session_state.risk_assessments.append(assessment_data)
                    
                    # Display results
                    display_risk_assessment_results(supply_chain_data, risk_metrics, analysis_result)
                    
                except Exception as e:
                    st.error(f"Risk assessment failed: {str(e)}")
                    logger.error(f"Assessment error: {e}")
    
    # Assessment history
    if st.session_state.risk_assessments:
        st.sidebar.markdown("---")
        st.sidebar.header("📚 Assessment History")
        
        history_df = pd.DataFrame([
            {
                "Event": r["supply_chain_data"]["event_type"].replace('_', ' ').title(),
                "Risk Level": r["risk_metrics"]["overall_risk"].title(),
                "Revenue Risk": f"${r['supply_chain_data']['potential_revenue_loss']/1000000:.1f}M",
                "Time": r["timestamp"][:16]
            }
            for r in st.session_state.risk_assessments[-10:]
        ])
        
        st.sidebar.dataframe(history_df, hide_index=True, use_container_width=True)
        
        if st.sidebar.button("🗑️ Clear History"):
            st.session_state.risk_assessments = []
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">
        <p>Supply Chain Risk Management - PRefLexOR v1.0.0</p>
        <p>Powered by PRefLexOR Framework | For Strategic Planning and Risk Management</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()