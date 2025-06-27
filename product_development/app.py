"""
Product Development Strategy Application with PRefLexOR
Streamlit-based interface for transparent product planning and strategic decision-making
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Ollama client
from models.ollama_client import OllamaClient

# Page configuration
st.set_page_config(
    page_title="Product Development Strategy - PRefLexOR",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
OLLAMA_CONFIG = {
    "base_url": "http://localhost:11434",
    "default_model": "llama3.1:8b-instruct-q4_K_M",
    "alternative_models": [
        "llama3.1:70b-instruct-q4_K_M",  # Best for strategic analysis
        "llama3.1:8b-instruct-q4_K_M",   # Balanced
        "llama3.2:3b-instruct-q4_K_M"    # Lightweight
    ]
}

SYSTEM_PROMPT = """You are a senior product manager and strategic consultant with expertise in product development, market analysis, and innovation strategy.
You have deep knowledge of product lifecycle management, competitive analysis, user research, and go-to-market strategies.
Provide detailed, step-by-step analysis using <|thinking|> tags to show your strategic reasoning process.
Focus on market opportunities, user needs, competitive positioning, and implementation roadmaps.
IMPORTANT: Always consider business viability, technical feasibility, and market desirability in your recommendations."""

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff6b35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .strategy-high {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .strategy-medium {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .strategy-low {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
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
    .product-info {
        background-color: #e7f3ff;
        border: 1px solid #b3d9ff;
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
    if 'product_strategies' not in st.session_state:
        st.session_state.product_strategies = []
    if 'current_model' not in st.session_state:
        st.session_state.current_model = OLLAMA_CONFIG["default_model"]

def setup_ollama_connection():
    """Setup and validate Ollama connection"""
    
    st.sidebar.header("🤖 Model Configuration")
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        OLLAMA_CONFIG["alternative_models"],
        index=OLLAMA_CONFIG["alternative_models"].index(st.session_state.current_model),
        help="Choose the Ollama model for product strategy analysis"
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

def create_product_form():
    """Create product development strategy form"""
    
    st.subheader("🚀 Product Development Strategy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Product Information**")
        product_name = st.text_input("Product Name", value="Smart Fitness Tracker Pro", help="Name of the product being developed")
        
        product_category = st.selectbox(
            "Product Category",
            ["consumer_electronics", "software_saas", "mobile_app", "healthcare", "fintech", "e_commerce", "gaming", "enterprise_software"]
        )
        
        development_stage = st.selectbox(
            "Development Stage",
            ["ideation", "concept", "mvp", "beta", "pre_launch", "launched"]
        )
        
        target_market = st.selectbox(
            "Target Market",
            ["b2c_mass_market", "b2c_premium", "b2b_small_business", "b2b_enterprise", "b2b2c", "niche_market"]
        )
        
        st.markdown("**Business Context**")
        budget_range = st.selectbox(
            "Development Budget",
            ["under_100k", "100k_500k", "500k_1m", "1m_5m", "5m_plus"]
        )
        
        timeline = st.selectbox(
            "Launch Timeline",
            ["3_months", "6_months", "12_months", "18_months", "24_months_plus"]
        )
        
    with col2:
        st.markdown("**Market Analysis**")
        market_size = st.selectbox(
            "Target Market Size",
            ["small_niche", "medium_specialized", "large_mainstream", "global_mass_market"]
        )
        
        competition_level = st.selectbox(
            "Competition Level",
            ["low", "moderate", "high", "saturated"]
        )
        
        competitive_advantage = st.multiselect(
            "Potential Competitive Advantages",
            ["technology_innovation", "cost_efficiency", "user_experience", "brand_strength", "distribution", "network_effects"],
            default=["technology_innovation", "user_experience"]
        )
        
        st.markdown("**Strategic Focus**")
        primary_goals = st.multiselect(
            "Primary Business Goals",
            ["revenue_growth", "market_share", "user_acquisition", "brand_building", "technology_platform", "data_collection"],
            default=["revenue_growth", "user_acquisition"]
        )
        
        risk_factors = st.multiselect(
            "Key Risk Factors",
            ["technology_risk", "market_risk", "competitive_risk", "regulatory_risk", "funding_risk", "execution_risk"],
            default=["technology_risk", "competitive_risk"]
        )
        
        success_metrics = st.multiselect(
            "Success Metrics",
            ["revenue", "user_growth", "market_share", "customer_satisfaction", "retention_rate", "time_to_market"],
            default=["revenue", "user_growth"]
        )
    
    # Additional strategic inputs
    st.markdown("**Strategic Details**")
    col3, col4 = st.columns(2)
    
    with col3:
        user_research_data = st.text_area(
            "User Research Insights",
            value="Target users are fitness enthusiasts aged 25-45 who want detailed health tracking with professional-grade accuracy",
            help="Key insights from user research and market analysis"
        )
        
        technical_requirements = st.text_area(
            "Technical Requirements",
            value="Advanced sensors, 7-day battery life, water resistance, mobile app integration, cloud analytics",
            help="Core technical features and capabilities needed"
        )
        
    with col4:
        market_positioning = st.text_area(
            "Market Positioning",
            value="Premium fitness tracker positioned between consumer and professional medical devices",
            help="How the product will be positioned in the market"
        )
        
        go_to_market_strategy = st.text_area(
            "Go-to-Market Approach",
            value="Direct-to-consumer online sales, fitness influencer partnerships, sports retail partnerships",
            help="Initial strategy for bringing product to market"
        )
    
    return {
        "product_name": product_name,
        "product_category": product_category,
        "development_stage": development_stage,
        "target_market": target_market,
        "budget_range": budget_range,
        "timeline": timeline,
        "market_size": market_size,
        "competition_level": competition_level,
        "competitive_advantage": competitive_advantage,
        "primary_goals": primary_goals,
        "risk_factors": risk_factors,
        "success_metrics": success_metrics,
        "user_research_data": user_research_data,
        "technical_requirements": technical_requirements,
        "market_positioning": market_positioning,
        "go_to_market_strategy": go_to_market_strategy
    }

def perform_strategy_analysis(product_data):
    """Perform AI-powered product strategy analysis"""
    
    analysis_prompt = f"""
    Develop a comprehensive product development strategy. Use <|thinking|> to show your detailed strategic reasoning.

    PRODUCT DEVELOPMENT REQUEST:
    {json.dumps(product_data, indent=2)}

    Please provide a thorough strategic analysis including:

    1. **Market Opportunity Assessment**: Market size, growth potential, and timing
    2. **Competitive Analysis**: Competitive landscape and differentiation strategy
    3. **Product Strategy**: Core value proposition and feature prioritization
    4. **User Experience Strategy**: User journey and experience design principles
    5. **Technical Strategy**: Technology stack, architecture, and development approach
    6. **Go-to-Market Strategy**: Launch approach, channels, and marketing strategy
    7. **Business Model**: Revenue streams, pricing strategy, and unit economics
    8. **Risk Assessment**: Key risks and mitigation strategies
    9. **Development Roadmap**: Phased development plan with milestones
    10. **Success Metrics**: KPIs and measurement framework

    Use detailed step-by-step reasoning in your <|thinking|> section, then provide clear strategic recommendations.

    Focus on:
    - Market viability and opportunity sizing
    - Product-market fit validation approach
    - Competitive positioning and differentiation
    - User-centered design principles
    - Technical feasibility and scalability
    - Business model sustainability
    - Implementation roadmap and timeline
    - Risk mitigation and contingency planning
    
    Ensure recommendations are actionable and consider resource constraints, timeline requirements, and market dynamics.
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

def create_strategy_visualizations(product_data):
    """Create strategy visualization charts"""
    
    # Market opportunity matrix
    fig_market = go.Figure()
    
    # Sample data for market positioning
    market_data = {
        "Product Categories": ["Consumer Basic", "Consumer Premium", "Semi-Professional", "Professional", "Our Product"],
        "Market Size": [100, 80, 40, 20, 60],
        "Competition": [90, 70, 40, 30, 50],
        "Opportunity": [20, 40, 60, 70, 80]
    }
    
    fig_market.add_trace(go.Scatter(
        x=market_data["Competition"],
        y=market_data["Opportunity"],
        mode='markers+text',
        marker=dict(
            size=[s/5 for s in market_data["Market Size"]],
            color=['red' if cat == 'Our Product' else 'blue' for cat in market_data["Product Categories"]],
            opacity=0.7
        ),
        text=market_data["Product Categories"],
        textposition="top center",
        name="Market Position"
    ))
    
    fig_market.update_layout(
        title="Market Opportunity Matrix",
        xaxis_title="Competition Level",
        yaxis_title="Market Opportunity",
        height=400
    )
    
    # Development timeline
    timeline_data = pd.DataFrame({
        "Phase": ["Research & Planning", "MVP Development", "Beta Testing", "Launch Preparation", "Market Launch"],
        "Start": [0, 2, 5, 7, 9],
        "Duration": [2, 3, 2, 2, 1],
        "Priority": ["High", "High", "Medium", "High", "High"]
    })
    
    fig_timeline = px.timeline(
        timeline_data,
        x_start="Start",
        x_end=[timeline_data["Start"][i] + timeline_data["Duration"][i] for i in range(len(timeline_data))],
        y="Phase",
        color="Priority",
        title="Product Development Timeline"
    )
    fig_timeline.update_layout(height=300)
    
    # Success metrics dashboard
    metrics_data = {
        "Metric": ["Revenue Target", "User Acquisition", "Market Share", "Customer Satisfaction", "Time to Market"],
        "Target": [100, 100, 100, 100, 100],
        "Current": [0, 0, 0, 0, 20]  # Starting values
    }
    
    fig_metrics = go.Figure()
    fig_metrics.add_trace(go.Bar(
        x=metrics_data["Metric"],
        y=metrics_data["Target"],
        name="Target",
        marker_color='lightblue',
        opacity=0.7
    ))
    fig_metrics.add_trace(go.Bar(
        x=metrics_data["Metric"],
        y=metrics_data["Current"],
        name="Current",
        marker_color='darkblue'
    ))
    fig_metrics.update_layout(
        title="Success Metrics Progress",
        yaxis_title="Progress (%)",
        height=400
    )
    
    return fig_market, fig_timeline, fig_metrics

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🚀 Product Development Strategy</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #7f8c8d; font-size: 1.2rem;">AI-Powered Product Strategy with Transparent Strategic Reasoning</p>', unsafe_allow_html=True)
    
    # Product info box
    st.markdown("""
    <div class="product-info">
        <strong>🚀 STRATEGIC DEVELOPMENT:</strong> This tool provides strategic analysis for product development decisions. 
        Use it to evaluate market opportunities, competitive positioning, and development roadmaps for informed product strategy.
    </div>
    """, unsafe_allow_html=True)
    
    # Setup Ollama connection
    if not setup_ollama_connection():
        st.error("Please configure Ollama connection in the sidebar before proceeding.")
        return
    
    # Main content
    product_data = create_product_form()
    
    # Analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🔍 Develop Product Strategy", type="primary", use_container_width=True):
            
            if st.session_state.ollama_client is None:
                st.error("❌ **Product Strategy Analysis Not Available**")
                st.error("Please ensure Ollama is properly configured in the sidebar.")
                return
            
            if not product_data["product_name"]:
                st.error("❌ Please enter a product name")
                return
            
            with st.spinner("🤖 Developing comprehensive product strategy with AI strategist..."):
                try:
                    strategy_result = perform_strategy_analysis(product_data)
                    
                    # Store result
                    product_strategy = {
                        "product_data": product_data,
                        "strategy_result": strategy_result,
                        "timestamp": datetime.now().isoformat()
                    }
                    st.session_state.product_strategies.append(product_strategy)
                    
                    # Display results
                    st.markdown("## 📊 Product Development Strategy")
                    
                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Product", product_data["product_name"])
                    with col2:
                        st.metric("Stage", product_data["development_stage"].replace('_', ' ').title())
                    with col3:
                        st.metric("Timeline", product_data["timeline"].replace('_', ' ').title())
                    with col4:
                        st.metric("Market", product_data["target_market"].replace('_', ' ').title())
                    
                    # Create tabs for results
                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "🧠 Strategic Reasoning", 
                        "📈 Market Analysis",
                        "🎯 Strategy Summary", 
                        "📊 Visualizations",
                        "📄 Strategic Plan"
                    ])
                    
                    with tab1:
                        st.markdown("### AI Strategic Analysis Process")
                        
                        if strategy_result.get("thinking"):
                            st.markdown(f"""
                            <div class="thinking-box">
                                <h4>🧠 Step-by-Step Strategic Analysis</h4>
                                {strategy_result["thinking"].replace(chr(10), '<br>')}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("### Strategy Development Summary")
                        st.write(strategy_result.get("answer", "No strategy analysis available"))
                    
                    with tab2:
                        st.markdown("### Market Opportunity Assessment")
                        
                        # Market analysis based on inputs
                        market_factors = {
                            "Market Size": product_data["market_size"].replace('_', ' ').title(),
                            "Competition Level": product_data["competition_level"].title(),
                            "Target Market": product_data["target_market"].replace('_', ' ').title(),
                            "Development Stage": product_data["development_stage"].replace('_', ' ').title()
                        }
                        
                        for factor, value in market_factors.items():
                            if "High" in value or "Large" in value or "Global" in value:
                                factor_class = "strategy-high"
                            elif "Low" in value or "Small" in value or "Niche" in value:
                                factor_class = "strategy-low"
                            else:
                                factor_class = "strategy-medium"
                                
                            st.markdown(f"""
                            <div class="{factor_class}">
                                <strong>{factor}:</strong> {value}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with tab3:
                        st.markdown("### Strategic Recommendations")
                        
                        st.markdown("**Competitive Advantages:**")
                        for advantage in product_data["competitive_advantage"]:
                            st.write(f"• {advantage.replace('_', ' ').title()}")
                        
                        st.markdown("**Primary Goals:**")
                        for goal in product_data["primary_goals"]:
                            st.write(f"• {goal.replace('_', ' ').title()}")
                        
                        st.markdown("**Key Risk Factors:**")
                        for risk in product_data["risk_factors"]:
                            st.write(f"• {risk.replace('_', ' ').title()}")
                        
                        st.markdown("**Success Metrics:**")
                        for metric in product_data["success_metrics"]:
                            st.write(f"• {metric.replace('_', ' ').title()}")
                    
                    with tab4:
                        st.markdown("### Strategy Visualizations")
                        
                        fig_market, fig_timeline, fig_metrics = create_strategy_visualizations(product_data)
                        
                        st.plotly_chart(fig_market, use_container_width=True)
                        st.plotly_chart(fig_timeline, use_container_width=True)
                        st.plotly_chart(fig_metrics, use_container_width=True)
                    
                    with tab5:
                        st.markdown("### Strategic Development Plan")
                        
                        st.markdown("**Phase 1: Market Validation**")
                        st.write("• Conduct user research and market validation")
                        st.write("• Validate core value proposition with target users")
                        st.write("• Analyze competitive landscape and positioning")
                        
                        st.markdown("**Phase 2: Product Development**")
                        st.write("• Develop MVP with core features")
                        st.write("• Implement user feedback loops")
                        st.write("• Build technical infrastructure and scalability")
                        
                        st.markdown("**Phase 3: Market Entry**")
                        st.write("• Execute go-to-market strategy")
                        st.write("• Launch marketing and user acquisition campaigns")
                        st.write("• Monitor success metrics and iterate based on data")
                        
                        # Export functionality
                        if st.button("📄 Download Strategy Report"):
                            report_json = json.dumps(product_strategy, indent=2, default=str)
                            st.download_button(
                                label="Download JSON Report",
                                data=report_json,
                                file_name=f"product_strategy_{product_data['product_name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                    
                except Exception as e:
                    st.error(f"Product strategy analysis failed: {str(e)}")
                    logger.error(f"Strategy analysis error: {e}")
    
    # Strategy history
    if st.session_state.product_strategies:
        st.sidebar.markdown("---")
        st.sidebar.header("📚 Strategy History")
        
        history_df = pd.DataFrame([
            {
                "Product": r["product_data"]["product_name"][:20] + "..." if len(r["product_data"]["product_name"]) > 20 else r["product_data"]["product_name"],
                "Stage": r["product_data"]["development_stage"].replace('_', ' ').title(),
                "Category": r["product_data"]["product_category"].replace('_', ' ').title(),
                "Time": r["timestamp"][:16]
            }
            for r in st.session_state.product_strategies[-10:]
        ])
        
        st.sidebar.dataframe(history_df, hide_index=True, use_container_width=True)
        
        if st.sidebar.button("🗑️ Clear History"):
            st.session_state.product_strategies = []
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">
        <p>Product Development Strategy - PRefLexOR v1.0.0</p>
        <p>Powered by PRefLexOR Framework | Strategic Analysis for Product Innovation</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()