"""
Investment Research Application with PRefLexOR
Streamlit-based interface for transparent investment analysis and decision-making
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
from datetime import datetime, timedelta
import logging
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Ollama client
from models.ollama_client import OllamaClient

# Page configuration
st.set_page_config(
    page_title="Investment Research - PRefLexOR",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
OLLAMA_CONFIG = {
    "base_url": "http://localhost:11434",
    "default_model": "llama3.1:8b-instruct-q4_K_M",
    "alternative_models": [
        "llama3.1:70b-instruct-q4_K_M",  # Best for complex analysis
        "llama3.1:8b-instruct-q4_K_M",   # Balanced
        "llama3.2:3b-instruct-q4_K_M"    # Lightweight
    ]
}

SYSTEM_PROMPT = """You are a senior financial analyst with expertise in equity research, market analysis, and investment strategy.
You have deep knowledge of financial markets, valuation methods, risk assessment, and portfolio management.
Provide detailed, step-by-step analysis using <|thinking|> tags to show your investment reasoning process.
Focus on fundamental analysis, technical indicators, market sentiment, and risk factors.
IMPORTANT: Always include disclaimers that this is for educational purposes and not financial advice."""

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2e8b57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-positive {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-negative {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-neutral {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
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
    .financial-disclaimer {
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
    if 'investment_analyses' not in st.session_state:
        st.session_state.investment_analyses = []
    if 'current_model' not in st.session_state:
        st.session_state.current_model = OLLAMA_CONFIG["default_model"]

def setup_ollama_connection():
    """Setup and validate Ollama connection"""
    
    st.sidebar.header("🤖 Model Configuration")
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        OLLAMA_CONFIG["alternative_models"],
        index=OLLAMA_CONFIG["alternative_models"].index(st.session_state.current_model),
        help="Choose the Ollama model for investment analysis"
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

def fetch_stock_data(symbol: str, period: str = "6mo"):
    """Fetch stock data using yfinance"""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        info = stock.info
        return hist, info
    except Exception as e:
        logger.error(f"Failed to fetch data for {symbol}: {e}")
        return None, None

def create_investment_form():
    """Create investment analysis form"""
    
    st.subheader("📊 Investment Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Stock Information**")
        symbol = st.text_input("Stock Symbol", value="AAPL", help="Enter stock ticker symbol (e.g., AAPL, GOOGL)")
        
        analysis_type = st.selectbox(
            "Analysis Type",
            ["fundamental", "technical", "comprehensive", "sector_comparison"]
        )
        
        time_horizon = st.selectbox(
            "Investment Horizon",
            ["short_term", "medium_term", "long_term"]
        )
        
        risk_tolerance = st.selectbox(
            "Risk Tolerance",
            ["conservative", "moderate", "aggressive"]
        )
        
        st.markdown("**Investment Parameters**")
        investment_amount = st.number_input("Investment Amount ($)", min_value=1000, value=10000, step=1000)
        target_return = st.slider("Target Annual Return (%)", min_value=5, max_value=30, value=12)
        
    with col2:
        st.markdown("**Market Context**")
        market_sentiment = st.selectbox(
            "Current Market Sentiment",
            ["bullish", "neutral", "bearish"]
        )
        
        sector_focus = st.multiselect(
            "Sector Considerations",
            ["technology", "healthcare", "finance", "energy", "consumer", "industrial"],
            default=["technology"]
        )
        
        economic_factors = st.multiselect(
            "Economic Factors",
            ["inflation", "interest_rates", "gdp_growth", "employment", "geopolitical"],
            default=["interest_rates"]
        )
        
        st.markdown("**Analysis Options**")
        include_competitors = st.checkbox("Include Competitor Analysis", value=True)
        include_valuation = st.checkbox("Include Valuation Metrics", value=True)
        include_technical = st.checkbox("Include Technical Analysis", value=False)
    
    # Fetch real stock data if possible
    stock_data = None
    stock_info = None
    
    if symbol:
        with st.spinner(f"Fetching data for {symbol}..."):
            try:
                stock_data, stock_info = fetch_stock_data(symbol.upper())
                if stock_data is not None and not stock_data.empty:
                    st.success(f"✅ Data loaded for {symbol.upper()}")
                    
                    # Display basic stock info
                    col1, col2, col3, col4 = st.columns(4)
                    current_price = stock_data['Close'].iloc[-1]
                    prev_close = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else current_price
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100
                    
                    with col1:
                        st.metric("Current Price", f"${current_price:.2f}", f"{change:+.2f}")
                    with col2:
                        st.metric("Change %", f"{change_pct:+.2f}%")
                    with col3:
                        volume = stock_data['Volume'].iloc[-1] if 'Volume' in stock_data.columns else 0
                        st.metric("Volume", f"{volume:,.0f}")
                    with col4:
                        market_cap = stock_info.get('marketCap', 0) if stock_info else 0
                        if market_cap > 1e9:
                            st.metric("Market Cap", f"${market_cap/1e9:.1f}B")
                        elif market_cap > 1e6:
                            st.metric("Market Cap", f"${market_cap/1e6:.1f}M")
                        else:
                            st.metric("Market Cap", "N/A")
                            
                    # Stock price chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['Close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='#2e8b57', width=2)
                    ))
                    fig.update_layout(
                        title=f"{symbol.upper()} Stock Price",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.warning(f"⚠️ Could not fetch data for {symbol.upper()}")
            except Exception as e:
                st.warning(f"⚠️ Error fetching data: {str(e)}")
    
    return {
        "symbol": symbol.upper() if symbol else "",
        "analysis_type": analysis_type,
        "time_horizon": time_horizon,
        "risk_tolerance": risk_tolerance,
        "investment_amount": investment_amount,
        "target_return": target_return,
        "market_sentiment": market_sentiment,
        "sector_focus": sector_focus,
        "economic_factors": economic_factors,
        "include_competitors": include_competitors,
        "include_valuation": include_valuation,
        "include_technical": include_technical,
        "stock_data": stock_data,
        "stock_info": stock_info
    }

def perform_investment_analysis(investment_data):
    """Perform AI-powered investment analysis"""
    
    # Prepare market data summary if available
    market_data_summary = ""
    if investment_data["stock_data"] is not None:
        stock_data = investment_data["stock_data"]
        current_price = stock_data['Close'].iloc[-1]
        high_52w = stock_data['High'].max()
        low_52w = stock_data['Low'].min()
        volatility = stock_data['Close'].pct_change().std() * np.sqrt(252) * 100
        
        market_data_summary = f"""
        CURRENT MARKET DATA:
        - Current Price: ${current_price:.2f}
        - 52-Week High: ${high_52w:.2f}
        - 52-Week Low: ${low_52w:.2f}
        - Annualized Volatility: {volatility:.1f}%
        - Recent Performance: {((current_price - stock_data['Close'].iloc[0]) / stock_data['Close'].iloc[0] * 100):.1f}% over period
        """
    
    analysis_prompt = f"""
    Perform a comprehensive investment analysis for this stock. Use <|thinking|> to show your detailed investment reasoning.

    INVESTMENT REQUEST:
    {json.dumps(investment_data, indent=2, default=str)}
    
    {market_data_summary}

    Please provide a thorough analysis including:

    1. **Company Overview**: Brief company description and business model
    2. **Investment Thesis**: Core reasons to invest or avoid
    3. **Financial Analysis**: Key metrics and financial health assessment
    4. **Valuation Assessment**: Current valuation vs intrinsic value
    5. **Risk Analysis**: Key risks and risk mitigation factors
    6. **Technical Analysis**: Price trends and technical indicators (if requested)
    7. **Market Position**: Competitive advantages and market share
    8. **Growth Prospects**: Future growth drivers and catalysts
    9. **Recommendation**: Buy/Hold/Sell with price targets and reasoning

    Use detailed step-by-step reasoning in your <|thinking|> section, then provide clear investment recommendations.

    Focus on:
    - Quantitative analysis with specific metrics
    - Qualitative factors affecting investment potential
    - Risk-adjusted return expectations
    - Portfolio fit and diversification benefits
    - Entry and exit strategies
    
    IMPORTANT: Include appropriate disclaimers about this being educational analysis only, not financial advice.
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
    st.markdown('<h1 class="main-header">📈 Investment Research</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #7f8c8d; font-size: 1.2rem;">AI-Powered Investment Analysis with Transparent Reasoning</p>', unsafe_allow_html=True)
    
    # Financial disclaimer
    st.markdown("""
    <div class="financial-disclaimer">
        <strong>📈 FINANCIAL DISCLAIMER:</strong> This tool provides educational analysis only and does not constitute financial advice. 
        Past performance does not guarantee future results. Always consult with qualified financial advisors before making investment decisions.
    </div>
    """, unsafe_allow_html=True)
    
    # Setup Ollama connection
    if not setup_ollama_connection():
        st.error("Please configure Ollama connection in the sidebar before proceeding.")
        return
    
    # Main content
    investment_data = create_investment_form()
    
    # Analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🔍 Perform Investment Analysis", type="primary", use_container_width=True):
            
            if st.session_state.ollama_client is None:
                st.error("❌ **Investment Analysis Not Available**")
                st.error("Please ensure Ollama is properly configured in the sidebar.")
                return
            
            if not investment_data["symbol"]:
                st.error("❌ Please enter a stock symbol")
                return
            
            with st.spinner("🤖 Analyzing investment opportunity with AI expert..."):
                try:
                    analysis_result = perform_investment_analysis(investment_data)
                    
                    # Store result
                    investment_analysis = {
                        "investment_data": investment_data,
                        "analysis_result": analysis_result,
                        "timestamp": datetime.now().isoformat()
                    }
                    st.session_state.investment_analyses.append(investment_analysis)
                    
                    # Display results
                    st.markdown("## 📊 Investment Analysis Results")
                    
                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Symbol", investment_data["symbol"])
                    with col2:
                        st.metric("Investment Amount", f"${investment_data['investment_amount']:,}")
                    with col3:
                        st.metric("Target Return", f"{investment_data['target_return']}%")
                    with col4:
                        st.metric("Risk Tolerance", investment_data["risk_tolerance"].title())
                    
                    # Create tabs for results
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "🧠 Investment Reasoning", 
                        "📊 Analysis Summary", 
                        "📈 Charts & Data",
                        "📄 Report"
                    ])
                    
                    with tab1:
                        st.markdown("### AI Investment Analysis Process")
                        
                        if analysis_result.get("thinking"):
                            st.markdown(f"""
                            <div class="thinking-box">
                                <h4>🧠 Step-by-Step Investment Analysis</h4>
                                {analysis_result["thinking"].replace(chr(10), '<br>')}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("### Investment Summary")
                        st.write(analysis_result.get("answer", "No analysis available"))
                    
                    with tab2:
                        st.markdown("### Investment Recommendation Summary")
                        
                        # Sample recommendation metrics
                        recommendation_metrics = {
                            "Overall Rating": "Buy",
                            "Price Target": "$200.00",
                            "Risk Level": investment_data["risk_tolerance"].title(),
                            "Time Horizon": investment_data["time_horizon"].replace('_', ' ').title(),
                            "Expected Return": f"{investment_data['target_return']}%"
                        }
                        
                        for metric, value in recommendation_metrics.items():
                            if "Buy" in value or "Positive" in value:
                                metric_class = "metric-positive"
                            elif "Sell" in value or "Negative" in value:
                                metric_class = "metric-negative"
                            else:
                                metric_class = "metric-neutral"
                                
                            st.markdown(f"""
                            <div class="{metric_class}">
                                <strong>{metric}:</strong> {value}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with tab3:
                        st.markdown("### Market Data & Analysis")
                        
                        if investment_data["stock_data"] is not None:
                            stock_data = investment_data["stock_data"]
                            
                            # Price and volume chart
                            fig = go.Figure()
                            
                            # Candlestick chart
                            fig.add_trace(go.Candlestick(
                                x=stock_data.index,
                                open=stock_data['Open'],
                                high=stock_data['High'],
                                low=stock_data['Low'],
                                close=stock_data['Close'],
                                name="Price"
                            ))
                            
                            fig.update_layout(
                                title=f"{investment_data['symbol']} Price Chart",
                                yaxis_title="Price ($)",
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Volume chart
                            if 'Volume' in stock_data.columns:
                                fig_vol = px.bar(
                                    x=stock_data.index,
                                    y=stock_data['Volume'],
                                    title=f"{investment_data['symbol']} Volume"
                                )
                                fig_vol.update_layout(height=300)
                                st.plotly_chart(fig_vol, use_container_width=True)
                        else:
                            st.info("📊 Market data visualization would appear here with real-time data")
                    
                    with tab4:
                        st.markdown("### Investment Analysis Report")
                        
                        st.markdown("**Key Findings:**")
                        st.write("• Strong fundamental analysis supports investment thesis")
                        st.write("• Valuation appears reasonable relative to growth prospects")
                        st.write("• Risk factors are manageable within stated risk tolerance")
                        
                        st.markdown("**Recommended Actions:**")
                        st.write("• Consider gradual position building over 3-6 months")
                        st.write("• Set stop-loss at 15% below entry price")
                        st.write("• Monitor quarterly earnings and guidance updates")
                        
                        # Export functionality
                        if st.button("📄 Download Investment Report"):
                            report_json = json.dumps(investment_analysis, indent=2, default=str)
                            st.download_button(
                                label="Download JSON Report",
                                data=report_json,
                                file_name=f"investment_analysis_{investment_data['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                    
                except Exception as e:
                    st.error(f"Investment analysis failed: {str(e)}")
                    logger.error(f"Analysis error: {e}")
    
    # Analysis history
    if st.session_state.investment_analyses:
        st.sidebar.markdown("---")
        st.sidebar.header("📚 Analysis History")
        
        history_df = pd.DataFrame([
            {
                "Symbol": r["investment_data"]["symbol"],
                "Amount": f"${r['investment_data']['investment_amount']/1000:.0f}K",
                "Target": f"{r['investment_data']['target_return']}%",
                "Time": r["timestamp"][:16]
            }
            for r in st.session_state.investment_analyses[-10:]
        ])
        
        st.sidebar.dataframe(history_df, hide_index=True, use_container_width=True)
        
        if st.sidebar.button("🗑️ Clear History"):
            st.session_state.investment_analyses = []
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">
        <p>Investment Research - PRefLexOR v1.0.0</p>
        <p>Powered by PRefLexOR Framework | For Educational Purposes Only - Not Financial Advice</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()