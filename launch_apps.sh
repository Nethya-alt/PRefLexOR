#!/bin/bash

# PRefLexOR Business Applications Launcher
# This script helps launch multiple Streamlit applications

echo "🏦 PRefLexOR Business Applications Suite"
echo "========================================"

# Check if Ollama is running
echo "🤖 Checking Ollama connection..."
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✅ Ollama is running"
else
    echo "❌ Ollama is not running"
    echo ""
    echo "Please start Ollama:"
    echo "  ollama serve"
    echo ""
    read -p "Press Enter when Ollama is ready..."
fi

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -i :$port >/dev/null 2>&1; then
        return 1  # Port is in use
    else
        return 0  # Port is available
    fi
}

# Function to launch application
launch_app() {
    local app_name=$1
    local app_dir=$2
    local port=$3
    
    echo ""
    echo "🚀 Launching $app_name on port $port..."
    
    if [ ! -d "$app_dir" ]; then
        echo "❌ Directory $app_dir not found"
        return 1
    fi
    
    if ! check_port $port; then
        echo "⚠️  Port $port is already in use"
        return 1
    fi
    
    cd "$app_dir"
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo "📦 Creating virtual environment for $app_name..."
        python -m venv venv
    fi
    
    # Activate virtual environment
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    
    # Install requirements
    echo "📥 Installing requirements for $app_name..."
    pip install -r requirements.txt > /dev/null 2>&1
    
    # Launch Streamlit app in background
    echo "🌐 Starting $app_name at http://localhost:$port"
    nohup streamlit run app.py --server.port $port > ${app_name}_app.log 2>&1 &
    
    # Store PID for cleanup
    echo $! > ${app_name}_app.pid
    
    cd ..
    sleep 2
}

# Application configurations
declare -A APPS
APPS[financial_risk]="financialrisk:8501"
APPS[medical_diagnosis]="medical_diagnosis:8502"
APPS[supply_chain]="supply_chain:8503"
APPS[legal_analysis]="legal_analysis:8504"
APPS[investment_research]="investment_research:8505"
APPS[product_development]="product_development:8506"

echo ""
echo "Available Applications:"
echo "1. Financial Risk Assessment (Port 8501)"
echo "2. Medical Diagnosis Support (Port 8502)"
echo "3. Supply Chain Risk Management (Port 8503)"
echo "4. Legal Document Analysis (Port 8504)"
echo "5. Investment Research (Port 8505)"
echo "6. Product Development Strategy (Port 8506)"
echo "7. Launch All Applications"
echo "8. Check Application Status"
echo "9. Stop All Applications"
echo ""

read -p "Select option (1-9): " choice

case $choice in
    1)
        launch_app "financial_risk" "financialrisk" "8501"
        echo "✅ Financial Risk Assessment launched at http://localhost:8501"
        ;;
    2)
        launch_app "medical_diagnosis" "medical_diagnosis" "8502"
        echo "✅ Medical Diagnosis Support launched at http://localhost:8502"
        ;;
    3)
        launch_app "supply_chain" "supply_chain" "8503"
        echo "✅ Supply Chain Risk Management launched at http://localhost:8503"
        ;;
    4)
        launch_app "legal_analysis" "legal_analysis" "8504"
        echo "✅ Legal Document Analysis launched at http://localhost:8504"
        ;;
    5)
        launch_app "investment_research" "investment_research" "8505"
        echo "✅ Investment Research launched at http://localhost:8505"
        ;;
    6)
        launch_app "product_development" "product_development" "8506"
        echo "✅ Product Development Strategy launched at http://localhost:8506"
        ;;
    7)
        echo "🚀 Launching all applications..."
        
        launch_app "financial_risk" "financialrisk" "8501"
        launch_app "medical_diagnosis" "medical_diagnosis" "8502"
        launch_app "supply_chain" "supply_chain" "8503"
        launch_app "legal_analysis" "legal_analysis" "8504"
        launch_app "investment_research" "investment_research" "8505"
        launch_app "product_development" "product_development" "8506"
        
        echo ""
        echo "✅ All applications launched successfully!"
        echo ""
        echo "Access your applications:"
        echo "🏦 Financial Risk Assessment: http://localhost:8501"
        echo "🩺 Medical Diagnosis Support: http://localhost:8502"
        echo "🚚 Supply Chain Risk Management: http://localhost:8503"
        echo "⚖️  Legal Document Analysis: http://localhost:8504"
        echo "📈 Investment Research: http://localhost:8505"
        echo "🚀 Product Development Strategy: http://localhost:8506"
        ;;
    8)
        echo "📊 Application Status:"
        echo ""
        
        for app in "${!APPS[@]}"; do
            IFS=':' read -r app_dir port <<< "${APPS[$app]}"
            
            if check_port $port; then
                echo "❌ $app: Not running (port $port available)"
            else
                echo "✅ $app: Running on port $port"
                echo "   📊 Access at: http://localhost:$port"
            fi
        done
        ;;
    9)
        echo "🛑 Stopping all applications..."
        
        # Kill processes by PID files
        for app in "${!APPS[@]}"; do
            IFS=':' read -r app_dir port <<< "${APPS[$app]}"
            
            if [ -f "${app_dir}/${app}_app.pid" ]; then
                pid=$(cat "${app_dir}/${app}_app.pid")
                if ps -p $pid > /dev/null 2>&1; then
                    kill $pid
                    echo "✅ Stopped $app (PID: $pid)"
                fi
                rm -f "${app_dir}/${app}_app.pid"
            fi
        done
        
        # Also kill any remaining streamlit processes
        pkill -f "streamlit run app.py" 2>/dev/null
        
        echo "✅ All applications stopped"
        ;;
    *)
        echo "❌ Invalid option. Please select 1-9."
        ;;
esac

echo ""
echo "💡 Tips:"
echo "  - Use Ctrl+C to stop individual applications"
echo "  - Check logs in each app directory: <app_name>_app.log"
echo "  - Ensure Ollama is running: ollama serve"
echo "  - Pull models: ollama pull llama3.1:8b-instruct-q4_K_M"