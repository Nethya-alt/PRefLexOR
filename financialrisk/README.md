# Financial Risk Assessment with PRefLexOR

A Streamlit application implementing PRefLexOR-style reasoning for transparent financial risk assessment and credit analysis.

## Features

- **Transparent Credit Analysis**: Step-by-step reasoning with thinking tokens
- **Multiple Risk Models**: Mortgage, personal loan, and business credit assessment
- **Regulatory Compliance**: Auditable decision trails for CFPB requirements
- **Interactive Dashboard**: Real-time analysis with visual risk indicators
- **Explanation Generation**: Human-readable reasoning for each decision

## Requirements

### Ollama Setup

1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)

2. **Recommended Model**: Install Llama 3.1 8B Instruct
   ```bash
   ollama pull llama3.1:8b-instruct-q4_K_M
   ```

3. **Alternative Models** (if you have sufficient hardware):
   ```bash
   # For better reasoning (requires 16GB+ RAM)
   ollama pull llama3.1:70b-instruct-q4_K_M
   
   # For faster responses (requires 4GB+ RAM)
   ollama pull llama3.2:3b-instruct-q4_K_M
   ```

4. **Start Ollama Service**:
   ```bash
   ollama serve
   ```

### Python Environment

1. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start the Application**:
   ```bash
   streamlit run app.py
   ```

2. **Access the Interface**:
   - Open browser to `http://localhost:8501`
   - Select risk assessment type
   - Input financial data
   - Review detailed analysis with reasoning

## Model Recommendations

### Primary Recommendation: Llama 3.1 8B Instruct
- **Model**: `llama3.1:8b-instruct-q4_K_M`
- **Memory**: ~6GB RAM required
- **Performance**: Excellent balance of reasoning quality and speed
- **Use Case**: Production deployment for most scenarios

### High-Performance Option: Llama 3.1 70B Instruct  
- **Model**: `llama3.1:70b-instruct-q4_K_M`
- **Memory**: ~45GB RAM required
- **Performance**: Superior reasoning capabilities
- **Use Case**: High-stakes decisions requiring maximum accuracy

### Lightweight Option: Llama 3.2 3B Instruct
- **Model**: `llama3.2:3b-instruct-q4_K_M`
- **Memory**: ~2.5GB RAM required  
- **Performance**: Good for basic analysis
- **Use Case**: Development/testing or resource-constrained environments

## Application Structure

```
financialrisk/
├── app.py                 # Main Streamlit application
├── models/
│   ├── __init__.py
│   ├── risk_assessor.py   # Core risk assessment logic
│   └── ollama_client.py   # Ollama integration
├── utils/
│   ├── __init__.py
│   ├── data_validation.py # Input validation utilities
│   └── visualization.py   # Chart and graph utilities
├── config/
│   ├── __init__.py
│   └── settings.py        # Configuration settings
├── requirements.txt
└── README.md
```

## Key Features

### 1. Risk Assessment Types
- **Mortgage Applications**: Full property and income analysis
- **Personal Loans**: Unsecured credit assessment
- **Business Credit**: Commercial lending evaluation

### 2. Analysis Components
- **Financial Metrics**: DTI, LTV, credit scores, cash flow
- **Risk Factors**: Employment stability, collateral, market conditions
- **Regulatory Compliance**: Automated compliance checking
- **Decision Reasoning**: Step-by-step explanation of recommendation

### 3. Output Formats
- **Executive Summary**: High-level recommendation
- **Detailed Analysis**: Complete reasoning process
- **Risk Visualization**: Interactive charts and risk indicators
- **Audit Trail**: Complete decision documentation

## Configuration

Edit `config/settings.py` to customize:
- Ollama model selection
- Risk thresholds and criteria
- Regulatory compliance rules
- Output formatting preferences

## Compliance & Auditing

The application generates complete audit trails including:
- Input data validation
- Step-by-step reasoning process
- Risk factor analysis
- Final recommendation with confidence scores
- Timestamp and version tracking

All outputs are designed to meet regulatory requirements for explainable AI in financial services.