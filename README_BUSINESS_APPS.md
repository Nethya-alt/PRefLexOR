# PRefLexOR Business Applications Suite

This directory contains multiple Streamlit applications demonstrating PRefLexOR's transparent reasoning capabilities across different business domains.

## Available Applications

### 1. 🏦 Financial Risk Assessment (`/financialrisk/`)
**Purpose**: Transparent credit analysis for mortgages, personal loans, and business credit
**Features**:
- CFPB-compliant reasoning trails
- Multiple risk assessment types
- Interactive financial visualizations
- Audit-ready decision documentation

**Run Command**:
```bash
cd financialrisk && streamlit run app.py
```

### 2. 🩺 Medical Diagnosis Support (`/medical_diagnosis/`)
**Purpose**: AI-powered clinical decision support with transparent reasoning
**Features**:
- Differential diagnosis analysis
- Laboratory result interpretation
- Treatment planning recommendations
- Clinical reasoning transparency

**Run Command**:
```bash
cd medical_diagnosis && streamlit run app.py
```

### 3. 🚚 Supply Chain Risk Management (`/supply_chain/`)
**Purpose**: Comprehensive supply chain risk assessment and mitigation planning
**Features**:
- Multi-dimensional risk analysis
- Scenario planning and impact assessment
- Interactive risk dashboards
- Actionable mitigation strategies

**Run Command**:
```bash
cd supply_chain && streamlit run app.py
```

### 4. ⚖️ Legal Document Analysis (`/legal_analysis/`)
**Purpose**: Contract risk assessment and legal document review
**Features**:
- Contract risk identification
- Regulatory compliance checking
- Clause analysis with recommendations
- Legal reasoning documentation

**Run Command**:
```bash
cd legal_analysis && streamlit run app.py --server.port 8504
```

### 5. 📈 Investment Research (`/investment_research/`)
**Purpose**: Investment analysis with transparent reasoning for regulatory compliance
**Features**:
- Real-time stock data integration  
- Company valuation analysis
- Risk factor assessment
- Portfolio fit analysis
- Technical and fundamental analysis
- Regulatory-compliant documentation

**Run Command**:
```bash
cd investment_research && streamlit run app.py --server.port 8505
```

### 6. 🚀 Product Development Strategy (`/product_development/`)
**Purpose**: Product opportunity analysis and go/no-go decision support
**Features**:
- Market opportunity assessment
- Competitive landscape analysis
- User research integration
- Resource requirement planning
- Development roadmap creation
- Strategic recommendation engine

**Run Command**:
```bash
cd product_development && streamlit run app.py --server.port 8506
```

## Prerequisites

### 1. Ollama Setup
All applications require Ollama with a compatible model:

```bash
# Install Ollama from https://ollama.ai
# Start Ollama service
ollama serve

# Pull recommended model (choose one)
ollama pull llama3.1:8b-instruct-q4_K_M    # Balanced (6GB RAM)
ollama pull llama3.2:3b-instruct-q4_K_M    # Lightweight (2.5GB RAM)
ollama pull llama3.1:70b-instruct-q4_K_M   # High-performance (45GB RAM)
```

### 2. Python Environment
Each application has its own requirements.txt:

```bash
# For each application directory:
cd <application_name>
pip install -r requirements.txt
```

## Quick Start Guide

### Option 1: Individual Applications
Run each application separately:
```bash
# Financial Risk Assessment (Port 8501)
cd financialrisk && streamlit run app.py --server.port 8501

# Medical Diagnosis (Port 8502 - in new terminal)
cd medical_diagnosis && streamlit run app.py --server.port 8502

# Supply Chain (Port 8503 - in new terminal)  
cd supply_chain && streamlit run app.py --server.port 8503

# Legal Analysis (Port 8504 - in new terminal)
cd legal_analysis && streamlit run app.py --server.port 8504

# Investment Research (Port 8505 - in new terminal)
cd investment_research && streamlit run app.py --server.port 8505

# Product Development (Port 8506 - in new terminal)
cd product_development && streamlit run app.py --server.port 8506
```

### Option 2: Application Launcher
Use the provided launcher script:
```bash
# Make executable
chmod +x launch_apps.sh

# Run launcher
./launch_apps.sh
```

The launcher provides options to:
- Launch individual applications on specific ports
- Launch all applications simultaneously  
- Check application status
- Stop all running applications

## Application Architecture

Each application follows the same PRefLexOR pattern:

```
<application_name>/
├── app.py                 # Main Streamlit application
├── models/
│   └── ollama_client.py   # Ollama integration
├── config/
│   └── settings.py        # Domain-specific configuration
├── utils/                 # Utility functions (when needed)
├── requirements.txt       # Python dependencies
└── README.md             # Application-specific documentation
```

## Common Features

### 🧠 Transparent Reasoning
All applications use thinking tokens (`<|thinking|>...<|/thinking|>`) to show:
- Step-by-step analysis process
- Decision rationale and logic
- Risk factor evaluation
- Regulatory compliance checking

### 📊 Interactive Visualizations
- Domain-specific dashboards
- Risk assessment charts
- Trend analysis graphs
- Decision support visualizations

### 📄 Export Capabilities
- JSON reports with complete reasoning
- CSV summaries for analysis
- Audit trails for compliance
- Integration-ready data formats

### 🔧 Configuration Options
- Multiple Ollama model support
- Adjustable risk thresholds
- Customizable analysis parameters
- Domain-specific compliance rules

## Model Recommendations by Use Case

| Application | Recommended Model | Memory Required | Use Case |
|-------------|------------------|-----------------|----------|
| Financial Risk | llama3.1:8b-instruct | 6GB | Regulatory compliance |
| Medical Diagnosis | llama3.1:70b-instruct | 45GB | Clinical accuracy critical |
| Supply Chain | llama3.1:8b-instruct | 6GB | Strategic planning |
| Legal Analysis | llama3.1:70b-instruct | 45GB | Legal precision required |
| Investment Research | llama3.1:8b-instruct | 6GB | Financial analysis |
| Product Development | llama3.2:3b-instruct | 2.5GB | Innovation ideation |

## Business Value Propositions

### Financial Services
- **Regulatory Compliance**: Auditable decision trails
- **Risk Management**: Transparent risk assessment
- **Customer Trust**: Explainable AI decisions

### Healthcare
- **Clinical Support**: Evidence-based reasoning
- **Documentation**: Complete analysis trails
- **Education**: Training and learning tool

### Manufacturing/Logistics
- **Risk Mitigation**: Proactive risk identification
- **Cost Optimization**: Data-driven decisions
- **Resilience Planning**: Scenario analysis

### Legal/Compliance
- **Due Diligence**: Systematic analysis
- **Risk Assessment**: Contract evaluation
- **Audit Support**: Complete documentation

### Investment Management
- **Fiduciary Responsibility**: Documented decisions
- **Research Quality**: Comprehensive analysis
- **Regulatory Reporting**: Compliant documentation

### Product Management
- **Strategic Planning**: Market analysis
- **Resource Allocation**: Evidence-based decisions
- **Innovation Pipeline**: Opportunity assessment

## Troubleshooting

### Common Issues
1. **Ollama Connection Failed**: Ensure `ollama serve` is running
2. **Model Not Found**: Pull required model with `ollama pull <model_name>`
3. **Out of Memory**: Use smaller model (llama3.2:3b-instruct)
4. **Slow Response**: Check system resources, consider model size

### Getting Help
- Check individual application README files
- Review troubleshooting guides in each directory
- Ensure Ollama service is properly configured
- Verify model compatibility with your hardware

## Development and Customization

### Adding New Applications
1. Create new directory following the standard structure
2. Copy base configuration from existing applications
3. Customize domain-specific logic and prompts
4. Add application to this README

### Customizing Existing Applications
- Modify `config/settings.py` for domain parameters
- Update system prompts for different reasoning styles
- Adjust risk thresholds and categories
- Add new visualization components

## Integration and Deployment

### Production Deployment
- Use Docker containers for consistent environments
- Configure load balancing for multiple model instances
- Implement proper authentication and authorization
- Set up monitoring and logging

### API Integration
- Each application can be modified to expose REST APIs
- Decision results are JSON-serializable
- Reasoning trails can be extracted programmatically
- Integration with existing business systems

## Compliance and Auditing

All applications generate complete audit trails including:
- Input data validation and sanitization
- Step-by-step reasoning processes
- Decision rationale and supporting evidence
- Model version and configuration details
- Timestamp and version tracking

This ensures compliance with:
- Financial services regulations (SOX, CFPB)
- Healthcare standards (HIPAA, FDA)
- Legal requirements (discovery, due diligence)
- Investment regulations (SEC, FINRA)
- Corporate governance standards

## License and Disclaimers

These applications are for demonstration and educational purposes. Always:
- Consult qualified professionals for actual business decisions
- Validate AI recommendations with domain experts
- Follow applicable regulations and compliance requirements
- Test thoroughly before production deployment
- Maintain appropriate human oversight and control