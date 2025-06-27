# Demo Guide - Financial Risk Assessment Application

This guide provides sample data and scenarios to demonstrate the PRefLexOR-powered financial risk assessment application.

## Quick Start

1. **Setup Ollama**:
   ```bash
   # Install Ollama from https://ollama.ai
   ollama serve
   ollama pull llama3.1:8b-instruct-q4_K_M
   ```

2. **Run Application**:
   ```bash
   ./run_app.sh
   # OR manually:
   pip install -r requirements.txt
   streamlit run app.py
   ```

3. **Access Interface**: Open http://localhost:8501

## Demo Scenarios

### Scenario 1: Low-Risk Mortgage Application ✅

**Profile**: Established professional with strong financials

```
Applicant Information:
- Name: Sarah Johnson
- Email: sarah.johnson@techcorp.com
- Phone: 555-123-4567
- Employment: Full-time, 8 years
- Monthly Income: $12,000

Loan Details:
- Loan Amount: $350,000
- Property Value: $500,000
- Down Payment: $150,000
- Property Type: Single Family
- Occupancy: Primary Residence

Financial Info:
- Credit Score: 780
- Monthly Debt Payments: $600
```

**Expected Result**: APPROVE with LOW risk
**Key Strengths**: 
- Excellent credit score (780)
- Strong DTI ratio (~25%)
- Conservative LTV (70%)
- Substantial down payment (30%)

### Scenario 2: Moderate-Risk Personal Loan 🟡

**Profile**: Young professional with good income but limited credit history

```
Applicant Information:
- Name: Michael Chen
- Email: m.chen@startup.com
- Phone: 555-987-6543
- Employment: Full-time, 2 years
- Monthly Income: $5,500

Loan Request:
- Loan Amount: $25,000
- Purpose: Debt Consolidation

Financial Info:
- Credit Score: 695
- Monthly Debt Payments: $800
- Housing Payment: $1,400
- Savings: $8,000
```

**Expected Result**: CONDITIONAL_APPROVAL with MODERATE risk
**Key Factors**:
- Good credit score but high DTI
- Limited employment history
- Reasonable savings buffer

### Scenario 3: High-Risk Business Credit Application ⚠️

**Profile**: New restaurant seeking working capital

```
Business Information:
- Name: Coastal Bistro LLC
- Industry: Food Service
- Business Age: 1.5 years
- Annual Revenue: $480,000

Credit Request:
- Requested Amount: $75,000
- Purpose: Working Capital

Financial Information:
- Net Operating Income: $45,000
- Business Credit Score: 45
- Current Assets: $25,000
- Current Liabilities: $35,000
- Total Debt: $60,000
- Total Equity: $15,000
- Annual Debt Service: $48,000
```

**Expected Result**: DENY with HIGH risk
**Risk Factors**:
- Poor debt service coverage ratio (0.94)
- Negative current ratio (0.71)
- High-risk industry (food service)
- Young business age
- Low business credit score

### Scenario 4: Excellent Credit Personal Loan ✅

**Profile**: High-income professional with excellent credit

```
Applicant Information:
- Name: Dr. Lisa Park
- Email: l.park@hospital.org
- Phone: 555-456-7890
- Employment: Full-time, 12 years
- Monthly Income: $18,000

Loan Request:
- Loan Amount: $40,000
- Purpose: Home Improvement

Financial Info:
- Credit Score: 820
- Monthly Debt Payments: $2,200
- Housing Payment: $3,500
- Savings: $75,000
```

**Expected Result**: APPROVE with LOW risk

### Scenario 5: Edge Case - Self-Employed Mortgage 🔍

**Profile**: Self-employed consultant with variable income

```
Applicant Information:
- Name: Robert Martinez
- Email: rob@consultingfirm.com
- Phone: 555-321-0987
- Employment: Self-employed, 6 years
- Monthly Income: $9,500 (average)

Loan Details:
- Loan Amount: $425,000
- Property Value: $550,000
- Down Payment: $125,000
- Property Type: Condo
- Occupancy: Primary Residence

Financial Info:
- Credit Score: 720
- Monthly Debt Payments: $1,800
```

**Expected Result**: CONDITIONAL_APPROVAL with MODERATE risk
**Analysis Points**:
- Self-employment adds complexity
- Good credit and down payment
- DTI on the edge
- Condo adds property risk

## Understanding the AI Reasoning

### Thinking Tokens Example

When you run an assessment, look for the **AI Reasoning** tab to see detailed thinking process:

```
<|thinking|>
**Financial Metrics Analysis**:
- Debt-to-income ratio: 32% (acceptable, below 36% threshold)
- Loan-to-value ratio: 70% (excellent, well below 80% threshold)
- Credit score: 780 (excellent tier, well above 740)

**Risk Factors Assessment**:
- Employment stability: 8 years (strong positive indicator)
- Down payment: 30% of property value (excellent)
- Property type: Single family (lowest risk category)
- Monthly payment estimate: $2,100 (18% of gross income)

**Regulatory Compliance Check**:
- Meets QM DTI requirements (32% < 43%)
- Sufficient documentation capability
- Property appraisal within normal range

**Overall Risk Profile**:
- Primary risks: Interest rate changes, market volatility
- Mitigation factors: Strong employment, conservative LTV, excellent credit
- Recommendation: APPROVE with standard terms
<|/thinking|>
```

### Key Features to Test

1. **Interactive Charts**: 
   - Risk gauge showing visual risk level
   - Financial metrics comparison with thresholds
   - DTI breakdown pie chart

2. **Compliance Checking**:
   - QM rule validation for mortgages
   - Automated regulatory compliance scoring

3. **Export Functionality**:
   - JSON reports with complete analysis
   - CSV summaries for spreadsheet analysis

4. **Model Selection**:
   - Test different Ollama models in sidebar
   - Compare reasoning quality between models

## Troubleshooting Common Issues

### 1. Ollama Connection Failed
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama service
ollama serve

# Verify model is available
ollama list
```

### 2. Model Not Found
```bash
# Pull recommended model
ollama pull llama3.1:8b-instruct-q4_K_M

# Alternative lightweight model
ollama pull llama3.2:3b-instruct-q4_K_M
```

### 3. Memory Issues
- Use smaller model: `llama3.2:3b-instruct-q4_K_M`
- Reduce max_tokens in config/settings.py
- Close other applications to free memory

### 4. Slow Response Times
- Check system resources (CPU/Memory usage)
- Try smaller model for faster inference
- Reduce temperature for more deterministic responses

## Performance Benchmarks

| Model | Memory Usage | Response Time | Quality Score |
|-------|-------------|---------------|---------------|
| Llama 3.1 8B | ~6GB | 15-30 seconds | Excellent |
| Llama 3.2 3B | ~2.5GB | 8-15 seconds | Good |
| Llama 3.1 70B | ~45GB | 45-90 seconds | Outstanding |

## Advanced Testing

### Custom Risk Thresholds
Edit `config/settings.py` to modify risk assessment criteria:
```python
RISK_THRESHOLDS = {
    "mortgage": {
        "debt_to_income": {
            "excellent": 0.25,  # Stricter than default 0.28
            "good": 0.33,       # Stricter than default 0.36
            "acceptable": 0.40, # Stricter than default 0.43
        }
    }
}
```

### API Integration Testing
The application stores assessment results that can be exported:
```python
# Access via session state
results = st.session_state.assessment_results
json_export = st.session_state.risk_assessor.export_assessment(result)
```

This comprehensive demo setup allows you to test all aspects of the PRefLexOR financial risk assessment system with realistic scenarios and edge cases.