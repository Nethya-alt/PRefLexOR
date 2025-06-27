# Diagram Fallbacks

If Mermaid diagrams are not rendering, use these text-based alternatives:

## Standard LLM vs PRefLexOR Flow

```
Standard LLM:
User Input → [BLACK BOX] → Output
     ↓           ???          ↓
"Approve loan"            "Approved"

PRefLexOR:
User Input → [THINKING PROCESS] → [REASONING] → Output
     ↓              ↓                   ↓          ↓
"Approve loan"  "Check credit"    "Score: 750"  "Approved"
                "Check DTI"       "DTI: 35%"    
                "Check history"   "Clean: ✓"
                "Apply rules"     "Qualified"
```

## Six Business Applications

```
                        PRefLexOR
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
   Financial        Healthcare        Supply Chain
    Services                            
        │                 │                 │
   - Loan Approval   - Diagnosis       - Risk Assessment
   - Risk Analysis   - Treatment       - Mitigation
   - Compliance      - Documentation   - Monitoring
        │                 │                 │
      Legal          Investment       Product Dev
        │                 │                 │
   - Contract        - Analysis        - Strategy
   - Risk Review     - Compliance      - Planning
   - Negotiation     - Documentation   - Decisions
```

## Training Process

```
Phase I: ORPO Training
Standard Model → Preference Learning → Learns to Think First

Phase II: DPO Refinement  
Think First → Quality Enhancement → Production Model

Enterprise Deployment
Production Model → Domain Config → Transparent Decisions → Audit Trails
```

## Risk Assessment Matrix

```
Risk Factors:           Assessment:
Geopolitical: 8/10  →    │████████░░│ HIGH
Financial: 4/10     →    │████░░░░░░│ LOW  
Operational: 6/10   →    │██████░░░░│ MED
Dependency: 8/10    →    │████████░░│ HIGH

Overall Risk: MEDIUM (6.2/10)
```

## Investment Analysis Framework

```
Market Data → PRefLexOR Analysis → Investment Decision
     ↓              ↓                       ↓
- Stock Price  - Fundamental          - Buy/Hold/Sell
- Financials   - Technical            - Price Target
- News         - Risk Assessment      - Risk Level
```

## Clinical Decision Tree

```
Patient Symptoms → Clinical Analysis → Diagnosis & Action
      ↓                    ↓                  ↓
"Chest pain"        Risk Assessment:    "High Risk"
"45yo male"         - Cardiac: 30%          ↓
                    - Pulmonary: 15%   Emergency
                    - GI: 20%          Evaluation
                    - Other: 35%       Required
```

Use these text-based diagrams if Mermaid rendering fails.