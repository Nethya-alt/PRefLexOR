"""
Configuration settings for Financial Risk Assessment Application
"""

import os
from typing import Dict, Any

# Ollama Configuration
OLLAMA_CONFIG = {
    "base_url": "http://localhost:11434",
    "default_model": "llama3.1:8b-instruct-q4_K_M",
    "alternative_models": [
        "llama3.1:70b-instruct-q4_K_M",  # High performance
        "llama3.2:3b-instruct-q4_K_M",   # Lightweight
        "llama3.1:8b-instruct-q4_K_M"    # Balanced (default)
    ],
    "timeout": 120,
    "temperature": 0.1,
    "max_tokens": 2048
}

# Risk Assessment Thresholds
RISK_THRESHOLDS = {
    "mortgage": {
        "debt_to_income": {
            "excellent": 0.28,
            "good": 0.36,
            "acceptable": 0.43,
            "high_risk": 1.0
        },
        "loan_to_value": {
            "excellent": 0.80,
            "good": 0.90,
            "acceptable": 0.95,
            "high_risk": 1.0
        },
        "credit_score": {
            "excellent": 740,
            "good": 680,
            "acceptable": 620,
            "poor": 0
        }
    },
    "personal_loan": {
        "debt_to_income": {
            "excellent": 0.20,
            "good": 0.30,
            "acceptable": 0.40,
            "high_risk": 1.0
        },
        "credit_score": {
            "excellent": 720,
            "good": 660,
            "acceptable": 600,
            "poor": 0
        }
    },
    "business_credit": {
        "debt_service_coverage": {
            "excellent": 1.5,
            "good": 1.25,
            "acceptable": 1.1,
            "poor": 0
        },
        "current_ratio": {
            "excellent": 2.0,
            "good": 1.5,
            "acceptable": 1.2,
            "poor": 0
        }
    }
}

# Regulatory Compliance Rules
COMPLIANCE_RULES = {
    "qualified_mortgage": {
        "max_debt_to_income": 0.43,
        "points_and_fees_limit": 0.03,
        "required_documentation": [
            "income_verification",
            "employment_verification",
            "asset_verification",
            "credit_report"
        ]
    },
    "fair_lending": {
        "prohibited_factors": [
            "race", "color", "religion", "national_origin",
            "sex", "marital_status", "age", "disability"
        ],
        "required_disclosures": [
            "adverse_action_notice",
            "fair_credit_reporting_act",
            "equal_credit_opportunity_act"
        ]
    }
}

# Application Settings
APP_SETTINGS = {
    "title": "Financial Risk Assessment - PRefLexOR",
    "description": "Transparent Credit Analysis with Explainable AI",
    "version": "1.0.0",
    "contact": {
        "email": "support@financialrisk.ai",
        "phone": "+1-555-0123"
    }
}

# Thinking Token Configuration
THINKING_TOKENS = {
    "start": "<|thinking|>",
    "end": "<|/thinking|>",
    "reflection_start": "<|reflect|>",
    "reflection_end": "<|/reflect|>"
}

# System Prompts for Different Assessment Types
SYSTEM_PROMPTS = {
    "mortgage": """You are an expert mortgage underwriter with 15+ years of experience in residential lending.
You have deep knowledge of GSE guidelines, FHA/VA requirements, and regulatory compliance including QM rules.
Provide detailed, step-by-step analysis using <|thinking|> tags to show your reasoning process.
Focus on risk assessment, regulatory compliance, and clear recommendations.""",

    "personal_loan": """You are a senior credit analyst specializing in unsecured consumer lending.
You understand credit risk modeling, behavioral scoring, and regulatory requirements for personal loans.
Provide thorough analysis using <|thinking|> tags to demonstrate your evaluation process.
Consider creditworthiness, ability to pay, and risk mitigation strategies.""",

    "business_credit": """You are a commercial lending specialist with expertise in business credit analysis.
You understand cash flow analysis, industry risk factors, and commercial lending regulations.
Use <|thinking|> tags to show detailed financial analysis and risk assessment.
Focus on business viability, repayment capacity, and collateral evaluation."""
}

# Risk Categories and Weights
RISK_CATEGORIES = {
    "financial_metrics": 0.40,
    "credit_history": 0.25,
    "employment_stability": 0.15,
    "collateral_quality": 0.10,
    "external_factors": 0.10
}

# Color Schemes for Risk Visualization
RISK_COLORS = {
    "low": "#28a745",      # Green
    "moderate": "#ffc107",  # Yellow
    "high": "#fd7e14",     # Orange
    "critical": "#dc3545"  # Red
}

def get_model_config(model_name: str = None) -> Dict[str, Any]:
    """Get configuration for specified model or default"""
    if model_name is None:
        model_name = OLLAMA_CONFIG["default_model"]

    config = OLLAMA_CONFIG.copy()
    config["model"] = model_name
    return config

def get_risk_threshold(assessment_type: str, metric: str, level: str) -> float:
    """Get risk threshold for specific assessment type and metric"""
    return RISK_THRESHOLDS.get(assessment_type, {}).get(metric, {}).get(level, 0.0)

def validate_compliance(assessment_type: str, data: Dict[str, Any]) -> Dict[str, bool]:
    """Validate compliance requirements for assessment type"""
    compliance_results = {}

    if assessment_type == "mortgage":
        qm_rules = COMPLIANCE_RULES["qualified_mortgage"]
        compliance_results["qm_compliant"] = data.get("debt_to_income", 1.0) <= qm_rules["max_debt_to_income"]
        compliance_results["documentation_complete"] = all(
            data.get(doc, False) for doc in qm_rules["required_documentation"]
        )

    return compliance_results
