"""
Configuration settings for Medical Diagnosis Support Application
"""

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
    "max_tokens": 3000
}

# Medical Condition Categories and Risk Levels
MEDICAL_CONDITIONS = {
    "cardiovascular": {
        "conditions": ["hypertension", "coronary_artery_disease", "heart_failure", "arrhythmia"],
        "urgency_levels": ["routine", "urgent", "emergent", "critical"]
    },
    "respiratory": {
        "conditions": ["asthma", "copd", "pneumonia", "pulmonary_embolism"],
        "urgency_levels": ["routine", "urgent", "emergent", "critical"]
    },
    "neurological": {
        "conditions": ["stroke", "seizure", "migraine", "dementia"],
        "urgency_levels": ["routine", "urgent", "emergent", "critical"]
    },
    "endocrine": {
        "conditions": ["diabetes", "thyroid_disorder", "adrenal_insufficiency"],
        "urgency_levels": ["routine", "urgent", "emergent", "critical"]
    },
    "rheumatologic": {
        "conditions": ["rheumatoid_arthritis", "lupus", "vasculitis"],
        "urgency_levels": ["routine", "urgent", "emergent", "critical"]
    }
}

# Diagnostic Confidence Thresholds
CONFIDENCE_THRESHOLDS = {
    "high_confidence": 0.85,
    "moderate_confidence": 0.65,
    "low_confidence": 0.40,
    "insufficient_data": 0.40
}

# Lab Value Reference Ranges
LAB_REFERENCES = {
    "complete_blood_count": {
        "hemoglobin": {"male": (13.8, 17.2), "female": (12.1, 15.1), "unit": "g/dL"},
        "hematocrit": {"male": (40.7, 50.3), "female": (36.1, 44.3), "unit": "%"},
        "white_blood_cells": {"normal": (4.5, 11.0), "unit": "K/uL"},
        "platelets": {"normal": (150, 450), "unit": "K/uL"}
    },
    "basic_metabolic_panel": {
        "glucose": {"normal": (70, 100), "unit": "mg/dL"},
        "sodium": {"normal": (136, 145), "unit": "mEq/L"},
        "potassium": {"normal": (3.5, 5.0), "unit": "mEq/L"},
        "creatinine": {"male": (0.7, 1.3), "female": (0.6, 1.1), "unit": "mg/dL"}
    },
    "inflammatory_markers": {
        "esr": {"normal": (0, 30), "unit": "mm/hr"},
        "crp": {"normal": (0, 3.0), "unit": "mg/L"},
        "rheumatoid_factor": {"normal": (0, 14), "unit": "IU/mL"}
    }
}

# Application Settings
APP_SETTINGS = {
    "title": "Medical Diagnosis Support - PRefLexOR",
    "description": "AI-Powered Clinical Decision Support with Transparent Reasoning",
    "version": "1.0.0",
    "disclaimer": "This tool is for educational and decision support purposes only. Always consult with qualified healthcare professionals for actual patient care."
}

# Thinking Token Configuration
THINKING_TOKENS = {
    "start": "<|thinking|>",
    "end": "<|/thinking|>",
    "reflection_start": "<|reflect|>",
    "reflection_end": "<|/reflect|>"
}

# System Prompts for Different Analysis Types
SYSTEM_PROMPTS = {
    "differential_diagnosis": """You are an experienced internal medicine physician with expertise in differential diagnosis. 
You have deep knowledge of pathophysiology, clinical presentation patterns, and evidence-based medicine.
Provide detailed, step-by-step analysis using <|thinking|> tags to show your clinical reasoning process.
Focus on differential diagnosis, risk stratification, and evidence-based recommendations.
IMPORTANT: Always include disclaimers that this is for educational/support purposes only.""",

    "lab_interpretation": """You are a clinical pathologist with expertise in laboratory medicine and diagnostic testing.
You understand reference ranges, test limitations, and clinical correlation of laboratory findings.
Provide thorough analysis using <|thinking|> tags to demonstrate your interpretation process.
Consider pre-test probability, test characteristics, and clinical context.""",

    "treatment_planning": """You are a clinical pharmacist and internist with expertise in evidence-based treatment protocols.
You understand drug interactions, contraindications, and treatment guidelines.
Use <|thinking|> tags to show your treatment planning reasoning.
Always emphasize the need for individualized patient care and specialist consultation."""
}

# Severity Assessment Criteria
SEVERITY_CRITERIA = {
    "critical": {
        "description": "Life-threatening condition requiring immediate intervention",
        "timeframe": "Minutes to hours",
        "examples": ["acute MI", "stroke", "sepsis", "anaphylaxis"]
    },
    "urgent": {
        "description": "Serious condition requiring prompt medical attention",
        "timeframe": "Hours to 24 hours", 
        "examples": ["pneumonia", "cellulitis", "depression with suicidal ideation"]
    },
    "semi_urgent": {
        "description": "Condition requiring medical evaluation within days",
        "timeframe": "1-7 days",
        "examples": ["hypertension", "diabetes management", "stable angina"]
    },
    "routine": {
        "description": "Stable condition for routine follow-up",
        "timeframe": "Weeks to months",
        "examples": ["routine screening", "chronic stable conditions"]
    }
}

# Color Schemes for Visualization
SEVERITY_COLORS = {
    "critical": "#dc3545",    # Red
    "urgent": "#fd7e14",      # Orange
    "semi_urgent": "#ffc107", # Yellow
    "routine": "#28a745"      # Green
}

def get_lab_reference(test_category: str, test_name: str, gender: str = "normal") -> tuple:
    """Get reference range for lab test"""
    try:
        test_data = LAB_REFERENCES.get(test_category, {}).get(test_name, {})
        if gender in test_data:
            return test_data[gender]
        elif "normal" in test_data:
            return test_data["normal"]
        else:
            return (0, 0)  # Default if not found
    except:
        return (0, 0)

def assess_lab_abnormality(value: float, reference_range: tuple) -> str:
    """Assess if lab value is abnormal"""
    if not reference_range or reference_range == (0, 0):
        return "unknown"
    
    low, high = reference_range
    if value < low:
        return "low"
    elif value > high:
        return "high"
    else:
        return "normal"

def calculate_confidence_level(supporting_factors: int, total_factors: int) -> str:
    """Calculate diagnostic confidence level"""
    if total_factors == 0:
        return "insufficient_data"
    
    ratio = supporting_factors / total_factors
    
    if ratio >= CONFIDENCE_THRESHOLDS["high_confidence"]:
        return "high_confidence"
    elif ratio >= CONFIDENCE_THRESHOLDS["moderate_confidence"]:
        return "moderate_confidence"
    elif ratio >= CONFIDENCE_THRESHOLDS["low_confidence"]:
        return "low_confidence"
    else:
        return "insufficient_data"