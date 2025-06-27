"""
Core risk assessment logic using PRefLexOR reasoning patterns
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict

from .ollama_client import OllamaClient
from config.settings import (
    RISK_THRESHOLDS, SYSTEM_PROMPTS, THINKING_TOKENS,
    COMPLIANCE_RULES, get_risk_threshold, validate_compliance
)

logger = logging.getLogger(__name__)

@dataclass
class RiskAssessmentResult:
    """Risk assessment result with detailed reasoning"""
    application_id: str
    assessment_type: str
    recommendation: str
    confidence_score: float
    risk_level: str
    thinking_process: str
    final_analysis: str
    compliance_status: Dict[str, bool]
    risk_factors: List[Dict[str, Any]]
    financial_metrics: Dict[str, float]
    timestamp: str
    model_used: str

class FinancialRiskAssessor:
    def __init__(self, ollama_client: OllamaClient):
        self.ollama_client = ollama_client
        self.assessment_history = []
        
    def assess_mortgage_risk(self, application_data: Dict[str, Any]) -> RiskAssessmentResult:
        """Assess mortgage application risk"""
        return self._perform_assessment(application_data, "mortgage")
    
    def assess_personal_loan_risk(self, application_data: Dict[str, Any]) -> RiskAssessmentResult:
        """Assess personal loan risk"""
        return self._perform_assessment(application_data, "personal_loan")
    
    def assess_business_credit_risk(self, application_data: Dict[str, Any]) -> RiskAssessmentResult:
        """Assess business credit risk"""
        return self._perform_assessment(application_data, "business_credit")
    
    def _perform_assessment(self, application_data: Dict[str, Any], assessment_type: str) -> RiskAssessmentResult:
        """Core assessment logic with PRefLexOR reasoning"""
        
        # Generate application ID if not provided
        app_id = application_data.get("application_id", f"{assessment_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Calculate financial metrics
        financial_metrics = self._calculate_financial_metrics(application_data, assessment_type)
        
        # Validate compliance
        compliance_status = validate_compliance(assessment_type, application_data)
        
        # Generate risk factors
        risk_factors = self._identify_risk_factors(application_data, assessment_type, financial_metrics)
        
        # Create detailed prompt for LLM analysis
        analysis_prompt = self._create_analysis_prompt(
            application_data, assessment_type, financial_metrics, risk_factors, compliance_status
        )
        
        # Get system prompt for assessment type
        system_prompt = SYSTEM_PROMPTS.get(assessment_type, SYSTEM_PROMPTS["mortgage"])
        
        # Generate LLM response with thinking tokens
        llm_response = self.ollama_client.generate_response(
            prompt=analysis_prompt,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=2048
        )
        
        if not llm_response["success"]:
            logger.error(f"LLM analysis failed: {llm_response.get('error', 'Unknown error')}")
            return self._create_fallback_assessment(app_id, assessment_type, application_data)
        
        # Extract thinking and final analysis
        parsed_response = self.ollama_client.extract_thinking_section(llm_response["content"])
        
        # Determine overall recommendation and risk level
        recommendation, confidence_score, risk_level = self._determine_final_recommendation(
            financial_metrics, risk_factors, compliance_status, parsed_response["answer"]
        )
        
        # Create assessment result
        result = RiskAssessmentResult(
            application_id=app_id,
            assessment_type=assessment_type,
            recommendation=recommendation,
            confidence_score=confidence_score,
            risk_level=risk_level,
            thinking_process=parsed_response["thinking"],
            final_analysis=parsed_response["answer"],
            compliance_status=compliance_status,
            risk_factors=risk_factors,
            financial_metrics=financial_metrics,
            timestamp=datetime.now().isoformat(),
            model_used=self.ollama_client.model
        )
        
        # Store in assessment history
        self.assessment_history.append(result)
        
        return result
    
    def _calculate_financial_metrics(self, data: Dict[str, Any], assessment_type: str) -> Dict[str, float]:
        """Calculate key financial metrics based on assessment type"""
        metrics = {}
        
        if assessment_type == "mortgage":
            # Debt-to-Income Ratio
            monthly_income = data.get("monthly_income", 0)
            monthly_debts = data.get("monthly_debt_payments", 0)
            proposed_payment = data.get("proposed_monthly_payment", 0)
            
            if monthly_income > 0:
                metrics["debt_to_income"] = (monthly_debts + proposed_payment) / monthly_income
                metrics["current_dti"] = monthly_debts / monthly_income
            
            # Loan-to-Value Ratio
            loan_amount = data.get("loan_amount", 0)
            property_value = data.get("property_value", 0)
            if property_value > 0:
                metrics["loan_to_value"] = loan_amount / property_value
            
            # Down Payment Percentage
            down_payment = data.get("down_payment", 0)
            if property_value > 0:
                metrics["down_payment_percentage"] = down_payment / property_value
                
            metrics["credit_score"] = data.get("credit_score", 0)
            
        elif assessment_type == "personal_loan":
            monthly_income = data.get("monthly_income", 0)
            monthly_debts = data.get("monthly_debt_payments", 0)
            requested_payment = data.get("requested_monthly_payment", 0)
            
            if monthly_income > 0:
                metrics["debt_to_income"] = (monthly_debts + requested_payment) / monthly_income
                metrics["current_dti"] = monthly_debts / monthly_income
                
            metrics["credit_score"] = data.get("credit_score", 0)
            metrics["loan_amount"] = data.get("loan_amount", 0)
            
        elif assessment_type == "business_credit":
            # Debt Service Coverage Ratio
            net_operating_income = data.get("net_operating_income", 0)
            annual_debt_service = data.get("annual_debt_service", 0)
            
            if annual_debt_service > 0:
                metrics["debt_service_coverage"] = net_operating_income / annual_debt_service
            
            # Current Ratio
            current_assets = data.get("current_assets", 0)
            current_liabilities = data.get("current_liabilities", 0)
            
            if current_liabilities > 0:
                metrics["current_ratio"] = current_assets / current_liabilities
                
            # Debt-to-Equity Ratio
            total_debt = data.get("total_debt", 0)
            total_equity = data.get("total_equity", 0)
            
            if total_equity > 0:
                metrics["debt_to_equity"] = total_debt / total_equity
                
            metrics["credit_score"] = data.get("business_credit_score", 0)
            
        return metrics
    
    def _identify_risk_factors(
        self, 
        data: Dict[str, Any], 
        assessment_type: str, 
        metrics: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Identify and categorize risk factors"""
        risk_factors = []
        
        # Financial metric risks
        thresholds = RISK_THRESHOLDS.get(assessment_type, {})
        
        for metric_name, value in metrics.items():
            if metric_name in thresholds:
                metric_thresholds = thresholds[metric_name]
                risk_level = self._categorize_metric_risk(value, metric_thresholds, metric_name)
                
                risk_factors.append({
                    "category": "financial_metrics",
                    "factor": metric_name,
                    "value": value,
                    "risk_level": risk_level,
                    "description": self._get_risk_description(metric_name, value, risk_level)
                })
        
        # Employment stability
        employment_years = data.get("employment_years", 0)
        employment_type = data.get("employment_type", "")
        
        employment_risk = "low"
        if employment_years < 1:
            employment_risk = "high"
        elif employment_years < 2:
            employment_risk = "moderate"
        elif employment_type.lower() in ["contract", "temporary", "self-employed"]:
            employment_risk = "moderate"
            
        risk_factors.append({
            "category": "employment",
            "factor": "employment_stability",
            "value": employment_years,
            "risk_level": employment_risk,
            "description": f"{employment_years} years of {employment_type} employment"
        })
        
        # Add more risk factors based on assessment type
        if assessment_type == "mortgage":
            self._add_mortgage_specific_risks(risk_factors, data)
        elif assessment_type == "business_credit":
            self._add_business_specific_risks(risk_factors, data)
            
        return risk_factors
    
    def _categorize_metric_risk(self, value: float, thresholds: Dict[str, float], metric_name: str) -> str:
        """Categorize risk level based on metric thresholds"""
        
        # Handle different metric types (higher is better vs lower is better)
        if metric_name in ["credit_score", "debt_service_coverage", "current_ratio"]:
            # Higher values are better
            if value >= thresholds.get("excellent", float('inf')):
                return "low"
            elif value >= thresholds.get("good", 0):
                return "moderate"
            elif value >= thresholds.get("acceptable", 0):
                return "high"
            else:
                return "critical"
        else:
            # Lower values are better (DTI, LTV, etc.)
            if value <= thresholds.get("excellent", 0):
                return "low"
            elif value <= thresholds.get("good", float('inf')):
                return "moderate"
            elif value <= thresholds.get("acceptable", float('inf')):
                return "high"
            else:
                return "critical"
    
    def _get_risk_description(self, metric_name: str, value: float, risk_level: str) -> str:
        """Generate human-readable risk descriptions"""
        descriptions = {
            "debt_to_income": f"DTI ratio of {value:.1%} is {risk_level} risk",
            "loan_to_value": f"LTV ratio of {value:.1%} is {risk_level} risk", 
            "credit_score": f"Credit score of {value:.0f} represents {risk_level} risk",
            "debt_service_coverage": f"DSCR of {value:.2f} indicates {risk_level} risk",
            "current_ratio": f"Current ratio of {value:.2f} shows {risk_level} liquidity risk"
        }
        return descriptions.get(metric_name, f"{metric_name}: {value} ({risk_level} risk)")
    
    def _add_mortgage_specific_risks(self, risk_factors: List[Dict[str, Any]], data: Dict[str, Any]):
        """Add mortgage-specific risk factors"""
        
        # Property type risk
        property_type = data.get("property_type", "").lower()
        property_risk = "low"
        if property_type in ["condo", "manufactured", "co-op"]:
            property_risk = "moderate"
        elif property_type in ["investment", "commercial"]:
            property_risk = "high"
            
        risk_factors.append({
            "category": "collateral",
            "factor": "property_type",
            "value": property_type,
            "risk_level": property_risk,
            "description": f"Property type: {property_type}"
        })
        
        # Occupancy type
        occupancy = data.get("occupancy_type", "").lower()
        occupancy_risk = "low"
        if occupancy == "second_home":
            occupancy_risk = "moderate"
        elif occupancy == "investment":
            occupancy_risk = "high"
            
        risk_factors.append({
            "category": "usage",
            "factor": "occupancy_type", 
            "value": occupancy,
            "risk_level": occupancy_risk,
            "description": f"Occupancy: {occupancy}"
        })
    
    def _add_business_specific_risks(self, risk_factors: List[Dict[str, Any]], data: Dict[str, Any]):
        """Add business-specific risk factors"""
        
        # Industry risk
        industry = data.get("industry", "").lower()
        industry_risk_map = {
            "technology": "moderate",
            "healthcare": "low", 
            "retail": "high",
            "restaurant": "high",
            "manufacturing": "moderate",
            "real_estate": "high"
        }
        industry_risk = industry_risk_map.get(industry, "moderate")
        
        risk_factors.append({
            "category": "industry",
            "factor": "industry_risk",
            "value": industry,
            "risk_level": industry_risk,
            "description": f"Industry: {industry} ({industry_risk} risk sector)"
        })
        
        # Business age
        business_age = data.get("business_age_years", 0)
        age_risk = "low"
        if business_age < 1:
            age_risk = "critical"
        elif business_age < 3:
            age_risk = "high"
        elif business_age < 5:
            age_risk = "moderate"
            
        risk_factors.append({
            "category": "stability",
            "factor": "business_age",
            "value": business_age,
            "risk_level": age_risk,
            "description": f"Business operating for {business_age} years"
        })
    
    def _create_analysis_prompt(
        self,
        data: Dict[str, Any],
        assessment_type: str,
        metrics: Dict[str, float],
        risk_factors: List[Dict[str, Any]],
        compliance_status: Dict[str, bool]
    ) -> str:
        """Create detailed prompt for LLM analysis"""
        
        prompt = f"""
Perform a comprehensive {assessment_type.replace('_', ' ')} risk assessment for this application. Use {THINKING_TOKENS['start']} to show your detailed reasoning process.

APPLICATION DATA:
{json.dumps(data, indent=2)}

CALCULATED FINANCIAL METRICS:
{json.dumps(metrics, indent=2)}

IDENTIFIED RISK FACTORS:
{json.dumps(risk_factors, indent=2)}

COMPLIANCE STATUS:
{json.dumps(compliance_status, indent=2)}

Please provide a thorough analysis including:

1. **Financial Strength Assessment**: Evaluate all financial metrics against industry standards
2. **Risk Factor Analysis**: Analyze each identified risk factor and its impact
3. **Regulatory Compliance**: Review compliance status and any concerns
4. **Market and External Factors**: Consider economic conditions and market risks
5. **Mitigation Strategies**: Suggest ways to address identified risks
6. **Final Recommendation**: Clear approve/deny/conditional approval with reasoning

Use detailed step-by-step reasoning in your {THINKING_TOKENS['start']} section, then provide a clear executive summary with your final recommendation.

Focus on:
- Quantitative analysis of financial ratios
- Qualitative assessment of risk factors
- Regulatory compliance verification
- Clear explanation of decision rationale
- Specific conditions or requirements if applicable
"""
        
        return prompt
    
    def _determine_final_recommendation(
        self,
        metrics: Dict[str, float],
        risk_factors: List[Dict[str, Any]],
        compliance_status: Dict[str, bool],
        llm_analysis: str
    ) -> tuple[str, float, str]:
        """Determine final recommendation, confidence score, and risk level"""
        
        # Calculate risk score based on factors
        total_risk_score = 0
        risk_count = 0
        
        for factor in risk_factors:
            risk_level = factor["risk_level"]
            risk_value = {"low": 1, "moderate": 2, "high": 3, "critical": 4}.get(risk_level, 2)
            total_risk_score += risk_value
            risk_count += 1
        
        average_risk = total_risk_score / max(risk_count, 1)
        
        # Check compliance issues
        compliance_issues = sum(1 for status in compliance_status.values() if not status)
        
        # Determine recommendation based on analysis
        llm_lower = llm_analysis.lower()
        
        if "approve" in llm_lower and "deny" not in llm_lower:
            if compliance_issues == 0 and average_risk <= 2:
                recommendation = "APPROVE"
                confidence_score = 0.85
                risk_level = "LOW" if average_risk <= 1.5 else "MODERATE"
            else:
                recommendation = "CONDITIONAL_APPROVAL"
                confidence_score = 0.70
                risk_level = "MODERATE"
        elif "conditional" in llm_lower or "conditions" in llm_lower:
            recommendation = "CONDITIONAL_APPROVAL"
            confidence_score = 0.65
            risk_level = "MODERATE" if average_risk <= 2.5 else "HIGH"
        else:
            recommendation = "DENY"
            confidence_score = 0.80
            risk_level = "HIGH" if average_risk <= 3 else "CRITICAL"
        
        # Adjust confidence based on data completeness
        if len(metrics) < 3:  # Insufficient data
            confidence_score *= 0.8
            
        return recommendation, confidence_score, risk_level
    
    def _create_fallback_assessment(
        self, 
        app_id: str, 
        assessment_type: str, 
        data: Dict[str, Any]
    ) -> RiskAssessmentResult:
        """Create fallback assessment when LLM fails"""
        
        return RiskAssessmentResult(
            application_id=app_id,
            assessment_type=assessment_type,
            recommendation="MANUAL_REVIEW_REQUIRED",
            confidence_score=0.0,
            risk_level="UNKNOWN",
            thinking_process="LLM analysis failed - manual review required",
            final_analysis="Unable to complete automated analysis. Please review manually.",
            compliance_status={},
            risk_factors=[],
            financial_metrics={},
            timestamp=datetime.now().isoformat(),
            model_used=self.ollama_client.model
        )
    
    def get_assessment_history(self) -> List[Dict[str, Any]]:
        """Get history of all assessments"""
        return [asdict(assessment) for assessment in self.assessment_history]
    
    def export_assessment(self, result: RiskAssessmentResult) -> Dict[str, Any]:
        """Export assessment result for external systems"""
        return asdict(result)