"""
Data validation utilities for financial risk assessment
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date
import re

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class DataValidator:
    """Comprehensive data validation for financial applications"""
    
    @staticmethod
    def validate_mortgage_application(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate mortgage application data"""
        errors = []
        
        # Required fields
        required_fields = [
            "applicant_name", "monthly_income", "loan_amount", 
            "property_value", "down_payment", "credit_score"
        ]
        
        for field in required_fields:
            if field not in data or data[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Numeric validations
        numeric_validations = [
            ("monthly_income", 0, 1000000, "Monthly income must be between $0 and $1,000,000"),
            ("loan_amount", 1000, 10000000, "Loan amount must be between $1,000 and $10,000,000"),
            ("property_value", 1000, 50000000, "Property value must be between $1,000 and $50,000,000"),
            ("down_payment", 0, 10000000, "Down payment must be between $0 and $10,000,000"),
            ("credit_score", 300, 850, "Credit score must be between 300 and 850"),
            ("employment_years", 0, 50, "Employment years must be between 0 and 50")
        ]
        
        for field, min_val, max_val, error_msg in numeric_validations:
            if field in data:
                try:
                    value = float(data[field])
                    if not (min_val <= value <= max_val):
                        errors.append(error_msg)
                except (ValueError, TypeError):
                    errors.append(f"{field} must be a valid number")
        
        # Logical validations
        if "loan_amount" in data and "property_value" in data:
            try:
                loan_amount = float(data["loan_amount"])
                property_value = float(data["property_value"])
                if loan_amount > property_value:
                    errors.append("Loan amount cannot exceed property value")
            except (ValueError, TypeError):
                pass  # Already caught in numeric validation
        
        if "down_payment" in data and "property_value" in data:
            try:
                down_payment = float(data["down_payment"])
                property_value = float(data["property_value"])
                if down_payment > property_value:
                    errors.append("Down payment cannot exceed property value")
            except (ValueError, TypeError):
                pass
        
        # Email validation
        if "email" in data and data["email"]:
            if not DataValidator.validate_email(data["email"]):
                errors.append("Invalid email format")
        
        # Phone validation
        if "phone" in data and data["phone"]:
            if not DataValidator.validate_phone(data["phone"]):
                errors.append("Invalid phone number format")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_personal_loan_application(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate personal loan application data"""
        errors = []
        
        # Required fields
        required_fields = [
            "applicant_name", "monthly_income", "loan_amount", 
            "credit_score", "employment_type"
        ]
        
        for field in required_fields:
            if field not in data or data[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Numeric validations
        numeric_validations = [
            ("monthly_income", 0, 100000, "Monthly income must be between $0 and $100,000"),
            ("loan_amount", 1000, 100000, "Loan amount must be between $1,000 and $100,000"),
            ("credit_score", 300, 850, "Credit score must be between 300 and 850"),
            ("employment_years", 0, 50, "Employment years must be between 0 and 50"),
            ("monthly_debt_payments", 0, 50000, "Monthly debt payments must be between $0 and $50,000")
        ]
        
        for field, min_val, max_val, error_msg in numeric_validations:
            if field in data:
                try:
                    value = float(data[field])
                    if not (min_val <= value <= max_val):
                        errors.append(error_msg)
                except (ValueError, TypeError):
                    errors.append(f"{field} must be a valid number")
        
        # Employment type validation
        valid_employment_types = ["full_time", "part_time", "self_employed", "contract", "retired", "unemployed"]
        if "employment_type" in data:
            if data["employment_type"].lower() not in valid_employment_types:
                errors.append(f"Employment type must be one of: {', '.join(valid_employment_types)}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_business_credit_application(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate business credit application data"""
        errors = []
        
        # Required fields
        required_fields = [
            "business_name", "business_age_years", "annual_revenue",
            "industry", "requested_amount", "business_credit_score"
        ]
        
        for field in required_fields:
            if field not in data or data[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Numeric validations
        numeric_validations = [
            ("business_age_years", 0, 100, "Business age must be between 0 and 100 years"),
            ("annual_revenue", 0, 1000000000, "Annual revenue must be between $0 and $1,000,000,000"),
            ("requested_amount", 1000, 50000000, "Requested amount must be between $1,000 and $50,000,000"),
            ("business_credit_score", 0, 100, "Business credit score must be between 0 and 100"),
            ("current_assets", 0, 1000000000, "Current assets must be non-negative"),
            ("current_liabilities", 0, 1000000000, "Current liabilities must be non-negative"),
            ("total_debt", 0, 1000000000, "Total debt must be non-negative"),
            ("total_equity", -1000000000, 1000000000, "Total equity can be negative but reasonable")
        ]
        
        for field, min_val, max_val, error_msg in numeric_validations:
            if field in data:
                try:
                    value = float(data[field])
                    if not (min_val <= value <= max_val):
                        errors.append(error_msg)
                except (ValueError, TypeError):
                    errors.append(f"{field} must be a valid number")
        
        # Industry validation
        valid_industries = [
            "technology", "healthcare", "retail", "manufacturing", 
            "construction", "professional_services", "food_service",
            "transportation", "real_estate", "finance", "other"
        ]
        if "industry" in data:
            if data["industry"].lower() not in valid_industries:
                errors.append(f"Industry must be one of: {', '.join(valid_industries)}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def validate_phone(phone: str) -> bool:
        """Validate phone number format"""
        # Remove all non-digits
        digits = re.sub(r'\D', '', phone)
        # Check if it's a valid US phone number (10 or 11 digits)
        return len(digits) in [10, 11] and (len(digits) != 11 or digits[0] == '1')
    
    @staticmethod
    def validate_ssn(ssn: str) -> bool:
        """Validate SSN format (XXX-XX-XXXX)"""
        pattern = r'^\d{3}-\d{2}-\d{4}$'
        return re.match(pattern, ssn) is not None
    
    @staticmethod
    def validate_ein(ein: str) -> bool:
        """Validate EIN format (XX-XXXXXXX)"""
        pattern = r'^\d{2}-\d{7}$'
        return re.match(pattern, ein) is not None
    
    @staticmethod
    def sanitize_input(value: Any) -> Any:
        """Sanitize input values"""
        if isinstance(value, str):
            # Remove leading/trailing whitespace
            value = value.strip()
            # Replace multiple spaces with single space
            value = re.sub(r'\s+', ' ', value)
            # Remove potentially dangerous characters
            value = re.sub(r'[<>"\']', '', value)
        return value
    
    @staticmethod
    def format_currency(amount: float) -> str:
        """Format amount as currency"""
        return f"${amount:,.2f}"
    
    @staticmethod
    def format_percentage(ratio: float) -> str:
        """Format ratio as percentage"""
        return f"{ratio:.1%}"
    
    @staticmethod
    def calculate_monthly_payment(principal: float, annual_rate: float, years: int) -> float:
        """Calculate monthly mortgage payment"""
        if annual_rate == 0:
            return principal / (years * 12)
        
        monthly_rate = annual_rate / 12
        num_payments = years * 12
        
        payment = principal * (monthly_rate * (1 + monthly_rate)**num_payments) / \
                 ((1 + monthly_rate)**num_payments - 1)
        
        return payment
    
    @staticmethod
    def validate_application_completeness(data: Dict[str, Any], assessment_type: str) -> Dict[str, Any]:
        """Validate application completeness and return summary"""
        
        validation_result = {
            "is_valid": False,
            "errors": [],
            "warnings": [],
            "completeness_score": 0.0,
            "missing_optional_fields": []
        }
        
        # Validate based on assessment type
        if assessment_type == "mortgage":
            is_valid, errors = DataValidator.validate_mortgage_application(data)
        elif assessment_type == "personal_loan":
            is_valid, errors = DataValidator.validate_personal_loan_application(data)
        elif assessment_type == "business_credit":
            is_valid, errors = DataValidator.validate_business_credit_application(data)
        else:
            errors = ["Invalid assessment type"]
            is_valid = False
        
        validation_result["is_valid"] = is_valid
        validation_result["errors"] = errors
        
        # Calculate completeness score
        total_fields = len(data)
        empty_fields = sum(1 for v in data.values() if v is None or v == "")
        validation_result["completeness_score"] = (total_fields - empty_fields) / max(total_fields, 1)
        
        # Identify optional fields that could improve assessment
        optional_fields = DataValidator._get_optional_fields(assessment_type)
        missing_optional = [field for field in optional_fields if field not in data or not data[field]]
        validation_result["missing_optional_fields"] = missing_optional
        
        return validation_result
    
    @staticmethod
    def _get_optional_fields(assessment_type: str) -> List[str]:
        """Get list of optional fields that improve assessment quality"""
        
        optional_fields = {
            "mortgage": [
                "assets_savings", "assets_checking", "assets_investments",
                "employment_industry", "co_borrower_income", "property_tax",
                "homeowners_insurance", "hoa_fees"
            ],
            "personal_loan": [
                "assets_savings", "employment_industry", "education_level",
                "housing_payment", "other_income"
            ],
            "business_credit": [
                "cash_flow_statement", "profit_loss_statement", "business_plan",
                "collateral_value", "personal_guarantee", "industry_experience"
            ]
        }
        
        return optional_fields.get(assessment_type, [])