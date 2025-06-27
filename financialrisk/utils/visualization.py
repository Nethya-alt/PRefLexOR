"""
Visualization utilities for financial risk assessment
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Any, List
import numpy as np

from config.settings import RISK_COLORS, RISK_THRESHOLDS

class RiskVisualizer:
    """Create interactive visualizations for risk assessment"""
    
    @staticmethod
    def create_risk_gauge(risk_level: str, confidence_score: float) -> go.Figure:
        """Create a risk level gauge chart"""
        
        # Map risk levels to numeric values
        risk_values = {
            "LOW": 0.2,
            "MODERATE": 0.5, 
            "HIGH": 0.8,
            "CRITICAL": 1.0
        }
        
        risk_value = risk_values.get(risk_level, 0.5)
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_value * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Risk Level: {risk_level}"},
            delta = {'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': RISK_COLORS.get(risk_level.lower(), "#ffc107")},
                'steps': [
                    {'range': [0, 25], 'color': RISK_COLORS["low"]},
                    {'range': [25, 50], 'color': RISK_COLORS["moderate"]},
                    {'range': [50, 75], 'color': RISK_COLORS["high"]},
                    {'range': [75, 100], 'color': RISK_COLORS["critical"]}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': confidence_score * 100
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            font_size=14
        )
        
        return fig
    
    @staticmethod
    def create_financial_metrics_chart(metrics: Dict[str, float], assessment_type: str) -> go.Figure:
        """Create financial metrics comparison chart"""
        
        thresholds = RISK_THRESHOLDS.get(assessment_type, {})
        
        # Prepare data for visualization
        metric_names = []
        current_values = []
        good_thresholds = []
        acceptable_thresholds = []
        colors = []
        
        for metric_name, value in metrics.items():
            if metric_name in thresholds:
                metric_thresholds = thresholds[metric_name]
                
                metric_names.append(metric_name.replace('_', ' ').title())
                current_values.append(value)
                
                # Get thresholds (handle different metric types)
                if metric_name in ["credit_score", "debt_service_coverage", "current_ratio"]:
                    good_thresholds.append(metric_thresholds.get("good", 0))
                    acceptable_thresholds.append(metric_thresholds.get("acceptable", 0))
                    # Color based on whether higher is better
                    if value >= metric_thresholds.get("good", 0):
                        colors.append(RISK_COLORS["low"])
                    elif value >= metric_thresholds.get("acceptable", 0):
                        colors.append(RISK_COLORS["moderate"])
                    else:
                        colors.append(RISK_COLORS["high"])
                else:
                    good_thresholds.append(metric_thresholds.get("good", 1.0))
                    acceptable_thresholds.append(metric_thresholds.get("acceptable", 1.0))
                    # Color based on whether lower is better
                    if value <= metric_thresholds.get("good", 1.0):
                        colors.append(RISK_COLORS["low"])
                    elif value <= metric_thresholds.get("acceptable", 1.0):
                        colors.append(RISK_COLORS["moderate"])
                    else:
                        colors.append(RISK_COLORS["high"])
        
        fig = go.Figure()
        
        # Add current values
        fig.add_trace(go.Bar(
            name='Current Value',
            x=metric_names,
            y=current_values,
            marker_color=colors,
            text=[f"{v:.2f}" if isinstance(v, float) else str(v) for v in current_values],
            textposition='auto',
        ))
        
        # Add threshold lines
        fig.add_trace(go.Scatter(
            name='Good Threshold',
            x=metric_names,
            y=good_thresholds,
            mode='markers',
            marker=dict(color='green', size=10, symbol='diamond'),
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter(
            name='Acceptable Threshold',
            x=metric_names,
            y=acceptable_thresholds,
            mode='markers',
            marker=dict(color='orange', size=10, symbol='triangle-up'),
            showlegend=True
        ))
        
        fig.update_layout(
            title=f"Financial Metrics Analysis - {assessment_type.replace('_', ' ').title()}",
            xaxis_title="Metrics",
            yaxis_title="Values",
            height=400,
            margin=dict(l=40, r=40, t=60, b=40),
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def create_risk_factors_chart(risk_factors: List[Dict[str, Any]]) -> go.Figure:
        """Create risk factors breakdown chart"""
        
        # Group risk factors by category
        categories = {}
        for factor in risk_factors:
            category = factor.get("category", "Other")
            if category not in categories:
                categories[category] = {"low": 0, "moderate": 0, "high": 0, "critical": 0}
            
            risk_level = factor.get("risk_level", "moderate")
            categories[category][risk_level] += 1
        
        # Prepare data for stacked bar chart
        category_names = list(categories.keys())
        low_counts = [categories[cat]["low"] for cat in category_names]
        moderate_counts = [categories[cat]["moderate"] for cat in category_names]
        high_counts = [categories[cat]["high"] for cat in category_names]
        critical_counts = [categories[cat]["critical"] for cat in category_names]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Low Risk',
            x=category_names,
            y=low_counts,
            marker_color=RISK_COLORS["low"]
        ))
        
        fig.add_trace(go.Bar(
            name='Moderate Risk',
            x=category_names,
            y=moderate_counts,
            marker_color=RISK_COLORS["moderate"]
        ))
        
        fig.add_trace(go.Bar(
            name='High Risk',
            x=category_names,
            y=high_counts,
            marker_color=RISK_COLORS["high"]
        ))
        
        fig.add_trace(go.Bar(
            name='Critical Risk',
            x=category_names,
            y=critical_counts,
            marker_color=RISK_COLORS["critical"]
        ))
        
        fig.update_layout(
            barmode='stack',
            title="Risk Factors by Category",
            xaxis_title="Risk Categories",
            yaxis_title="Number of Factors",
            height=400,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        return fig
    
    @staticmethod
    def create_debt_to_income_breakdown(
        monthly_income: float,
        current_debts: float, 
        proposed_payment: float = 0
    ) -> go.Figure:
        """Create debt-to-income ratio breakdown"""
        
        # Calculate ratios
        current_dti = (current_debts / monthly_income) if monthly_income > 0 else 0
        proposed_dti = ((current_debts + proposed_payment) / monthly_income) if monthly_income > 0 else 0
        remaining_income = monthly_income - current_debts - proposed_payment
        
        # Create pie chart
        labels = ['Current Debt Payments', 'Proposed Payment', 'Remaining Income']
        values = [current_debts, proposed_payment, max(0, remaining_income)]
        colors = [RISK_COLORS["high"], RISK_COLORS["moderate"], RISK_COLORS["low"]]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker_colors=colors,
            textinfo='label+percent+value',
            texttemplate='%{label}<br>%{percent}<br>$%{value:,.0f}',
            hovertemplate='%{label}<br>Amount: $%{value:,.0f}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title=f"Monthly Income Breakdown<br>Current DTI: {current_dti:.1%} | Proposed DTI: {proposed_dti:.1%}",
            height=400,
            margin=dict(l=40, r=40, t=80, b=40),
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_credit_score_distribution(credit_score: int) -> go.Figure:
        """Create credit score distribution visualization"""
        
        # Credit score ranges
        score_ranges = {
            "Poor": (300, 579),
            "Fair": (580, 669), 
            "Good": (670, 739),
            "Very Good": (740, 799),
            "Excellent": (800, 850)
        }
        
        # Create distribution curve
        x_vals = list(range(300, 851))
        y_vals = [1 if score_ranges["Poor"][0] <= x <= score_ranges["Poor"][1] else
                 2 if score_ranges["Fair"][0] <= x <= score_ranges["Fair"][1] else
                 3 if score_ranges["Good"][0] <= x <= score_ranges["Good"][1] else
                 4 if score_ranges["Very Good"][0] <= x <= score_ranges["Very Good"][1] else
                 5 for x in x_vals]
        
        fig = go.Figure()
        
        # Add score ranges as colored areas
        for i, (range_name, (start, end)) in enumerate(score_ranges.items()):
            color = ["#dc3545", "#fd7e14", "#ffc107", "#28a745", "#17a2b8"][i]
            fig.add_trace(go.Scatter(
                x=[start, end, end, start, start],
                y=[0, 0, 6, 6, 0],
                fill='toself',
                fillcolor=color,
                opacity=0.3,
                line=dict(width=0),
                name=range_name,
                showlegend=True
            ))
        
        # Add current credit score line
        fig.add_trace(go.Scatter(
            x=[credit_score, credit_score],
            y=[0, 6],
            mode='lines',
            line=dict(color='black', width=3, dash='dash'),
            name=f'Your Score: {credit_score}',
            showlegend=True
        ))
        
        fig.update_layout(
            title=f"Credit Score Position: {credit_score}",
            xaxis_title="Credit Score",
            yaxis_title="Score Range",
            yaxis=dict(showticklabels=False),
            height=300,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        return fig
    
    @staticmethod
    def create_loan_affordability_chart(
        income: float,
        current_debts: float,
        loan_amount: float,
        interest_rate: float,
        loan_term: int
    ) -> go.Figure:
        """Create loan affordability analysis chart"""
        
        # Calculate monthly payment
        monthly_rate = interest_rate / 12 / 100
        num_payments = loan_term * 12
        
        if monthly_rate > 0:
            monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / \
                            ((1 + monthly_rate)**num_payments - 1)
        else:
            monthly_payment = loan_amount / num_payments
        
        # Calculate affordability scenarios
        scenarios = []
        dti_ratios = []
        
        for payment_multiplier in [0.8, 0.9, 1.0, 1.1, 1.2]:
            adjusted_payment = monthly_payment * payment_multiplier
            total_debts = current_debts + adjusted_payment
            dti = total_debts / income if income > 0 else 0
            
            scenarios.append(f"{payment_multiplier:.0%} of Payment")
            dti_ratios.append(dti)
        
        # Create bar chart
        colors = [RISK_COLORS["low"] if dti <= 0.28 else
                 RISK_COLORS["moderate"] if dti <= 0.36 else
                 RISK_COLORS["high"] if dti <= 0.43 else
                 RISK_COLORS["critical"] for dti in dti_ratios]
        
        fig = go.Figure(data=[go.Bar(
            x=scenarios,
            y=[ratio * 100 for ratio in dti_ratios],
            marker_color=colors,
            text=[f"{ratio:.1%}" for ratio in dti_ratios],
            textposition='auto'
        )])
        
        # Add threshold lines
        fig.add_hline(y=28, line_dash="dash", line_color="green", 
                     annotation_text="Excellent DTI (28%)")
        fig.add_hline(y=36, line_dash="dash", line_color="orange",
                     annotation_text="Good DTI (36%)")
        fig.add_hline(y=43, line_dash="dash", line_color="red",
                     annotation_text="Max QM DTI (43%)")
        
        fig.update_layout(
            title=f"Debt-to-Income Analysis<br>Base Monthly Payment: ${monthly_payment:,.0f}",
            xaxis_title="Payment Scenarios",
            yaxis_title="Debt-to-Income Ratio (%)",
            height=400,
            margin=dict(l=40, r=40, t=80, b=40)
        )
        
        return fig
    
    @staticmethod
    def create_assessment_summary_table(result) -> go.Figure:
        """Create a summary table for the assessment result"""
        
        # Prepare table data
        headers = ["Metric", "Value", "Status"]
        
        table_data = [
            ["Application ID", result.application_id, ""],
            ["Assessment Type", result.assessment_type.replace('_', ' ').title(), ""],
            ["Recommendation", result.recommendation, ""],
            ["Risk Level", result.risk_level, ""],
            ["Confidence Score", f"{result.confidence_score:.1%}", ""],
            ["Model Used", result.model_used, ""],
            ["Timestamp", result.timestamp[:19], ""]
        ]
        
        # Add financial metrics
        for metric, value in result.financial_metrics.items():
            if isinstance(value, float):
                if metric in ["debt_to_income", "loan_to_value"]:
                    formatted_value = f"{value:.1%}"
                else:
                    formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
            
            table_data.append([
                metric.replace('_', ' ').title(),
                formatted_value,
                "📊"
            ])
        
        # Create table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=headers,
                fill_color='paleturquoise',
                align='left',
                font=dict(size=14, color='black')
            ),
            cells=dict(
                values=list(zip(*table_data)),
                fill_color='lavender',
                align='left',
                font=dict(size=12)
            )
        )])
        
        fig.update_layout(
            title="Assessment Summary",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig