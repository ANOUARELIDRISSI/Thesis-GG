"""
Morocco Génération Green - Interactive Flask Dashboard
Visualize data, train models, select best models, forecast, and check SDG goal achievement
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template, jsonify, request, send_file

# Add shared utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shared', 'utils'))
from models import (adaptive_model_selection, HoltWinters, ARIMALite, 
                    PolynomialTrend, RFForecaster, GBForecaster, SVRForecaster)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'morocco-gg-2030'

# Global data storage
DATA_CACHE = {}

# ─────────────────────────────────────────────────────────────
# Data Loading Functions
# ─────────────────────────────────────────────────────────────

def load_question_data(question_num):
    """Load data for a specific question"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if question_num == 1:
        raw_path = os.path.join(base_dir, "question_1_food_security", "data", "raw", "morocco_food_security_1990_2023.csv")
        df = pd.read_csv(raw_path)
        # Compute derived metrics
        df["cereal_total_supply_1000MT"] = df["cereal_production_1000MT"] + df["cereal_imports_1000MT"]
        df["cereal_self_sufficiency_pct"] = (
            df["cereal_production_1000MT"] / df["cereal_total_supply_1000MT"] * 100
        ).clip(5, 99)
        
        targets = {
            "caloric_availability_kcal_cap_day": {"value": 3500, "direction": "higher"},
            "cereal_self_sufficiency_pct": {"value": 70, "direction": "higher"},
            "import_dependency_cereal_pct": {"value": 30, "direction": "lower"},
            "irrigated_area_1000ha": {"value": 1600, "direction": "higher"},
            "undernourishment_pct": {"value": 2.0, "direction": "lower"},
        }
        
    elif question_num == 2:
        raw_path = os.path.join(base_dir, "question_2_agricultural_gdp", "data", "raw", "morocco_agri_gdp_1990_2023.csv")
        df = pd.read_csv(raw_path)
        baseline_gdp = df[df["Year"] == 2020]["agri_gdp_billion_MAD"].values[0]
        target_gdp = baseline_gdp * 2.0
        
        targets = {
            "agri_gdp_billion_MAD": {"value": target_gdp, "direction": "higher"},
            "agri_investment_billion_MAD": {"value": None, "direction": "higher"},
            "agri_labor_productivity_1000MAD_worker": {"value": None, "direction": "higher"},
        }
        
    elif question_num == 3:
        raw_path = os.path.join(base_dir, "question_3_agricultural_exports", "data", "raw", "morocco_agri_exports_1990_2023.csv")
        df = pd.read_csv(raw_path)
        df["export_gdp_ratio"] = df["agri_exports_billion_MAD"] / df["agri_gdp_billion_MAD"]
        baseline_ratio = df[df["Year"] == 2020]["export_gdp_ratio"].values[0]
        target_ratio = baseline_ratio * 2.0
        
        targets = {
            "export_gdp_ratio": {"value": target_ratio, "direction": "higher"},
            "agri_exports_billion_MAD": {"value": None, "direction": "higher"},
            "citrus_exports_billion_MAD": {"value": None, "direction": "higher"},
        }
    
    return df, targets


def get_available_variables(question_num):
    """Get list of numeric variables for a question"""
    df, _ = load_question_data(question_num)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'Year']
    return numeric_cols


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')


@app.route('/api/questions')
def get_questions():
    """Get list of available questions"""
    questions = [
        {"id": 1, "name": "Food Security", "description": "Caloric availability, self-sufficiency, import dependency"},
        {"id": 2, "name": "Agricultural GDP", "description": "Doubling agricultural GDP by 2030"},
        {"id": 3, "name": "Agricultural Exports", "description": "Export/GDP ratio doubling"}
    ]
    return jsonify(questions)


@app.route('/api/question/<int:question_id>/variables')
def get_variables(question_id):
    """Get target variables (with GG 2030 goals) for a question"""
    try:
        df, targets = load_question_data(question_id)
        
        # Only return variables that have targets (GG 2030 goals)
        target_variables = []
        for variable, target_info in targets.items():
            if variable in df.columns and target_info.get("value") is not None:
                target_variables.append({
                    "name": variable,
                    "label": variable.replace('_', ' ').title(),
                    "target": target_info.get("value"),
                    "direction": target_info.get("direction")
                })
        
        return jsonify({"variables": target_variables})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/question/<int:question_id>/data/<variable>')
def get_data(question_id, variable):
    """Get historical data for a variable"""
    try:
        df, targets = load_question_data(question_id)
        
        if variable not in df.columns:
            return jsonify({"error": f"Variable {variable} not found"}), 404
        
        data = {
            "years": df["Year"].tolist(),
            "values": df[variable].tolist(),
            "variable": variable,
            "target": targets.get(variable, {}).get("value"),
            "direction": targets.get(variable, {}).get("direction")
        }
        
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/question/<int:question_id>/train', methods=['POST'])
def train_models(question_id):
    """Train all models and return comparison"""
    try:
        data = request.json
        variable = data.get('variable')
        
        df, targets = load_question_data(question_id)
        
        if variable not in df.columns:
            return jsonify({"error": f"Variable {variable} not found"}), 404
        
        y = df[variable].dropna().values
        
        # Run adaptive model selection
        model_df, best_model, best_name, length_label = adaptive_model_selection(y, verbose=False)
        
        # Store in cache
        cache_key = f"q{question_id}_{variable}"
        DATA_CACHE[cache_key] = {
            "best_model": best_model,
            "best_name": best_name,
            "y": y,
            "years": df["Year"].tolist()
        }
        
        # Convert model comparison to dict
        models_comparison = model_df.to_dict('records')
        
        return jsonify({
            "models": models_comparison,
            "best_model": best_name,
            "series_info": length_label
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/question/<int:question_id>/forecast', methods=['POST'])
def forecast(question_id):
    """Generate forecast using best model with policy scenarios"""
    try:
        data = request.json
        variable = data.get('variable')
        horizon = data.get('horizon', 7)
        
        cache_key = f"q{question_id}_{variable}"
        
        if cache_key not in DATA_CACHE:
            return jsonify({"error": "Please train models first"}), 400
        
        cached = DATA_CACHE[cache_key]
        best_model = cached["best_model"]
        y = cached["y"]
        years = cached["years"]
        
        # Generate baseline forecast
        forecast_values = best_model.predict(horizon)
        forecast_years = list(range(years[-1] + 1, years[-1] + 1 + horizon))
        
        # Confidence intervals
        std_err = np.std(y[-8:]) * 0.15 * np.sqrt(np.arange(1, horizon + 1))
        lower_95 = forecast_values - 1.96 * std_err
        upper_95 = forecast_values + 1.96 * std_err
        
        # Check goal achievement
        df, targets = load_question_data(question_id)
        target_info = targets.get(variable, {})
        target_value = target_info.get("value")
        direction = target_info.get("direction")
        
        goal_achieved = None
        policy_scenarios = []
        
        if target_value is not None:
            forecast_2030 = forecast_values[-1]
            if direction == "higher":
                goal_achieved = bool(forecast_2030 >= target_value)
            elif direction == "lower":
                goal_achieved = bool(forecast_2030 <= target_value)
            
            # If goal not achieved, generate policy scenarios
            if not goal_achieved:
                policy_scenarios = generate_policy_scenarios_for_gap(
                    forecast_values, forecast_years, target_value, direction, variable
                )
        
        return jsonify({
            "historical_years": years,
            "historical_values": y.tolist(),
            "forecast_years": forecast_years,
            "forecast_values": forecast_values.tolist(),
            "lower_95": lower_95.tolist(),
            "upper_95": upper_95.tolist(),
            "model_name": cached["best_name"],
            "target_value": target_value,
            "target_direction": direction,
            "goal_achieved": goal_achieved,
            "forecast_2030": float(forecast_values[-1]) if len(forecast_values) > 0 else None,
            "policy_scenarios": policy_scenarios
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400


def generate_policy_scenarios_for_gap(baseline_forecast, forecast_years, target_value, direction, variable):
    """Generate policy intervention scenarios when goal is NOT achieved"""
    scenarios = []
    
    # Define policy multipliers based on variable type and direction
    if "gdp" in variable.lower() or "production" in variable.lower() or "export" in variable.lower():
        policies = [
            {"name": "Moderate Investment (+20%)", "multiplier": 1.20, "description": "Increase agricultural investment by 20%"},
            {"name": "Strong Investment (+40%)", "multiplier": 1.40, "description": "Increase agricultural investment by 40%"},
            {"name": "Comprehensive Reform (+60%)", "multiplier": 1.60, "description": "Full policy package: investment, tech, irrigation"}
        ]
    elif "self_sufficiency" in variable.lower() or "availability" in variable.lower() or "area" in variable.lower():
        policies = [
            {"name": "Moderate Intervention (+15%)", "multiplier": 1.15, "description": "Expand irrigated area and improve yields"},
            {"name": "Strong Intervention (+30%)", "multiplier": 1.30, "description": "Major productivity and infrastructure boost"},
            {"name": "Comprehensive Package (+50%)", "multiplier": 1.50, "description": "Full agricultural transformation program"}
        ]
    elif "import" in variable.lower() or "dependency" in variable.lower() or "undernourishment" in variable.lower():
        # For "lower is better" metrics, use reduction multipliers
        policies = [
            {"name": "Moderate Reduction (-15%)", "multiplier": 0.85, "description": "Targeted intervention to reduce dependency"},
            {"name": "Strong Reduction (-30%)", "multiplier": 0.70, "description": "Comprehensive program to reduce dependency"},
            {"name": "Transformative Reduction (-45%)", "multiplier": 0.55, "description": "Full transformation to achieve self-sufficiency"}
        ]
    else:
        # Default policies based on direction
        if direction == "higher":
            policies = [
                {"name": "Policy Intervention (+20%)", "multiplier": 1.20, "description": "Moderate policy intervention"},
                {"name": "Strong Policy (+40%)", "multiplier": 1.40, "description": "Strong policy measures"},
                {"name": "Transformative Policy (+60%)", "multiplier": 1.60, "description": "Transformative policy package"}
            ]
        else:
            policies = [
                {"name": "Policy Intervention (-20%)", "multiplier": 0.80, "description": "Moderate policy intervention"},
                {"name": "Strong Policy (-40%)", "multiplier": 0.60, "description": "Strong policy measures"},
                {"name": "Transformative Policy (-60%)", "multiplier": 0.40, "description": "Transformative policy package"}
            ]
    
    for policy in policies:
        # Apply policy multiplier with gradual increase over time
        policy_forecast = []
        for i, val in enumerate(baseline_forecast):
            # Gradual change: starts at 1.0, reaches multiplier by end
            progress = (i + 1) / len(baseline_forecast)
            current_multiplier = 1.0 + (policy["multiplier"] - 1.0) * progress
            policy_forecast.append(val * current_multiplier)
        
        # Check if this policy achieves the goal
        policy_2030 = policy_forecast[-1]
        if direction == "higher":
            achieves_goal = policy_2030 >= target_value
        else:
            achieves_goal = policy_2030 <= target_value
        
        scenarios.append({
            "name": policy["name"],
            "description": policy["description"],
            "forecast": policy_forecast,
            "forecast_2030": float(policy_2030),
            "achieves_goal": bool(achieves_goal),
            "gap": float(abs(target_value - policy_2030))
        })
    
    return scenarios


@app.route('/api/question/<int:question_id>/scenarios', methods=['POST'])
def generate_scenarios(question_id):
    """Generate what-if scenarios for any variable (achieved or not)"""
    try:
        data = request.json
        variable = data.get('variable')
        horizon = data.get('horizon', 7)
        
        cache_key = f"q{question_id}_{variable}"
        
        if cache_key not in DATA_CACHE:
            return jsonify({"error": "Please train models first"}), 400
        
        cached = DATA_CACHE[cache_key]
        best_model = cached["best_model"]
        y = cached["y"]
        years = cached["years"]
        
        # Generate baseline forecast
        baseline_forecast = best_model.predict(horizon)
        forecast_years = list(range(years[-1] + 1, years[-1] + 1 + horizon))
        
        # Get target info
        df, targets = load_question_data(question_id)
        target_info = targets.get(variable, {})
        target_value = target_info.get("value")
        direction = target_info.get("direction")
        
        # Generate scenarios (optimistic, moderate, pessimistic, transformative)
        scenarios = generate_what_if_scenarios(
            baseline_forecast, forecast_years, target_value, direction, variable, y
        )
        
        return jsonify({
            "baseline_forecast": baseline_forecast.tolist(),
            "forecast_years": forecast_years,
            "scenarios": scenarios,
            "target_value": target_value,
            "direction": direction
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400


def generate_what_if_scenarios(baseline_forecast, forecast_years, target_value, direction, variable, historical_data):
    """Generate what-if scenarios for exploration"""
    scenarios = []
    
    baseline_2030 = baseline_forecast[-1]
    historical_growth = np.mean(np.diff(historical_data[-10:])) if len(historical_data) > 10 else 0
    
    # Scenario 1: Pessimistic (slower growth or faster decline)
    if direction == "higher":
        pessimistic_mult = 0.70  # 30% slower growth
        pessimistic_desc = "Pessimistic: Climate shocks, reduced investment, policy delays"
    else:
        pessimistic_mult = 1.30  # 30% slower improvement
        pessimistic_desc = "Pessimistic: Increased dependency, market challenges"
    
    # Scenario 2: Business as Usual (baseline)
    bau_desc = "Business as Usual: Current trends continue"
    
    # Scenario 3: Moderate Improvement
    if direction == "higher":
        moderate_mult = 1.25  # 25% faster growth
        moderate_desc = "Moderate: Increased investment, better weather, stable policies"
    else:
        moderate_mult = 0.75  # 25% faster improvement
        moderate_desc = "Moderate: Targeted interventions, efficiency gains"
    
    # Scenario 4: Optimistic (strong growth or rapid improvement)
    if direction == "higher":
        optimistic_mult = 1.50  # 50% faster growth
        optimistic_desc = "Optimistic: Major breakthroughs, optimal conditions, strong policies"
    else:
        optimistic_mult = 0.50  # 50% faster improvement
        optimistic_desc = "Optimistic: Comprehensive reforms, technological adoption"
    
    # Scenario 5: Transformative (game-changing)
    if direction == "higher":
        transformative_mult = 1.80  # 80% faster growth
        transformative_desc = "Transformative: Paradigm shift, massive investment, innovation"
    else:
        transformative_mult = 0.30  # 70% faster improvement
        transformative_desc = "Transformative: Complete restructuring, self-sufficiency achieved"
    
    # Generate forecasts for each scenario
    for name, mult, desc in [
        ("Pessimistic", pessimistic_mult, pessimistic_desc),
        ("Business as Usual", 1.0, bau_desc),
        ("Moderate", moderate_mult, moderate_desc),
        ("Optimistic", optimistic_mult, optimistic_desc),
        ("Transformative", transformative_mult, transformative_desc)
    ]:
        scenario_forecast = []
        for i, val in enumerate(baseline_forecast):
            progress = (i + 1) / len(baseline_forecast)
            current_mult = 1.0 + (mult - 1.0) * progress
            scenario_forecast.append(val * current_mult)
        
        scenario_2030 = scenario_forecast[-1]
        
        # Check if achieves goal
        achieves_goal = None
        if target_value is not None:
            if direction == "higher":
                achieves_goal = bool(scenario_2030 >= target_value)
            else:
                achieves_goal = bool(scenario_2030 <= target_value)
        
        gap = abs(target_value - scenario_2030) if target_value is not None else 0
        
        scenarios.append({
            "name": name,
            "description": desc,
            "forecast": scenario_forecast,
            "forecast_2030": float(scenario_2030),
            "achieves_goal": achieves_goal,
            "gap": float(gap),
            "multiplier": mult
        })
    
    return scenarios


@app.route('/api/question/<int:question_id>/summary')
def get_summary(question_id):
    """Get summary of all key indicators and goal achievement"""
    try:
        df, targets = load_question_data(question_id)
        
        summary = {
            "question_id": question_id,
            "indicators": []
        }
        
        for variable, target_info in targets.items():
            if variable in df.columns:
                y = df[variable].dropna().values
                
                # Quick forecast
                model_df, best_model, best_name, _ = adaptive_model_selection(y, verbose=False)
                forecast_values = best_model.predict(7)
                
                target_value = target_info.get("value")
                direction = target_info.get("direction")
                
                goal_achieved = None
                if target_value is not None:
                    forecast_2030 = float(forecast_values[-1])
                    if direction == "higher":
                        goal_achieved = bool(forecast_2030 >= float(target_value))
                    elif direction == "lower":
                        goal_achieved = bool(forecast_2030 <= float(target_value))
                
                summary["indicators"].append({
                    "variable": variable,
                    "current_value": float(y[-1]),
                    "forecast_2030": float(forecast_values[-1]),
                    "target_value": target_value,
                    "goal_achieved": goal_achieved,
                    "best_model": best_name
                })
        
        return jsonify(summary)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("\n" + "="*60)
    print("  Morocco Génération Green - Interactive Dashboard")
    print("="*60)
    print("\n  Starting Flask server...")
    print("  Open your browser and go to: http://127.0.0.1:5000")
    print("\n  Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
