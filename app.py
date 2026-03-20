from flask import Flask, request, jsonify
import numpy as np
import os
import xgboost as xgb

app = Flask(__name__)

# ==========================================
# 1. LOAD THE MACHINE LEARNING MODEL
# ==========================================
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
    MODEL_PATH = os.path.join(BASE_DIR, 'partnex_credibility_model.json')

    model = xgb.XGBClassifier() 
    model.load_model(MODEL_PATH)
    print("XGBoost JSON Model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load model. Error: {e}")

# ==========================================
# 2. SMART METRICS DEDUCTION ENGINE
# ==========================================
def calculate_smart_metrics(data_payload):
    revenue = float(data_payload.get('revenue', data_payload.get('annual_revenue_amount_1', 0)) or 0)
    expenses = float(data_payload.get('expenses', data_payload.get('monthly_expenses', 0)) or 0)
    debt = float(data_payload.get('debt', data_payload.get('existing_liabilities', 0)) or 0)
    
    # ----------------------------------------------------
    # NEW MULTI-FACTOR IMPACT SCORE
    # ----------------------------------------------------
    impact_score = 0.2 # Baseline score for all SMEs
    
    # Factor 1: Job Creation (Up to +0.3)
    employees = data_payload.get('number_of_employees')
    if employees is not None and str(employees).isdigit():
        emp_count = int(employees)
        if emp_count >= 50: impact_score += 0.3
        elif emp_count >= 20: impact_score += 0.2
        elif emp_count >= 5: impact_score += 0.1

    # Factor 2: Economic Footprint / Revenue (Up to +0.3)
    if revenue >= 50000000: impact_score += 0.3      # 50M+ Naira
    elif revenue >= 15000000: impact_score += 0.2    # 15M+ Naira
    elif revenue >= 5000000: impact_score += 0.1     # 5M+ Naira

    # Factor 3: Industry Sector / Purpose (Up to +0.2)
    sector = str(data_payload.get('industry_sector', '')).lower()
    
    high_impact_sectors = ['health', 'education', 'agriculture', 'farming', 'clean energy']
    medium_impact_sectors = ['manufacturing', 'technology', 'logistics', 'fintech']
    
    if any(keyword in sector for keyword in high_impact_sectors):
        impact_score += 0.2
    elif any(keyword in sector for keyword in medium_impact_sectors):
        impact_score += 0.1
    
    # ----------------------------------------------------
    # REPORTING CONSISTENCY SCORE
    # ----------------------------------------------------
    consistency = 0.5 
    if revenue > 0:
        if expenses < revenue: consistency += 0.25 
        if debt <= (revenue * 0.5): consistency += 0.25 

    # Strictly cap both at a maximum of 1.0
    raw_impact = min(round(impact_score, 2), 1.0)
    raw_consistency = min(round(consistency, 2), 1.0)
    
    return raw_impact, raw_consistency

# ==========================================
# 3. THE AI PREDICTION ENDPOINT
# ==========================================
@app.route('/score', methods=['POST'], strict_slashes=False)
@app.route('//score', methods=['POST'], strict_slashes=False)
@app.route('/predict', methods=['POST'], strict_slashes=False)
@app.route('/api/score', methods=['POST'], strict_slashes=False)
def predict_score():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # EXTRACT DIRECTLY FROM NODE.JS PAYLOAD
        raw_impact = float(data.get('impact_score', 0.2))
        raw_consistency = float(data.get('consistency_score', 0.5))
        
        revenue = float(data.get('revenue', data.get('annual_revenue_amount_1', 0)) or 0)
        expenses = float(data.get('expenses', data.get('monthly_expenses', 0)) or 0)
        debt = float(data.get('debt', data.get('existing_liabilities', 0)) or 0)
        revenue_growth = float(data.get('revenue_growth', 0) or 0)
        
        # Scale UP for the XGBoost model exactly as before
        ai_impact = int(min(round(raw_impact * 100), 100))
        ai_consistency = int(min(round(raw_consistency * 10), 10))

        features = np.array([[revenue, expenses, debt, revenue_growth, ai_consistency, ai_impact]])
        # 1. Your features array is set up
        features = np.array([[revenue, expenses, debt, revenue_growth, ai_consistency, ai_impact]])
        
        # THE FIX: Support both XGBClassifier and raw Booster models!
        try:
            probabilities = model.predict_proba(features)[0]
        except AttributeError:
            import xgboost as xgb
            dmatrix = xgb.DMatrix(features)
            preds = model.predict(dmatrix)
            # Flatten to a 1D list safely
            probabilities = preds[0] if len(preds.shape) > 1 else preds
        
        # THE PADDING FIX: Ensure we always have at least 3 numbers to prevent IndexErrors!
        probs_list = list(probabilities)
        while len(probs_list) < 3:
            probs_list.append(0.0)
        
        # 3. Your math formula executes flawlessly
        raw_score = (probs_list[0] * 20) + (probs_list[1] * 75) + (probs_list[2] * 100)
        
        # 3. Your math formula executes flawlessly
        raw_score = (probabilities[0] * 20) + (probabilities[1] * 75) + (probabilities[2] * 100)
        
        
        # ==========================================
        # THE FAIR MARKET SCORE WEIGHTING (STRICT)
        # ==========================================
        raw_score = (probabilities[0] * 20) + (probabilities[1] * 75) + (probabilities[2] * 100)
        
        score = int(round(raw_score)) 
        score = max(0, min(100, score))
        
        # ==========================================
        # SYNCED RISK LEVEL CATEGORIZATION
        # ==========================================
        if score >= 70:
            risk_level = "LOW"
        elif score >= 40:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        response_data = {
            "credibility_score": score,
            "credible_class": risk_prediction,
            "risk_level": risk_level,
            "explanation": {
                "source": "ai-service",
                "model_inputs": {
                    "revenue": revenue,
                    "expenses": expenses,
                    "debt": debt,
                    "revenue_growth": revenue_growth,
                    "reporting_consistency": raw_consistency, 
                    "impact_score": raw_impact                
                },
                "note": f"Score successfully generated. Impact: {raw_impact}, Consistency: {raw_consistency}."
            },
            "model_version": "ai-v2.2"
        }

        print("\n" + "="*40)
        print("OUTGOING PAYLOAD TO NODE.JS:")
        print(response_data)
        print("="*40 + "\n")

        return jsonify(response_data), 200

    except Exception as e:
        print(f"\n Prediction Error: {str(e)}\n")
        return jsonify({"error": "Internal AI Server Error", "details": str(e)}), 500

# ==========================================
# HEALTH CHECK / ROOT ENDPOINT
# ==========================================
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "online", "version": "2.4-strict"}), 200

@app.route('/ping', methods=['GET'])
def ping():
    return "Server is awake!", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
