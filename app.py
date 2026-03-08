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
    
    # Calculate base decimals (0 to 1 scale)
    impact_score = 0.4 
    employees = data_payload.get('number_of_employees')
    if employees is not None and str(employees).isdigit():
        emp_count = int(employees)
        if emp_count >= 20: impact_score += 0.5
        elif emp_count >= 5: impact_score += 0.3
    else:
        if revenue >= 15000000: impact_score += 0.5
        elif revenue >= 5000000: impact_score += 0.3
            
    consistency = 0.5 
    if revenue > 0:
        if expenses < revenue: consistency += 0.25 
        if debt <= (revenue * 0.5): consistency += 0.25 

    # Keep them bounded strictly between 0 and 1 for the payload
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
            
        # ----------------------------------------------------
        # 🪵 LOGGING: INCOMING DATA
        # ----------------------------------------------------
        print("\n" + "="*40)
        print("📥 INCOMING PAYLOAD FROM NODE.JS:")
        print(data)
        print("="*40 + "\n")
        
        # 1. Grab the 0 to 1 decimals
        raw_impact, raw_consistency = calculate_smart_metrics(data)
        
        revenue = float(data.get('revenue', data.get('annual_revenue_amount_1', 0)) or 0)
        expenses = float(data.get('expenses', data.get('monthly_expenses', 0)) or 0)
        debt = float(data.get('debt', data.get('existing_liabilities', 0)) or 0)
        revenue_growth = float(data.get('revenue_growth', 0) or 0)
        
        # ==========================================
        # 🚨 THE SCALE FIX FOR THE AI'S BRAIN
        # ==========================================
        # Scale them UP only for the XGBoost model (0.9 -> 90, 1.0 -> 10)
        ai_impact = int(min(round(raw_impact * 100), 100))
        ai_consistency = int(min(round(raw_consistency * 10), 10))

        # Build features array using the SCALED integers
        features = np.array([[revenue, expenses, debt, revenue_growth, ai_consistency, ai_impact]])
        
        # Predict
        probabilities = model.predict_proba(features)
        risk_prediction = int(model.predict(features)[0])
        
        # Calculate raw score, convert to pure integer, and bound 0-100
        raw_score = (1.0 - probabilities[0][0]) * 100
        score = int(round(raw_score)) 
        score = max(0, min(100, score))
        
        # Map to the 3 classes from Colab training
        if risk_prediction == 2: 
            risk_level = "LOW"
        elif risk_prediction == 1: 
            risk_level = "MEDIUM"
        else: 
            risk_level = "HIGH"

        # ==========================================
        # 🚨 THE DECIMAL FIX FOR THE PAYLOAD
        # ==========================================
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
                    "reporting_consistency": raw_consistency, # Using original 0-1 decimal
                    "impact_score": raw_impact                # Using original 0-1 decimal
                },
                "note": f"Score successfully generated. Impact: {raw_impact}, Consistency: {raw_consistency}."
            },
            "model_version": "ai-v2.3"
        }

        # ----------------------------------------------------
        # 🪵 LOGGING: OUTGOING DATA
        # ----------------------------------------------------
        print("\n" + "="*40)
        print("📤 OUTGOING PAYLOAD TO NODE.JS:")
        print(response_data)
        print("="*40 + "\n")

        return jsonify(response_data), 200

    except Exception as e:
        print(f"\n❌ Prediction Error: {str(e)}\n")
        return jsonify({"error": "Internal AI Server Error", "details": str(e)}), 500


# ==========================================
# 4. HEALTH CHECK / ROOT ENDPOINT
# ==========================================
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "online", "version": "2.3"}), 200

@app.route('/ping', methods=['GET'])
def ping():
    return "Server is awake!", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
