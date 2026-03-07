from flask import Flask, request, jsonify
import joblib 
import numpy as np
import os
import xgboost as xgb

app = Flask(__name__)

# ==========================================
# 1. LOAD THE MACHINE LEARNING MODEL
# ==========================================
try:
    # Use absolute paths so the cloud server never gets lost
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
    MODEL_PATH = os.path.join(BASE_DIR, 'partnex_credibility_model.json')

    # Load the JSON model natively using XGBoost
    model = xgb.XGBClassifier() 
    model.load_model(MODEL_PATH)
    
    print("XGBoost JSON Model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load model. Ensure 'partnex_credibility_model.json' exists. Error: {e}")

    
# ==========================================
# 2. SMART METRICS DEDUCTION ENGINE
# ==========================================
def calculate_smart_metrics(data_payload):
    """
    Deduces dynamic Impact and Consistency scores purely from available financial data.
    This acts as a failsafe if the Node.js backend forgets to send the full profile.
    """
    revenue = float(data_payload.get('revenue', data_payload.get('annual_revenue_amount_1', 0)) or 0)
    expenses = float(data_payload.get('expenses', data_payload.get('monthly_expenses', 0)) or 0)
    debt = float(data_payload.get('debt', data_payload.get('existing_liabilities', 0)) or 0)
    
    # CALCULATE IMPACT SCORE
    impact_score = 0.4 
    
    employees = data_payload.get('number_of_employees')
    if employees is not None and str(employees).isdigit():
        emp_count = int(employees)
        if emp_count >= 20: impact_score += 0.5
        elif emp_count >= 5: impact_score += 0.3
    else:
        if revenue >= 15000000: 
            impact_score += 0.5
        elif revenue >= 5000000: 
            impact_score += 0.3
            
    # CALCULATE REPORTING CONSISTENCY
    consistency = 0.5 
    
    if revenue > 0:
        if expenses < revenue: 
            consistency += 0.25 
        if debt <= (revenue * 0.5): 
            consistency += 0.25 

    final_impact = min(round(impact_score, 2), 1.0)
    final_consistency = min(round(consistency, 2), 1.0)
    
    return final_impact, final_consistency


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
        #    LOGGING: INCOMING DATA
        # ----------------------------------------------------
        print("\n" + "="*40)
        print("INCOMING PAYLOAD FROM NODE.JS:")
        print(data)
        print("="*40 + "\n")
        
        dynamic_impact, dynamic_consistency = calculate_smart_metrics(data)
        
        revenue = float(data.get('revenue', data.get('annual_revenue_amount_1', 0)) or 0)
        expenses = float(data.get('expenses', data.get('monthly_expenses', 0)) or 0)
        debt = float(data.get('debt', data.get('existing_liabilities', 0)) or 0)
        revenue_growth = float(data.get('revenue_growth', 0) or 0)
        
        # Build features array: [Revenue, Expenses, Debt, Growth, Consistency, Impact]
        features = np.array([[revenue, expenses, debt, revenue_growth, dynamic_consistency, dynamic_impact]])
        
        probabilities = model.predict_proba(features)
        risk_prediction = int(model.predict(features)[0])
        
        score = float(round((1.0 - probabilities[0][0]) * 100, 1))
        
        risk_level = "LOW"
        if risk_prediction == 1: 
            risk_level = "MEDIUM"
        elif risk_prediction == 0: 
            risk_level = "HIGH"

        # Construct the final response dictionary
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
                    "reporting_consistency": dynamic_consistency, 
                    "impact_score": dynamic_impact                
                },
                "note": f"Score dynamically generated. Impact factor: {dynamic_impact}, Consistency: {dynamic_consistency}."
            },
            "model_version": "ai-v2.1"
        }

        # ----------------------------------------------------
        # LOGGING: OUTGOING DATA
        # ----------------------------------------------------
        print("\n" + "="*40)
        print("OUTGOING PAYLOAD TO NODE.JS:")
        print(response_data)
        print("="*40 + "\n")

        # Return the exact JSON structure expected by Node.js
        return jsonify(response_data), 200

    except Exception as e:
        print(f"\n❌ Prediction Error: {str(e)}\n")
        return jsonify({"error": "Internal AI Server Error", "details": str(e)}), 500


# ==========================================
# 4. HEALTH CHECK ENDPOINTS
# ==========================================
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "online", "version": "2.1"}), 200

@app.route('/ping', methods=['GET'])
def ping():
    return "Server is awake!", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

