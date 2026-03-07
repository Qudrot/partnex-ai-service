from flask import Flask, request, jsonify
import joblib 
import numpy as np

app = Flask(__name__)

# ==========================================
# 1. LOAD THE MACHINE LEARNING MODEL
# ==========================================
import os
import xgboost as xgb

try:
    # Use absolute paths so the cloud server never gets lost
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
    MODEL_PATH = os.path.join(BASE_DIR, 'partnex_credibility_model.json')

    # Load the JSON model natively using XGBoost
    model = xgb.XGBClassifier() # Use XGBRegressor() if your model predicts a continuous number
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
    # Grab whatever financial numbers Node.js sent us (handling both camelCase and snake_case)
    revenue = float(data_payload.get('revenue', data_payload.get('annual_revenue_amount_1', 0)))
    expenses = float(data_payload.get('expenses', data_payload.get('monthly_expenses', 0)))
    debt = float(data_payload.get('debt', data_payload.get('existing_liabilities', 0)))
    
    # ---------------------------------------------------------
    # CALCULATE IMPACT SCORE (Socioeconomic Value)
    # ---------------------------------------------------------
    impact_score = 0.4 # Base score for a registered micro-business
    
    # Did Node.js secretly send employees? If yes, use it!
    employees = data_payload.get('number_of_employees')
    if employees is not None and str(employees).isdigit():
        emp_count = int(employees)
        if emp_count >= 20: impact_score += 0.5
        elif emp_count >= 5: impact_score += 0.3
    else:
        # FALLBACK: Use Revenue Size as a proxy for job creation / socioeconomic impact
        if revenue >= 15000000: # 15M+ Naira 
            impact_score += 0.5
        elif revenue >= 5000000: # 5M+ Naira
            impact_score += 0.3
            
    # ---------------------------------------------------------
    # CALCULATE REPORTING CONSISTENCY (Data Trustworthiness)
    # ---------------------------------------------------------
    consistency = 0.5 # Base score for basic manual entry
    
    if revenue > 0:
        # Sanity Check 1: Do expenses make sense? (Healthy businesses track expenses properly)
        if expenses < revenue: 
            consistency += 0.25 
        
        # Sanity Check 2: Are liabilities clearly tracked and manageable?
        if debt <= (revenue * 0.5): 
            consistency += 0.25 

    # Ensure scores never go above 1.0 (Maximum AI limit)
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
        # 1. Get the JSON payload sent by Node.js
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # 2. Generate the dynamic metrics (Overwriting the 0.9/0.7 hardcoded Node.js values)
        dynamic_impact, dynamic_consistency = calculate_smart_metrics(data)
        
        # 3. Extract the standard financial numbers safely
        revenue = float(data.get('revenue', data.get('annual_revenue_amount_1', 0)))
        expenses = float(data.get('expenses', data.get('monthly_expenses', 0)))
        debt = float(data.get('debt', data.get('existing_liabilities', 0)))
        revenue_growth = float(data.get('revenue_growth', 0))
        
        # 4. Build the exact array your XGBoost model expects
        # MUST MATCH TRAINING COLUMNS: [Revenue, Expenses, Debt, Growth, Consistency, Impact]
        features = np.array([[revenue, expenses, debt, revenue_growth, dynamic_consistency, dynamic_impact]])
        
        # 5. Run the Prediction
        probabilities = model.predict_proba(features)
        risk_prediction = int(model.predict(features)[0])
        
        # Calculate final 0-100 score based on the probability of NOT being high risk (Class 0)
        score = float(round((1.0 - probabilities[0][0]) * 100, 1))
        
        # Map the class to a readable Risk Level string
        risk_level = "LOW"
        if risk_prediction == 1: 
            risk_level = "MEDIUM"
        elif risk_prediction == 0: 
            risk_level = "HIGH"

        # 6. Return the exact JSON structure the Flutter app is waiting for
        return jsonify({
            "score": score,
            "risk_level": risk_level,
            "explanation": {
                "source": "ai-service",
                "credible_class": risk_prediction,
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
        }), 200

    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        return jsonify({"error": "Internal AI Server Error", "details": str(e)}), 500


# ==========================================
# 4. HEALTH CHECK / ROOT ENDPOINT
# ==========================================
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "online",
        "service": "Partnex AI Scoring Engine",
        "version": "2.1"
    }), 200


if __name__ == '__main__':
    # Runs on port 5000 by default. Adjust if your hosting provider requires a different port.
    app.run(host='0.0.0.0', port=5000, debug=True)
    
@app.route('/ping', methods=['GET'])
def ping():
    return "Server is awake!", 200



