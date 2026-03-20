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
# 2. THE AI PREDICTION ENDPOINT
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
        
        # 👉 ADDED: Log the incoming payload exactly as Node.js sends it
        print("\n" + "="*40)
        print("INCOMING PAYLOAD FROM NODE.JS:")
        print(data)
        print("="*40 + "\n")
        
        # EXTRACT DIRECTLY FROM NODE.JS PAYLOAD
        raw_impact = float(data.get('impact_score', 0.2))
        raw_consistency = float(data.get('consistency_score', 0.5))
        
        revenue = float(data.get('revenue', data.get('annual_revenue_amount_1', 0)) or 0)
        expenses = float(data.get('expenses', data.get('monthly_expenses', 0)) or 0)
        debt = float(data.get('debt', data.get('existing_liabilities', 0)) or 0)
        revenue_growth = float(data.get('revenue_growth', 0) or 0)
        
        # Scale UP for the XGBoost model
        ai_impact = int(min(round(raw_impact * 100), 100))
        ai_consistency = int(min(round(raw_consistency * 10), 10))

        # 1. Your features array is set up
        features = np.array([[revenue, expenses, debt, revenue_growth, ai_consistency, ai_impact]])
        
        # 2. Support both XGBClassifier and raw Booster models
        try:
            probabilities = model.predict_proba(features)[0]
        except AttributeError:
            import xgboost as xgb
            dmatrix = xgb.DMatrix(features)
            preds = model.predict(dmatrix)
            # Flatten to a 1D list safely
            probabilities = preds[0] if len(preds.shape) > 1 else preds
        
        # THE PADDING FIX: Ensure we always have at least 3 numbers to prevent IndexErrors
        probs_list = list(probabilities)
        while len(probs_list) < 3:
            probs_list.append(0.0)
        
        # 3. Your math formula executes safely using the padded probs_list
        raw_score = (probs_list[0] * 20) + (probs_list[1] * 75) + (probs_list[2] * 100)
        
        score = int(round(raw_score)) 
        score = max(0, min(100, score))
        
        # ==========================================
        # SYNCED RISK LEVEL CATEGORIZATION
        # ==========================================
        if score >= 75:
            risk_level = "LOW"
        elif score >= 50:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        response_data = {
            "credibility_score": score,
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
    return jsonify({"status": "online", "version": "2.5-clean"}), 200

@app.route('/ping', methods=['GET'])
def ping():
    return "Server is awake!", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
