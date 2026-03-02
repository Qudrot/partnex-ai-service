from flask import Flask, request, jsonify
import xgboost as xgb
import pandas as pd
import os

app = Flask(__name__)

print("Loading XGBoost Model V2...")
# Forces XGBoost to only use 1 CPU thread to prevent Docker deadlocks
model = xgb.XGBClassifier(n_jobs=1)
# Make sure you are using the V2 model trained on the 6 new columns!
model.load_model('partnex_credibility_model.json')

# FIX 1: Match the backend's hardcoded "/score" route
@app.route('/score', methods=['POST'])
def predict_credibility():
    try:
        # Get the SME data sent from Node.js
        data = request.json
        print(f"Incoming Payload: {data}")
        
        # Convert to DataFrame
        features = pd.DataFrame([data])
        
        # Predict the 0-100 Credibility Score
        probabilities = model.predict_proba(features)
        score = float(round(probabilities[0][2] * 100, 1))
        
        # Predict the Risk Level (returns 0, 1, or 2)
        risk_prediction = int(model.predict(features)[0])
        
        # FIX 2: Send the integer instead of text so Node.js Number() doesn't crash
        response_data = {
            'credibility_score': score,
            'credible_class': risk_prediction 
        }
        
        print(f"Sending Response: {response_data}")
        return jsonify(response_data), 200

    except Exception as e:
        print(f"AI CRASHED: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Render assigns a dynamic port, and host='0.0.0.0' opens it to the internet
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)


