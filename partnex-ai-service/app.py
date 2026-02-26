from flask import Flask, request, jsonify
import xgboost as xgb
import pandas as pd

app = Flask(__name__)

# 1. Load the model you downloaded from Colab
model = xgb.XGBClassifier()
model.load_model('partnex_credibility_model.json')

# 2. Create the REST API Endpoint
@app.route('/predict', methods=['POST'])
def predict_credibility():
    try:
        # Get the SME data sent from your Node.js backend
        data = request.json
        
        # Convert the JSON data into a format the model understands
        features = pd.DataFrame([data])
        
        # Predict the 0-100 Credibility Score
        probabilities = model.predict_proba(features)
        score = float(round(probabilities[0][2] * 100, 1)) # Probability of "Low Risk" class
        
        # Predict the Risk Level (0=High, 1=Medium, 2=Low)
        risk_prediction = int(model.predict(features)[0])
        risk_labels = ['High', 'Medium', 'Low']
        
        return jsonify({
            'success': True,
            'credibility_score': score,
            'risk_level': risk_labels[risk_prediction]
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)