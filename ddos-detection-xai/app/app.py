from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import shap

app = Flask(__name__)

# Load models
rf = joblib.load('../models/trained_rf_model.pkl')
lstm = load_model('../models/trained_lstm_model.h5')
scaler = joblib.load('../models/scaler.pkl')

def generate_shap_plot(sample):
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(sample)
    shap.force_plot(explainer.expected_value[1], shap_values[1], sample, matplotlib=True, show=False)
    plt.savefig('app/static/shap_plots/plot.png', bbox_inches='tight')
    plt.close()

@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    scaled_data = scaler.transform([data])
    
    # RF Prediction
    rf_prob = rf.predict_proba(scaled_data)[0][1]
    
    # LSTM Prediction
    lstm_input = scaled_data.reshape(1, 1, len(data))
    lstm_prob = lstm.predict(lstm_input)[0][0]
    
    # Ensemble
    final_prob = (rf_prob + lstm_prob) / 2
    generate_shap_plot(scaled_data[0])
    
    return jsonify({
        'status': 'DDoS' if final_prob > 0.5 else 'Normal',
        'confidence': float(final_prob),
        'shap_plot': '/static/shap_plots/plot.png'
    })

if __name__ == '__main__':
    app.run(debug=True)

    