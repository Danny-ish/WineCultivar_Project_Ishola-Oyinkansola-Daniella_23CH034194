from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model & scaler (adjust path if needed during local testing vs Render)
MODEL_PATH = os.path.join('model', 'wine_cultivar_model.pkl')
SCALER_PATH = os.path.join('model', 'scaler.pkl')

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# The exact 6 features in the order used during training
FEATURE_ORDER = [
    'flavanoids',
    'proline',
    'color_intensity',
    'od280/od315_of_diluted_wines',
    'alcohol',
    'total_phenols'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    cultivar_map = {0: "Cultivar 0", 1: "Cultivar 1", 2: "Cultivar 2"}

    if request.method == 'POST':
        try:
            # Get form values (convert to float)
            inputs = [
                float(request.form.get('flavanoids', 0)),
                float(request.form.get('proline', 0)),
                float(request.form.get('color_intensity', 0)),
                float(request.form.get('od280', 0)),
                float(request.form.get('alcohol', 0)),
                float(request.form.get('total_phenols', 0))
            ]

            # Convert to 2D array and scale
            features = np.array([inputs])
            features_scaled = scaler.transform(features)

            # Predict
            pred = model.predict(features_scaled)[0]
            prediction = cultivar_map.get(pred, "Unknown")

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 