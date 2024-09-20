from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open('diabetes_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    features = [float(x) for x in request.form.values()]
    # Transform features using the scaler
    final_features = scaler.transform([np.array(features)])
    # Make prediction
    prediction = model.predict(final_features)
    
    # Interpret the result
    result = 'Positive for diabetes' if prediction[0] == 1 else 'Negative for diabetes'

    return render_template('index.html', prediction_text=f'Result: {result}')

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
