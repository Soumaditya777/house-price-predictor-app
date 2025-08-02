from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html', predicted_price=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['bedrooms']),
            float(request.form['bathrooms']),
            float(request.form['sqft_living']),
            float(request.form['sqft_lot']),
            float(request.form['floors']),
            float(request.form['waterfront']),
            float(request.form['view']),
            float(request.form['condition']),
            float(request.form['sqft_above']),
            float(request.form['sqft_basement']),
            float(request.form['yr_built']),
            float(request.form['yr_renovated']),
        ]
        prediction = model.predict([features])[0]
        predicted_price = f"${prediction:,.2f}"
    except Exception as e:
        predicted_price = "Error in input!"

    return render_template('index.html', predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
