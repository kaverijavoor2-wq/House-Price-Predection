from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load('house_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    area = float(request.form['area'])
    bedrooms = float(request.form['bedrooms'])
    bathrooms = float(request.form['bathrooms'])

    prediction = model.predict([[area, bedrooms, bathrooms]])
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text=f'Estimated Price: {output} Lakhs')

if __name__ == "__main__":
    app.run(debug=True)