import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Sample dataset
data = {
    'area': [1000, 1200, 1500, 1800, 2000, 2200, 2500],
    'bedrooms': [2, 2, 3, 3, 4, 4, 5],
    'bathrooms': [1, 2, 2, 3, 3, 4, 4],
    'price': [30, 35, 45, 55, 65, 75, 90]  # in lakhs
}

df = pd.DataFrame(data)

X = df[['area', 'bedrooms', 'bathrooms']]
y = df['price']

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, 'house_model.pkl')

print("Model trained and saved!")