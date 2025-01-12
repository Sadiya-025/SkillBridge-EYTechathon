from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import pandas as pd


filename = './data/Students_Performance.csv'
data = pd.read_csv(filename)

# Encode categorical data
label_encoder = LabelEncoder()
data['parental_level_of_education'] = label_encoder.fit_transform(data['parental_level_of_education'])

# Define features (X) and target (y)
X = data.drop(columns=['average_score'])
y = data['average_score']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R-squared: {r2:.2f}")
# Save the model, scaler, and label encoder

joblib.dump(model, 'student_performance_model.pkl', compress=3)
joblib.dump(scaler, 'scaler.pkl')

feature_columns = X.columns
joblib.dump(feature_columns, 'feature_columns.pkl')