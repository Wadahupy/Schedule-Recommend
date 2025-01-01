from flask import Flask, render_template, request
import pandas as pd
from datetime import timedelta, datetime
import joblib

app = Flask(__name__)

# Load the saved model
model = joblib.load('RandomForestRegressor.pkl')

@app.route('/')
def index():
    today = datetime.now().strftime('%Y-%m-%d')
    return render_template('index.html', current_date=today)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        death_date_input = request.form['death_date']
        death_date = pd.to_datetime(death_date_input)
        today = pd.Timestamp(datetime.now().date())
        
        # Validate death date
        if death_date > today:
            return render_template('index.html', error="Pet Death date cannot be in the future.")
        
        # Prepare test features (ensure consistency with training data)
        test_features = pd.DataFrame({
            'Death Month': [death_date.month],
            'Burial Weekday': [0],  # Dummy placeholder
            'Burial Month': [death_date.month]  # Default assumption
        })

        # Predict days between
        predicted_days_between = model.predict(test_features)[0]

        # Recommend burial date
        recommended_burial_date = death_date + timedelta(days=round(predicted_days_between))

        # Get the weekday of the recommended burial date
        burial_weekday = recommended_burial_date.day_name()

        # Format dates to MM-DD-YYYY
        formatted_death_date = death_date.strftime("%m-%d-%Y")
        formatted_burial_date = recommended_burial_date.strftime("%m-%d-%Y")

        # Render the result
        return render_template(
            'index.html',
            death_date=formatted_death_date,
            predicted_days=round(predicted_days_between),
            burial_date=formatted_burial_date,
            burial_weekday=burial_weekday
        )
    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
