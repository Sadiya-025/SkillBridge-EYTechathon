import random
import pandas as pd
import os, cohere, markdown, joblib
from flask_sqlalchemy import SQLAlchemy
from sklearn.preprocessing import LabelEncoder
from flask import Flask, jsonify, render_template, request, redirect, url_for, session

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.sqlite3'

db=SQLAlchemy(app)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'fallback_secret_key')
co = cohere.Client('9xtb0BFwgzrTgGQIG7QJrdZAUBH8oQoW30EK2Z7I')

model = joblib.load('./models/student_performance_model.pkl')
scaler = joblib.load('./models/scaler.pkl')
feature_columns = joblib.load('./models/feature_columns.pkl')
encoder = LabelEncoder()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    date_joined = db.Column(db.DateTime, default=db.func.current_timestamp())

app.app_context().push()
db.create_all()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        new_user=User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()        
        return redirect(url_for('signin'))
    
    return render_template('sign-up.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Authenticates user
        existing_user=User.query.filter_by(username=username, password=password).first()
        if existing_user:
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            return render_template('not-found.html')
    
    return render_template('sign-in.html')

@app.route('/signout', methods=['GET', 'POST'])
def signout():
    # Clears the session
    session.clear()
    return redirect(url_for('signin'))

@app.route('/about', methods=['GET'])
def about():
    if request.method == 'GET':
        return render_template('about.html')
    
@app.route('/contact', methods=['GET'])
def contact():
    if request.method == 'GET':
        return render_template('contact.html')

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('signin'))
    
    username = session['username']
    return render_template('dashboard.html', username=username)

@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('signin'))
    
    data = {
        'gender': [int(request.form['gender'])],  # 0 or 1
        'parental_level_of_education': [int(request.form['parental_level_of_education'])],  # Categorical
        'lunch': [int(request.form['lunch'])],  # 0 or 1
        'test_preparation_course': [int(request.form['test_preparation_course'])],  # 0 or 1
        'math_score': [int(request.form['math_score'])],  # Numerical
        'reading_score': [int(request.form['reading_score'])],  # Numerical
        'writing_score': [int(request.form['writing_score'])]  # Numerical
    }
    
    new_data = pd.DataFrame(data)

    data['parental_level_of_education'] = encoder.fit_transform(data['parental_level_of_education'])
    print(data)
    new_data_scaled = scaler.transform(new_data)

    # Predict the average score
    average_score_pred = model.predict(new_data_scaled)

    if average_score_pred<=40:
        result="Poor"
    elif average_score_pred>40 and average_score_pred<=70:
        result="Moderate"
    else:
        result="Excellent"
    
    # Generates a personalized health report using Generative AI (Cohere)
    prompt = f"""
    Generate a personalized educational and skill set report based on the following user data:
    Gender: {request.form['Gender']}
    Parental Level of Education: {request.form['parental_level_of_education']}
    Has Lunch: {request.form['lunch']} Yes/No
    Test Preparation Course: {request.form['test_preparation_course']} Taken/Has Not Taken the Test Preparation Course
    Math Score: {request.form['math_score']}
    Reading Score: {request.form['reading_score']} 
    Writing Score: {request.form['writing_score']}

    Provide a detailed skill set report with the following:
    1. Evaluation of current educational qualification based on the user's data.
    2. Personalized educational suggestions and skillset improvement recommendations based on the math, reading, writing scores out of 100.
    3. Long-term skills improvement tips based on the data (including managing current job situation, increasing skills set, educational improvements, etc).
    
    Restructure it with Bullet Points
    """

    response = co.generate(
        model="command-r-plus-08-2024",
        prompt=prompt,
        max_tokens=1000
    )

    recommendations = response.generations[0].text.strip()
    recommendations = markdown.markdown(recommendations)
    username = session['username']

    return render_template('dashboard.html', prediction=result, educational_report=recommendations, username=username)

if __name__ == '__main__':
    app.run(debug=True)