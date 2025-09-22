# app.py
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template, request
import psycopg2
from psycopg2.extras import RealDictCursor

# Connect to PostgreSQL (replace with your Render credentials)
conn = psycopg2.connect(
    host="dpg-d38b8g15pdvs738ec890-a.render.com",
    port=5432,
    user="customer_database_90u1_user",
    password="0K54IYnIXIlgtEEJINsEO2DKGjr5JlEy",
    dbname="customer_database"
)

# Load data into pandas
query = 'SELECT * FROM customers'
df = pd.read_sql(query, conn)

# Feature engineering
df['LastPurchaseDate'] = pd.to_datetime(df['LastPurchaseDate'])
df['DaysSinceLastPurchase'] = (datetime.today() - df['LastPurchaseDate']).dt.days
df['Churn'] = df['Churn'].astype(int)
df['AgeGroup'] = pd.cut(df["Age"], bins=[17,25,35,50,70,100], labels=[0,1,2,3,4])
df['PurchaseFrequency'] = df['purchaseHistory'] / (df['DaysSinceLastPurchase'] + 1)

# Features & labels
X = df[['Age','AgeGroup','DaysSinceLastPurchase','PurchaseFrequency']]
Y = df['Churn']
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Prediction function
def predict_churn(age, last_purchase_date, purchase_history):
    if age < 18 or age > 99:
        return "Invalid", "Please input an age between 18 and 99"
    
    last_purchase_date = pd.to_datetime(last_purchase_date)
    days_since_last = (datetime.today() - last_purchase_date).days
    age_group = pd.cut([age], bins=[17,25,35,50,70,100], labels=[0,1,2,3,4]).astype(int)[0]
    purchase_freq = purchase_history / (days_since_last + 1)
    
    input_data = pd.DataFrame([{
        'Age': age,
        'AgeGroup': age_group,
        'DaysSinceLastPurchase': days_since_last,
        'PurchaseFrequency': purchase_freq
    }])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    result = 'Likely to Churn' if prediction == 1 else "Not likely to Churn"
    return result, round(probability, 2)

# Flask app
app = Flask(__name__)

@app.route("/")
def home():
    table_html = df.to_html(classes='table table-striped', index=False)
    return render_template("home.html", table=table_html)

@app.route("/predict", methods=['POST'])
def predict():
    age = int(request.form["Age"])
    last_purchase_date = request.form["lastPurchaseDate"]
    purchase_history = int(request.form["purchase_history"])
    
    result, prob = predict_churn(age, last_purchase_date, purchase_history)
    
    if result == "Invalid":
        return render_template("result.html",
                               age=age,
                               last_purchase_date=last_purchase_date,
                               purchase_history=purchase_history,
                               result=prob,
                               probability="N/A")
    
    return render_template("result.html",
                           age=age,
                           last_purchase_date=last_purchase_date,
                           purchase_history=purchase_history,
                           result=result,
                           probability=prob)

if __name__ == "__main__":
    app.run(debug=True)
