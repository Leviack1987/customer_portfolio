import pandas as pd
import mysql.connector
from datetime import datetime
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score
from flask import Flask, render_template, request

#Connect to sql
conn=mysql.connector.connect(
    host='localhost',
    port=3306,
    user='root',
    password='Halomaster17*',
    database='portfolio_db'
)

#load into panda translator
query='SELECT * FROM customers'
df=pd.read_sql(query,conn)

#feature engineering
df['LastPurchaseDate']=pd.to_datetime(df['LastPurchaseDate'])
df['DaysSinceLastPurchase']=(datetime.today()-df['LastPurchaseDate']).dt.days
df['Churn']=df['Churn'].astype(int)
df['AgeGroup']=pd.cut(df["Age"],bins=[17,25,35,50,70,99], labels=[0,1,2,3,4])
df['PurchaseFrequency']=df['purchaseHistory']/(df['DaysSinceLastPurchase']+1)
#features
X=df[['Age','AgeGroup','DaysSinceLastPurchase','PurchaseFrequency']]
Y=df['Churn']
#convert into numeric
X=X.apply(pd.to_numeric,errors='coerce').fillna(0)
#put into training
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
#learning
model=LogisticRegression()
model.fit(X_train,Y_train)

#now make an input function
def predict_churn(age,last_purchase_date,purchase_history):
    if age<18 or age>99:
        return "Invalid", "Please input an age between 18 and 99"
    last_purchase_date=pd.to_datetime(last_purchase_date)
    daysSinceLastPurchase=(datetime.today()-last_purchase_date).days
    age_group=pd.cut([age],bins=[17,25,35,50,70,99], labels=[0,1,2,3,4]).astype(int)[0]
    purchase_frequency=purchase_history/(daysSinceLastPurchase+1)

    input_data=pd.DataFrame([{
        'Age':age,
        'AgeGroup':age_group,
        'DaysSinceLastPurchase':daysSinceLastPurchase,
        'PurchaseFrequency':purchase_frequency
    }])

    prediction=model.predict(input_data)
    probability=model.predict_proba(input_data)[0][1]

    result='Likely to Churn' if prediction==1 else "Not likely to Churn"
    return result,round(probability,2)

#FLask section
app=Flask(__name__)

@app.route("/")
def home():
    data_html=df.to_html(classes='table table-striped',index=False)
    return render_template("home.html",table=data_html)

@app.route("/predict",methods=['POST'])
def predict():
    age=int(request.form["Age"])
    last_purchase_date=request.form["lastPurchaseDate"]
    purchase_history=int(request.form["purchase_history"])
    
    result,prob=predict_churn(age,last_purchase_date,purchase_history)
    if result=="Invalid":
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

if __name__=="__main__":
    app.run(debug=True)