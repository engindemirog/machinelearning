import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import random
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

def generateData(m=1000):
    data=[]
    for _ in range(1000):
        age = random.randint(20,65)
        income = round(random.uniform(2.5,15.0),2)
        credit_score = random.randint(300,800)
        has_default = random.choice([0,1])
        approved = 1 if credit_score>650 and income>5 and not has_default else 0
        data.append([age,income,credit_score,has_default,approved])
    return pd.DataFrame(data,columns=["age","income","credit_score","has_default","approved"])

df = generateData()

X = df[["age","income","credit_score","has_default"]]
y = df["approved"]

model = DecisionTreeClassifier(random_state=42)
model.fit(X,y)

joblib.dump(model,"credit_model.pkl")

app = FastAPI(title="Credit Approval API",description="Credit Approval API using Desicion Trees")


class Applicant(BaseModel):
    age:int
    income:float
    credit_score:int
    has_default:int

@app.post("/predict",tags=["prediction"])
def predict_approval(applicant:Applicant):
  data_model = joblib.load("credit_model.pkl")
  input_data = [[applicant.age,applicant.income,applicant.credit_score,applicant.has_default]]
  prediction = data_model.predict(input_data)[0]
  result = "Approved" if prediction==1 else "Rejected"

  return {
      "prediction": result,
      "details":{
          "age" : applicant.age,
          "income": applicant.income,
          "credit_score":applicant.credit_score,
          "has_default":applicant.has_default
      }
  }

#Ödev 1 - ARGE : DesicionTrees'de gini yerine alternatif ne kullanılabilir? Farkı nedir?
#Ödev 2 - ARGE : Pydantic ile başka neler yapılabilir?
#Ödev 3 - ARGE : Faker kütüphanesi ne işe yarar? Detaylı araştırınız.