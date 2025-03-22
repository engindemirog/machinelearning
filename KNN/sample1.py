import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#midterm,final
data = [
    [30,40],
    [60,70],
    [90,80],
    [20,45],
    [30,49],
    [60,54],
    [90,64],
    [100,78],
    [10,40],
    [20,100],
    [80,60],
    [70,100],
    [70,90],
    [50,80],
    [50,77],
]

def calculate(mid,final):
    average = mid*0.4+final*0.6
    return 1 if average>=50 else 0

labels = [calculate(x[0],x[1]) for x in data]

df = pd.DataFrame(data, columns=["mid","final"])
df["status"] = labels

X = df[["mid","final"]]
y = df["status"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,y_train)

y_prediction = model.predict(X_test)

print(accuracy_score(y_test,y_prediction))

student = np.array([[0,70]])
prediction = model.predict(student)

print("Geçti" if prediction[0]==1 else "Kaldı")