import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from flask import Flask,request,jsonify

#1 eğitim seviyesi 0=lise , 1 lisans, 2 YL
#2 tecrübe yılı
#3 hired?
data = [
    [0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0],
    [1, 0, 0], [1, 2, 0], [1, 2, 1], [1, 2, 0],
    [1, 4, 1], [1, 5, 1], [2, 0, 0], [2, 1, 1],
    [2, 2, 1], [2, 3, 1], [2, 4, 1], [2, 5, 1],
    [2, 6, 1], [2, 7, 1], [2, 8, 1], [2, 9, 1]
]

df = pd.DataFrame(data, columns=["school","year","hired"])

X = df[["school","year"]] #features
y = df["hired"] #target

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X,y)

joblib.dump(model,"knn_model.pkl")


app = Flask(__name__) #json,restful


model = joblib.load("knn_model.pkl")

@app.route("/")
def home():
    return "KNN API hazır 🚀"    #http://localhost:5000/

@app.route("/prediction", methods=["POST"]) #http://localhost:5000/prediction  POST
def predict():
    data = request.get_json()
    try:
        school = int(data["school"])
        year = int(data["year"])

        testData = np.array([[school, year]])
        result = model.predict(testData)[0]

        return jsonify({
            "school": school,
            "year": year,
            "hired": "Alındı" if result == 1 else "Alınmadı"
        })

    except Exception as e:
        return jsonify({"hata": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)