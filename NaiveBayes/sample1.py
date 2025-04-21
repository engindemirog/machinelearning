import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = {
    'text': [
        'Kredi kartı borcunuzu hemen ödeyin',
        'Tebrikler! Kazandınız. Hemen tıklayın!', 
        'Yarın toplantıyı unutma',
        'Bedava hediye seni bekliyor',
        'Önemli bir fatura bildirimi var',
        'Bu hafta sonu kahve içelim mi?',
        'Ücretsiz tatil kazandınız!',
        'Bu ay çok çalıştın, tebrikler'
    ],
    'label': [1, 1, 0, 1, 0, 0, 1, 0]  # 1: spam, 0: normal
}

df =pd.DataFrame(data)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])

y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = MultinomialNB()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("Accuracy : ", accuracy_score(y_test,y_pred))