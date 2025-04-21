import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


np.random.seed(42)

n_samples = 300
incomes = np.random.uniform(2,12,n_samples)
debpts = np.random.uniform(10,90,n_samples)

labels = []

for income,debpt in zip(incomes,debpts):
    if income<6 and debpt>70:
        labels.append(1)#Riskli
    else:
        labels.append(0)#Güvenli

X = np.column_stack((incomes,debpts))
y = np.array(labels)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=42)

model = SVC(kernel="sigmoid")
model.fit(X_train,y_train)

accuracy = model.score(X_test,y_test)
print("accuracy : ", accuracy)

def plot_decision_boundary(model, X, y):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=60, edgecolors='k', alpha=0.7)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1],
               linestyles=['--', '-', '--'])

    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
               s=150, linewidth=1.5, facecolors='none', edgecolors='k')

    plt.title("Faker Verisiyle SVM: Kredi Riski Tahmini")
    plt.xlabel("Gelir (standardize)")
    plt.ylabel("Borç Oranı (standardize)")
    plt.grid(True)
    plt.show()

plot_decision_boundary(model, X_scaled, y)