import numpy as np

#inputs
temperature = 5
humidity = 60

X = np.array([temperature,humidity])

#nöron  #weights
weights = np.array([0.4,0.6])

#eşik değer
bias = -20

#noron çıktısı(output)

output = np.dot(X,weights) + bias

print("Nöronun ham çıktısı", output)

def sigmoid(x):
    return 1/(1+np.exp(-x))

activated_output = sigmoid(output)

print("Nöronun aktivasyon sonrası çıktısı : ", activated_output)