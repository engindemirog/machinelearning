import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


#ages
X = np.array([5,6,7,8,9,10],dtype=float).reshape(-1,1)

#heights
Y = np.array([110,116,123,130,136,142],dtype=float).reshape(-1,1)

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = x_scaler.fit_transform(X)
Y_scaled = y_scaler.fit_transform(Y)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=10, activation="relu",input_shape=[1]),
    tf.keras.layers.Dense(units=1)
])

model.compile(optimizer="adam",loss="mean_squared_error")

model.fit(X_scaled,Y_scaled,epochs=500, verbose=0)


test_age = np.array([[7.5]])
test_age_scaled = x_scaler.transform(test_age)



predicted_height_scaled = model.predict(np.array([test_age_scaled]))

predicted_height = y_scaler.inverse_transform(predicted_height_scaled)

print(f"{test_age} için boy tahmini = {predicted_height}")