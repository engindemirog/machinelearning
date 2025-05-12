import tensorflow as tf

def build_model(input_shape):

  model = tf.keras.Sequential(
      [
          tf.keras.layers.Dense(8,activation="relu",input_shape=(input_shape,)),
          tf.keras.layers.Dense(4,activation="relu"),
          tf.keras.layers.Dense(1,activation="sigmoid")
      ]
  )

  model.compile(optimizer="adam",loss="mean_squared_error",metrics=["accuracy"])
  return model