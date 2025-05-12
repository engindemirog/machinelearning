def train_model(model,X_train,y_train,X_test,y_test):
  model.fit(X_train,y_train,epochs=50,validation_data=(X_test,y_test),verbose=1)
  model.save("outputs/model.h5")
  return model