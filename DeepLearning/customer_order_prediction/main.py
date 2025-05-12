from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate_model


def main():
  print("Loading data")
  df = load_data("data/query.sql")

  print("Preprocessing data")
  X_train,X_test,y_train,y_test =  preprocess_data(df,class_imbalance=2)

  print("Building model")
  model = build_model(input_shape=X_train.shape[1])

  print("Training model")
  model = train_model(model,X_train,y_train, X_test,y_test)

  print("Evaluating model")
  evaluate_model(model,X_test,y_test)



if __name__ =="__main__":
  main()