import joblib
import pandas as pd

def predict(data,model_path):
  model = joblib.load(model_path)
  return model.predict(data)