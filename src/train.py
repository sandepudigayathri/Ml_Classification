import pandas as pd
import joblib
from sklearn.tree import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from src.preprocessing import Preprocessing

def Train_data(data_path, model_path):
  data = pd.read_csv(data_path)
  X = data.drop("placed", axis = 1)
  Y = data['placed']

  preprocessor = Preprocessing(['gender','ssc_board','hsc_board','hsc_stream'],['student_id','specialization'],['city_tier'])

  model = Pipeline(
      steps=[
          ('preprocessor', preprocessor),
          ('classifier', RandomForestClassifier())
      ]
  )

  model.fit(X,Y)
  joblib.dump(model, model_path)

  print("model Saved successfully")

