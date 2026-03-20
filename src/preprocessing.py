from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,OrdinalEncoder

def Preprocessing(one_hot_enc_columns,drop_cols,ord_cols):
  preprocessor = ColumnTransformer(
      transformers=[
          ('cat', OneHotEncoder(handle_unknown='ignore',dtype='int64'),one_hot_enc_columns),
          ('drop_col', 'drop', drop_cols),
         ('ord', OrdinalEncoder(categories=[['Tier 1','Tier 2','Tier 3']]),ord_cols)
    ],
      remainder='passthrough'
  )
  return preprocessor