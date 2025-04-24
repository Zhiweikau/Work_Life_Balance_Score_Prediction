import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
import joblib

df = pd.read_csv('.\Wellbeing_and_lifestyle_data_Kaggle.csv')
df = df.drop_duplicates()
df = df[df['DAILY_STRESS'] != '1/1/00']
df['DAILY_STRESS'] = df['DAILY_STRESS'].astype('int64')
df = df.drop(columns=['Timestamp'])

x = df.drop(columns=['WORK_LIFE_BALANCE_SCORE'])
y = df['WORK_LIFE_BALANCE_SCORE']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

categorical_cols = ["AGE", "GENDER"]
numerical_cols = df.select_dtypes('int64').columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=0))
])

pipeline.fit(x_train, y_train)

# Use joblib for models with heavy Numpy, Pandas or scikit-learn components, Pipelines with StandardScaler, OneHotEncoder, etc. (XGBoost + Scikit-learn pipeline)
joblib.dump(pipeline, "xgb_pipeline_model.joblib")