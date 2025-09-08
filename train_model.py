import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor
import pickle


# โหลดข้อมูล
df = pd.read_csv('car.csv')
df['service_history'] = df['service_history'].fillna('None')

df['Car_Age'] = 2025 - df['make_year']

# Features และ Target
scaler = StandardScaler()
features=['mileage_kmpl','engine_cc','owner_count','insurance_valid','accidents_reported','service_history','fuel_type','brand','transmission','color','Car_Age']
target='price_usd'
numerical_cols = df[features].select_dtypes(include=['number']).columns.tolist()
X=df[features]
y=df[target]

# Train-Test Split จะได้: train 60%, val 20%, test 20%
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

cat_model = CatBoostRegressor(
    silent=True,
    random_state=42)
    
cat_features = X_train.select_dtypes(include='object').columns.tolist()
cat_feature_indices = [X_train.columns.get_loc(col) for col in cat_features]
    

best_model = CatBoostRegressor(
    iterations=400,
    learning_rate=0.04,
    depth=4,
    l2_leaf_reg=16,
    cat_features=cat_feature_indices,
    silent=True,
    random_state=42
)

best_model.fit(X_train, y_train)

with open(r"C:/Users/HP/Desktop/web_app/model.pkl", "wb") as f:
    pickle.dump(best_model, f)
