# scripts/fit.py

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders import CatBoostEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
import yaml
import os
import joblib

# обучение модели
def fit_model():
    # Прочитайте файл с гиперпараметрами params.yaml
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    model_params = params.get("train", {}).get("model_params", {})
    random_state = params.get("train", {}).get("random_state", 42)
    
    # Загрузите результат предыдущего шага: initial_data.csv
    data_path = os.path.join("data/initial_data.csv")
    df = pd.read_csv(data_path)

    # Предположим, что 'target' — это целевая переменная
    X = df.drop(columns=["target"])
    y = df["target"]

    # Определим числовые и категориальные признаки
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Преобразования признаков
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(), categorical_features),
        ]
    )

    model = LogisticRegression(**model_params, random_state=random_state, verbose=0)
    
    # Создание pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Обучение модели
    pipeline.fit(X, y)

    # Сохранение модели
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, os.path.join("models/fitted_model.pkl"))


if __name__ == '__main__':
	fit_model()