# scripts/evaluate.py

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
import joblib
import json
import yaml
import os

# оценка качества модели
def evaluate_model():
    # Прочитайте файл с гиперпараметрами params.yaml
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    cv_params = params.get("evaluate", {}).get("cv_params", {})
    scoring = params.get("evaluate", {}).get("scoring", ['f1', 'roc_auc'])
    n_splits = cv_params.get("n_splits", 5)
    random_state = cv_params.get("random_state", 42)
    shuffle = cv_params.get("shuffle", True)

    # Загрузите данные
    data = pd.read_csv('data/initial_data.csv')
    X = data.drop(columns=["target"])
    y = data["target"]

    # Загрузите обученную модель
    # или загрузка модели
    with open('models/fitted_model.pkl', 'rb') as fd:
        model = joblib.load(fd) 


    # Настройка кросс-валидации
    cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    # Выполнение кросс-валидации
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=True)
    
    for key, value in cv_results.items():
        cv_results[key] = round(value.mean(), 3) 

    os.makedirs('cv_results', exist_ok=True) # создание директории, если её ещё нет
    with open("cv_results/cv_res.json", "w") as f:
        json.dump({k: v.tolist() for k, v in cv_results.items()}, f, indent=4)

if __name__ == '__main__':
    evaluate_model()
    