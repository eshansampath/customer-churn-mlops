from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from collections import Counter
import pandas as pd
import joblib

from src.data_loader import load_data
from src.config import DATA_PATH
from src.preprocessing import clean_data
from src.feature_engineering import create_features


def train_model():
    # Load & clean
    df = load_data(DATA_PATH)
    df = clean_data(df)

    # Split
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["Churn"]
    )

    train_df = create_features(train_df)
    test_df = create_features(test_df)

    X_train = pd.get_dummies(train_df.drop("Churn", axis=1))
    y_train = train_df["Churn"]

    X_test = pd.get_dummies(test_df.drop("Churn", axis=1))
    y_test = test_df["Churn"]

    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Imbalance handling
    counter = Counter(y_train)
    scale_pos_weight = counter[0] / counter[1]

    model = XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )

    param_dist = {
        "n_estimators": [300, 500, 800],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.6, 0.7, 0.8],
        "colsample_bytree": [0.6, 0.7, 0.8],
        "gamma": [0, 0.1, 0.3],
        "min_child_weight": [1, 3, 5],
        "reg_alpha": [0, 0.1, 1],
        "reg_lambda": [1, 5, 10]
    }

    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=50,
        scoring="f1",
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    print("Best Params:", search.best_params_)

    # Early stopping
    best_model.set_params(early_stopping_rounds=20)

    best_model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    print("Best iteration:", best_model.best_iteration)

    # Save
    joblib.dump(best_model, "models/model.pkl")
    joblib.dump(X_train.columns, "models/columns.pkl")

    return best_model, X_test, y_test, X_train, search.best_params_