import optuna
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def tune_random_forest(X, y):

    def objective(trial):

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5)
        }

        model = RandomForestClassifier(**params)

        score = cross_val_score(
            model,
            X,
            y,
            cv=5,
            scoring="accuracy"
        ).mean()

        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    best_params = study.best_params

    best_model = RandomForestClassifier(**best_params)
    best_model.fit(X, y)

    return best_model, best_params


def tune_xgboost(X, y):

    def objective(trial):

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0)
        }

        model = XGBClassifier(
            eval_metric="logloss",
            **params
        )

        score = cross_val_score(
            model,
            X,
            y,
            cv=5,
            scoring="accuracy"
        ).mean()

        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    best_params = study.best_params

    best_model = XGBClassifier(
        eval_metric="logloss",
        **best_params
    )

    best_model.fit(X, y)

    return best_model, best_params