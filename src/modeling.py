import ee
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from xgboost import XGBClassifier

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_ee_random_forest(training, feature_bands, num_trees: int = 50, seed: int = 42, label_band: str = "label"):
    """
    Train a random forest classifier.
    Args:
        training: the training data
        feature_bands: the feature bands
        num_trees: the number of trees
        seed: the seed
        label_band: the label band
    Returns:
        classifier: the random forest classifier
    """
    classifier = (
        ee.Classifier.smileRandomForest(numberOfTrees=num_trees, seed=seed).train(
            features=training,
            classProperty=label_band,
            inputProperties=feature_bands,
        )
    )

    return classifier


def baseline_models():
    """
    Define base estimators once to avoid repetition.
    Returns:
        dict: a dictionary of the baseline models
    """
    lr = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    xgb = XGBClassifier(n_estimators=50, eval_metric="logloss", random_state=42)

    return {
        "Logistic_Regression": clone(lr),
        "Random_Forest": clone(rf),
        "XGBoost": clone(xgb),
        "Voting_Classifier": VotingClassifier(
            estimators=[("lr", clone(lr)), ("rf", clone(rf)), ("xgb", clone(xgb))],
            voting="soft",
        ),
    }


def get_custom_cv_splits(df: pd.DataFrame, cv_type: str = "spatial", n_splits: int = 5):
    """
    Get the custom CV splits.
    Args:
        df: the dataframe to split
        cv_type: the type of CV (time-series or spatial)
        n_splits: the number of splits for the spatial CV
    Returns:
        splits: the splits
        df: the sorted dataframe
    """
    df = df.sort_values("year").reset_index(drop=True)
    splits = []

    if cv_type == "time-series":
        years = sorted(df["year"].unique())
        for i in range(1, len(years)):
            train_idx = df[df["year"].isin(years[:i])].index.values
            val_idx = df[df["year"] == years[i]].index.values
            splits.append((train_idx, val_idx))

    elif cv_type == "spatial":
        gk = GroupKFold(n_splits=n_splits)
        for train_idx, val_idx in gk.split(df, df["label"], groups=df["spatial_cluster"]):
            splits.append((train_idx, val_idx))

    return splits, df


def compute_f1score(y_true, y_pred):
    """
    Compute the F1 score.
    Args:
        y_true: the true labels
        y_pred: the predicted labels
    Returns:
        f1_score: the F1 score
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    return report["1"]["f1-score"]  # '1' is the positive class (disturbance)


def evaluate_models(df: pd.DataFrame, feature_cols: list, models: dict, splits: list, cv_name: str):
    """
    Evaluate the models.
    Args:
        df: the dataframe to evaluate
        feature_cols: the feature columns
        models: the models to evaluate
        splits: the splits
        cv_name: the name of the CV
    Returns:
        results: a dictionary of the results of the evaluation
    """

    results = {}

    for name, model in tqdm(models.items(), desc=f"Evaluating models {cv_name} CV"):
        fold_accuracies, fold_f1_scores = [], []

        for train_idx, val_idx in splits:
            X_train, y_train = df.iloc[train_idx][feature_cols], df.iloc[train_idx]["label"]
            X_val, y_val = df.iloc[val_idx][feature_cols], df.iloc[val_idx]["label"]

            # Fit and Predict
            model.fit(X_train, y_train)
            acc = model.score(X_val, y_val)
            y_pred = model.predict(X_val)
            f1 = compute_f1score(y_val, y_pred)

            fold_accuracies.append(acc)
            fold_f1_scores.append(f1)

        avg_acc = np.mean(fold_accuracies)
        std_acc = np.std(fold_accuracies)
        avg_f1 = np.mean(fold_f1_scores)
        std_f1 = np.std(fold_f1_scores)
        results[name] = (avg_acc, avg_f1, std_acc, std_f1)
        print(f"{name}: Average {cv_name} CV Accuracy: {avg_acc:.4f}, Average CV F1-Score: {avg_f1:.4f} (±{std_acc:.4f} acc, ±{std_f1:.4f} f1)")

    return results


def get_spatial_groups(gdf: gpd.GeoDataFrame, n_groups: int = 25):
    """
    Get the spatial groups.
    Args:
        gdf: the GeoDataFrame to group
        n_groups: the number of groups
    Returns:
        list: the spatial groups
    """
    # Project to UTM (meters)
    gdf_utm = gdf.to_crs("EPSG:32633")

    # Extract X and Y coordinates into a 2D array
    coords = np.column_stack((gdf_utm.geometry.x, gdf_utm.geometry.y))

    # Cluster based on meter-distances
    kmeans = KMeans(n_clusters=n_groups, random_state=42, n_init=10)
    groups = kmeans.fit_predict(coords)

    return groups


def tune_models(
    train_val_df,
    feature_cols,
    models,
    n_splits: int = 5,
    param_configs=None,
    scoring: str = "f1",
    cv_type: str = "time-series",
):
    """
    Tune the models.
    Args:
        train_val_df: the training and validation dataframe
        feature_cols: the feature columns
        models: the models to tune
        n_splits: the number of splits for the CV
    Returns:
        results_df: a dataframe of the results of the tuning
            model: the name of the model
            best_cv_score: the best CV score
            best_params: the best parameters
        best_model_name: the name of the best model
        best_model: the best model
    """
    custom_cv, df_sorted = get_custom_cv_splits(train_val_df, cv_type=cv_type, n_splits=n_splits)

    X = df_sorted[feature_cols]
    y = df_sorted["label"]

    model_space = {
        "RandomForestClassifier": (models["Random_Forest"], param_configs.get("Random_Forest")),
        "XGBClassifier": (models["XGBoost"], param_configs.get("XGBoost")),
        "LogisticRegression": (models["Logistic_Regression"], param_configs.get("Logistic_Regression")),
    }

    all_results = []
    best_models = {}

    for name, (estimator, param_dist) in model_space.items():
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_dist,
            cv=custom_cv,
            scoring=scoring,
            n_jobs=-1,
        )

        # This will use our custom CV splits instead of the default KFold
        search.fit(X, y)

        best_models[name] = search.best_estimator_

        all_results.append(
            {
                "model": name,
                "best_cv_score": search.best_score_,
                "best_params": search.best_params_,
            }
        )

    results_df = pd.DataFrame(all_results).sort_values("best_cv_score", ascending=False).reset_index(drop=True)
    best_model_name = results_df.loc[0, "model"]
    best_model = best_models[best_model_name]

    return results_df, best_model_name, best_model


def report_performance(model, name: str, X_test, y_test, features):
    """
    Report the performance of the model.
    Args:
        model: the model to report the performance of
        name: the name of the model
        X_test: the test features
        y_test: the test labels
        features: the feature columns
    Returns:
        y_pred: the predicted labels
    """
    # Predictions and Basic Metrics
    y_pred = model.predict(X_test)
    print(f"\nTest Classification Report for {name}:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    print(f"Test Confusion Matrix for {name}:")
    print(confusion_matrix(y_test, y_pred))

    # Feature Importances (if applicable)
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        print(f"\nTop 5 Feature Importances for {name}:")
        print(importances.head(5))
    elif hasattr(model, "coef_"):  # For Logistic Regression
        importances = pd.Series(model.coef_[0], index=features).sort_values(ascending=False)
        print(f"\nTop 5 Coefficients for {name}:")
        print(importances.head(5))

    return y_pred


def compute_metrics(error_matrix: ee.ConfusionMatrix):
    """
    Compute the metrics.
    Args:
        error_matrix: the confusion matrix
    Returns:
        dict: a dictionary of the metrics
    """
    cm_val = error_matrix.getInfo()
    TN, FP = cm_val[0]
    FN, TP = cm_val[1]
    total = TN + FP + FN + TP

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / total if total > 0 else 0
    kappa = error_matrix.kappa().getInfo()

    return {
        "accuracy": accuracy,
        "kappa": kappa,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }

