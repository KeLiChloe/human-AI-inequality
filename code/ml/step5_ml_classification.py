import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import pickle

def initial_screen_features_RF(X, y, threshold):
    """
    Select important features based on Random Forest importance scores.

    Parameters:
        X (DataFrame): Feature matrix.
        y (Series): Target variable.
        threshold (float): Minimum importance score to retain a feature.

    Returns:
        DataFrame: Filtered feature matrix.
        list: Selected feature names.
    """
    rf_model = RandomForestClassifier(random_state=random_seed)
    rf_model.fit(X, y)
    feature_importances = rf_model.feature_importances_

    # Create a DataFrame for feature importance
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # Filter features based on importance threshold
    selected_features = importance_df[importance_df['Importance'] > threshold]['Feature']
    X_filtered = X[selected_features]

    print("\nFeature Importance:")
    print(importance_df.head(10))
    print(f"\nSelected {len(selected_features)} features based on importance > {threshold}")

    return X_filtered

def initial_screen_features_lasso(X, y, threshold):
    """
    Select important features based on LASSO coefficients.

    Parameters:
        X (DataFrame): Feature matrix.
        y (Series): Target variable.
        threshold (float): Regularization strength (LASSO parameter).

    Returns:
        DataFrame: Filtered feature matrix.
        list: Selected feature names.
    """

    # Train LASSO model
    random_seed = np.random.randint(10000)
    lasso_model = Lasso(alpha=threshold, random_state=random_seed)
    lasso_model.fit(X, y)

    # Get coefficients and feature importance
    coefficients = lasso_model.coef_
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': coefficients
    }).sort_values(by='Coefficient', key=abs, ascending=False)

    # Filter features based on non-zero coefficients
    selected_features = importance_df[importance_df['Coefficient'] != 0]['Feature']
    X_filtered = X[selected_features]

    print("\nLASSO Feature Coefficients:")
    print(importance_df[importance_df['Coefficient'] != 0])
    print(f"\nSelected {len(selected_features)} features with non-zero coefficients")

    return X_filtered

def tune_hyperparameters(X_train, y_train, model_save_dir, add_second_order_interaction):
    """
    Tune hyperparameters for models using GridSearchCV.

    Parameters:
        X_train (array-like): Features for training.
        y_train (array-like): Labels for training.

    Returns:
        dict: Best hyperparameters for each model.
    """
    # Define hyperparameter grids
    param_grid_rf = {
        "n_estimators": [100, 150],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
    }
    param_grid_gb = {
        "n_estimators": [100, 150],
        "learning_rate": [0.01, 0.05, 0.01],
        "max_depth": [5, 10],
    }

    # Initialize models
    random_seed = np.random.randint(10000)
    models = {
        "random_forest": (RandomForestClassifier(random_state=random_seed), param_grid_rf),
        "gradient_boosting": (GradientBoostingClassifier(random_state=random_seed), param_grid_gb),
        # "cart": (DecisionTreeClassifier(random_state=random_seed), param_grid_cart),
    }

    # Tune models
    best_params = {}
    for model_name, (model, param_grid) in models.items():
        print(f"Tuning {model_name}...")
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring="f1_macro", n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_params[model_name] = grid_search.best_params_
    
    # save best_params
    with open(f"{model_save_dir}/best_params{'_soi' if add_second_order_interaction else ''}.pkl", "wb") as file:
        pickle.dump(best_params, file)
    

    return best_params

# Function to load and prepare data
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    
    # Create the target variable
    df['target'] = df['count_frequency_inequality_words'].apply(lambda x: 1 if x > 0 else 0)
    y = df['target']

    # Select features, dropping non-relevant columns
    drop_columns = ['count_frequency_inequality_words', 'target', 'title', 'paper_abstract', # lables
                    # 'mixed', 'other', 'native_hawaiian_or_other_pacific_islander', 'native_americans',
                    # 'first_author_race_other', 'first_author_race_native_hawaiian_or_other_pacific_islander',
                    ]

    # Select features 
    X = df.drop(columns=drop_columns)  
    return X, y, df

def add_second_order_interactions(X):
    # Step 1: Initialize PolynomialFeatures with degree 2 for second-order interactions
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

    # Step 2: Transform the feature matrix
    X_interactions = poly.fit_transform(X)

    # Step 3: Get the names of the features (optional)
    interaction_feature_names = poly.get_feature_names_out(input_features=X.columns)

    # Step 4: Create a DataFrame with interaction terms
    X = pd.DataFrame(X_interactions, columns=interaction_feature_names)
    return X

# Function to scale data
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Function to train and predict models
def train_and_predict_models(X_train_scaled, y_train, X_test_scaled, best_params=None):
    """
    Train models using the best parameters and make predictions.

    Parameters:
        X_train_scaled (array-like): Scaled training features.
        y_train (array-like): Training labels.
        X_test_scaled (array-like): Scaled test features.
        best_params (dict): Tuned hyperparameters for each model (optional).

    Returns:
        dict: Predicted probabilities for each model.
    """
    random_seed = np.random.randint(10000)
    models = {
        "logistic_regression": LogisticRegression(max_iter=500),
        "random_forest": RandomForestClassifier(random_state=random_seed),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_seed),
        # "cart": DecisionTreeClassifier(random_state=random_seed),
    }

    trained_models = {}
    predictions = {}

    # Update models with tuned hyperparameters
    if best_params:
        for model_name in models.keys():
            if model_name in best_params:
                models[model_name].set_params(**best_params[model_name])

    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[model_name] = model
        predictions[model_name] = model.predict_proba(X_test_scaled)[:, 1]

    return trained_models, predictions


# Function to calculate metrics for varying thresholds
def calculate_metrics(y_test, y_pred_proba):
    thresholds = np.arange(0.0, 1.05, 0.05)
    accuracy, precision, recall, f1 = [], [], [], []
    for thresh in thresholds:
        y_pred_thresh = (y_pred_proba >= thresh).astype(int)
        accuracy.append(accuracy_score(y_test, y_pred_thresh))
        precision.append(precision_score(y_test, y_pred_thresh, zero_division=0))
        recall.append(recall_score(y_test, y_pred_thresh))
        f1.append(f1_score(y_test, y_pred_thresh))
    return thresholds, accuracy, precision, recall, f1

# Function to perform cross-validation on the train set
def cross_validate_with_metrics(X_train, y_train, n_splits=5):
    random_seed = np.random.randint(10000)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    # Initialize dictionaries to store metrics and ROC data

    cv_results = {
    "logistic_regression": {"accuracy": [], "precision": [], "recall": [], "f1": [], "auc": []},
    "random_forest": {"accuracy": [], "precision": [], "recall": [], "f1": [], "auc": []},
    "gradient_boosting": {"accuracy": [], "precision": [], "recall": [], "f1": [], "auc": []},
    # "cart": {"accuracy": [], "precision": [], "recall": [], "f1": [], "auc": []},  # Add CART here
    }

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        

        _, predictions = train_and_predict_models(X_train_fold, y_train_fold, X_val_fold)

        for model_name, y_pred_proba in predictions.items():
            _, acc, prec, rec, f1 = calculate_metrics(y_val_fold, y_pred_proba)
            cv_results[model_name]["accuracy"].append(acc)
            cv_results[model_name]["precision"].append(prec)
            cv_results[model_name]["recall"].append(rec)
            cv_results[model_name]["f1"].append(f1)
            cv_results[model_name]["auc"].append(roc_auc_score(y_val_fold, y_pred_proba))

    # Average metrics across folds
    for model_name, metrics in cv_results.items():
        for metric, values in metrics.items():
            cv_results[model_name][metric] = np.mean(values, axis=0)

    return cv_results

# Function to plot final metrics and ROC curves
def plot_metrics_and_roc(predictions, y_test, save_dir, add_second_order_interaction):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    thresholds = np.arange(0.0, 1.05, 0.05)
    for ax, (model_name, y_pred_proba) in zip(axes[:3], predictions.items()):
        _, acc, prec, rec, f1 = calculate_metrics(y_test, y_pred_proba)
        ax.plot(thresholds, acc, label="Accuracy")
        ax.plot(thresholds, prec, label="Precision")
        ax.plot(thresholds, rec, label="Recall")
        ax.plot(thresholds, f1, label="F1 Score")
        ax.set_title(f"{model_name} (AUC = {roc_auc_score(y_test, y_pred_proba):.3f}) {'With SOI' if add_second_order_interaction else ''}")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Metric Value")
        ax.legend()
        ax.grid()

    # Plot ROC Curve
    ax_roc = axes[3]
    for model_name, y_pred_proba in predictions.items():
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        ax_roc.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.4f})")
    ax_roc.plot([0, 1], [0, 1], 'k--', label="Random Chance")
    ax_roc.set_title("ROC Curve for All Models")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend()
    ax_roc.grid()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/metrics_and_roc{'_soi' if add_second_order_interaction else ''}.jpg")

    plt.show()
    

# Main function
def main(file_path, model_save_dir, add_second_order_interaction, use_best_params):

    # Step 1: Load and prepare data
    X, y, _ = load_and_prepare_data(file_path)
    
    if add_second_order_interaction:
        X = add_second_order_interactions(X)
    
    # X = initial_screen_features_lasso(X, y, threshold=0.0001)

    # Step 2: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    # X_train, X_test = scale_data(X_train, X_test)

    feature_names = X_train.columns.tolist()
    print(f"Number of features: {len(feature_names)}")

    # Step 3: Cross-Validation and Tuning Hyperparameters
    if use_best_params:
        with open(f"{model_save_dir}/best_params{'_soi' if add_second_order_interaction else ''}.pkl", "rb") as file:
            best_params = pickle.load(file)
    else:
        print("Tuning hyperparameters...")
        best_params = tune_hyperparameters(X_train, y_train, model_save_dir, add_second_order_interaction)
        print("\nBest Hyperparameters:")
    for model_name, params in best_params.items():
        print(f"{model_name}: {params}")

    print("\nCross-Validation Results:") 
    cv_results = cross_validate_with_metrics(X_train, y_train)
    for model_name, metrics in cv_results.items():
        print(f"\nModel: {model_name}")
        print(f"AUC (Cross-Validation): {np.mean(metrics['auc']):.4f}")

    # Step 4: Train on full train set and predict on test set
    
    train_models, predictions = train_and_predict_models(X_train, y_train, X_test, best_params)
    
    print("\nTest Set Results:")
    for model_name, y_pred_proba in predictions.items():
        print(f"\nModel: {model_name}")
        print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
        # print(f"Accuracy: {accuracy_score(y_test, y_pred_proba >= 0.5):.4f}")
        # print(f"Precision: {precision_score(y_test, y_pred_proba >= 0.5):.4f}")
        # print(f"Recall: {recall_score(y_test, y_pred_proba >= 0.5):.4f}")
        # print(f"F1 Score: {f1_score(y_test, y_pred_proba >= 0.5):.4f}")


    # Step 5: Plot results based on test set
    plot_metrics_and_roc(predictions, y_test, model_save_dir, add_second_order_interaction)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python step6_ml_classification.py <file_path> <model_save_dir>")
        sys.exit(1)

    file_path = sys.argv[1]
    model_save_dir = sys.argv[2]
    
    add_second_order_interaction = False
    use_best_params = False

    main(file_path, model_save_dir, add_second_order_interaction, use_best_params)

