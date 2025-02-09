import pandas as pd
from sklearn.linear_model import Lasso
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import pickle
import matplotlib.cm as cm
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def initial_screen_features_RF(X, y, threshold=0.01):
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
    rf_model = RandomForestClassifier(random_state=42)
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

def initial_screen_features_lasso(X, y, alpha=0.001):
    """
    Select important features based on LASSO coefficients.

    Parameters:
        X (DataFrame): Feature matrix.
        y (Series): Target variable.
        alpha (float): Regularization strength (LASSO parameter).

    Returns:
        DataFrame: Filtered feature matrix.
        list: Selected feature names.
    """

    # Train LASSO model
    lasso_model = Lasso(alpha=alpha, random_state=42)
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

def tune_hyperparameters(X_train, y_train, subset_id, model_save_dir):
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
    models = {
        "random_forest": (RandomForestClassifier(random_state=42), param_grid_rf),
        "gradient_boosting": (GradientBoostingClassifier(random_state=42), param_grid_gb),
    }

    # Tune models
    best_params = {}
    for model_name, (model, param_grid) in models.items():
        print(f"Tuning {model_name}...")
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring="f1_macro", n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_params[model_name] = grid_search.best_params_
    
    # save best_params
    with open(f"{model_save_dir}/best_params_subset_{subset_id}.pkl", "wb") as file:
        pickle.dump(best_params, file)
    

    return best_params

# Function to scale data
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train, X_test

# load and prepare data with second-order interactions, and do initail feature screening
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    
    # Create the target variable
    df['target'] = df['count_frequency_inequality_words'].apply(lambda x: 1 if x > 0 else 0)
    y = df['target']

    # Select features, dropping non-relevant columns
    drop_columns = ['count_frequency_inequality_words', 'target', 'title', 'paper_abstract', # lables
                    'mixed', 'other', 'native_hawaiian_or_other_pacific_islander', 'native_americans',
                    'first_author_race_other', 'first_author_race_native_hawaiian_or_other_pacific_islander',
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

def main(file_path, load_existing_best_params, params_pkl_dir, model_name, model_save_dir, second_order_interaction):
    """
    Main function to find feature importance rankings robustly using shuffled and split subsets.

    Parameters:
        file_path (str): Path to the dataset.
        model_save_dir (str): Directory to save model parameters.

    Returns:
        DataFrame: Robust feature importance ranking.
    """
    # Step 1: Load and prepare data with second-order interactions
    X, y, _ = load_and_prepare_data(file_path)
    if second_order_interaction:
        X = add_second_order_interactions(X)
    
    X = initial_screen_features_RF(X, y, 0.01)
    
    print(f"Dataset shape: {X.shape}")

    # Step 2: Shuffle the entire dataset
    random_seed = np.random.randint(100000)
    dataset = pd.concat([X, y], axis=1).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    X = dataset.iloc[:, :-1]  # Features
    y = dataset.iloc[:, -1]   # Target
    
    # run a simple logistic regression model and check the coefficients
    lr_model = LogisticRegression(random_state=random_seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    lr_model.fit(X_train, y_train)
    
    print("\nLogistic Regression Coefficients:")
    coefficients = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': lr_model.coef_[0]
    }).sort_values(by='Coefficient', key=abs, ascending=True)
    
    # test the performance
    y_pred = lr_model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    
        # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 8))

    # Separate positive and negative coefficients
    coefficients['Sign'] = np.where(coefficients['Coefficient'] > 0, 1, -1)

    # Bar plot with direction
    for index, row in coefficients.iterrows():
        if index <= 15:
            coef = row['Coefficient']
            feature = row['Feature']
            color = 'blue' if coef > 0 else 'red'  # Choose colors for positive and negative

            ax.barh(feature, coef, color=color)

    # Add a vertical line at 0 to separate positive and negative contributions
    ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8)

    # Add labels and title
    ax.set_xlabel('Coefficient Value', fontsize=14)
    ax.set_ylabel('Features', fontsize=14)
    ax.set_title(f"Logistic Regression {'With SOI' if second_order_interaction else ''}", fontsize=16, fontweight='bold')

    # Improve layout and readability
    plt.tight_layout()
    plt.savefig(f"{model_save_dir}/lr_feature_importance{'_soi' if second_order_interaction else ''}.png")
    plt.show()


    # Step 3: Split the shuffled dataset into 10 subsets
    subsets_X = np.array_split(X, 10)
    subsets_y = np.array_split(y, 10)

    # Initialize a list to store feature importance scores
    all_feature_importances = []

    # Step 4: Iterate over the subsets
    for i in range(10):
        print(f"\nProcessing subset {i + 1}...")

        # Create training and test sets
        X_split = subsets_X[i]
        y_split = subsets_y[i]

        # Step 5: Tune hyperparameters on the training set
        if load_existing_best_params:
            with open(f"{params_pkl_dir}/best_params_subset_{i}.pkl", "rb") as file:
                best_params = pickle.load(file)
        else:
            best_params = tune_hyperparameters(X_split, y_split, i, model_save_dir)

        # Step 6: Train a Random Forest model with the best parameters
        if model_name == 'random_forest':
            model = RandomForestClassifier(**best_params["random_forest"], random_state=42)
            model.fit(X_split, y_split)
            explainer = shap.TreeExplainer(model)
            shap_values_class1 = explainer.shap_values(X_split)[:,:,1]  # SHAP values for each sample and feature
        elif model_name == 'gradient_boosting':
            model = GradientBoostingClassifier(**best_params["gradient_boosting"], random_state=42)
            model.fit(X_split, y_split)
            explainer = shap.TreeExplainer(model)
            shap_values_class1 = explainer.shap_values(X_split)
        
        print(f"SHAP values shape: {shap_values_class1.shape}")

        # Aggregate mean absolute SHAP values to determine feature importance
        feature_importances = pd.DataFrame({
            'Feature': X_split.columns,
            'Directional SHAP Value': np.mean(shap_values_class1, axis=0),  # Mean SHAP values across all samples
            'Feature Importance': model.feature_importances_ # feature importance of 
        })

        all_feature_importances.append(feature_importances)
        

    # Step 8: Aggregate feature importance scores over all subsets
    average_importance = pd.concat(all_feature_importances).groupby("Feature").mean().sort_values(by="Feature Importance", ascending=False)

    print("\Average Feature Importance Ranking:")
    print(average_importance)
    

    
    Top_N = 10
    feature_votes = {}

    for importance_df in all_feature_importances:
        top_features = importance_df.nlargest(Top_N, 'Feature Importance')['Feature']
        for feature in top_features:
            feature_votes[feature] = feature_votes.get(feature, 0) + 1

    # Convert votes to a DataFrame
    votes_df = pd.DataFrame(list(feature_votes.items()), columns=['Feature', 'Votes'])
    # Step 3: Merge average importance with votes
    combined_df = votes_df.merge(average_importance, on='Feature', how='left')

    # Step 4: Sort by Votes (descending) and Importance (descending)
    combined_df = combined_df.sort_values(by=['Votes', 'Feature Importance'], ascending=[False, False])
    
    # Verify sorting
    print("\nFinal Feature Importance Values Ranking (Combined Votes and Feature Importance):")
    print(combined_df)

    combined_df = combined_df.sort_values(by=['Votes', 'Feature Importance'], ascending=[True, True])
    
    # Generate gradient colors in descending order
    colors = cm.viridis(np.linspace(0, 1, len(combined_df)))

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar plot for Votes with gradient colors
    bars = ax.barh(combined_df["Feature"], combined_df["Votes"], color=colors, label="Votes")

    # Add importance as text
    for i, (votes, avg_imp) in enumerate(zip(combined_df["Votes"], combined_df["Feature Importance"] * np.where(combined_df["Directional SHAP Value"] > 0, 1, -1))):
        if avg_imp > 0:
            ax.text(votes + 0.2, i, f'{avg_imp:.5f}', va='center', fontsize=10, color="black")
        else:
            ax.text(votes + 0.2, i, f'{avg_imp:.5f}', va='center', fontsize=10, color="red")

    # Labels and title
    ax.set_xlabel("Votes", fontsize=14)
    ax.set_ylabel("Features", fontsize=14)
    ax.set_title(f"Feature Votes ({model_name}) {'With SOI' if second_order_interaction else ''}", fontsize=16, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Adjust y-axis ticks for better readability
    ax.set_yticks(range(len(combined_df["Feature"])))
    ax.set_yticklabels(combined_df["Feature"], fontsize=12)

    # Remove legend for simplicity
    plt.tight_layout()
    plt.savefig(f"{model_save_dir}/{model_name}_feature_votes{'_soi' if second_order_interaction else ''}.png")
    plt.show()



    
    
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python feature_importance_random_split_validation.py <file_path> <model_save_dir>")
        sys.exit(1)

    file_path = sys.argv[1]
    model_save_dir = sys.argv[2]
    
    load_existing_best_params = False
    params_pkl_dir='models/race'
    
    second_order_interaction = True

    model_name = 'random_forest' # random_forest, gradient_boosting, cart

    main(file_path, load_existing_best_params, params_pkl_dir, model_name, model_save_dir, second_order_interaction)
    
    