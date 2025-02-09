from sklearn.inspection import PartialDependenceDisplay
import pickle
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier

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
    X = df.drop(columns=drop_columns)  # Adjust feature selection as needed


    # Step 1: Initialize PolynomialFeatures with degree 2 for second-order interactions
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

    # Step 2: Transform the feature matrix
    X_interactions = poly.fit_transform(X)

    # Step 3: Get the names of the features (optional)
    interaction_feature_names = poly.get_feature_names_out(input_features=X.columns)

    # Step 4: Create a DataFrame with interaction terms
    X = pd.DataFrame(X_interactions, columns=interaction_feature_names)

    return X, y, df

def initial_screen_features_lasso(X, y, alpha=0.005):
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

    return X_filtered, selected_features.tolist()

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

    return X_filtered, selected_features.tolist()


random_forest_model_file = "models/race/random_forest_model.pkl"
random_forest_model = pickle.load(open(random_forest_model_file, "rb"))

# Step 1: Load and prepare data with second-order interactions
X, y, _ = load_and_prepare_data('/Users/like/Desktop/Research/Human-AI/data/samples/test/ml_pos_neg.csv')
    
# Initial feature screening using Random Forest
X, _ = initial_screen_features_lasso(X, y)
    
PartialDependenceDisplay.from_estimator(random_forest_model, X, )
