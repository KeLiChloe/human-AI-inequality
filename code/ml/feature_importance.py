import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the directory containing the model files
model_dir = 'models/tmp'

# Iterate over all files in the directory with `_model.pkl` suffix
for model_file in os.listdir(model_dir):
    if model_file.endswith('_model.pkl'):
        model_path = os.path.join(model_dir, model_file)
        # Step 1: Load the model
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        
        # Step 2: Extract feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = abs(model.coef_.flatten())
        else:
            print(f"Model in {model_file} does not have feature importance attributes.")
            continue

        # If feature names are not available in the model, generate generic names
        feature_file = f"{model_dir}/feature_names.pkl"
        with open(feature_file, 'rb') as file:
            feature_names = pickle.load(file)

        # Step 3: Create a DataFrame for plotting
        print(len(feature_names))
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        importance_df = importance_df[:15]
        print(importance_df)

        # Step 4: Plot feature importance
        plt.figure(figsize=(10, 8))
        sns.barplot(
            y='Feature',
            x='Importance',
            data=importance_df,
            palette='viridis'  # Beautiful color palette
        )
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(f'Feature Importance: {model_file.replace("_model.pkl", "")}', fontsize=14)
        plt.tight_layout()

        # Save the figure dynamically with model name
        save_path = os.path.join(model_dir, f'Feature_Importance_{model_file.replace("_model.pkl", "")}.png')
        plt.savefig(save_path)
        plt.show()
