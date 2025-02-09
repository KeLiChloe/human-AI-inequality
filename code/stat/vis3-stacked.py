import os
import json
import pandas as pd
import matplotlib.pyplot as plt

def load_json_file(file_path):
    """Load data from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file {file_path}: {e}")
        return {}

def aggregate_frequencies_over_time(directory_path, fields, start_year, end_year, file_type):
    """Aggregate frequencies over time for specified fields in a given directory."""
    data = {str(year): {field: 0 for field in fields} for year in range(start_year, end_year + 1)}

    for field in fields:
        field_path = os.path.join(directory_path, field)
        if os.path.isdir(field_path):
            file_path = os.path.join(field_path, f'{file_type}.json')
            if os.path.exists(file_path):
                frequencies = load_json_file(file_path)
                for year in range(start_year, end_year + 1):
                    data[str(year)][field] += frequencies.get(str(year), 0)

    return data

def process_and_plot_stacked_area_chart(start_year, end_year):
    """Process data from JSON files and plot stacked area charts."""
    root_directory_path = '/Users/like/Desktop/Research/Human-AI/data/frequency_by_fields'
    categories = [d for d in os.listdir(root_directory_path) if os.path.isdir(os.path.join(root_directory_path, d))]

    for category in categories:
        directory_path = os.path.join(root_directory_path, category)
        fields = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
        
        for file_type in ['sum-count', 'absolute-percent']:
            data = aggregate_frequencies_over_time(directory_path, fields, start_year, end_year, file_type)
            if data:
                df = pd.DataFrame(data).T  # Transpose the data frame to have years as index
                df.index = pd.to_datetime(df.index, format='%Y')
                
                df.plot.area(figsize=(14, 8), colormap='tab20c')
                tmp = 'count' if file_type == 'sum-count' else file_type
                plt.title(f'Stacked Area Chart for {category.capitalize()} ({tmp.capitalize()} from {start_year} to {end_year})', fontsize=16)
                plt.xlabel('Year', fontsize=14)
                plt.ylabel('Total Frequency' if file_type == 'sum-count' else 'Percentage (%)', fontsize=14)
                plt.legend(title="Fields", bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.savefig(os.path.join(directory_path, f'{category}_{tmp}_stacked_area.png'))

# Example call with year range as parameters
process_and_plot_stacked_area_chart(1900, 2020)
