import os
import json
import matplotlib.pyplot as plt
import pandas as pd

def load_json_file(file_path):
    """Load data from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file {file_path}: {e}")
        return {}

def aggregate_frequencies_for_fields(directory_path, fields, start_year, end_year, file_name):
    """Aggregate frequencies for specified fields in a given directory."""
    aggregated_data = {field: {} for field in fields}

    for field in fields:
        field_path = os.path.join(directory_path, field)
        file_path = os.path.join(field_path, file_name)
        if os.path.exists(file_path):
            frequencies = load_json_file(file_path)
            for year, value in frequencies.items():
                if start_year <= int(year) <= end_year:
                    aggregated_data[field][year] = value

    return aggregated_data

def plot_line_trends(aggregated_data, category, output_dir, start_year, end_year, title_suffix):
    """Plot and save line trends for the given data."""
    plt.figure(figsize=(14, 8))

    final_year_values = {}

    for field, data in aggregated_data.items():
        years = sorted(map(int, data.keys()))
        values = [data[str(year)] for year in years]
        plt.plot(years, values, linestyle='-', label=field)
        if years:
            final_year_values[field] = values[-1]

    # Sort fields by their value in the final year and get the top 5


    plt.title(f'Occurrences of Inequality-Related Keywords in {category.capitalize()} Category ({start_year}-{end_year}) - {title_suffix}', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Frequency' if "Count" in title_suffix else 'Percentage (%)', fontsize=14)
    
    # Get legend handles and labels, then sort them by final year values
    handles, labels = plt.gca().get_legend_handles_labels()
    sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: final_year_values[x[1]], reverse=True)
    handles, labels = zip(*sorted_handles_labels)
    
    plt.legend(handles, labels, title="Fields (Ordered by Last Year Value)")
    plt.grid(True)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{category}_line_trends_{title_suffix}.png')
    plt.savefig(output_path)
    print(f"Saved line trends to {output_path}")

def process_and_plot_line_trends(root_directory_path, start_year, end_year):
    """Process data from JSON files and plot line trends."""
    categories = [d for d in os.listdir(root_directory_path) if os.path.isdir(os.path.join(root_directory_path, d))]

    file_types = ['sum-count.json', 'absolute-percent.json', 'relative-percent.json']
    title_suffixes = ['Count', 'Absolute Percent (%)', 'Relative Percent (%)']

    for category in categories:
        category_path = os.path.join(root_directory_path, category)
        fields = [d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))]

        for file_name, title_suffix in zip(file_types, title_suffixes):
            aggregated_data = aggregate_frequencies_for_fields(category_path, fields, start_year, end_year, file_name)

            output_dir = os.path.join(root_directory_path, category)
            os.makedirs(output_dir, exist_ok=True)

            plot_line_trends(aggregated_data, category, output_dir, start_year, end_year, title_suffix)

# Root directory path
root_directory_path = '/Users/like/Desktop/Research/Human-AI/data/frequency_by_fields'

# Process and plot line trends for all categories and fields for the specified year range
process_and_plot_line_trends(root_directory_path, 1950, 2020)
