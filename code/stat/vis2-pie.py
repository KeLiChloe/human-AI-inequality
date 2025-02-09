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

def aggregate_frequencies(directory_path, start_year, end_year):
    """Aggregate frequencies for all fields in a given directory."""
    data = {}

    for field in os.listdir(directory_path):
        field_path = os.path.join(directory_path, field)
        total_frequency = 0

        if os.path.isdir(field_path):
            file_path = os.path.join(field_path, 'sum-count.json')
            if os.path.exists(file_path):
                frequencies = load_json_file(file_path)
                for year in range(start_year, end_year + 1):
                    total_frequency += frequencies.get(str(year), 0)
        
        data[field] = total_frequency

    return data

def plot_pie_chart(data, category, start_year, end_year, output_path, title_suffix="", mode="granular"):
    """Plot and save a pie chart for the given data."""
    df = pd.DataFrame(list(data.items()), columns=['Field', 'Total Frequency'])
    df['Percentage'] = (df['Total Frequency'] / df['Total Frequency'].sum()) * 100

    # Sort data by Total Frequency in descending order
    df = df.sort_values(by='Total Frequency', ascending=False)

    if mode == "rough":
        colors = plt.get_cmap('tab20')(range(3))  # Distinct colors for rough level categories
    else:
        colors = plt.get_cmap('tab20c')(range(len(df)))  # Original color scheme for granular level
    total = df['Total Frequency'].sum()
    explode = [0.1 if (freq / total) > 0.1 else 0 for freq in df['Total Frequency']]

    def autopct_format(pct):
        absolute = int(round(pct / 100. * total))
        if mode == "granular":
            return f'{pct:.1f}%\n({df["Field"][df["Total Frequency"] == absolute].values[0]})' if pct > 5 else (f'{pct:.1f}%' if pct > 0 else '')
        if mode == "rough":
            return f'{pct:.1f}%\n({df["Field"][df["Total Frequency"] == absolute].values[0]})'

    plt.figure(figsize=(24, 20))
    wedges, texts, autotexts = plt.pie(df['Total Frequency'], colors=colors, startangle=140, autopct=autopct_format, pctdistance=0.85, explode=explode)
    for autotext in autotexts:
        autotext.set_fontsize(18)

    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    plt.axis('equal')
    plt.title(f'{category.capitalize()} category: total frequency counts by fields ({start_year}-{end_year}) {title_suffix}', fontsize=30, fontweight='bold')

    # Create legend with sorted labels
    legend_labels = [f'{row["Field"]} ({row["Percentage"]:.1f}%)' for index, row in df.iterrows()]
    sorted_legend = sorted(zip(wedges, legend_labels), key=lambda x: float(x[1].split()[-1][1:-2]), reverse=True)

    if sorted_legend:
        handles, labels = zip(*sorted_legend)
        plt.legend(handles, labels, title="Fields (Ordered by Value)", loc="center left", bbox_to_anchor=(1.1, 0, 0.5, 1), title_fontsize=25, fontsize=20)

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved pie chart to {output_path}")

def process_and_plot_data(start_year, end_year):
    """Process data from JSON files and plot pie charts."""
    root_directory_path = '/Users/like/Desktop/Research/Human-AI/data/frequency_by_fields'
    categories = [d for d in os.listdir(root_directory_path) if os.path.isdir(os.path.join(root_directory_path, d))]

    field_groups = {
        'Natural Sciences': [
            'Biology', 'Chemistry', 'Physics', 'Geology', 
            'Environmental Science', 'Medicine', 'Materials Science'
        ],
        'Engineering and Technology': [
            'Engineering', 'Computer Science', 'Mathematics'
        ],
        'Social Sciences': [
            'Sociology', 'Political Science', 'Psychology', 
            'Economics', 'Geography', 'History', 'Art', 'Philosophy'
        ],
    }

    for category in categories:
        directory_path = os.path.join(root_directory_path, category)
        
        # Granular Level Pie Chart
        granular_data = aggregate_frequencies(directory_path, start_year, end_year)
        if granular_data:
            output_path = os.path.join(directory_path, f'{category}_category_fields_pie_granular.png')
            plot_pie_chart(granular_data, category, start_year, end_year, output_path, title_suffix="(Granular Level)", mode = "granular")
        
        # Rough Level Pie Chart
        rough_data = {group: 0 for group in field_groups}
        for group, fields in field_groups.items():
            for field in fields:
                if field in granular_data:
                    rough_data[group] += granular_data[field]
        
        if rough_data:
            output_path = os.path.join(directory_path, f'{category}_category_fields_pie_rough.png')
            plot_pie_chart(rough_data, category, start_year, end_year, output_path, title_suffix="(Rough Level)", mode = "rough")

# Example call with year range as parameters
process_and_plot_data(1800, 2020)
