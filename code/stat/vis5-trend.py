import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    """Load data from a JSON file."""
    with open(filepath, 'r') as file:
        return json.load(file)

def plot_data(data, start_year, end_year, ax, color, label, point_interval=10):
    """Helper function to plot data."""
    filtered_data = {int(year): value for year, value in data.items() if start_year <= int(year) <= end_year}
    years = sorted(filtered_data.keys())
    values = [filtered_data[year] for year in years]

    # Applying point interval to reduce plot points
    plot_years = years[::point_interval]
    plot_values = values[::point_interval]


    ax.plot(plot_years, plot_values, linestyle='-', color=color, label=label)

def plot_category_data(pos_data, neg_data, category, root_dir, start_year, end_year, point_interval, x_label_interval):
    """Plot individual category data."""
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    plot_data(pos_data, start_year, end_year, ax, 'blue', f"{category.capitalize()} Positive", point_interval)
    plot_data(neg_data, start_year, end_year, ax, 'red', f"{category.capitalize()} Negative", point_interval)

    # Adjust x-axis label intervals
    ax.set_xticks([year for year in range(start_year, end_year + 1) if (year - start_year) % x_label_interval == 0])
    ax.set_xticklabels([str(year) for year in range(start_year, end_year + 1) if (year - start_year) % x_label_interval == 0], rotation=45)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    
    ax.set_xlabel('Year', fontsize=18, weight='bold', labelpad=10)
    ax.set_ylabel('Count', fontsize=18, weight='bold')
    ax.set_title(f'Occurrences of Inequality-Related Keywords in {category.capitalize()} From {start_year} to {end_year}', fontsize=16, weight='bold')
    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(root_dir, f"{category.capitalize()}_percent_trends.png"))
    plt.close(fig)

import matplotlib.pyplot as plt
import seaborn as sns

def plot_multiple_data(categories, root_dir, start_year, end_year, point_interval, x_label_interval):
    """Plot multiple datasets for positive and negative frequencies on plots."""
    sns.set_theme(style="darkgrid")

    # Generate a color palette with a number of colors equal to the number of categories * 2 for pos and neg
    colors = sns.color_palette("Set2", len(categories) * 2)

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    legend_data = []

    for index, category in enumerate(categories):
        pos_file_path = os.path.join(root_dir, category, "percent-pos.json")
        neg_file_path = os.path.join(root_dir, category, "percent-neg.json")

        pos_data = load_data(pos_file_path)
        neg_data = load_data(neg_file_path)

        # Plot individual category data
        plot_category_data(pos_data, neg_data, category, root_dir, start_year, end_year, point_interval, x_label_interval)

        # Add to combined plot
        plot_data(pos_data, start_year, end_year, ax, colors[index * 2], f"{category.capitalize()} Positive", point_interval)
        plot_data(neg_data, start_year, end_year, ax, colors[index * 2 + 1], f"{category.capitalize()} Negative", point_interval)

        # Collect data for sorting the legend
        final_pos_value = pos_data.get(str(end_year), 0)  # Ensure data is loaded correctly
        final_neg_value = neg_data.get(str(end_year), 0)  # Ensure data is loaded correctly
        legend_data.append((final_pos_value, f"{category.capitalize()} Positive", colors[index * 2]))
        legend_data.append((final_neg_value, f"{category.capitalize()} Negative", colors[index * 2 + 1]))

    # Adjust x-axis label intervals for combined plot
    ax.set_xticks([year for year in range(start_year, end_year + 1) if (year - start_year) % x_label_interval == 0])
    ax.set_xticklabels([str(year) for year in range(start_year, end_year + 1) if (year - start_year) % x_label_interval == 0], rotation=45)

    ax.set_xlabel('Year', fontsize=18, weight='bold', labelpad=10)
    ax.set_ylabel('Count', fontsize=18, weight='bold')
    ax.set_title('Occurrences of Inequality-Related Keywords (Percentage %) in All Categories From {} To {}'.format(start_year, end_year), fontsize=16, weight='bold')

    # Sort legend data and create custom legend
    legend_data.sort(reverse=True, key=lambda x: x[0])  # Sort based on final values
    handles = [plt.Line2D([0], [0], color=item[2], marker='o', linestyle='-', label=item[1]) for item in legend_data]
    ax.legend(handles=handles, title="Categories (Ordered by Last Year Value)", loc='upper left', title_fontsize=20, fontsize=20)

    plt.tight_layout()
    fig.savefig(os.path.join(root_dir, "all_categories_percent_trends.png"))
    plt.close(fig)

# Parameters
root_dir = "/Users/like/Desktop/Research/Human-AI/data/frequency"
categories = ["race", "gender", "economic", "general"]  # Define your categories here
start_year = 1800  # Define the start year
end_year = 2020    # Define the end year
point_interval = 1  # Points to plot
x_label_interval = 10 # X-axis label interval

# Call the function to generate the plot
plot_multiple_data(categories, root_dir, start_year, end_year, point_interval, x_label_interval)



