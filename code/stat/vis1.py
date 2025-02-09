import json
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def load_data(filepath):
    """ Load data from a JSON file specified by the full filepath. """
    with open(filepath, 'r') as file:
        return json.load(file)

def plot_data(data, start_year, end_year, ax, color, label, point_interval=1):
    """Helper function to plot data."""
    filtered_data = {int(year): value for year, value in data.items() if start_year <= int(year) <= end_year}
    years = sorted(filtered_data.keys())
    values = [filtered_data[year] for year in years]

    plot_years = years[::point_interval]
    plot_values = values[::point_interval]

    ax.plot(plot_years, plot_values, linestyle='-', color=color, label=label)


def plot_multiple_data(dirs, root_path, start_year, end_year, x_interval=1, point_interval=1, plot_type='percentage'):
    """ Plot multiple datasets read from JSON files within specified year range on plots. """
    sns.set_theme(style="darkgrid")

    # Generate a color palette with a number of colors equal to the number of directories
    colors = sns.color_palette("Set1", len(dirs))

    fig, ax = plt.subplots(figsize=(12, 8))  # Slightly larger figure size for better readability

    for index, dir_name in enumerate(dirs):
        # Set file paths based on plot_type
        if plot_type == "percentage":
            filepath = f"{root_path}/{dir_name}/percent.json"
        elif plot_type == "count":
            
            filepath = f"{root_path}/{dir_name}/sum-count.json"
        else:
            print("error in file path")
        
        # Load data from file
        data = load_data(filepath)

        # Assign plot label
        label = "all (including general)" if dir_name == "all" else dir_name

        # Plot data
        plot_data(data, start_year, end_year, ax, colors[index], label, point_interval)
    
    ax.set_xticks([year for year in range(start_year, end_year+1) if year % x_interval == 0])
    ax.set_xticklabels([str(year) for year in range(start_year, end_year+1) if year % x_interval == 0], rotation=45)
    ax.set_xlabel('Year', fontsize=14, weight='bold', labelpad=10)
    if plot_type == "percentage":
        y_label = "Percentage %"
    elif plot_type == "count":
        y_label = "Count"
    ax.set_ylabel(y_label, fontsize=14, weight='bold')
    ax.set_title(f'Occurrences of Inequality-Related Keywords ({y_label}) From {start_year} to {end_year}', fontsize=16, weight='bold')
    # After all datasets are plotted
    ax.legend(loc='upper left',fontsize=15)

    # plt.show()
    fig.savefig(f"{root_path}/{plot_type}.jpg")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python vis.py mode")
        sys.exit(1)

    plot_type = sys.argv[1] # plot_type = ["percentage", "count", "both"]

    # Parameters
    dirs = ["gender", "economic", "race", "susceptibility", "all"]
    root_path = "/Users/like/Desktop/Research/Human-AI/data/frequency"

    # Execution
    if plot_type == "both":
        plot_multiple_data(dirs, root_path, 1800, 2020, x_interval=10, point_interval=1, plot_type="percentage")
        plot_multiple_data(dirs, root_path, 1800, 2020, x_interval=10, point_interval=1, plot_type="count")
    else:
        plot_multiple_data(dirs, root_path, 1800, 2020, x_interval=10, point_interval=1, plot_type=plot_type)



        