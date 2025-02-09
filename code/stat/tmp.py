import os
import json

def load_json_file(file_path):
    """Load data from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file {file_path}: {e}")
        return {}

def save_json_file(data, file_path):
    """Save data to a JSON file."""
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def create_percent_files_for_category(category_path, all_paper_num):
    """Create percent-neg.json and percent-pos.json for a given category."""
    sum_neg_data = load_json_file(os.path.join(category_path, 'sum-neg.json'))
    sum_pos_data = load_json_file(os.path.join(category_path, 'sum-pos.json'))
    sum_data = load_json_file(os.path.join(category_path, 'sum-count.json'))

    percent_neg_data = {
        year: round(sum_neg_data[year] * 100 / all_paper_num.get(year, 1), 5)
        for year in sum_neg_data if year in all_paper_num
    }
    percent_pos_data = {
        year: round(sum_pos_data[year] * 100 / all_paper_num.get(year, 1), 5)
        for year in sum_pos_data if year in all_paper_num
    }

    percent_data = {
        year: round(sum_data[year] * 100 / all_paper_num.get(year, 1), 5)
        for year in sum_data if year in all_paper_num
    }

    save_json_file(percent_neg_data, os.path.join(category_path, 'percent-neg.json'))
    save_json_file(percent_pos_data, os.path.join(category_path, 'percent-pos.json'))
    save_json_file(percent_data, os.path.join(category_path, 'percent.json'))

    print(f"Created percent-neg.json and percent-pos.json in {category_path}")


def process_categories(root_directory_path, all_paper_num_path):
    """Process all categories in the root directory."""
    categories = [d for d in os.listdir(root_directory_path) if os.path.isdir(os.path.join(root_directory_path, d))]

    all_paper_num = load_json_file(all_paper_num_path)

    for category in categories:
        category_path = os.path.join(root_directory_path, category)
        sum_files()
        create_percent_files_for_category(category_path, all_paper_num)

# Root directory path
root_directory_path = '/Users/like/Desktop/Research/Human-AI/data/frequency_test'
# Path to all-paper-num.json
all_paper_num_path = '/Users/like/Desktop/Research/Human-AI/data/meta_data/all-paper-num.json'

# Process and create sum-neg.json, sum-pos.json, sum-count.json, percent-neg.json, and percent-pos.json for all categories
process_categories(root_directory_path, all_paper_num_path)
