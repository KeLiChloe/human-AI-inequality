import requests
from requests.auth import HTTPBasicAuth
import sys
import json
import os
from tqdm import tqdm

def save_progress(progress_file, category, keyword):
    with open(progress_file, 'w') as file:
        json.dump({'category': category, 'keyword': keyword}, file)

def load_progress(progress_file):
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as file:
            return json.load(file)
    return None

def calculate_completed_keywords(wordlist, category_list, saved_progress, sentiments):
    completed_keywords = 0
    print("saved_progress:", saved_progress['category'], saved_progress['keyword'])
    if saved_progress:
        for i, category in enumerate(category_list):
            category_sentiments = sentiments[i]
            for sentiment in category_sentiments:
                keywords = wordlist.get(category, {}).get(str(sentiment), [])
                for keyword in keywords:
                    
                    if category == saved_progress['category'] and keyword == saved_progress['keyword']:
                        completed_keywords += 1
                        break
                    completed_keywords += 1
                    print(keyword, completed_keywords)
                else:
                    continue
                break
            else:
                continue
            break
    return completed_keywords


def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def write_to_json(data, file_path):
    # Create the directory if it does not exist
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        print(f"create directory {os.path.dirname(file_path)}")
    except Exception as e:
        print(f"Failed to create {file_path}: {e}")
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"Results successfully saved to {file_path}")
    except Exception as e:
        print(f"Failed to write to {file_path}: {e}")

def aggregate_and_write_to_json(new_data, file_path):
    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Read existing data if the file exists
    existing_data = {}
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                existing_data = json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}")
        except IOError as e:
            print(f"Error reading file {file_path}: {e}")

    for year, authors in new_data.items():
        if str(year) in existing_data:
            # Combine author lists and remove duplicates
            existing_data[str(year)] = list(set(existing_data[str(year)] + authors))
            
        else:
            existing_data[str(year)] = authors

    # Write aggregated data to the JSON file
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=4)
    except IOError as e:
        print(f"Failed to write to {file_path}: {e}")


def process_pivot_data(pivot_data):
    results = {}
    for field_entry in pivot_data:
        field = field_entry['value']
        year_dict = {}
        for year_entry in field_entry.get('pivot', []):
            year = year_entry['value']
            authors_list = [author['value'] for author in year_entry.get('pivot', [])]
            year_dict[year] = authors_list
        results[field] = year_dict
    return results

def fetch_authors_by_keyword(solr_url, keyword, pivot):
    params = {
        'q': f'paper_abstract_lookup:"{keyword}" OR title_lookup:"{keyword}"',
        'wt': 'json',
        'rows': 0,  # We don't need the actual documents, just the faceted information
        'facet': 'on',
        'facet.pivot': pivot,
        'facet.limit': -1  # Get all authors
    }

    try:
        response = requests.get(solr_url, params=params, auth=HTTPBasicAuth(username, password))
        response.raise_for_status()  # Raises an HTTPError for bad responses
        # Extract pivot faceting from response
        pivot_data = response.json()['facet_counts']['facet_pivot'][pivot]
        results = process_pivot_data(pivot_data)
        return results
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


def fetch_authors(solr_url, wordlist, pivot, base_dir, category_list, sentiments, progress_file='progress.json'):
    progress = load_progress(progress_file)
    start = False if progress else True

    # Count total and completed keywords for the progress bar
    total_keywords = sum(len(wordlist.get(category, {}).get(str(sentiment), []))
                         for category, sentiments_for_category in zip(category_list, sentiments)
                         for sentiment in sentiments_for_category) - len(filter_out)

    completed_keywords = calculate_completed_keywords(wordlist, category_list, progress, sentiments) if progress else 0

    # Start or resume processing
    with tqdm(total=total_keywords, initial=completed_keywords, desc="Processing keywords", unit="keyword") as pbar:
        for i, category in enumerate(category_list):
            category_sentiments = sentiments[i]
            for sentiment in category_sentiments:
                keywords = wordlist.get(category, {}).get(str(sentiment), [])
                for keyword in keywords:
                    if keyword not in filter_out:
                        if start or (progress and category == progress['category'] and keyword == progress['keyword']):
                            start = True  # Start processing from this keyword
                            print(f"Querying keyword {keyword}")
                            result = fetch_authors_by_keyword(solr_url, keyword, pivot)
                            if result:
                                for field_of_study, data in result.items():
                                    file_path = f"{base_dir}/{category}/{field_of_study}/author_names.json"
                                    aggregate_and_write_to_json(data, file_path)
                                save_progress(progress_file, category, keyword)  # Save progress after each keyword

                            pbar.update(1)  # Update the progress bar after each keyword is processed

    pbar.close()

# Example usage
if __name__ == "__main__":
    

    solr_url = 'https://solr-semantic-scholar-raw.totosearch.org/solr/semantic_scholar_raw/select'
    username = 'research'
    password = 'insead123456'
    

    if len(sys.argv) < 2:
        print("Usage: python fetch_authors.py <wordlist_path> ")
        sys.exit(1)
    
    wordlist_path = sys.argv[1]
    wordlist = read_json_file(wordlist_path)
    if wordlist is None:
        sys.exit("Failed to load wordlist.")

    pivot = 'fields_of_study,year,authors'
    category_list = ["general"]
    sentiments = [[-1, 1]]


    # delete these when calculation solo occurrences because they are too general
    filter_out = ["bias", "biases",  "equality", "equalities",  "inequality", "inequalities"]

    base_dir = '/Users/like/Desktop/Research/Human-AI/data/author_names'

    fetch_authors(solr_url, wordlist, pivot, base_dir, category_list, sentiments)
    
    