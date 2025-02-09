import requests
from requests.auth import HTTPBasicAuth
import json
import time
from tqdm import tqdm
import sys
import os


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

# Search for documents where both keywords co-occur either in the title or in the paperAbstract
def get_keyword_cooccurrences(keyword1, keyword2, pivot):
    params = {
        'q': f'((title_lookup:"{keyword1}" AND title_lookup:"{keyword2}") OR (paper_abstract_lookup:"{keyword1}" AND paper_abstract_lookup:"{keyword2}"))',
        'wt': 'json',
        'rows': 0,  # We don't need the actual documents, just the counts
        'facet': 'on',
        'facet.pivot': pivot, # Pivot faceting by fields_of_study and year
        'facet.limit': -1,  # No limit on facet counts
        'facet.mincount': 1,  # Exclude zero count
    }
    
    # Make the HTTP GET request to Solr
    response = requests.get(solr_url, params=params, auth=HTTPBasicAuth(username, password))
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the response JSON
        response_json = response.json()
        pivot_counts = response_json['facet_counts']['facet_pivot'][pivot]
        return pivot_counts  # Convert list to a dict
    else:
        print(f"Failed to query Solr for co-occurrences '{keyword1}' and '{keyword2}'. Status code:", response.status_code)
        return None


def aggregate_cooccurrences(wordlist, bases, base_sentiments, categories, categories_sentiments, root_dir, pivot):
    total_counts_by_field = {}
    
    
    # Calculate total keyword pairs for the progress bar
    total_keyword_pairs = sum(len(wordlist.get(category, {}).get(str(sentiment), [])) 
                                    for category, sentiments_for_category in zip(categories, categories_sentiments) 
                                        for sentiment in sentiments_for_category) * sum(
                                len(wordlist.get(base, {}).get(str(sentiment), [])) 
                                    for base, sentiments_for_base in zip(bases, base_sentiments) 
                                        for sentiment in sentiments_for_base)
    
    with tqdm(total=total_keyword_pairs, desc="Processing keyword pairs", unit="pair") as pbar:
        for i, category in enumerate(categories):
            category_sentiments = categories_sentiments[i]
            category_counts_by_field = {}
            for category_sentiment in category_sentiments:
                category_keywords = wordlist.get(category, {}).get(str(category_sentiment), [])
                for j, base in enumerate(bases):
                    base_sentiments = bases_sentiments[j]
                    for base_sentiment in base_sentiments:
                        base_keywords = wordlist.get(base, {}).get(str(base_sentiment), [])
                        for base_keyword in base_keywords:
                            for category_keyword in category_keywords:
                                pivot_results = get_keyword_cooccurrences(base_keyword, category_keyword, pivot)
                                
                                if pivot_results:
                                    for result in pivot_results:
                                        field_of_study = result['value']
                                        for pivot_data in result['pivot']:
                                            year = pivot_data['value']
                                            count = pivot_data['count']

                                            if field_of_study not in category_counts_by_field:
                                                category_counts_by_field[field_of_study] = {}
                                            if year not in category_counts_by_field[field_of_study]:
                                                category_counts_by_field[field_of_study][year] = 0
                                            category_counts_by_field[field_of_study][year] += count

                                            if field_of_study not in total_counts_by_field:
                                                total_counts_by_field[field_of_study] = {}
                                            if year not in total_counts_by_field[field_of_study]:
                                                total_counts_by_field[field_of_study][year] = 0
                                            total_counts_by_field[field_of_study][year] += count

                                # Update progress bar after each keyword pair is processed
                                pbar.update(1)
            for field, counts in category_counts_by_field.items():
                field_dir = os.path.join(root_dir, category, field)
                write_to_json(counts, os.path.join(field_dir, 'co-count.json'))
    for field, counts in total_counts_by_field.items():
        field_dir = os.path.join(root_dir, 'all', field)
        write_to_json(counts, os.path.join(field_dir, 'co-count.json'))
                
        



if __name__ == "__main__":
    start_time = time.time()
    
    if len(sys.argv) < 2:
        print("Usage: python script.py <wordlist_path>")
        sys.exit(1)

    root_dir = '/Users/like/Desktop/Research/Human-AI/data/frequency_by_fields/'

    wordlist_path = sys.argv[1]

    # The URL to Solr 
    solr_url = 'https://solr-semantic-scholar-raw.totosearch.org/solr/semantic_scholar_raw/select'
    username = 'research'
    password = 'insead123456'

    
    bases = ["general"]
    bases_sentiments = [[-1, 1]] 

    categories = ["economic", "race", "gender", "susceptibility"]
    categories_sentiments = [[0], [0], [0], [-1, 1]]

    # categories = ["susceptibility"]
    # categories_sentiments = [[-1, 1]]


    wordlist = read_json_file(wordlist_path)
    if wordlist is None:
        sys.exit("Failed to load wordlist.")
        
    pivot = "fields_of_study,year"

    aggregate_cooccurrences(wordlist, bases, bases_sentiments, categories, categories_sentiments, root_dir, pivot)
        
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time} seconds")


