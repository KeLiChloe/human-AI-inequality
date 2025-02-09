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
def get_keyword_cooccurrences(keyword1, keyword2):
    params = {
        'q': f'((title_lookup:"{keyword1}" AND title_lookup:"{keyword2}") OR (paper_abstract_lookup:"{keyword1}" AND paper_abstract_lookup:"{keyword2}"))',
        'wt': 'json',
        'rows': 0,  # We don't need the actual documents, just the counts
        'facet': 'on',
        'facet.field': 'year',
        'facet.limit': -1,  # No limit on facet counts
        'facet.mincount': 1,  # Exclude zero count
    }
    
    # Make the HTTP GET request to Solr
    response = requests.get(solr_url, params=params, auth=HTTPBasicAuth(username, password))
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the response JSON
        response_json = response.json()
        facet_counts = response_json['facet_counts']['facet_fields']['year']
        return dict(zip(facet_counts[::2], facet_counts[1::2]))  # Convert list to a dict
    else:
        print(f"Failed to query Solr for co-occurrences '{keyword1}' and '{keyword2}'. Status code:", response.status_code)
        return None


def aggregate_cooccurrences(wordlist, bases, base_sentiments, categories, categories_sentiments, root_dir):
    total_counts_by_year = {}

    all_category_counts_by_year_pos_or_neg = {}
    all_category_counts_by_year_pos_or_neg["pos"] = {}
    all_category_counts_by_year_pos_or_neg["neg"] = {}
    
    
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
            category_counts_by_year = {}
            for category_sentiment in category_sentiments:
                category_keywords = wordlist.get(category, {}).get(str(category_sentiment), [])
                for j, base in enumerate(bases):
                    base_sentiments = bases_sentiments[j]
                    for base_sentiment in base_sentiments:
                        category_counts_by_year_pos_or_neg = {}
                        if category == "susceptibility":
                            susceptibility = "mutability" if category_sentiment == 1 else "fixedness"

                        pos_or_neg = "pos" if base_sentiment == 1 else "neg"
                        base_keywords = wordlist.get(base, {}).get(str(base_sentiment), [])
                        for base_keyword in base_keywords:
                            for category_keyword in category_keywords:
                                yearly_counts = get_keyword_cooccurrences(base_keyword, category_keyword)
                                
                                if yearly_counts:
                                    for year, count in yearly_counts.items():
                                        if year not in total_counts_by_year:
                                            total_counts_by_year[year] = 0
                                        if year not in category_counts_by_year:
                                            category_counts_by_year[year] = 0
                                        if year not in category_counts_by_year_pos_or_neg:
                                            category_counts_by_year_pos_or_neg[year] = 0
                                        if year not in all_category_counts_by_year_pos_or_neg[pos_or_neg]:
                                            all_category_counts_by_year_pos_or_neg[pos_or_neg][year] = 0

                                        category_counts_by_year[year] += count
                                        category_counts_by_year_pos_or_neg[year] += count
                                            
                                        all_category_counts_by_year_pos_or_neg[pos_or_neg][year] += count
                                        total_counts_by_year[year] += count
                                
                                # Update progress bar after each keyword pair is processed
                                pbar.update(1)
                        
                        if category == "susceptibility":
                            write_to_json(category_counts_by_year_pos_or_neg, root_dir+f'{category}/co-count-{pos_or_neg}-{susceptibility}.json') 
                        else:
                            write_to_json(category_counts_by_year_pos_or_neg, root_dir+f'{category}/co-count-{pos_or_neg}.json') 
            
            write_to_json(category_counts_by_year, root_dir+f'{category}/co-count.json')

    write_to_json(all_category_counts_by_year_pos_or_neg["pos"], root_dir+'all/co-count-pos.json')
    write_to_json(all_category_counts_by_year_pos_or_neg["neg"], root_dir+'all/co-count-neg.json')
    write_to_json(total_counts_by_year, root_dir+'all/co-count.json')




if __name__ == "__main__":
    start_time = time.time()
    
    if len(sys.argv) < 2:
        print("Usage: python script.py <wordlist_path>")
        sys.exit(1)

    root_dir = '/Users/like/Desktop/Research/Human-AI/data_without_filtering/frequency/'

    wordlist_path = sys.argv[1]

    # The URL to Solr 
    solr_url = 'https://solr-semantic-scholar-raw.totosearch.org/solr/semantic_scholar_raw/select'
    username = 'research'
    password = 'insead123456'

    
    bases = ["general"]
    bases_sentiments = [[-1, 1]] 

    categories = ["economic", "race", "gender", "susceptibility"]

    categories_sentiments = [[0], [0], [0], [-1, 1]]

    wordlist = read_json_file(wordlist_path)
    if wordlist is None:
        sys.exit("Failed to load wordlist.")
        
    
    aggregate_cooccurrences(wordlist, bases, bases_sentiments, categories, categories_sentiments, root_dir)
        
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time} seconds")


