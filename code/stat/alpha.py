import requests
from requests.auth import HTTPBasicAuth
import json
import time
from tqdm import tqdm
import sys
import os


# Does it make sense/acceptable to also check co-occurrences of susceptibility words with race/gender/economic category (neural)





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

# Function to query Solr and get the count of documents by year for a keyword
def get_keyword_counts(keyword):
    # Solr query parameters
    params = {
        'q': f'title_lookup:"{keyword}" OR paper_abstract_lookup:"{keyword}"',
        'wt': 'json',
        'rows': 0,  # We don't need the actual documents, just the counts
        'facet': 'on',
        'facet.field': 'year',
        'facet.limit': -1,  # No limit on facet counts
        'facet.mincount': 1,  # Exclude zero count
        # 'fq': 'year:["1922" TO "2010"]'  # Filter query to include only documents from 2000 to 2019
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
        print(f"Failed to query Solr for keyword '{keyword}'. Status code:", response.status_code)
        return None
    
def aggregate_occurrences(wordlist, categories, sentiments, filter_out, root_dir):
    total_counts_by_year = {}

    all_category_counts_by_year_pos_or_neg = {}
    all_category_counts_by_year_pos_or_neg["pos"] = {}
    all_category_counts_by_year_pos_or_neg["neg"] = {}

    # Calculate total keywords for progress bar initialization
    total_keywords = sum(len(wordlist.get(category, {}).get(str(sentiment), [])) 
                     for category, sentiments_for_category in zip(categories, sentiments) 
                     for sentiment in sentiments_for_category) - len(filter_out)
    
    with tqdm(total=total_keywords, desc="Processing keywords", unit="keyword") as pbar:
        for i, category in enumerate(categories):
            category_sentiments = sentiments[i]
            category_counts_by_year = {}
            for sentiment in category_sentiments:
                pos_or_neg = "pos" if sentiment == 1 else "neg"
                category_counts_by_year_pos_or_neg = {}
                keywords = wordlist.get(category, {}).get(str(sentiment), [])
                for keyword in keywords:
                    if keyword not in filter_out:
                        yearly_counts = get_keyword_counts(keyword)
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
                
                        # Update progress bar after each keyword is processed
                        pbar.update(1)
                write_to_json(category_counts_by_year_pos_or_neg, root_dir+f'{category}/solo-count-{pos_or_neg}.json') 
            write_to_json(category_counts_by_year, root_dir+f'{category}/solo-count.json')
            
    write_to_json(all_category_counts_by_year_pos_or_neg["pos"], root_dir+'all/solo-count-pos.json')
    write_to_json(all_category_counts_by_year_pos_or_neg["neg"], root_dir+'all/solo-count-neg.json')
    write_to_json(total_counts_by_year, root_dir+'all/solo-count.json')



if __name__ == "__main__":
    start_time = time.time()

    if len(sys.argv) < 2:
        print("Usage: python script.py <wordlist_path> ")
        sys.exit(1)
    
    wordlist_path = sys.argv[1]
    wordlist = read_json_file(wordlist_path)
    if wordlist is None:
        sys.exit("Failed to load wordlist.")

    # The URL to Solr 
    solr_url = 'https://solr-semantic-scholar-raw.totosearch.org/solr/semantic_scholar_raw/select'
    username = 'research'
    password = 'insead123456'

    categories = ["economic", "race", "gender", "general"]
    sentiments = [[-1, 1], [-1, 1], [-1, 1], [-1, 1]]

    # delete these when calculation solo occurrences because they are too general (math)
    filter_out = []
    save_dir = '/Users/like/Desktop/Research/Human-AI/data_without_filtering/frequency/'


    aggregate_occurrences(wordlist, categories, sentiments, filter_out, save_dir)
    
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time} seconds")