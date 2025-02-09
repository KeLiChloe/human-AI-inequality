
import requests
from requests.auth import HTTPBasicAuth
import json
import time
from tqdm import tqdm
import sys
import os
import random

def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def get_query_result(keyword):
    # Solr query parameters
    random_seed = random.randint(0, 1000)
    params = {
        # 'q': f'(title_lookup:"{keyword}" OR paper_abstract_lookup:"{keyword}") AND year:[2008 TO *]',  # Year filter added
        'q': f'year:[2011 TO *]',
        'wt': 'csv',
        'rows': 5000,
        'fl': 'title,paper_abstract,year,authors,journal_name,fields_of_study',  # Fields to include in the CSV
        'sort': f'random_{random_seed} asc'
    }
    
    # Make the HTTP GET request to Solr
    response = requests.get(solr_url, params=params, auth=HTTPBasicAuth(username, password))
    
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to query Solr for keyword '{keyword}'. Status code:", response.status_code)


if __name__ == "__main__":
    start_time = time.time()

    if len(sys.argv) < 3:
        print("Usage: python create_data_sample.py <wordlist_path> <output_csv>")
        sys.exit(1)
    
    wordlist_path = sys.argv[1]
    output_csv = sys.argv[2]

    wordlist = read_json_file(wordlist_path)
    if wordlist is None:
        sys.exit("Failed to load wordlist.")

    # The URL to Solr 
    solr_url = 'https://solr-semantic-scholar-raw.totosearch.org/solr/semantic_scholar_raw/select'
    username = 'research'
    password = 'insead123456'

    categories = ["gender"]
    sentiments = [[1,-1]]
    

    # delete these when calculation solo occurrences because they are too general (math)
    filter_out = []

    with open(output_csv, 'w', newline='', encoding='utf-8') as file:
        file.write('')  # This ensures the file is empty before appending data

   # Calculate total keywords for progress bar initialization
    total_iterations = sum(len(wordlist.get(category, {}).get(str(sentiment), [])) 
                     for category, sentiments_for_category in zip(categories, sentiments) 
                     for sentiment in sentiments_for_category)

    count = 0
    with tqdm(total=total_iterations, desc="Processing Keywords") as pbar:
        for i, category in enumerate(categories):
            category_sentiments = sentiments[i]
            for sentiment in category_sentiments:
                keywords = wordlist.get(category, {}).get(str(sentiment), [])
                for keyword in keywords:
                    response_csv = get_query_result(keyword)
                    if response_csv:  # Ensure the query was successful
                        with open(output_csv, 'a', newline='', encoding='utf-8') as file:
                            lines = response_csv.splitlines()  # response_csv is already text, no .text needed
                            if os.stat(output_csv).st_size == 0:
                                # Write header if file is empty (first time writing to the file)
                                file.write(response_csv)
                            else:
                                # Skip the first line and append the rest (to avoid writing the header again)
                                file.write("\n".join(lines[1:]))
                    pbar.update(1)  # Update the progress bar after each keyword


    with open(output_csv, 'a', newline='', encoding='utf-8') as file:
        lines = response_csv.splitlines()
        if os.stat(output_csv).st_size == 0:
            file.write(response_csv)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time} seconds")

