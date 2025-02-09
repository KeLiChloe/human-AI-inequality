import pandas as pd
import re
import requests
from requests.auth import HTTPBasicAuth  # Only if Solr requires authentication
from tqdm import tqdm
import numpy as np
import sys
import os


# Generally, the facebook has reliable data for firt/last name identification, but not for country.
## need to change file_path and output_dir

solr_url = 'https://solr-facebooknames.totosearch.org/solr/facebook_lastname/select'
username = 'research'
password = 'insead123456'

def find_country_with_rank_1(data):
    # Loop through the keys in the data
    for key in data.keys():
        # Check if the key contains 'lastname_rank' and if its value is 1
        if key.endswith('lastname_rank') and data[key] == 1:
            # Extract the corresponding country
            country_key = key.replace('_lastname_rank', '')
            return data.get(f'{country_key}', None)
    
    # Return None if no country with rank 1 is found
    return None

def get_query_from_facebook(name, solr_url, username=None, password=None):
    params = {
        'q': f'lastname:"{name}"',
        'wt': 'json',  # Requesting JSON response
        "sort":"lastname_count desc",
        "rows":"1"
    }
    # Make the HTTP GET request to Solr
    response = requests.get(solr_url, params=params, auth=HTTPBasicAuth(username, password) if username and password else None)
    if response.status_code == 200:
        # Parse the response JSON
        data = response.json()
        if 'response' in data and 'docs' in data['response'] and len(data['response']['docs']) > 0:
            result = data['response']['docs'][0]
            return result['lastname_percent']/100,  result['lastname_count'], result['female_percent']/100, find_country_with_rank_1(result)
        else:
            return 0, 0, 0.5, None  # Name not found
        
    else:
        print(f"Failed to query Solr for name '{name}'. Status code:", response.status_code)
        return None


def contains_weird_characters(text):
    # Check if the text contains non-ASCII characters
    if isinstance(text, str):
        return bool(re.search(r'[^\x00-\xFF]', text))  # 覆盖了 ASCII（0-127）和 Latin-1（128-255）字符集
    return False

# Corrected function with proper return order: first name, last name, female_percent, country
def determine_first_last_female_country_facebook(name, solr_url, username=None, password=None):
    # Split the name into parts (only consider the first and last parts)
    
    name_parts = name.strip().split()
    first_word = name_parts[0].replace("\\", "")  # remove "\" in the string
    last_word = name_parts[-1].replace("\\", "")  # remove "\" in the string
    
    # determining which are first and last names
    first_word_percent, first_word_count, female_percent_first_word, country_first_word  = get_query_from_facebook(first_word, solr_url, username, password)
    last_word_percent, last_word_count, female_percent_last_word, country_last_word = get_query_from_facebook(last_word, solr_url, username, password)
    
    if len(name_parts) == 1:
        return name_parts[0], name_parts[0], female_percent_first_word, country_first_word # Treat as both first and last name
    if '-' in first_word:
        return first_word, last_word, female_percent_first_word, country_last_word  # First word as first name
    elif '-' in last_word:
        return last_word, first_word, female_percent_last_word, country_first_word  # Last word as first name
    
    if '.' in first_word:
        return first_word, last_word, female_percent_first_word, country_last_word # First word as first name
    elif '.' in last_word:
        return last_word, first_word, female_percent_last_word, country_first_word # Last word as first name
    
    if len(first_word)==1:
        return first_word, last_word, female_percent_first_word, country_last_word # First word as first name
    elif len(last_word)==1:
        return last_word, first_word, female_percent_last_word, country_first_word  # Last word as first name
    

    # Decide which percent to trust based on the higher lastname_count
    # Under this logic, if both parts of the names cannot be queried, always treat first word as first name

    if first_word_count >= last_word_count:
        
        # Trust first word's percent
        if first_word_percent < 0.5:
            return first_word, last_word, female_percent_first_word, country_last_word # First word as first name
        else:
            return last_word, first_word, female_percent_last_word, country_first_word   # last word is first name
            
    else:
        # Trust last word's percent
        if last_word_percent >= 0.5:
            return first_word, last_word, female_percent_first_word, country_last_word  # First word as first name
        else:
            return last_word, first_word, female_percent_last_word, country_first_word  ## last word is first name



if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: python step2_data_cleaning.py <input_file_path> <output_dir>")
        # example: python step2_data_cleaning.py data/samples/race/test.csv data/samples/race
        sys.exit(1)

    # Load the CSV file with error handling for malformed rows
    file_path = sys.argv[1]
    output_dir = sys.argv[2]

    df = pd.read_csv(file_path, error_bad_lines=False, warn_bad_lines=True)

    # Step 1: cleaninig (remove rows with empty value, or more than 6 fields, or weird characters)
    df_cleaned = df.dropna(how='any')
    df_cleaned = df_cleaned[df_cleaned.apply(lambda row: len(row) == 6, axis=1)]
    df_cleaned = df_cleaned[~df_cleaned.applymap(contains_weird_characters).any(axis=1)]
    df_cleaned = df_cleaned.drop_duplicates(subset='title', keep='first')
    # print how many dropped
    print(f"Dropped {len(df) - len(df_cleaned)} rows due to missing values, more than 6 fields, weird characters, or duplicates.")
    print(f"Remaining rows: {len(df_cleaned)}")

    # Step 2: create first name and last name list
    df_cleaned['author_list'] = df_cleaned['authors'].str.split(',').apply(lambda x: [author.lower().strip() for author in x if author.strip()])

    df_cleaned['fields_of_study_list'] = df_cleaned['fields_of_study'].str.split(',')


    df_parts = np.array_split(df_cleaned, 30)
    os.makedirs(output_dir, exist_ok=True)

    # Loop over each part and process separately
    for idx, df_part in enumerate(df_parts):
        # Define the output filename for the current part
        output_file = os.path.join(output_dir, f"df_cleaned_part_{idx+1}.csv")
        
        # If the file already exists, skip this part
        if os.path.exists(output_file):
            print(f"Part {idx+1} already processed. Skipping...")
            continue
        
        firstnames = []
        lastnames = []
        female_percents = []
        countries = []
        
        # Process the author lists in each part
        for author_list in tqdm(df_part['author_list'], desc=f"Processing Authors for Part {idx+1}"):
            firstname_row = []
            lastname_row = []
            female_percent_row = []
            country_row = []
            for author in author_list:
                firstname, lastname, female_percent, country= determine_first_last_female_country_facebook(author, solr_url, username, password)
                firstname_row.append(firstname)
                lastname_row.append(lastname)
                female_percent_row.append(female_percent)
                country_row.append(country)
            
            firstnames.append(firstname_row)
            lastnames.append(lastname_row)
            female_percents.append(female_percent_row)
        
        # Add the first names and last names as new columns to this part of the DataFrame
        df_part['fb_firstname'] = firstnames
        df_part['fb_lastname'] = lastnames
        df_part['fb_female_percent'] = female_percents
        
        # Save the processed part to a file
        df_part.to_csv(output_file, index=False)
        print(f"Part {idx+1} processed and saved to {output_file}.")
