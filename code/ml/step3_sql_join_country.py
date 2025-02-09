import sqlite3
import os
import glob
import pandas as pd
import json
import pandas as pd
import re
import sys



def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
# Function to count word occurrences in a given text
def count_words(text, words):
    # Convert the text to lowercase for case-insensitive matching
    text = text.lower()
    
    # Initialize the count to 0
    total_count = 0
    
    # Count the occurrences of each word
    for word in words:
        # Escape special characters and count occurrences using regular expressions
        total_count += len(re.findall(r'\b' + re.escape(word) + r'\b', text))
    
    return total_count


def process_file(df, output_file_path, all_words):
    print(len(df))
    df = df.drop_duplicates(subset='title', keep='first')
    print(len(df))

    df['count_frequency_inequality_words'] = df.apply(
        lambda row: count_words(str(row['title']) + ' ' + str(row['paper_abstract']), all_words), axis=1)
    
    df.to_csv(output_file_path, index=False)
    print(f"Updated CSV with inequality word count saved to {output_file_path}")

def get_country_by_name_sql(db_path: str, firstname: str, lastname: str) -> str:
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query using the last name first
    query_lastname = """
    SELECT place
    FROM surnames
    WHERE name_lookup = ?
    ORDER BY incidence DESC
    LIMIT 1;
    """
    cursor.execute(query_lastname, (lastname,))
    result = cursor.fetchone()

    # If a result is found with the last name, return it
    if result:
        cursor.close()
        conn.close()
        return result[0]

    # If no result, query using the first name
    query_firstname = """
    SELECT place
    FROM surnames
    WHERE name_lookup = ?
    ORDER BY incidence DESC
    LIMIT 1;
    """
    cursor.execute(query_firstname, (firstname,))
    result = cursor.fetchone()

    # Close the database connection
    cursor.close()
    conn.close()

    # If a result is found with the first name, return it
    if result:
        return result[0]
    
    # If no match for both names, return None
    return None


def populate_sql_country(df):
    sql_countries = []
    
    for _, row in df.iterrows():
        firstnames = eval(row['fb_firstname'])  # Convert string list back to actual list
        lastnames = eval(row['fb_lastname'])
        
        country_results = []
        
        # Iterate over the pairs of first and last names
        for i in range(len(firstnames)):
            country = get_country_by_name_sql(db_path, firstnames[i], lastnames[i])
            country_results.append(country)
        
        # Store the list of countries in the new column
        sql_countries.append(country_results)
    
    df['sql_country'] = sql_countries
    return df

def process_all_and_merge_csv_files(directory):
    # Use glob to find all files that start with 'df_cleaned_part_' and end with '.csv'
    csv_files = glob.glob(os.path.join(directory, "df_cleaned_part_*.csv"))
    
    # Initialize an empty list to store DataFrames
    df_list = []
    
    for csv_file in csv_files:
        # Load each CSV into a DataFrame
        df = pd.read_csv(csv_file)
        df_list.append(df)
    
    # Concatenate all DataFrames into a single DataFrame
    merged_df = pd.concat(df_list, ignore_index=True)
    
    # Apply the populate_sql_country function to the merged DataFrame
    df_updated = populate_sql_country(merged_df)  
    return df_updated

def create_country_race_diversity_columns(json_file_path, df):
    with open(json_file_path, 'r') as f:
        country_data = json.load(f)

    # Initialize new columns
    df['country_highest_ratio_race'] = None
    df['country_race_shannon_entropy'] = None
    df['country_race_simpson_index'] = None
    df['country_race_inverse_dominance'] = None

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        countries = row['sql_country']  # Convert string representation of list into actual list
        if None in countries:
            continue
        highest_ratio_race_list = []
        shannon_entropy_list = []
        simpson_index_list = []
        inverse_dominance_list = []
        
        # For each country in the author's list
        for country in countries:
            if country == 'England':
                country = 'UNITED KINGDOM'
            if country == 'South Korea':
                country = 'KOREA, REPUBLIC OF'
            if country == 'Vietnam':
                country = 'VIET NAM'
            if country == 'Iran':
                country = 'IRAN, ISLAMIC REPUBLIC OF'
            if country == 'Russia':
                country = 'RUSSIAN FEDERATION'
            country_info = country_data.get(country.upper())  # Match the country in JSON (case insensitive)
            if country_info:
                highest_ratio_race_list.append(country_info['highest_ratio_race'])
                shannon_entropy_list.append(country_info['shannon_entropy'])
                simpson_index_list.append(country_info['simpson_index'])
                inverse_dominance_list.append(country_info['inverse_dominance'])
            else:
                break

        if len(highest_ratio_race_list) > 0:
            # Assign the lists as new column values
            df.at[index, 'country_highest_ratio_race'] = highest_ratio_race_list
            df.at[index, 'country_race_shannon_entropy'] = shannon_entropy_list
            df.at[index, 'country_race_simpson_index'] = simpson_index_list
            df.at[index, 'country_race_inverse_dominance'] = inverse_dominance_list
    return df


if __name__ == '__main__':
    if len(sys.argv) < 5:
        # e.g., python code/ml/step3_sql_join_country.py data/samples/test data/samples/test/sql_country_with_inequality_word_count.csv meta_data/wordlist.json race
        print("Usage: python step3_sql_join_country.py <input_files_dir> <output_file_path> <word_list_file_path> <count_category> ")
        # python <> data/samples/test-neg data/samples/test-neg/no-ineq-sql_country_with_inequality_word_count.csv meta_data/wordlist.json "['race']"

json_file_path = 'meta_data/country_race_diversity_data.json'  
db_path = '/Users/like/Desktop/Research/Human-AI/database/forebears-surnames.sqlite'

directory = sys.argv[1]
output_file_path = sys.argv[2]
word_list_file_path = sys.argv[3]
count_category = eval(sys.argv[4])

wordlist = read_json_file(word_list_file_path)

all_words = []
for category in count_category:
    all_words += wordlist[category]["0"] + wordlist[category]["1"] + wordlist[category]["-1"]

print(len(all_words))
print(all_words)


df_tmp = process_all_and_merge_csv_files(directory)
df = create_country_race_diversity_columns(json_file_path, df_tmp)
df = df.dropna()
process_file(df,output_file_path, all_words)
