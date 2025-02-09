import pandas as pd
import sys
import ast
import numpy as np
from create_country_json import calculate_simpson_index, calculate_shannon_entropy, calculate_inverse_dominance

# Load the two CSV files

field_groups = {
    'natural_sciences': [
        'Biology', 'Chemistry', 'Physics', 'Geology', 'Geography',
        'Environmental Science', 'Medicine', 'Materials Science'
    ],
    'engineering_and_technology': [
        'Engineering', 'Computer Science', 'Mathematics'
    ],
    'social_sciences': [
        'Sociology', 'Political Science', 'Psychology', 
        'Economics', 'History', 'Art', 'Philosophy'
    ]
}

def map_fields_of_study(fields_list, field_groups):
    if isinstance(fields_list, str):
        fields_list = eval(fields_list)
    mapped_fields = {key: 0 for key in field_groups.keys()}
    for field in fields_list:
        for category, fields in field_groups.items():
            if field in fields:
                mapped_fields[category] = 1
    return pd.Series(mapped_fields)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        # e.g., python code/ml/step3_sql_join_country.py data/samples/test data/samples/test/sql_country_with_inequality_word_count.csv meta_data/wordlist.json race
        print("Usage: python step5.py <file1_path> <file2_path> <output_file_path>")

    file1_path = sys.argv[1]
    file2_path = sys.argv[2]
    output_file_path = sys.argv[3]

    print(f"Loading CSV files: {file1_path}, {file2_path}")

    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)


    df_cleaned = pd.concat([df1, df2], axis=0)
    df_cleaned = df_cleaned.reset_index(drop=True) # don't delete this line, otherwise the index will be duplicated
    df_cleaned = df_cleaned.dropna()
    

    df_transformed = pd.DataFrame()

    df_transformed['title'] = df_cleaned['title']
    df_transformed['paper_abstract'] = df_cleaned['paper_abstract']
    df_transformed['count_frequency_inequality_words'] = df_cleaned['count_frequency_inequality_words']
    df_transformed['number_of_authors'] = df_cleaned['author_list'].apply(lambda x: len(eval(x)))

    df_cleaned['country_highest_ratio_race'] = df_cleaned['country_highest_ratio_race'].apply(ast.literal_eval)
    # unique_races = set(race for sublist in data['highest_ratio_race'] for race in sublist)
    unique_races = ['white', 'black', 'asian', 'hispanic', 'native_hawaiian_or_other_pacific_islander',
                    'native_americans', 'mixed', 'other']
    
    race_percent_col_names = {col: col + "_percent" for col in unique_races}
    
    for race in unique_races:
        df_transformed[f'{race}'] = df_cleaned['country_highest_ratio_race'].apply(lambda x: x.count(race) if isinstance(x, list) else 0)

    # Convert to percentages and rename
    # race_sum = df_transformed[unique_races].sum(axis=1)
    # df_transformed[unique_races] = df_transformed[unique_races].div(race_sum, axis=0)
    # df_transformed.rename(columns=race_percent_col_names, inplace=True)


    df_transformed[['natural_sciences', 'engineering_and_technology', 'social_sciences']] = df_cleaned['fields_of_study_list'].apply(lambda x: map_fields_of_study(eval(x), field_groups))

    df_transformed['female_score_mean'] = df_cleaned['fb_female_percent'].apply(lambda x: sum(eval(x)) / len(eval(x)))
    df_transformed['female_score_max'] = df_cleaned['fb_female_percent'].apply(lambda x: max(eval(x)))
    df_transformed['female_score_min'] = df_cleaned['fb_female_percent'].apply(lambda x: min(eval(x)))
    df_transformed['first_author_female_score'] = df_cleaned['fb_female_percent'].apply(lambda x: eval(x)[0])

    df_transformed['country_race_shannon_entropy_mean'] = df_cleaned['country_race_shannon_entropy'].apply(lambda x: sum(eval(x)) / len(eval(x)))
    df_transformed['country_race_simpson_index_mean'] = df_cleaned['country_race_simpson_index'].apply(lambda x: sum(eval(x)) / len(eval(x)))
    df_transformed['country_race_inverse_dominance_mean'] = df_cleaned['country_race_inverse_dominance'].apply(lambda x: sum(eval(x)) / len(eval(x)))
    
    # df_transformed['country_race_shannon_entropy_max'] = df_cleaned['country_race_shannon_entropy'].apply(lambda x: max(eval(x)))
    # df_transformed['country_race_simpson_index_max'] = df_cleaned['country_race_simpson_index'].apply(lambda x: max(eval(x)))
    # df_transformed['country_race_inverse_dominance_max'] = df_cleaned['country_race_inverse_dominance'].apply(lambda x: max(eval(x)))

    race_sum = df_transformed[unique_races].sum(axis=1)
    
    df_race_proportions = df_transformed[unique_races].div(race_sum, axis=0)

    # first author race 
    dummies = pd.get_dummies(df_cleaned['country_highest_ratio_race'].apply(lambda x: x[0]), prefix='first_author_race')
    df_transformed = pd.concat([df_transformed, dummies], axis=1)

    df_transformed['paper_race_shannon_entropy'] = df_race_proportions.apply(lambda x: calculate_shannon_entropy(x), axis=1)
    df_transformed['paper_race_simpson_index'] = df_race_proportions.apply(lambda x: calculate_simpson_index(x), axis=1)
    df_transformed['paper_race_inverse_dominance'] = df_race_proportions.apply(lambda x: calculate_inverse_dominance(x), axis=1)


    df_transformed.to_csv(output_file_path, index=False)
    print(f"Updated CSV saved to {output_file_path}")

