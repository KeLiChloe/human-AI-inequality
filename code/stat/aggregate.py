import json
import sys

def read_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def write_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def aggregate_json_values(file1, file2, output_file, mode):
    """ Sum values from two JSON files and write the results to a new JSON file. """
    # Read the two JSON files
    data1 = read_json(file1)
    data2 = read_json(file2)
    
    # Initialize the result dictionary
    result = {}

    # Get all unique keys from both dictionaries
    all_keys = set(data1.keys()) | set(data2.keys())

    # Sum values for each key, assuming missing keys as 0
    for key in all_keys:
        if mode == "percent": # add
            result[key] = round(100*(float(data1.get(key, 0)) / float(data2.get(key, 0))), 5)
        elif mode == "sum ": # divide (percentage)
            result[key] = data1.get(key, 0) + data2.get(key, 0)
        else: 
            Exception
            
    write_json(result, output_file)




if __name__ == "__main__":

    if len(sys.argv) < 5:
        print("Usage: python aggregate.py <file1> <file2> <output_file> <mode>")
        sys.exit(1)
    
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    output = sys.argv[3]
    mode = sys.argv[4]
    
    aggregate_json_values(file1, file2, output, mode)