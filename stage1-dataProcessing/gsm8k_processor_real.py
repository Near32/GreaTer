import json
import csv

# Function to extract goal, target, and final_target from each JSONL line
def extract_data_from_jsonl(jsonl_file):
    with open(jsonl_file, 'r') as f:
        lines = f.readlines()
        data = []
        for line in lines:
            line_data = json.loads(line.strip())
            question = line_data['question']
            answer = line_data['answer']
            final_target = str(answer.split("####")[-1].strip())
            data.append({'goal': question, 'target': answer, 'final_target': final_target})
    return data

# Function to write extracted data into a CSV file
def write_to_csv(data, csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['goal', 'target', 'final_target'])
        writer.writeheader()
        for row in data:
            writer.writerow(row)

# Path to your x.jsonl file
jsonl_file = '/scratch1/sfd5525/prompt_optimization/data/grade_school_math/data/test.jsonl'

# Path to the new CSV file
csv_file = '/scratch1/sfd5525/prompt_optimization/data/grade_school_math/processed/test_new.csv'

# Extract data from JSONL file
extracted_data = extract_data_from_jsonl(jsonl_file)

# Write extracted data to CSV file
write_to_csv(extracted_data, csv_file)

print("CSV file generated successfully.")
