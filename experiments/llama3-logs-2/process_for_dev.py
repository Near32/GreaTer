import os
import re

# Directory containing the log files
log_directory = "."
# Directory where the processed files will be saved
processed_directory = "processed_fffff"

# Function to clean up the candidate string
def clean_string(s):
    s = s.replace('▁', ' ')  # Replace `▁` with spaces
    if s.endswith('_'):
        s = s[:-1]  # Remove trailing `_`
    return s

# Create the processed directory if it doesn't exist
os.makedirs(processed_directory, exist_ok=True)

# Regular expression to match the Step lines and extract the candidates
step_line_pattern = re.compile(r"Step:\s+\d+\s+Candidates:\s+\[(.*)\]")

# Loop through all the log files in the directory
for filename in os.listdir(log_directory):
    if filename.endswith("fffff.log"):
        log_filepath = os.path.join(log_directory, filename)
        processed_filename = f"{os.path.splitext(filename)[0]}.txt"
        processed_filename = processed_filename.replace("_fffff", "")
        processed_filepath = os.path.join(processed_directory, processed_filename)

        last_candidates_str = None

        with open(log_filepath, "r") as log_file:
            for line in log_file:
                match = step_line_pattern.search(line)
                if match:
                    # Keep updating last_candidates_str with the latest matched line
                    last_candidates_str = match.group(1)


        if last_candidates_str is not None:
            # Process only the last matched line
            with open(processed_filepath, "w") as processed_file:
                # Use regex to find all tuples and extract the second part (the string)
                candidate_strings = re.findall(r"'(.*?)'", last_candidates_str)
                # Add ". Use" as a prefix and write to the processed file
                for candidate in candidate_strings:
                    cleaned_candidate = clean_string(candidate)
                    processed_file.write(f'. Use{cleaned_candidate}\n')
