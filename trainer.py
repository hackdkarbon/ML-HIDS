import pandas as pd
import os
import mltrainer as mlcode

# Replace these with your column names
columns = ['ip_src','tcp_srcport','ip_dst',	'tcp_dstport',	'frame_time',	'frame_protocols',	'http_request_method',	'http_request_uri',	'http_host',	'http_user_agent',
           	'http_referer',	'http_response_code',	'http_content_type']  

# Input CSV file path
csv_file_path = 'smalldata.csv'

# Specify the directory to save unique values
unique_values_directory = 'pickle'
data_directory = 'data'
# Create the directory if it doesn't exist
if not os.path.exists(unique_values_directory):
    os.makedirs(unique_values_directory)

# Read the CSV file into a Pandas DataFrame
csv_files = [f for f in os.listdir(data_directory) if f.endswith(".csv")]
for csv_file in csv_files:
    # Read the CSV file into a Pandas DataFrame
    csv_file_path = os.path.join(data_directory, csv_file)
    df = pd.read_csv(csv_file_path)

# Create and save files with unique values for each column
for col_name in columns:
    # Extract unique values from the column
    unique_values = df[col_name].unique()

    # Save the unique values to a file
    with open(os.path.join(unique_values_directory, f'{col_name}.pkl'), 'w') as file:
        file.write('\n'.join(str(value) for value in unique_values))
print("#### Non-ML training finished successfully")
##call ml/training code to perform ML training 
mlcode.main(csv_file_path)
print("#### all trainings finished successfully")

# To load and lookup unique values later in a different program:
# Example usage:
# col_name = "column1"  # Specify the column name you want to look up
# with open(os.path.join(unique_values_directory, f'{col_name}_unique_values.txt'), 'r') as file:
#     unique_values = file.read().splitlines()
# print(unique_values)
