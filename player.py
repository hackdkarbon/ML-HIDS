import socket
import csv
import time

# Define the CSV file path
csv_file_path = 'test_data.csv'

# Define the host and port of the receiving program
host = 'localhost'  # Replace with the actual host
port = 9000  # Replace with the actual port

# Open a socket connection
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((host, port))

# Read data from the CSV file and send it
with open(csv_file_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # Assuming the first row is the header

    for row in reader:
        # Format data as needed (e.g., join the columns with commas)
        data = ','.join(row)
        
        # Send data to the receiving program
        client.send(data.encode())
        print(f"Sent: {data}")

        # Wait for one second before sending the next line
        time.sleep(1)

# Close the connection
client.close()
