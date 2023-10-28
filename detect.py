import socket
import os
import datetime
import argparse
import mldetect as mldetect
import whatsapp as whatsapp

# Specify the directory containing the unique values files
unique_values_directory = 'pickle'

# Create the anomaly-dat directory if it doesn't exist
anomaly_dat_directory = 'anomaly-data'
if not os.path.exists(anomaly_dat_directory):
    os.makedirs(anomaly_dat_directory)

# Function to load unique values from a file
def load_unique_values(column_name):
    unique_values = set()
    unique_values_file = os.path.join(unique_values_directory, f'{column_name}.pkl')
    if os.path.exists(unique_values_file):
        with open(unique_values_file, 'r') as file:
            unique_values = set(file.read().splitlines())
    return unique_values

# Define the column sequence
column_sequence = [
    "ip_src", "tcp_srcport", "ip_dst", "tcp_dstport",
    "frame_time", "frame_protocols", "http_request_method",
    "http_request_uri", "http_host", "http_user_agent",
    "http_referer", "http_response_code", "http_content_type"
]

def process_data(client_socket):
    while True:
        data = client_socket.recv(1024).decode('utf-8')
        if not data:
            print("no data")
            continue

        # Split the received data into columns
        columns = data.split(',')

        if len(columns) != len(column_sequence):
            print("Received data does not match the expected column sequence.")
            continue
        anomaly = 0
        anomalycols = ""
        for col_name, col_value in zip(column_sequence, columns):
            unique_values = load_unique_values(col_name)
            if col_value not in unique_values:
                current_datetime = datetime.datetime.now()
                file_name = current_datetime.strftime('%d%m%Y') + '.csv'
                with open(os.path.join(anomaly_dat_directory, file_name), 'a') as file:
                    file.write(data + '\n')      
                anomaly = anomaly + 1
                anomalycols += col_name +":" + col_value + ","

        if (anomaly >= 1):
            print(f"#################### Anomaly Detected ############### Mismatch  { round((anomaly/len(columns)),2) * 100 }% for mismatched coloum(s) {anomalycols}")
            whatsapp.sendmessage(f"*Anomaly Detected* check anomaly folder for details ## Mismatch  {  round((anomaly/len(columns)),2) * 100 } % for mismatched coloum(s) {anomalycols}")
        else:
            print("#### known traffic pattern")
        try:
            mldetect.process_data(data)
        except:
            print("#### ML model need correction")

        #client_socket.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9001, help='Port number to listen on')
    args = parser.parse_args()

    # Create a socket to listen on the specified port    
    server_address = ('localhost', args.port)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(server_address)
    server_socket.listen(1)

    print(f"Awaitng traffic on port {args.port}...")

    while True:
        client_socket, client_address = server_socket.accept()
        print(f"started connection from {client_address}")
        process_data(client_socket)
