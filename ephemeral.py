import socket
import pickle
import csv
import threading

# Load the trained ML model
with open('trained_model.pkl', 'rb') as model_file:
    trained_model = pickle.load(model_file)

# Define the host and port to listen on
host = '0.0.0.0'  # Listen on all available network interfaces
port = 1234

# Create a CSV log file
csv_log_file = 'traffic_log.csv'
csv_log_header = ['Source IP', 'Original Destination IP', 'Classification']

# Create a dictionary to store mappings from source port to original destination IP
port_to_destination = {}

# Function to forward traffic to its original destination
def forward_traffic(client_socket, destination_ip):
    try:
        destination_port = port_to_destination[destination_ip]
        destination_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        destination_socket.connect((destination_ip, destination_port))
        while True:
            data = client_socket.recv(1024)
            if not data:
                break
            destination_socket.send(data)
        destination_socket.close()
        client_socket.close()
    except Exception as e:
        print(f"Error forwarding traffic: {e}")

# Create a socket to listen for incoming connections
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    server_socket.bind((host, port))
    server_socket.listen(1)  # Listen for one incoming connection at a time

    print(f"Listening on port {port}...")

    # Infinite loop to continuously handle incoming connections
    while True:
        try:
            client_socket, client_address = server_socket.accept()
            print(f"Accepted connection from {client_address}")

            # Receive data from the client
            data = client_socket.recv(1024)  # Adjust buffer size as needed

            if not data:
                print("No data received. Closing connection.")
                client_socket.close()
                continue

            # Preprocess the received data (replace with your preprocessing logic)
            def preprocess_input_data(input_data):
                # Your preprocessing code here
                # Ensure that the input_data is in the correct format for your model
                return preprocessed_data

            preprocessed_data = preprocess_input_data(data)

            # Use the ML model to make predictions
            prediction = trained_model.predict([preprocessed_data])

            # Interpret the model's prediction
            if prediction == 0:
                result = "Normal Traffic"

                # Forward the normal traffic to its original destination
                source_ip, source_port = client_address
                if source_ip in port_to_destination:
                    forwarding_thread = threading.Thread(target=forward_traffic, args=(client_socket, source_ip))
                    forwarding_thread.start()
                else:
                    print("No mapping found for source IP. Closing connection.")
                    client_socket.close()

            else:
                result = "Anomalous Traffic"

                # Log anomalous traffic to the CSV file
                with open(csv_log_file, 'a', newline='') as log_file:
                    csv_writer = csv.writer(log_file)
                    csv_writer.writerow([client_address[0], port_to_destination.get(client_address[0], 'Unknown'), result])

            print(f"Classification result: {result}")

        except Exception as e:
            print(f"Error handling connection: {e}")
