import socket
import random
import time
import configparser

# Initialize ctr and servicectr
ctr = 0
servicectr = -1

# Function to load configuration
def load_config():
    config = configparser.ConfigParser()
    config.read('canary.ini')
    services = []

    for service in config.sections():
        services.append({
            "name": service,
            "host": config[service]["host"],
            "port": int(config[service]["port"]),
            "weight": int(config[service]["weight"]),
            "available": True  # Indicate if the service is available
        })

    return services

# Function to create and manage a pool of service connections
def create_connection_pool(services):
    connection_pool = {}

    for service in services:
        host, port = service["host"], service["port"]
        connection_pool[(host, port)] = []

    return connection_pool

# Function to send data to a service
def send_data_to_service(service, data_line, connection_pool):
    host, port = service["host"], service["port"]

    if not connection_pool.get((host, port)):
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            connection.connect((host, port))
            connection_pool[(host, port)].append(connection)
        except Exception as e:
            print(f"Error connecting to service {service['name']}: {e}")
            service["available"] = False
            return

    connection = connection_pool[(host, port)].pop()
    try:
        connection.send(data_line.encode('utf-8'))
        connection_pool[(host, port)].append(connection)
        #print(f"Sent data {data_line} to service {service['name']}")
    except Exception as e:
        print(f"Error sending data to service {service['name']}: {e}")
        connection.close()
        service["available"] = False

# Function to select a service based on weightage
def select_service(services):
    global ctr, servicectr  # Use the global keyword to modify these variables
    available_services = [service for service in services if service["weight"]]
    
    if not available_services:
        return None

    sorted_services = sorted(available_services, key=lambda item: item["weight"])

    if ctr > sorted_services[servicectr]["weight"]:
        ctr = 0
    
    if servicectr >= len(sorted_services) -1:
            servicectr = 0
    else:
            servicectr = servicectr + 1

    return sorted_services[servicectr]

# Create a socket to listen on port 9000
server_address = ('localhost', 9000)
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(server_address)
server_socket.listen(1)

print("Listening on port 9000...")

# Initialize the last config update time and connection pool
last_config_update_time = 0
services = load_config()
connection_pool = create_connection_pool(services)

# Main loop
while True:
    client_socket, client_address = server_socket.accept()
    print(f"Accepted connection from {client_address}")

    try:
        # Check if it's time to reload the configuration (every 5 minutes)
        if time.time() - last_config_update_time >= 300:
            services = load_config()
            connection_pool = create_connection_pool(services)
            last_config_update_time = time.time()

        # Read data line by line from the client socket and send it to the selected service
        ctr = 0
        servicectr = -1
        while True:
            data_line = client_socket.recv(1024).decode('utf-8')
            selected_service = select_service(services)
            print(f"Relaying to service {selected_service}")
            if selected_service:
                if not data_line:
                    print("no data line")
                    break
                try:
                    send_data_to_service(selected_service, data_line, connection_pool)
                except Exception as e:
                    print(f"Error sending data to service {selected_service['name']}: {e}")
                ctr = ctr + 1
            else:
                print("No available service selected based on load balancing configuration.")

    except Exception as e:
        print(f"Error: {e}")

    # Close the client socket
    client_socket.close()
