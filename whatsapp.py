import requests

# Define the URL and request payload
def sendmessage(message):
    url = " http://localhost:7071/whatsapp/sendmessage"  # Replace with the actual URL
    headers = {
        "Content-Type": "application/json",
        "X-FIREBASE-IDTOKEN": "{{token}}",  # Replace with your actual token
    }
    data = {
    "messaging_product": "whatsapp",
    "recipient_type": "individual",
    "to": "917838351911",
    "type": "interactive",
    "interactive": {
        "type": "button",
        "body": {
            "text": message
        },
        "action": {
            "buttons": [
                {
                    "type": "reply",
                    "reply": {
                        "id": "block",
                        "title": "Block"
                    }
                },
                {
                    "type": "reply",
                    "reply": {
                        "id": "false",
                        "title": "False"
                    }
                }
            ]
        }
    }
}
    #print(f"Data: {data}")

    try:
        # Make a POST request to the API
        response = requests.post(url, json=data, headers=headers)
        #print(f"status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed with an error: {e} {response}")
