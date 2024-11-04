import requests
import os

# API URL (adjust if hosted elsewhere or using a different port)
ip = "0.0.0.0" # Change to the IP address of the server
port = "0000" # Change to the port number of the server

url = f"http://{ip}:{port}/synthesize" # URL for the TTS API

# Data payload for the TTS request
data = {
    "text": "What are you doing? Oh, you're a programmer! That's awesome! What project are you working on?",
    "f0_up_key": 8,                       # Optional pitch adjustment
    "f0_method": "rmvpe",                 # Pitch extraction method
    "index_rate": 0,                      # Index rate for model processing
    "protect": 0.33                       # Protection rate for the synthesis
}

# Sending POST request to the API
print(f"Sending TTS request to {url}...")
response = requests.post(url, json=data)
file_name = "U.E.P_reply"

def get_unique_filename(base_name, extension=".wav"):
    counter = 1
    unique_name = f"{base_name}{extension}"
    # Increment file name until a non-existing name is found
    while os.path.exists(unique_name):
        unique_name = f"{base_name} ({counter}){extension}"
        counter += 1
    return unique_name

# Handling the response
if response.status_code == 200:
    unique_file_name = get_unique_filename(file_name)
    
    with open(unique_file_name, "wb") as f:
        f.write(response.content)
    print(f"Audio saved as {unique_file_name}")
else:
    print(f"Failed to synthesize audio. Status code: {response.status_code}")
    print("Error:", response.json().get("error"))
