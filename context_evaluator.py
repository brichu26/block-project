import requests
import json 

def send_message(server_url, message, session_id):
    payload = {
        "session_id": session_id,
        "message": message
    }
    
    response = requests.post(f"{server_url}/interact", json=payload)
    if response.status_code == 200:
        return response.json().get('response')
    else:
        print(f"Failed to interact with the server: {response.status_code}")
        return None

def evaluate_interaction_context(server_url):
    session_id = "test-session-123"
    context_ability_score = 0

    # Define a series of interactions
    interactions = [
        ("Hello, can you help me find a book?", "What kind of book are you looking for?"),
        ("I'm looking for science fiction.", "Do you have a specific author in mind?"),
        ("Yes, Arthur C. Clarke.", "You might enjoy 'Rendezvous with Rama'."),
    ]

    for user_message, expected_response in interactions:
        response = send_message(server_url, user_message, session_id)
        if response and expected_response in response:
            context_ability_score += 10  # Award points for expected behavior
        else:
            print(f"Context issue: expected: '{expected_response}', got: '{response}'")

    context_ability_score = context_ability_score / len(interactions)
    return context_ability_score

def main():
    # Load the saved servers
    with open("pulsemcp_servers.json", "r") as f:
        servers = json.load(f)

    for server in servers:
        server_url = server.get('url')
        if server_url:
            context_ability_score = evaluate_interaction_context(server_url)
            print(f"Server URL: {server_url}\nContext Ability Score: {context_ability_score}\n")

if __name__ == "__main__":
    main()