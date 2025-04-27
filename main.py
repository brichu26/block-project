import csv
from github_evaluator import get_popularity_score, get_documentation_score
from api_evaluator import test_oauth_implementation
import json
import requests
from datetime import datetime, timedelta

def fetch_and_save_pulsemcp_servers(num_servers):
    url = "https://api.pulsemcp.com/v0beta/servers"
    headers = {
        "User-Agent": "MyToolManager/1.0 (https://mytoolmanager.com)"
    }
    params = {
        "count_per_page": num_servers
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        servers = data.get("servers", [])
        cleaned_servers = []
        for s in servers:
            source_code_url = s.get("source_code_url")
            if source_code_url:
                # Extract owner and repo name from the source_code_url
                parts = source_code_url.rstrip('/').split('/')
                if len(parts) >= 2:
                    owner, repo = parts[-2], parts[-1]
                    cleaned_servers.append({
                        "owner": owner,
                        "repo": repo
                    })
        # Save to JSON
        with open("pulsemcp_servers.json", "w") as f:
            json.dump(cleaned_servers, f, indent=2)

        print(f" Saved {len(cleaned_servers)} server(s) to pulsemcp_servers.json")

    except requests.exceptions.RequestException as e:
        print(f" API call failed: {e}")

def fetch_pulsemcp_csv(num_servers, name):
    url = "https://api.pulsemcp.com/v0beta/servers"
    headers = {
        "User-Agent": "MyToolManager/1.0 (https://mytoolmanager.com)"
    }
    params = {
        "count_per_page": num_servers
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        servers = data.get("servers", [])

        # Extract relevant fields
        cleaned_servers = []
        for s in servers:
            source_code_url = s.get("source_code_url")
            if source_code_url:
                parts = source_code_url.rstrip('/').split('/')
                if len(parts) >= 2:
                    owner, repo = parts[-2], parts[-1]
                    cleaned_servers.append({
                        "owner": owner,
                        "repo": repo,
                        "github_stars": s.get("github_stars", 0),
                        "download_count": s.get("package_download_count", 0),
                        "experimental_ai_generated_description": s.get("EXPERIMENTAL_ai_generated_description", "")
                    })

        # Save to CSV
        with open(name, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["owner", "repo", "github_stars", "download_count", "experimental_ai_generated_description"])
            writer.writeheader()
            writer.writerows(cleaned_servers)

        print(f"✅ Saved {len(cleaned_servers)} server(s) to pulsemcp_servers.csv")

    except requests.exceptions.RequestException as e:
        print(f"❌ API call failed: {e}")

def fetch_pulsemcp_json(num_servers, name):
    url = "https://api.pulsemcp.com/v0beta/servers"
    headers = {
        "User-Agent": "MyToolManager/1.0 (https://mytoolmanager.com)"
    }
    params = {
        "count_per_page": num_servers
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        servers = data.get("servers", [])

        # Extract relevant fields
        cleaned_servers = []
        for s in servers:
            source_code_url = s.get("source_code_url")
            if source_code_url:
                parts = source_code_url.rstrip('/').split('/')
                if len(parts) >= 2:
                    owner, repo = parts[-2], parts[-1]
                    cleaned_servers.append({
                        "owner": owner,
                        "repo": repo,
                        "github_stars": s.get("github_stars", 0),
                        "download_count": s.get("package_download_count", 0),
                        "experimental_ai_generated_description": s.get("EXPERIMENTAL_ai_generated_description", "")
                    })

        # Save to JSON
        with open(name, "w", encoding="utf-8") as f:
            json.dump(cleaned_servers, f, ensure_ascii=False, indent=4)

        print(f"✅ Saved {len(cleaned_servers)} server(s) to {name}")

    except requests.exceptions.RequestException as e:
        print(f"❌ API call failed: {e}")

#fetch_pulsemcp_json(4000, "pulsemcp_servers_all.json") #save pulsemcp servers to json
fetch_pulsemcp_csv(4000, "pulsemcp_servers_all.csv") #save pulsemcp servers to csv
#fetch_pulsemcp_csv(10, 'small_test') #save pulsemcp servers to csv


#fetch_and_save_pulsemcp_servers(200) #save pulsemcp servers to json

#load the saved servers
# with open("pulsemcp_servers.json", "r") as f:
#     servers = json.load(f)

# for server in servers[:5]:
#     owner = server["owner"]
#     repo = server["repo"]
#     # Fetch GitHub popularity score 
#     repo_score = get_popularity_score(owner, repo)
#     #print(f"Popularity score for {owner}/{repo}: {repo_score}")
#     doc_score = get_documentation_score(owner, repo)
#     #print(f"doc score for {owner}/{repo}: {doc_score}")
#     url = f"https://api.github.com/repos/{owner}/{repo}"
#     print(test_oauth_implementation(url))
#     #print(f"Repo URL: {url}")


