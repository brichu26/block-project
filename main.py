import csv
import json
import requests
from datetime import datetime, timedelta
from typing import Optional


#wenjia's fetch function for pulsemcp
# def fetch_and_save_pulsemcp_servers(num_servers):
#     url = "https://api.pulsemcp.com/v0beta/servers"
#     headers = {
#         "User-Agent": "MyToolManager/1.0 (https://mytoolmanager.com)"
#     }
#     params = {
#         "count_per_page": num_servers
#     }
#     try:
#         response = requests.get(url, headers=headers, params=params)
#         response.raise_for_status()
#         data = response.json()
#         servers = data.get("servers", [])
#         cleaned_servers = []
#         for s in servers:
#             source_code_url = s.get("source_code_url")
#             if source_code_url:
#                 # Extract owner and repo name from the source_code_url
#                 parts = source_code_url.rstrip('/').split('/')
#                 if len(parts) >= 2:
#                     owner, repo = parts[-2], parts[-1]
#                     cleaned_servers.append({
#                         "owner": owner,
#                         "repo": repo
#                     })
#         # Save to JSON
#         with open("pulsemcp_servers.json", "w") as f:
#             json.dump(cleaned_servers, f, indent=2)

#         print(f" Saved {len(cleaned_servers)} server(s) to pulsemcp_servers.json")

#     except requests.exceptions.RequestException as e:
#         print(f" API call failed: {e}")

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

# def fetch_pulsemcp_json(num_servers, name):
#     url = "https://api.pulsemcp.com/v0beta/servers"
#     headers = {
#         "User-Agent": "MyToolManager/1.0 (https://mytoolmanager.com)"
#     }
#     params = {
#         "count_per_page": num_servers
#     }

#     try:
#         response = requests.get(url, headers=headers, params=params)
#         response.raise_for_status()
#         data = response.json()
#         servers = data.get("servers", [])

#         # Extract relevant fields
#         cleaned_servers = []
#         for s in servers:
#             source_code_url = s.get("source_code_url")
#             if source_code_url:
#                 parts = source_code_url.rstrip('/').split('/')
#                 if len(parts) >= 2:
#                     owner, repo = parts[-2], parts[-1]
#                     cleaned_servers.append({
#                         "owner": owner,
#                         "repo": repo,
#                         "github_stars": s.get("github_stars", 0),
#                         "download_count": s.get("package_download_count", 0),
#                         "experimental_ai_generated_description": s.get("EXPERIMENTAL_ai_generated_description", "")
#                     })

#         # Save to JSON
#         with open(name, "w", encoding="utf-8") as f:
#             json.dump(cleaned_servers, f, ensure_ascii=False, indent=4)

#         print(f"✅ Saved {len(cleaned_servers)} server(s) to {name}")

#     except requests.exceptions.RequestException as e:
#         print(f"❌ API call failed: {e}")


#THIS IS HALUK'S FETCH FUNCTION
def fetch_servers(
                count: int = 100,
                query: Optional[str] = None,
                offset: int = 0,
                output_file: Optional[str] = None):
    """
    Fetch MCP servers and sort by package download count.
    
    Args:
        count (int): Number of servers to fetch (max 5000)
        query (str, optional): Search term to filter servers
        offset (int): Number of results to skip for pagination
        output_file (str, optional): Name of the CSV file to save results
        
    Returns:
        List[Dict]: List of server data
        
    Raises:
        requests.exceptions.RequestException: If the API request fails
    """
    BASE_URL = "https://api.pulsemcp.com/v0beta"
    headers = {
           "User-Agent": "MyToolManager/1.0 (https://mytoolmanager.com)"
       }
    params = {
        "count_per_page": 5000,  # Get maximum allowed to ensure we have all servers
        "offset": offset
    }
    
    if query:
        params["query"] = query
        
    try:
        response = requests.get(
            f"{BASE_URL}/servers",
            headers=headers,
            params=params,
            timeout=10
        )
        response.raise_for_status()
        
        data = response.json()
        servers = data.get("servers", [])
        
        # Sort servers by download count
        sorted_servers = sorted(
            servers,
            key=lambda x: x.get("github_stars", 0) or 0,
            reverse=True
        )
        
        # Return only the requested count
        list_mcp_information = []
        sorted_servers_final = sorted_servers #deleting the [:count] from here
        
        for server in sorted_servers_final:
            # Extract owner and repo from GitHub URL
            github_url = server.get("source_code_url", "")
            owner, repo = "", ""
            if github_url:
                # Remove any trailing slashes and split
                parts = github_url.rstrip('/').split('/')
                if len(parts) >= 2:
                    # Find the index of 'github.com' and get the next two parts
                    try:
                        github_index = parts.index('github.com')
                        if len(parts) > github_index + 2:
                            owner = parts[github_index + 1]
                            repo = parts[github_index + 2]
                    except ValueError:
                        pass
            


            list_mcp_information.append({
                'server_name': server.get("name", ""),
                'server_url': server.get("url", ""),
                'github_url': github_url,
                'owner': owner,
                'repo': repo,
                'github_stars': server.get("github_stars", 0), #changed name 
                'download_count': server.get("package_download_count", 0),
                'experimental_ai_generated_description': server.get("EXPERIMENTAL_ai_generated_description", "")
            })

        # Save to CSV if output_file is provided
        if output_file:
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list_mcp_information[0].keys())
                writer.writeheader()
                writer.writerows(list_mcp_information)
            print(f"✅ Saved {len(list_mcp_information)} server(s) to {output_file}")

        return list_mcp_information
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching servers from PulseMCP API: {e}")
        return []

#fetch_servers(output_file="pulsemcp_servers_all.csv") #save pulsemcp servers to csv

#fetch_pulsemcp_csv(5000, "pulsemcp_servers_all_updated.csv") #save pulsemcp servers to csv
#fetch_pulsemcp_json(4000, "pulsemcp_servers_all_updated.json") #save pulsemcp servers to json
#fetch_pulsemcp_csv(4000, "pulsemcp_servers_all_updated.csv") #save pulsemcp servers to csv
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

# import csv

# # Input and output filenames
# input_file = 'pulsemcp_servers_all_updated.csv'
# output_file = 'sorted_data_by_stars.csv'

# # Read and sort the CSV
# with open(input_file, newline='', encoding='utf-8') as csvfile:
#     reader = csv.DictReader(csvfile)
#     sorted_rows = sorted(reader, key=lambda row: int(row['github_stars']) if row['github_stars'].isdigit() else 0, reverse=True)

# # Write only the top 300 sorted data back to a new CSV
# with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
#     writer = csv.DictWriter(csvfile, fieldnames=reader.fieldnames)
#     writer.writeheader()
#     writer.writerows(sorted_rows[:500])

# print(f"CSV sorted by 'github_stars' and saved to {output_file}")



#wenjia's get offical repos from readme
# import re
# import csv

# def extract_github_repos_from_readme(file_path):
#     """
#     Extracts GitHub repository owner/repo pairs and URLs from a README.md file.

#     Args:
#         file_path (str): Path to the README.md file

#     Returns:
#         list[tuple]: List of (owner, repo, url) tuples
#     """
#     github_repo_pattern = re.compile(r'(https://github\.com/([\w\.-]+)/([\w\.-]+))')
#     repos = set()

#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             matches = github_repo_pattern.findall(line)
#             for url, owner, repo in matches:
#                 repos.add((owner, repo, url))

#     return sorted(repos)

# def write_repos_to_csv(repos, output_path):
#     """
#     Writes the list of repositories to a CSV file.

#     Args:
#         repos (list[tuple]): List of (owner, repo, url) tuples
#         output_path (str): Path to output CSV file
#     """
#     with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['owner', 'repo', 'server_github_url'])  # Header
#         writer.writerows(repos)

# if __name__ == "__main__":
#     readme_path = "mcp_README_official.md"  # Adjust path as needed
#     repositories = extract_github_repos_from_readme(readme_path)
#     output_csv = "official_repos_name.csv"
#     write_repos_to_csv(repositories, output_csv)
#     print(f"Extracted {len(repositories)} repos to {output_csv}")



import pandas as pd

# Load CSVs
source_df = pd.read_csv('pulsemcp_servers_all.csv')
target_df = pd.read_csv('official_repos_name.csv')
#target_df = pd.read_csv('reference_mcp.csv')


# Drop columns with no name or unnamed columns
source_df = source_df.loc[:, source_df.columns.notna()]
target_df = target_df.loc[:, target_df.columns.notna()]

source_df = source_df.loc[:, ~source_df.columns.str.fullmatch('Unnamed.*|\\s*')]
target_df = target_df.loc[:, ~target_df.columns.str.fullmatch('Unnamed.*|\\s*')]

# Merge
common_keys = ['owner', 'repo']
merged_df = pd.merge(
    target_df,
    source_df[['owner', 'repo', 'github_stars', 'download_count', 'experimental_ai_generated_description']],
    on=common_keys,
    how='left'
)

# Drop empty-named columns post-merge (just in case)
merged_df = merged_df.loc[:, merged_df.columns.str.strip().astype(bool)]

# Save
merged_df.to_csv('official_repos_with_metadata.csv', index=False)




