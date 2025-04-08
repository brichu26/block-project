import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from typing import Dict, List, Tuple, Optional
import csv
import re
from bs4 import BeautifulSoup

# GitHub API configuration
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
if not GITHUB_TOKEN:
    print("Warning: GITHUB_TOKEN environment variable not set. GitHub API requests may be rate-limited.")
    HEADERS = {
        'Accept': 'application/vnd.github.v3+json'
    }
else:
    HEADERS = {
        'Authorization': f'token {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3+json'
    }

# Target repository for fetching community servers
MCP_COMMUNITY_REPO_URL = "https://github.com/modelcontextprotocol/servers"
MCP_COMMUNITY_REPO_API_README_URL = "https://api.github.com/repos/modelcontextprotocol/servers/readme"

class MCPServerScorer:
    def __init__(self):
        pass

    def calculate_popularity_score(self, stars: int, forks: int, last_commit: datetime) -> float:
        # GitHub Stars (Score component: 15 points max)
        if stars >= 1000:
            star_score = 15
        elif stars >= 500:
            star_score = 12
        elif stars >= 100:
            star_score = 10
        elif stars >= 50:
            star_score = 7
        elif stars > 0:
             star_score = 4
        else:
            star_score = 0 # Assign 0 if no stars

        # GitHub Forks (Score component: 15 points max)
        if forks >= 200:
            fork_score = 15
        elif forks >= 100:
            fork_score = 12
        elif forks >= 50:
            fork_score = 10
        elif forks >= 10:
            fork_score = 7
        elif forks > 0:
            fork_score = 4
        else:
            fork_score = 0 # Assign 0 if no forks

        # Recent Commits (Score component: 10 points max)
        now = datetime.now()
        if last_commit: # Check if last_commit is not None
             time_since_commit = now - last_commit
             if time_since_commit <= timedelta(days=30):
                 commit_score = 10
             elif time_since_commit <= timedelta(days=180):
                 commit_score = 8
             elif time_since_commit <= timedelta(days=365):
                 commit_score = 6
             elif time_since_commit <= timedelta(days=730): # 2 years
                 commit_score = 4
             else:
                 commit_score = 2
        else:
             commit_score = 0 # Assign 0 if no commit data

        # Total score out of 40, normalized to 0-100 scale
        total_score = star_score + fork_score + commit_score
        return round((total_score / 40) * 100, 2)

def fetch_github_repo_data(repo_path: str) -> Optional[Dict]:
    """Fetch repository data from GitHub API for a specific owner/repo path."""
    if not repo_path or '/' not in repo_path:
        print(f"Invalid repository path format: {repo_path}")
        return None

    base_url = f"https://api.github.com/repos/{repo_path}"
    repo_info = None
    commits = None
    last_commit_date = None

    try:
        # Fetch basic repository info
        response_repo = requests.get(base_url, headers=HEADERS, timeout=10)
        response_repo.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        repo_info = response_repo.json()

        # Fetch commits
        commits_url = f"{base_url}/commits?per_page=1" # Only need the most recent
        response_commits = requests.get(commits_url, headers=HEADERS, timeout=10)
        response_commits.raise_for_status()
        commits = response_commits.json()

        if commits:
            # Handle different commit structures (commit.author or commit.committer)
            commit_date_str = None
            if 'commit' in commits[0] and commits[0]['commit']:
                 if 'author' in commits[0]['commit'] and commits[0]['commit']['author'] and 'date' in commits[0]['commit']['author']:
                      commit_date_str = commits[0]['commit']['author']['date']
                 elif 'committer' in commits[0]['commit'] and commits[0]['commit']['committer'] and 'date' in commits[0]['commit']['committer']:
                      commit_date_str = commits[0]['commit']['committer']['date']

            if commit_date_str:
                 # Parse date, removing the 'Z' for timezone if present
                 last_commit_date = datetime.strptime(commit_date_str.replace('Z', ''), '%Y-%m-%dT%H:%M:%S')
            else:
                 print(f"Could not find commit date for {repo_path}")
        else:
            print(f"No commits found for {repo_path}")

        return {
            'stars': repo_info.get('stargazers_count', 0),
            'forks': repo_info.get('forks_count', 0),
            'last_commit': last_commit_date,
            'name': repo_info.get('name', repo_path.split('/')[-1]),
            'description': repo_info.get('description', 'N/A'),
            'repo_path': repo_path # Add repo_path for reference
        }

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {repo_path}: {e}")
        # Return partial data if possible, or None
        if repo_info: # If basic info was fetched but commits failed
             return {
                  'stars': repo_info.get('stargazers_count', 0),
                  'forks': repo_info.get('forks_count', 0),
                  'last_commit': None, # Indicate commit fetch failed
                  'name': repo_info.get('name', repo_path.split('/')[-1]),
                  'description': repo_info.get('description', 'N/A'),
                  'repo_path': repo_path
             }
        return None # If initial repo fetch failed
    except Exception as e: # Catch other potential errors like JSON decoding or date parsing
         print(f"An unexpected error occurred processing {repo_path}: {e}")
         return None


def extract_github_repo_path(url: str) -> Optional[str]:
    """Extracts owner/repo path from various GitHub URL formats."""
    if not url:
        return None
    # Match standard github.com URLs
    match = re.search(r"github\.com/([^/]+/[^/]+)", url)
    if match:
        repo_path = match.group(1)
        # Remove trailing .git if present
        if repo_path.endswith('.git'):
            repo_path = repo_path[:-4]
        return repo_path
    return None

def fetch_community_servers_from_readme() -> List[Dict[str, str]]:
    """Fetches and parses the MCP community servers README to extract server names and repo paths."""
    servers = []
    try:
        # Fetch README content using GitHub API to get the raw markdown/content
        response = requests.get(MCP_COMMUNITY_REPO_API_README_URL, headers=HEADERS, timeout=15)
        response.raise_for_status()
        readme_data = response.json()

        # Decode the base64 encoded content
        import base64
        readme_content = base64.b64decode(readme_data['content']).decode('utf-8')

        # Use BeautifulSoup to parse the HTML rendering of the README (or just process markdown)
        # Simpler approach: Process raw markdown for list items under relevant sections
        server_section_found = False
        potential_sections = ["🌟 Reference Servers", "Community Servers", "MCP Servers"] # Add known section headers
        lines = readme_content.splitlines()

        for line in lines:
            line = line.strip()
            # Check if we entered a relevant section
            if any(line.startswith(f"## {section}") or line.startswith(f"# {section}") for section in potential_sections):
                 server_section_found = True
                 continue # Move to the next line after finding the header

            # Stop processing if we hit another major section or end of relevant list
            if server_section_found and (line.startswith("## ") or line.startswith("# ") or not line):
                 # Heuristic: stop if we encounter another header or an empty line after finding the section
                 # This might need refinement based on exact README structure
                 # server_section_found = False # Optionally reset if multiple sections could exist
                 pass # Continue checking in case list items are separated by blank lines

            # If in a server section, look for list items likely containing links
            if server_section_found and line.startswith("*"):
                 # Extract name (text before potential link/parenthesis)
                 server_name = line.split('](')[0].split('[')[-1].strip()
                 if not server_name: # Handle cases like "* **Server Name** - Description"
                     match_bold = re.match(r"\* \*\*([^*]+)\*\*", line)
                     if match_bold:
                         server_name = match_bold.group(1)
                     else: # Fallback to the start of the line if no clear name pattern
                          server_name = line[1:].strip().split('-')[0].strip()


                 # Find a GitHub link in the line
                 repo_path = None
                 links = re.findall(r'\(https://github\.com/[^)]+\)', line)
                 if links:
                     repo_path = extract_github_repo_path(links[0])

                 # If a valid repo path is found, add the server
                 if repo_path:
                     print(f"Found Server: {server_name}, Repo: {repo_path}")
                     servers.append({"name": server_name, "repo_path": repo_path})
                 else:
                     print(f"Could not extract GitHub repo path for line: {line}")


    except requests.exceptions.RequestException as e:
        print(f"Error fetching README from {MCP_COMMUNITY_REPO_API_README_URL}: {e}")
    except Exception as e:
        print(f"Error parsing README content: {e}")

    # Deduplicate based on repo_path as some servers might be listed multiple times or have variations
    unique_servers = []
    seen_repos = set()
    for server in servers:
        if server['repo_path'] not in seen_repos:
            unique_servers.append(server)
            seen_repos.add(server['repo_path'])

    print(f"Found {len(unique_servers)} unique community servers.")
    return unique_servers

def main():
    print("Fetching community server list from MCP README...")
    community_servers = fetch_community_servers_from_readme()

    if not community_servers:
        print("No community servers found or error fetching README. Exiting.")
        return

    print(f"Fetching GitHub data for {len(community_servers)} servers...")
    server_data = []
    scorer = MCPServerScorer()
    processed_count = 0

    for server in community_servers:
        repo_path = server.get("repo_path")
        server_name = server.get("name", "Unknown")
        if not repo_path:
            print(f"Skipping server '{server_name}' due to missing repo path.")
            continue

        print(f"Processing {repo_path} ({server_name})...")
        data = fetch_github_repo_data(repo_path)
        processed_count += 1

        if data:
            popularity_score = scorer.calculate_popularity_score(
                data['stars'],
                data['forks'],
                data['last_commit']
            )
            server_data.append({
                'Server Name': server_name,
                'Repository': data['repo_path'],
                'Stars': data['stars'],
                'Forks': data['forks'],
                'Last Commit Date': data['last_commit'].strftime('%Y-%m-%d %H:%M:%S') if data['last_commit'] else 'N/A',
                'Popularity Score (%)': popularity_score
                # Add description if needed: 'Description': data['description']
            })
            print(f"-> Stars: {data['stars']}, Forks: {data['forks']}, Last Commit: {data['last_commit']}, Score: {popularity_score}%")
        else:
            print(f"-> Failed to fetch data for {repo_path}")
            # Optionally add failed repos to the CSV with N/A values
            server_data.append({
                'Server Name': server_name,
                'Repository': repo_path,
                'Stars': 'N/A',
                'Forks': 'N/A',
                'Last Commit Date': 'N/A',
                'Popularity Score (%)': 'N/A'
            })

        # Add a small delay to avoid hitting rate limits, especially without a token
        time.sleep(0.5) # Adjust as needed

    print(f"\nProcessed {processed_count} servers.")

    if not server_data:
        print("No data collected for servers. Cannot generate CSV.")
        return

    # Output to CSV
    output_filename = 'community_server_scores.csv'
    print(f"Writing results to {output_filename}...")
    try:
        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            # Define fieldnames based on the keys of the first valid data entry
            fieldnames = server_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(server_data)
        print("Successfully wrote results to CSV.")
    except IOError as e:
        print(f"Error writing CSV file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during CSV writing: {e}")


if __name__ == "__main__":
    main()
