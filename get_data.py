import pandas as pd
import requests
from urllib.parse import urlparse
import os
import time
from datetime import datetime
from tqdm import tqdm

GITHUB_TOKEN = ""
import pandas as pd
import os
import time
import requests
from urllib.parse import urlparse
from tqdm import tqdm

HEADERS = {"Authorization": f"Bearer {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
GRAPHQL_URL = "https://api.github.com/graphql"

BATCH_SIZE = 50
OUTPUT_CSV = "enriched_repos_graphql2.csv"


def parse_repo_url(url):
    try:
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.strip("/").split("/")
        
        # Use full path as identifier if necessary
        owner = path_parts[0] if len(path_parts) > 0 else None
        repo = path_parts[1] if len(path_parts) > 1 else None
        
        # Include more of the URL for unique identification
        unique_identifier = "/".join(path_parts)
        
        return unique_identifier if owner and repo else (None, None)
    except Exception as e:
        print(f"Error parsing URL: {e}")
        return None, None



def build_graphql_batch_query(batch):
    queries = []
    for i, repo_data in enumerate(batch):
        if len(repo_data) >= 2:
            owner, repo = repo_data
            queries.append(f'''
            r{i}: repository(owner: "{owner}", name: "{repo}") {{
                stargazerCount
                forkCount
                defaultBranchRef {{
                    target {{
                        ... on Commit {{
                            committedDate
                        }}
                    }}
                }}
            }}
            ''')
    return f"query {{\n{''.join(queries)}\n}}"


def send_graphql_query(query):
    response = requests.post(GRAPHQL_URL, json={"query": query}, headers=HEADERS)
    if response.status_code != 200:
        raise Exception(f"GraphQL query failed: {response.text}")
    return response.json()


def enrich_csv(input_csv_path, output_csv_path=OUTPUT_CSV):
    df = pd.read_csv(input_csv_path)

    if os.path.exists(output_csv_path):
        enriched_df = pd.read_csv(output_csv_path)
        start_index = len(enriched_df)
        print(f"Resuming from row {start_index}")
    else:
        enriched_df = pd.DataFrame(columns=df.columns.tolist() + ["github_stars", "forks", "latest_commit_date"])
        start_index = 0

    repos = []
    index_map = {}
    for i in range(start_index, len(df)):
        url = df.iloc[i].get("github_url", "")
        if pd.notna(url):
            unique_identifier = parse_repo_url(url)
            if unique_identifier:
                path_parts = unique_identifier.split("/", 2)  # Limit splits to handle first two items
                if len(path_parts) >= 2:
                    index_map[len(repos)] = i
                    repos.append(path_parts[:2])  # Only take first two as [owner, repo]

    with tqdm(total=len(repos), unit="repo") as pbar:
        for i in range(0, len(repos), BATCH_SIZE):
            batch = repos[i:i + BATCH_SIZE]
            query = build_graphql_batch_query(batch)

            try:
                result = send_graphql_query(query)
                enriched_rows = []

                for j, (owner, repo) in enumerate(batch):
                    repo_data = result.get("data", {}).get(f"r{j}", {})
                    metadata = {
                        "github_stars": repo_data.get("stargazerCount"),
                        "forks": repo_data.get("forkCount"),
                        "latest_commit_date": repo_data.get("defaultBranchRef", {}).get("target", {}).get("committedDate")
                    }
                    original_row = df.iloc[index_map[i + j]]
                    enriched_row = pd.concat([original_row, pd.Series(metadata)])
                    enriched_rows.append(enriched_row.to_frame().T)

                enriched_df = pd.concat([enriched_df] + enriched_rows, ignore_index=True)
                enriched_df.to_csv(output_csv_path, index=False)
                pbar.update(len(batch))

            except Exception as e:
                print(f"Batch {i//BATCH_SIZE + 1} failed: {e}")
                time.sleep(10)  # wait a bit before retrying

    print("Enrichment complete.")

#enrich_csv("pulsemcp_servers_all.csv")
import pandas as pd
from datetime import datetime

# Load the CSV file
df = pd.read_csv("enriched_repos_graphql2.csv")

# Convert 'latest_commit_date' to datetime
df['latest_commit_date'] = pd.to_datetime(df['latest_commit_date'], errors='coerce')

# Calculate days since last commit
now = pd.Timestamp.utcnow()
df['days_since_commit'] = (now - df['latest_commit_date']).dt.total_seconds() / (60 * 60 * 24)

# Select relevant columns
numeric_cols = ['github_stars', 'download_count', 'forks', 'days_since_commit']

# Convert numeric columns (in case some are strings or missing)
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Filter out rows where the days_since_commit could not be calculated
df_clean = df[numeric_cols].dropna(subset=['days_since_commit'], how='all')

# Calculate stats
stats = df_clean.agg(['mean', 'median', 'std'])

# Display stats
print("Statistics for github_stars, download_count, forks, days_since_commit:")
print(stats)

# Additional display of days_since_commit stats
if 'days_since_commit' in stats.columns:
    print(f"\n Days since last commit:")
    print(f"Mean:   {stats.loc['mean', 'days_since_commit']:.2f} days")
    print(f"Median: {stats.loc['median', 'days_since_commit']:.2f} days")
    print(f"Std:    {stats.loc['std', 'days_since_commit']:.2f} days")
else:
    print("\n Days since last commit: N/A")
