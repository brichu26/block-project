import csv
import json
import requests
from datetime import datetime, timedelta
from typing import Optional, List
import pandas as pd
import requests
from urllib.parse import urlparse
import os
import time
from datetime import datetime
from tqdm import tqdm
import random


class MCPDataGenerator:
    HEADERS = {"Authorization": f"Bearer {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    GRAPHQL_URL = "https://api.github.com/graphql"
    BATCH_SIZE = 50
    OUTPUT_CSV = "enriched_repos_all.csv"

    def __init__(self):
        self.df: pd.DataFrame = None
        self.enriched_df: pd.DataFrame = None
        self.stats: dict = {}
        self.GITHUB_TOKEN = ""
        self.HEADERS = {"Authorization": f"Bearer {self.GITHUB_TOKEN}"} if self.GITHUB_TOKEN else {}
        self.GRAPHQL_URL = "https://api.github.com/graphql"
        self.BATCH_SIZE = 50
        self.OUTPUT_CSV = "enriched_repos_all.csv"


    def fetch_servers(self,
                count: int = 100,
                query: Optional[str] = None,
                offset: int = 0,
                output_file: Optional[str] = None,
                fetch_all: bool = False):
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
            if fetch_all:
                sorted_servers_final = sorted_servers
            else:
                sorted_servers_final = sorted_servers[:count] #change this later back to 
            
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
                    'server_github_url': github_url,
                    'owner': owner,
                    'repo': repo,
                    'github_stars': server.get("github_stars", 0),
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
        
    
    @staticmethod
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


    @staticmethod
    def build_graphql_batch_query(batch):
        """
        This function builds a graphql query for a batch of repositories.
        It takes a list of repository data and returns a string of a graphql query.

        Args:
            batch (list): A list of repository data

        Returns:
            str: A string of a graphql query
        """
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


    def send_graphql_query(self, query):
        response = requests.post(self.GRAPHQL_URL, json={"query": query}, headers=self.HEADERS)
        if response.status_code != 200:
            raise Exception(f"GraphQL query failed: {response.text}")
        return response.json()


    def enrich_csv(self, input_csv_path, output_csv_path=None):
        if output_csv_path is None:
            output_csv_path = self.OUTPUT_CSV
            
        df = pd.read_csv(input_csv_path or self.csv_file_path)

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
            url = df.iloc[i].get("server_github_url", "")
            if pd.notna(url):
                unique_identifier = self.parse_repo_url(url)
                if unique_identifier:
                    path_parts = unique_identifier.split("/", 2)  # Limit splits to handle first two items
                    if len(path_parts) >= 2:
                        index_map[len(repos)] = i
                        repos.append(path_parts[:2])  # Only take first two as [owner, repo]

        with tqdm(total=len(repos), unit="repo") as pbar:
            for i in range(0, len(repos), self.BATCH_SIZE):
                batch = repos[i:i + self.BATCH_SIZE]
                query = self.build_graphql_batch_query(batch)

                try:
                    result = self.send_graphql_query(query)
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
                    print(f"Batch {i//self.BATCH_SIZE + 1} failed: {e}")
                    time.sleep(10)  # wait a bit before retrying
        
        print("Enrichment complete.")
        return enriched_df


    def calculate_statistics(self, input_csv_path) -> dict:

        """
        This function calculates the statistics of the repositories.
        It takes an input csv path and returns a dictionary of statistics.
        It calculates the mean, median, and standard deviation of the repositories.
        It also calculates the number of days since the last commit.

        Args:
            input_csv_path (str): The path to the input csv file

        Returns:
            dict: A dictionary of statistics
        """
        df = pd.read_csv(input_csv_path)
        df['latest_commit_date'] = pd.to_datetime(df['latest_commit_date'], errors='coerce')
        now = pd.Timestamp.utcnow()
        df['days_since_commit'] = (now - df['latest_commit_date']).dt.total_seconds() / 86400
        cols = ['github_stars', 'download_count', 'forks', 'days_since_commit']
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
        clean = df.dropna(subset=['days_since_commit'], how='all')
        agg = clean[cols].agg(['mean', 'median', 'std'])
        stats = {}
        for col in cols:
            key = col if col != 'days_since_commit' else 'commit_days'
            stats[f"mean_{key}"]   = agg.loc['mean', col]
            stats[f"median_{key}"] = agg.loc['median', col]
            stats[f"std_{key}"]    = agg.loc['std', col]
        self.stats = stats
        return stats


    def find_missing_rows(
        self,
        file1_path: str,
        file2_path: str,
        column: str,
        output_path: str,
        columns_to_compare: list = None
    ) -> None:
        """
        This function finds the missing rows between two csv files.
        It takes two csv file paths, a column name, an output path, and a list of columns to compare.
        It returns the missing rows.

        Should be used if there is an issue with processing the csv files in secure.py. New CSV should be generated with this function and rerun the secure.py file.

        Args:
            file1_path (str): The path to the first csv file

        """
        try:
            df1 = pd.read_csv(file1_path)
            df2 = pd.read_csv(file2_path)
            if column not in df1 or column not in df2:
                raise ValueError(f"Column '{column}' not found in both files")
            missing = set(df1[column]) - set(df2[column])
            out = df1[df1[column].isin(missing)]
            if columns_to_compare:
                out = out[columns_to_compare]
            out.to_csv(output_path, index=False)
            print(f"Missing rows: {len(out)} saved to {output_path}")
        except Exception as e:
            print(f"Error: {e}")
            raise


    def get_random_rows(self, csv_file_path, count, output_file_path):
        """
        Get N random rows from a CSV file and save them to a new CSV file with renamed column.
        Should be used to generate random sample of servers.
        
        Args:
            csv_file_path (str): Path to the input CSV file
            count (int): Number of random rows to select
            output_file_path (str): Path where the new CSV file will be saved
            
        Returns:
            str: Path to the created output file
        """
        try:
            with open(csv_file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                all_rows = list(reader)
                
                if not all_rows:
                    print("Input CSV file is empty")
                    return None
                    
                # If count is greater than total rows, use all rows
                if count >= len(all_rows):
                    selected_rows = all_rows
                else:
                    # Randomly select rows
                    selected_rows = random.sample(all_rows, count)
                
                # Rename the column from github_url to server_github_url
                for row in selected_rows:
                    if 'github_url' in row:
                        row['server_github_url'] = row.pop('github_url')
                
                # Get fieldnames from the first row
                fieldnames = list(selected_rows[0].keys())
                
                # Write to new CSV file
                with open(output_file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(selected_rows)
                
                print(f"Successfully created {output_file_path} with {len(selected_rows)} random rows")
                return output_file_path
                
        except Exception as e:
            print(f"Error processing CSV files: {e}")
            return None

"""
Fetch server is used to fetch the servers from the PulseMCP API in a sorted manner. Enrich_csv
is used to get the metadata of the servers. Calculate_statistics is used to calculate the statistics of the servers and the csv generated from enrich_csv should
be used as input for calculate_staistics.

We have already added a csv called enriched_repos_metadata.csv (detailed information regarding all of the servers present in PulseMCP) which is the output of enrich_csv. 
This is used as input for calculate_statistics. If you want to fetch updated servers, you can use the enrich_csv function with the csv file you want to enrich as input.
"""

if __name__ == "__main__":
    mcp_analyzer = MCPDataGenerator()
    
    mcp_analyzer.fetch_servers(40, output_file="trial_servers.csv", fetch_all=False) #Select True to fetch all servers, also fetches servers by github stars in descending order
    #mcp_analyzer.enrich_csv("trial_servers.csv", output_csv_path="trial_enriched_repos_40_servers.csv")
    #print(mcp_analyzer.calculate_statistics("enriched_repos_metadata.csv"))