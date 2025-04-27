import requests
import csv
import json
import re
import time
from collections import defaultdict
from datetime import datetime
from urllib.parse import urlparse

GITHUB_TOKEN = "" 
GITHUB_TOKENS = [GITHUB_TOKEN]

# Popularity scoring thresholds
STAR_THRESHOLDS = [10000, 5000, 1000, 500, 0]
FORK_THRESHOLDS = [2000, 1000, 500, 100, 0]
COMMIT_THRESHOLDS = [30, 180, 365, 730, float('inf')]  # Days since last commit

STAR_SCORES = [10, 8, 6, 4, 2]
FORK_SCORES = [10, 8, 6, 4, 2]
COMMIT_SCORES = [10, 8, 6, 4, 2]

import requests
import time
import re
import base64
import csv
from datetime import datetime
from urllib.parse import urlparse
from random import choice


class MCPSecurityAnalyzer:
    def __init__(self, csv_file, GITHUB_TOKENS=None):
        self.csv_file = csv_file
        self.servers = []
        self.results = []
        self.GITHUB_TOKENS = GITHUB_TOKENS or []
        self.headers = {
            "Accept": "application/vnd.github.v3+json"
        }
        self.current_token_index = 0
        self.cache = {
            "repo_info": {},
            "file_content": {},
            "search": {},
            "contents": {}
        }

        # Suspicious patterns & domains (same as your original code)
        self.suspicious_patterns = [
            r"eval\s*\(", r"exec\s*\(", r"base64\.b64decode", r"send.*data",
            r"collect.*data", r"tracking", r"analytics", r"fetch\s*\(",
            r"axios\.post", r"request\.post", r"\.send\(", r"webhook",
            r"store_all_queries", r"log_all"
        ]

        self.suspicious_domains = [
            "data-collector", "analytics", "tracker", "metrics", "telemetry", "logging"
        ]

    def rotate_token(self):
        """Switch to the next token in the list and reset the headers."""
        if not self.GITHUB_TOKENS:
            raise ValueError("No GitHub tokens provided.")
        self.current_token_index = (self.current_token_index + 1) % len(self.GITHUB_TOKENS)
        self.headers["Authorization"] = f"token {self.GITHUB_TOKENS[self.current_token_index]}"

    def _make_github_request(self, url, cache_key=None, category=None):
        if category and cache_key in self.cache.get(category, {}):
            return self.cache[category][cache_key]

        response = requests.get(url, headers=self.headers)
        
        # Handle rate limiting
        if response.status_code == 403 and 'X-RateLimit-Remaining' in response.headers and int(response.headers['X-RateLimit-Remaining']) == 0:
            reset_time = int(response.headers['X-RateLimit-Reset'])
            sleep_time = max(0, reset_time - time.time() + 1)
            print(f"Rate limit exceeded for token {self.GITHUB_TOKENS[self.current_token_index]}. Sleeping for {sleep_time} seconds...")
            time.sleep(sleep_time)
            self.rotate_token()  # Switch token
            return self._make_github_request(url, cache_key, category)

        if response.status_code == 200:
            data = response.json()
            if category:
                self.cache[category][cache_key] = data
            return data
        else:
            print(f"GitHub API error {response.status_code}: {response.text}")
            return None

    def get_repo_info(self, owner, repo):
        key = f"{owner}/{repo}"
        if key in self.cache["repo_info"]:
            return self.cache["repo_info"][key]

        url = f"https://api.github.com/repos/{owner}/{repo}"
        data = self._make_github_request(url, key, "repo_info")
        if not data:
            return None

        # Fetch commits separately (needed for recency)
        commits_url = f"https://api.github.com/repos/{owner}/{repo}/commits"
        commits = self._make_github_request(commits_url, f"{key}/commits", "repo_info")

        latest_commit_date = None
        if isinstance(commits, list) and commits:
            latest_commit_date = commits[0]['commit']['committer']['date']

        data['stars'] = data.get('stargazers_count', 0)
        data['forks'] = data.get('forks_count', 0)
        data['latest_commit_date'] = latest_commit_date

        return data

    def calculate_popularity_score(self, repo_info):
        stars = repo_info.get('stars', 0)
        forks = repo_info.get('forks', 0)
        latest_commit_date = repo_info.get('latest_commit_date')

        if not latest_commit_date:
            return None

        last_commit_days = (datetime.utcnow() - datetime.strptime(latest_commit_date, "%Y-%m-%dT%H:%M:%SZ")).days

        star_score = self.calculate_score(stars, STAR_THRESHOLDS, STAR_SCORES)
        fork_score = self.calculate_score(forks, FORK_THRESHOLDS, FORK_SCORES)
        commit_score = self.calculate_score(last_commit_days, COMMIT_THRESHOLDS, COMMIT_SCORES)

        popularity_score = (star_score + fork_score + commit_score) / 3
        return {
            "star_score": star_score,
            "fork_score": fork_score,
            "commit_score": commit_score,
            "popularity_score": popularity_score
        }

    @staticmethod
    def calculate_score(value, thresholds, scores):
        for threshold, score in zip(thresholds, scores):
            if value >= threshold:
                return score
        return 0

    def check_github_repo(self, owner, repo):
        result = {
            "owner": owner,
            "repo": repo,
            "risk_score": 0,
            "risk_factors": [],
            "popularity_data": None,
            "documentation_score": None,
            "suspicious_files": [],
            "oauth_implementation": False,
            "direct_api_tokens": False,
            "outbound_connections": [],
        }

        repo_info = self.get_repo_info(owner, repo)
        if not repo_info:
            result["error"] = "Repo info not found."
            return result

        #result["popularity_data"] = self.calculate_popularity_score(repo_info)
        #result["documentation_score"] = self.check_documentation(owner, repo)

        config_files = self.find_config_files(owner, repo, repo_info.get('default_branch', 'main'))

        for file in config_files:
            analysis = self.analyze_file(owner, repo, file)
            if analysis['suspicious']:
                result["suspicious_files"].append(analysis)
                result["risk_score"] += analysis["risk_score"]
                result["risk_factors"].extend(analysis["risk_factors"])
                result["outbound_connections"].extend(analysis["outbound_connections"])

        auth_results = self.check_authentication_methods(owner, repo)
        result.update(auth_results)

        result["outbound_connections"] = list(set(result["outbound_connections"]))
        result["risk_factors"] = list(set(result["risk_factors"]))

        return result

    def check_authentication_methods(self, owner, repo):
        result = {"oauth_implementation": False, "direct_api_tokens": False}
        
        # Combine searches for multiple keywords to reduce calls
        keywords = ["oauth", "authorization_code", "client_id", "api_token", "api_key", "bearer_token"]
        keyword_query = "+".join(keywords)

        url = f"https://api.github.com/search/code?q={keyword_query}+in:file+repo:{owner}/{repo}"
        cache_key = f"{owner}/{repo}/{keyword_query}"
        data = self._make_github_request(url, cache_key, "search")

        if data and data.get("total_count", 0) > 0:
            if any(keyword in data["items"] for keyword in ["oauth", "authorization_code", "client_id"]):
                result["oauth_implementation"] = True
            else:
                result["direct_api_tokens"] = True

        return result

    def update_from_csv(self, input_csv, output_csv):
        repositories = []

        try:
            with open(input_csv, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    owner = row.get('owner')
                    repo = row.get('repo')
                    if owner and repo:
                        repositories.append(row)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return

        print(f"Loaded {len(repositories)} repositories from {input_csv}")

        updated_repositories = []

        for repo_info in repositories:
            owner = repo_info['owner']
            repo = repo_info['repo']

            print(f"Evaluating security for {owner}/{repo}...")

            security_score = self.evaluate_security(owner, repo)
            if security_score is not None:
                repo_info['security_score'] = security_score
            else:
                print(f"Warning: Could not evaluate {owner}/{repo}")

            updated_repositories.append(repo_info)

        if updated_repositories:
            fieldnames = updated_repositories[0].keys()
            try:
                with open(output_csv, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for repo_info in updated_repositories:
                        writer.writerow(repo_info)
                print(f"Saved updated security scores to {output_csv}")
            except Exception as e:
                print(f"Error saving CSV: {e}")


def main():
    print("Starting MCPSecurityAnalyzer...")
    tokens = ["your_first_token", "your_second_token"]  # Add your GitHub tokens here
    analyzer = MCPSecurityAnalyzer("small_test.csv", GITHUB_TOKENS=tokens)
    if not analyzer.load_servers():
        return

    analyzer.update_from_csv("small_test.csv", "small_updated_test.csv")

def main():
    print("Starting MCPSecurityAnalyzer...")
    analyzer = MCPSecurityAnalyzer("small_test.csv", GITHUB_TOKEN)
    if not analyzer.load_servers():
        return

    # Update security scores
    analyzer.update_from_csv("small_test.csv", "small_updated_test.csv")

if __name__ == "__main__":
    main()
