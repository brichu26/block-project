import requests
import pandas as pd
from datetime import datetime
import time
import os
from typing import Dict, List, Tuple, Optional, Any
import csv
import re
import json
from bs4 import BeautifulSoup

# Import evaluation components
from mcp_documentation_eval import DocumentationEvaluator
from mcp_security_eval import SecurityEvaluator
from mcp_popularity_eval import PopularityEvaluator

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
    """Central scoring system for MCP servers based on the scorecard criteria."""
    
    def __init__(self, github_token: Optional[str] = None):
        """Initialize with optional GitHub token for API access."""
        self.github_token = github_token
        self.documentation_evaluator = DocumentationEvaluator(github_token)
        self.security_evaluator = SecurityEvaluator(github_token)
        self.popularity_evaluator = PopularityEvaluator(github_token)
    
    def score_server(self, owner: str, repo: str) -> Dict[str, Any]:
        """Score a server based on all evaluation criteria."""
        print(f"\nEvaluating {owner}/{repo}...")
        
        # Run documentation evaluation
        print("Evaluating documentation quality...")
        doc_result = self.documentation_evaluator.evaluate(owner, repo)
        
        # Run security evaluation
        print("Evaluating security practices...")
        security_result = self.security_evaluator.evaluate(owner, repo)
        
        # Run popularity evaluation
        print("Evaluating popularity and community support...")
        popularity_result = self.popularity_evaluator.evaluate(owner, repo)
        
        # Calculate weighted overall score
        # Assuming equal weights for each category (adjust as needed)
        overall_score = (
            doc_result["score"] * 0.33 +
            security_result["score"] * 0.33 +
            popularity_result["score"] * 0.34
        )
        
        return {
            "repo": f"{owner}/{repo}",
            "overall_score": round(overall_score, 2),
            "documentation": doc_result,
            "security": security_result,
            "popularity": popularity_result
        }
    
    def generate_scorecard(self, result: Dict[str, Any]) -> str:
        """Generate a human-readable scorecard from evaluation results."""
        repo = result["repo"]
        overall_score = result["overall_score"]
        doc = result["documentation"]
        security = result["security"]
        popularity = result["popularity"]
        
        scorecard = f"""
{'=' * 80}
MCP SERVER SCORECARD: {repo}
{'=' * 80}

OVERALL SCORE: {overall_score}/10

1. DOCUMENTATION QUALITY: {doc['score']}/10 - {doc['rating']}
{'-' * 40}
- Base Section Score: {doc['details']['base_section_score']} (30% weight)
- Essential Sections Bonus: {doc['details']['essential_sections_bonus']} (20% weight)
- Readability Score: {doc['details']['readability_score']} (15% weight)
- Size Bonus: {doc['details']['size_bonus']} (10% weight)
- Heading Structure Bonus: {doc['details']['heading_structure_bonus']} (10% weight)
- Code Examples Bonus: {doc['details']['code_examples_bonus']} (10% weight)
- Completeness Bonus: {doc['details']['completeness_bonus']} (5% weight)

2. SECURITY PRACTICES: {security['score']} - Risk Level: {security['risk_level']}
{'-' * 40}
- Authentication Score: {security['details']['authentication_score']} (40% weight)
- Configuration Security Score: {security['details']['configuration_security_score']} (30% weight)
- Network Security Score: {security['details']['network_security_score']} (30% weight)
- OAuth Implementation: {'Yes' if security['details']['oauth_implementation'] else 'No'}
- Direct API Tokens: {'Yes' if security['details']['direct_api_tokens'] else 'No'}

3. POPULARITY & COMMUNITY SUPPORT: {popularity['score']}/10
{'-' * 40}
- GitHub Stars: {popularity['details']['stars']} (z-score: {popularity['details']['stars_zscore']})
- Forks: {popularity['details']['forks']} (z-score: {popularity['details']['forks_zscore']})
- Days Since Last Commit: {popularity['details']['days_since_commit']} (z-score: {popularity['details']['days_since_commit_zscore']})
- Download Count: {popularity['details']['download_count']} (z-score: {popularity['details']['download_count_zscore']})

Stars Score: {popularity['details']['stars_score']} (15% weight)
Forks Score: {popularity['details']['forks_score']} (15% weight)
Recent Activity Score: {popularity['details']['days_since_commit_score']} (10% weight)
Download Score: {popularity['details']['download_count_score']} (10% weight)

{'=' * 80}
        """
        return scorecard

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

        # Process raw markdown for list items under relevant sections
        server_section_found = False
        potential_sections = ["🌟 Reference Servers", "Community Servers", "MCP Servers"]
        lines = readme_content.splitlines()

        for line in lines:
            line = line.strip()
            # Check if we entered a relevant section
            if any(line.startswith(f"## {section}") or line.startswith(f"# {section}") for section in potential_sections):
                server_section_found = True
                continue  # Move to the next line after finding the header

            # Stop processing if we hit another major section
            if server_section_found and (line.startswith("## ") or line.startswith("# ")):
                server_section_found = False

            # If in a server section, look for list items likely containing links
            if server_section_found and line.startswith("*"):
                # Extract name (text before potential link/parenthesis)
                server_name = line.split('](')[0].split('[')[-1].strip()
                if not server_name:  # Handle cases like "* **Server Name** - Description"
                    match_bold = re.match(r"\* \*\*([^*]+)\*\*", line)
                    if match_bold:
                        server_name = match_bold.group(1)
                    else:  # Fallback to the start of the line if no clear name pattern
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

    # Deduplicate based on repo_path
    unique_servers = []
    seen_repos = set()
    for server in servers:
        if server['repo_path'] not in seen_repos:
            unique_servers.append(server)
            seen_repos.add(server['repo_path'])

    print(f"Found {len(unique_servers)} unique community servers.")
    return unique_servers

def export_results_to_csv(results: List[Dict[str, Any]], filename: str = "mcp_server_scores.csv"):
    """Export evaluation results to a CSV file."""
    if not results:
        print("No results to export.")
        return
    
    # Prepare data for CSV
    csv_data = []
    for result in results:
        repo = result["repo"]
        doc_score = result["documentation"]["score"]
        doc_rating = result["documentation"]["rating"]
        security_score = result["security"]["score"]
        security_risk = result["security"]["risk_level"]
        popularity_score = result["popularity"]["score"]
        overall_score = result["overall_score"]
        
        csv_data.append({
            "Repository": repo,
            "Overall Score": overall_score,
            "Documentation Score": doc_score,
            "Documentation Rating": doc_rating,
            "Security Score": security_score,
            "Security Risk Level": security_risk,
            "Popularity Score": popularity_score,
            "Stars": result["popularity"]["details"]["stars"],
            "Forks": result["popularity"]["details"]["forks"],
            "Days Since Last Commit": result["popularity"]["details"]["days_since_commit"]
        })
    
    # Write to CSV
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = csv_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        print(f"Successfully exported results to {filename}")
    except Exception as e:
        print(f"Error exporting results to CSV: {e}")

def export_results_to_json(results: List[Dict[str, Any]], filename: str = "mcp_server_scores.json"):
    """Export full evaluation results to a JSON file."""
    if not results:
        print("No results to export.")
        return
    
    try:
        with open(filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(results, jsonfile, indent=2)
        print(f"Successfully exported detailed results to {filename}")
    except Exception as e:
        print(f"Error exporting results to JSON: {e}")

def main():
    """Main function to run the MCP server scoring system."""
    github_token = os.getenv('GITHUB_TOKEN')
    
    # Parse command line arguments for specific repos
    import argparse
    parser = argparse.ArgumentParser(description='Score MCP servers based on documentation, security, and popularity.')
    parser.add_argument('--repos', nargs='+', help='Specific GitHub repositories to evaluate (format: owner/repo)')
    parser.add_argument('--from-readme', action='store_true', help='Fetch servers from MCP community README')
    parser.add_argument('--output-csv', type=str, default="mcp_server_scores.csv", help='Output CSV filename')
    parser.add_argument('--output-json', type=str, default="mcp_server_scores.json", help='Output JSON filename')
    parser.add_argument('--verbose', action='store_true', help='Print detailed scorecards')
    args = parser.parse_args()
    
    # Initialize the scorer
    scorer = MCPServerScorer(github_token)
    
    # Determine which repositories to evaluate
    repos_to_evaluate = []
    
    if args.repos:
        # Use provided repos
        for repo_path in args.repos:
            if '/' in repo_path:
                owner, repo = repo_path.split('/', 1)
                repos_to_evaluate.append({"name": repo, "owner": owner, "repo": repo})
            else:
                print(f"Invalid repository format: {repo_path}. Expected format: owner/repo")
    
    if args.from_readme or not repos_to_evaluate:
        print("Fetching community server list from MCP README...")
        community_servers = fetch_community_servers_from_readme()
        for server in community_servers:
            if '/' in server['repo_path']:
                owner, repo = server['repo_path'].split('/', 1)
                repos_to_evaluate.append({"name": server['name'], "owner": owner, "repo": repo})
    
    if not repos_to_evaluate:
        print("No repositories to evaluate. Exiting.")
        return
    
    print(f"Will evaluate {len(repos_to_evaluate)} repositories.")
    
    # Evaluate each repository
    results = []
    for repo_info in repos_to_evaluate:
        try:
            owner = repo_info["owner"]
            repo = repo_info["repo"]
            
            result = scorer.score_server(owner, repo)
            results.append(result)
            
            if args.verbose:
                scorecard = scorer.generate_scorecard(result)
                print(scorecard)
            else:
                print(f"{owner}/{repo}: Overall Score: {result['overall_score']}/10")
            
            # Add a small delay to avoid rate limits
            time.sleep(1)
        except Exception as e:
            print(f"Error evaluating {repo_info.get('name', repo_info.get('repo', 'unknown'))}: {e}")
    
    # Export results
    if results:
        export_results_to_csv(results, args.output_csv)
        export_results_to_json(results, args.output_json)
    
    print("Evaluation complete!")

if __name__ == "__main__":
    main()
