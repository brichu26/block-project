import requests
from datetime import datetime
import os
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

class PopularityEvaluator:
    """
    Evaluates the popularity and community support of a repository based on the scorecard criteria
    using z-score statistics from a larger dataset.
    """
    
    # Statistics from the 4000 server dataset (as provided in the scorecard)
    STATS = {
        "github_stars": {"mean": 322.103774, "median": 5, "std": 3227.825429},
        "download_count": {"mean": 36.669556, "median": 2, "std": 363.238564},
        "forks": {"mean": 43.918024, "median": 2, "std": 35.352834},
        "days_since_commit": {"mean": 322.103774, "median": 39.316045, "std": 363.238564}
    }
    
    # Z-score thresholds for point assignment
    THRESHOLDS = [3.0, 2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5]
    POINTS = [10, 8, 7, 6, 5, 4, 3, 2, 1]
    
    # For days_since_commit, lower is better, so we reverse the points
    POINTS_COMMIT_DAYS = [1, 2, 3, 4, 5, 6, 7, 8, 10]
    
    def __init__(self, github_token: Optional[str] = None):
        """Initialize with optional GitHub token for API access."""
        self.github_token = github_token
        self.headers = {"Authorization": f"token {github_token}"} if github_token else {}
    
    def fetch_repo_data(self, owner: str, repo: str) -> Dict[str, Any]:
        """Fetch repository data from GitHub API."""
        repo_url = f"https://api.github.com/repos/{owner}/{repo}"
        commits_url = f"https://api.github.com/repos/{owner}/{repo}/commits"
        
        # Default values
        data = {
            "stars": 0,
            "forks": 0,
            "days_since_commit": 365,  # Default to 1 year if no commits found
            "download_count": 0
        }
        
        try:
            # Get repo details
            repo_response = requests.get(repo_url, headers=self.headers, timeout=10)
            if repo_response.status_code == 200:
                repo_data = repo_response.json()
                data["stars"] = repo_data.get("stargazers_count", 0)
                data["forks"] = repo_data.get("forks_count", 0)
            else:
                print(f"Error fetching repo data: Status {repo_response.status_code}")
                return data
            
            # Get latest commit date
            commits_response = requests.get(commits_url, headers=self.headers, timeout=10)
            if commits_response.status_code == 200:
                commits = commits_response.json()
                if commits and isinstance(commits, list) and len(commits) > 0:
                    latest_commit = commits[0]
                    if "commit" in latest_commit and "committer" in latest_commit["commit"]:
                        commit_date_str = latest_commit["commit"]["committer"]["date"]
                        commit_date = datetime.strptime(commit_date_str, "%Y-%m-%dT%H:%M:%SZ")
                        days_since = (datetime.utcnow() - commit_date).days
                        data["days_since_commit"] = days_since
            else:
                print(f"Error fetching commit data: Status {commits_response.status_code}")
            
            # Try to get package download count (for npm packages)
            try:
                package_json_url = f"https://api.github.com/repos/{owner}/{repo}/contents/package.json"
                pkg_response = requests.get(package_json_url, headers=self.headers, timeout=10)
                if pkg_response.status_code == 200:
                    import base64
                    import json
                    pkg_data = pkg_response.json()
                    if "content" in pkg_data and pkg_data.get("encoding") == "base64":
                        content = base64.b64decode(pkg_data["content"]).decode("utf-8")
                        package_info = json.loads(content)
                        package_name = package_info.get("name", "")
                        if package_name:
                            # Try to get NPM download stats
                            npm_url = f"https://api.npmjs.org/downloads/point/last-month/{package_name}"
                            npm_response = requests.get(npm_url, timeout=5)
                            if npm_response.status_code == 200:
                                npm_data = npm_response.json()
                                data["download_count"] = npm_data.get("downloads", 0)
            except Exception as e:
                # Not critical if we can't get download count
                print(f"Error fetching download data: {e}")
            
            return data
            
        except Exception as e:
            print(f"Error in fetch_repo_data: {e}")
            return data
    
    def calculate_zscore(self, value: float, metric: str) -> float:
        """Calculate the z-score for a given value and metric."""
        stats = self.STATS.get(metric, {"mean": 0, "std": 1})
        mean = stats["mean"]
        std = stats["std"]
        
        # Avoid division by zero
        if std == 0:
            return 0
            
        return (value - mean) / std
    
    def get_points_from_zscore(self, zscore: float, is_days_since_commit: bool = False) -> int:
        """Convert a z-score to points based on the thresholds."""
        points = self.POINTS.copy()
        
        # For days_since_commit, lower is better, so we use the reversed point scale
        if is_days_since_commit:
            points = self.POINTS_COMMIT_DAYS.copy()
        
        for i, threshold in enumerate(self.THRESHOLDS):
            if zscore >= threshold:
                return points[i]
        
        return points[-1]  # Lowest score if below all thresholds
    
    def evaluate(self, owner: str, repo: str) -> Dict[str, Any]:
        """
        Evaluate the popularity and community support of a GitHub repository.
        
        Returns:
            Dictionary with popularity assessment details and overall score
        """
        # Fetch repository data
        repo_data = self.fetch_repo_data(owner, repo)
        
        # Calculate z-scores
        stars_zscore = self.calculate_zscore(repo_data["stars"], "github_stars")
        forks_zscore = self.calculate_zscore(repo_data["forks"], "forks")
        days_zscore = self.calculate_zscore(repo_data["days_since_commit"], "days_since_commit")
        downloads_zscore = self.calculate_zscore(repo_data["download_count"], "download_count")
        
        # Calculate points
        stars_points = self.get_points_from_zscore(stars_zscore)
        forks_points = self.get_points_from_zscore(forks_zscore)
        days_points = self.get_points_from_zscore(days_zscore, is_days_since_commit=True)
        downloads_points = self.get_points_from_zscore(downloads_zscore)
        
        # Apply weights from the scorecard
        stars_score = stars_points * 0.15
        forks_score = forks_points * 0.15
        days_score = days_points * 0.10
        downloads_score = downloads_points * 0.10
        
        # Calculate total score
        total_score = stars_score + forks_score + days_score + downloads_score
        
        # Scale to a 0-10 range (total weight is 50% of the total possible)
        normalized_score = total_score / 0.5
        
        return {
            "score": round(normalized_score, 2),
            "details": {
                "stars": repo_data["stars"],
                "forks": repo_data["forks"],
                "days_since_commit": repo_data["days_since_commit"],
                "download_count": repo_data["download_count"],
                "stars_zscore": round(stars_zscore, 2),
                "forks_zscore": round(forks_zscore, 2),
                "days_since_commit_zscore": round(days_zscore, 2),
                "download_count_zscore": round(downloads_zscore, 2),
                "stars_points": stars_points,
                "forks_points": forks_points,
                "days_since_commit_points": days_points,
                "download_count_points": downloads_points,
                "stars_score": round(stars_score, 2),
                "forks_score": round(forks_score, 2),
                "days_since_commit_score": round(days_score, 2),
                "download_count_score": round(downloads_score, 2)
            }
        }

def test_popularity_evaluation(owner: str, repo: str):
    """Test the popularity evaluator with a single repository."""
    github_token = os.getenv('GITHUB_TOKEN')
    evaluator = PopularityEvaluator(github_token)
    result = evaluator.evaluate(owner, repo)
    
    print(f"Popularity Evaluation for {owner}/{repo}:")
    print(f"Overall Score: {result['score']}/10")
    print("\nRepository Stats:")
    print(f"- Stars: {result['details']['stars']} (z-score: {result['details']['stars_zscore']}, points: {result['details']['stars_points']})")
    print(f"- Forks: {result['details']['forks']} (z-score: {result['details']['forks_zscore']}, points: {result['details']['forks_points']})")
    print(f"- Days Since Last Commit: {result['details']['days_since_commit']} (z-score: {result['details']['days_since_commit_zscore']}, points: {result['details']['days_since_commit_points']})")
    print(f"- Download Count: {result['details']['download_count']} (z-score: {result['details']['download_count_zscore']}, points: {result['details']['download_count_points']})")
    
    print("\nWeighted Scores:")
    print(f"- Stars Score: {result['details']['stars_score']} (15% weight)")
    print(f"- Forks Score: {result['details']['forks_score']} (15% weight)")
    print(f"- Recent Activity Score: {result['details']['days_since_commit_score']} (10% weight)")
    print(f"- Download Score: {result['details']['download_count_score']} (10% weight)")
    print("-" * 50)

# Example usage
if __name__ == "__main__":
    # Test with a few example repositories
    for repo_info in [
        ("modelcontextprotocol", "servers"),
        ("modelcontextprotocol", "python-sdk")
    ]:
        test_popularity_evaluation(*repo_info)
