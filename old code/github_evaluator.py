import requests
from datetime import datetime

# GitHub API token (optional but recommended for higher rate limits)
GITHUB_TOKEN = ""  # Replace with your GitHub token or leave empty for unauthenticated requests
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

# Popularity scoring thresholds
STAR_THRESHOLDS = [10000, 5000, 1000, 500, 0]
FORK_THRESHOLDS = [2000, 1000, 500, 100, 0]
COMMIT_THRESHOLDS = [30, 180, 365, 730, float('inf')]  # Days since last commit

STAR_SCORES = [10, 8, 6, 4, 2]
FORK_SCORES = [10, 8, 6, 4, 2]
COMMIT_SCORES = [10, 8, 6, 4, 2]

def get_github_repo_popularity_data(owner, repo):
    """Fetch GitHub stars, forks, and latest commit activity for a given repository."""
    repo_url = f"https://api.github.com/repos/{owner}/{repo}"
    commits_url = f"https://api.github.com/repos/{owner}/{repo}/commits"

    try:
        # Get repo details
        repo_response = requests.get(repo_url, headers=HEADERS).json()
        stars = repo_response.get("stargazers_count", 0)
        forks = repo_response.get("forks_count", 0)

        # Get latest commits
        commits_response = requests.get(commits_url, headers=HEADERS).json()
        latest_commit_date = None

        if isinstance(commits_response, list) and len(commits_response) > 0:
            latest_commit_date = commits_response[0]["commit"]["committer"]["date"]

        return stars, forks, latest_commit_date

    except Exception as e:
        print(f"Error fetching data for {owner}/{repo}: {e}")
        return None, None, None

def calculate_score(value, thresholds, scores):
    """Assign a score based on predefined thresholds."""
    for i, threshold in enumerate(thresholds):
        if value >= threshold:
            return scores[i]
    return 0

def calculate_popularity_score(owner, repo):
    """Calculate overall popularity score for a GitHub repository."""
    stars, forks, latest_commit_date = get_github_repo_popularity_data(owner, repo)

    if stars is None or forks is None or latest_commit_date is None:
        return None

    # Calculate days since last commit
    last_commit_days = (datetime.utcnow() - datetime.strptime(latest_commit_date, "%Y-%m-%dT%H:%M:%SZ")).days

    # Assign scores
    star_score = calculate_score(stars, STAR_THRESHOLDS, STAR_SCORES)
    fork_score = calculate_score(forks, FORK_THRESHOLDS, FORK_SCORES)
    commit_score = calculate_score(last_commit_days, COMMIT_THRESHOLDS, COMMIT_SCORES)

    # Compute final popularity score (weighted sum)
    popularity_score = (0.15 * star_score) + (0.15 * fork_score) + (0.10 * commit_score)

    return {
        "stars": stars,
        "forks": forks,
        "last_commit_days": last_commit_days,
        "star_score": star_score,
        "fork_score": fork_score,
        "commit_score": commit_score,
        "popularity_score": round(popularity_score, 2)
    }

# Used by main evaluator function
def get_popularity_score(owner, repo):
    return calculate_popularity_score(owner, repo)

# Keywords for Documentation Quality Check
DOC_KEYWORDS = [
    "API Reference", "Installation", "Setup", "Authentication",
    "Error Handling", "Example Usage", "SDK", "CLI", "Configuration"
]

def check_documentation(owner, repo):
    """Fetch README content and analyze documentation quality."""
    url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    headers = {"Accept": "application/vnd.github.v3.raw"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            readme_content = response.text.lower()
            doc_score = sum(1 for keyword in DOC_KEYWORDS if keyword.lower() in readme_content)
            max_score = len(DOC_KEYWORDS)
            final_score = round((doc_score / max_score) * 10, 2)
            return final_score
        else:
            return 0  # No documentation found
    except requests.exceptions.RequestException:
        return 0

def check_license(owner, repo):
    """Check if a license exists for the repository."""
    url = f"https://api.github.com/repos/{owner}/{repo}/license"
    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            return True  # License exists
        else:
            return False  # No license found
    except requests.exceptions.RequestException:
        return False
    
def evaluate_documentation(owner, repo):
    """Evaluate documentation quality for MCP repositories."""
    score = check_documentation(owner, repo)
    score += 0.3 if check_license(owner, repo) else 0  # Add points for license
    return str(f"- Documentation Score: {score}/10")

def get_documentation_score(owner, repo):
    """Get documentation score for a given repository."""
    return evaluate_documentation(owner, repo)