import requests
from datetime import datetime

# GitHub API token (optional but recommended for higher rate limits)
GITHUB_TOKEN = "your_github_token_here"  # Replace with your GitHub token or leave empty for unauthenticated requests
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

# Popularity scoring thresholds
STAR_THRESHOLDS = [10000, 5000, 1000, 500, 0]
FORK_THRESHOLDS = [2000, 1000, 500, 100, 0]
COMMIT_THRESHOLDS = [30, 180, 365, 730, float('inf')]  # Days since last commit

STAR_SCORES = [10, 8, 6, 4, 2]
FORK_SCORES = [10, 8, 6, 4, 2]
COMMIT_SCORES = [10, 8, 6, 4, 2]

def get_github_repo_data(owner, repo):
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
    stars, forks, latest_commit_date = get_github_repo_data(owner, repo)

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

# List of actual MCP server repositories to evaluate
repositories = [
    {"owner": "modelcontextprotocol", "repo": "servers"},
    {"owner": "modelcontextprotocol", "repo": "python-sdk"},
    {"owner": "modelcontextprotocol", "repo": "typescript-sdk"},
    {"owner": "modelcontextprotocol", "repo": "kotlin-sdk"},
    {"owner": "modelcontextprotocol", "repo": "java-sdk"}
]

# Calculate and display popularity scores
for repo in repositories:
    result = calculate_popularity_score(repo['owner'], repo['repo'])
    if result:
        print(f"Repository: {repo['owner']}/{repo['repo']}")
        print(f"Stars: {result['stars']}")
        print(f"Forks: {result['forks']}")
        print(f"Days Since Last Commit: {result['last_commit_days']}")
        print(f"Star Score: {result['star_score']}/10")
        print(f"Fork Score: {result['fork_score']}/10")
        print(f"Commit Activity Score: {result['commit_score']}/10")
        print(f"Total Popularity Score: {result['popularity_score']}/40")
        print("-" * 50)
    else:
        print(f"Failed to retrieve data for {repo['owner']}/{repo['repo']}")
