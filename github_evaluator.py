import requests
import datetime

def fetch_github_repo_info(repo_url):
    api_url = f"https://api.github.com/repos/{repo_url}"
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def evaluate_github_metrics(repo_data):
    stars = repo_data.get('stargazers_count', 0)
    forks = repo_data.get('forks_count', 0)
    last_commit_date_str = repo_data.get('pushed_at', None)
    popularity_score = evaluate_popularity(stars, forks)
    activity_score = evaluate_activity(last_commit_date_str)
    return popularity_score, activity_score

def evaluate_popularity(stars, forks):
    star_score = 2
    fork_score = 2

    if stars >= 10000:
        star_score = 10
    elif stars >= 5000:
        star_score = 8
    elif stars >= 1000:
        star_score = 6
    elif stars >= 500:
        star_score = 4

    if forks >= 2000:
        fork_score = 10
    elif forks >= 1000:
        fork_score = 8
    elif forks >= 500:
        fork_score = 6
    elif forks >= 100:
        fork_score = 4

    return star_score, fork_score

def evaluate_activity(last_commit_date_str):
    if last_commit_date_str:
        last_commit_date = datetime.datetime.strptime(last_commit_date_str, '%Y-%m-%dT%H:%M:%SZ')
        time_since_last_commit = datetime.datetime.now() - last_commit_date
        days_since_last_commit = time_since_last_commit.days

        if days_since_last_commit < 30:
            return 10
        elif days_since_last_commit < 180:
            return 8
        elif days_since_last_commit < 365:
            return 6
        elif days_since_last_commit < 730:
            return 4
        else:
            return 2
    return 2