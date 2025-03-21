from github_evaluator import fetch_github_repo_info, evaluate_github_metrics
from server_evaluator import evaluate_performance_scalability, evaluate_security_reliability, evaluate_documentation_developer_experience

def evaluate_server(repo_url, weights):
    repo_data = fetch_github_repo_info(repo_url)

    if not repo_data:
        print("Failed to fetch data from GitHub.")
        return

    # Evaluate GitHub Metrics
    popularity_score, activity_score = evaluate_github_metrics(repo_data)

    # Evaluate other metrics
    #performance_scalability_score = evaluate_performance_scalability()
    #security_reliability_score = evaluate_security_reliability()
    #documentation_experience_score = evaluate_documentation_developer_experience()

    # Calculate total score with customizable weights
    total_score = (
        sum(popularity_score) * weights['popularity'] +
        activity_score * weights['activity'] 
        #sum(performance_scalability_score) * weights['performance_scalability'] +
        #sum(security_reliability_score) * weights['security_reliability'] +
        #sum(documentation_experience_score) * weights['documentation_experience']
    )

    print(f"Total Score for {repo_url}: {total_score:.2f}")

if __name__ == '__main__':
    repos = [
        "JetBrains/mcp-jetbrains",
        "stripe/agent-toolkit",
        "googleapis/python-firestore",
        "laulauland/bluesky-context-server",
        "openbnb-org/mcp-server-airbnb",
        "delorenj/mcp-server-ticketmaster",
        "financial-datasets/mcp-server"
        

    ]

    # Define the weights for each category
    weights = {
        'popularity': 0.4,
        'activity': 0.1,
        'performance_scalability': 0.2,
        'security_reliability': 0.2,
        'documentation_experience': 0.1,
    }

    for repo in repos:
        evaluate_server(repo, weights)