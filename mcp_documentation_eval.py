import requests

# MCP Repositories for Documentation Check
mcp_repos = [
    {"name": "MCP Servers", "owner": "modelcontextprotocol", "repo": "servers"},
    {"name": "Python SDK Server", "owner": "modelcontextprotocol", "repo": "python-sdk"},
    {"name": "TypeScript SDK Server", "owner": "modelcontextprotocol", "repo": "typescript-sdk"},
    {"name": "Kotlin SDK Server", "owner": "modelcontextprotocol", "repo": "kotlin-sdk"},
    {"name": "Java SDK Server", "owner": "modelcontextprotocol", "repo": "java-sdk"}
]

# GitHub API Token (optional)
GITHUB_TOKEN = "your_github_token_here"  # Replace with your token
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

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

def evaluate_documentation():
    """Evaluate documentation quality for MCP repositories."""
    for repo in mcp_repos:
        print(f"Checking documentation for {repo['name']} ({repo['owner']}/{repo['repo']})...")
        score = check_documentation(repo['owner'], repo['repo'])
        print(f"- Documentation Score: {score}/10")
        print("-" * 50)

# Run documentation evaluation
evaluate_documentation()
