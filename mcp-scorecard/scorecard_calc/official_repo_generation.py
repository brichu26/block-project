import re
import csv
import pandas as pd


def extract_github_repos_from_readme(file_path):
    """
    Extracts GitHub repository owner/repo pairs and URLs from a README.md file.

    Args:
        file_path (str): Path to the README.md file

    Returns:
        list[tuple]: List of (owner, repo, url) tuples
    """
    github_repo_pattern = re.compile(r'(https://github\.com/([\w\.-]+)/([\w\.-]+))')
    repos = set()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            matches = github_repo_pattern.findall(line)
            for url, owner, repo in matches:
                repos.add((owner, repo, url))

    return sorted(repos)

def write_repos_to_csv(repos, output_path):
    """
    Writes the list of repositories to a CSV file.

    Args:
        repos (list[tuple]): List of (owner, repo, url) tuples
        output_path (str): Path to output CSV file
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['owner', 'repo', 'server_github_url'])  # Header
        writer.writerows(repos)





def generate_official_repos(source_df, target_df):
# Load CSVs
    
    #target_df = pd.read_csv('reference_mcp.csv')
    source_df = pd.read_csv(source_df)
    target_df = pd.read_csv(target_df)

    # Drop columns with no name or unnamed columns
    source_df = source_df.loc[:, source_df.columns.notna()]
    target_df = target_df.loc[:, target_df.columns.notna()]

    source_df = source_df.loc[:, ~source_df.columns.str.fullmatch('Unnamed.*|\\s*')]
    target_df = target_df.loc[:, ~target_df.columns.str.fullmatch('Unnamed.*|\\s*')]

    # Merge
    common_keys = ['owner', 'repo']
    merged_df = pd.merge(
        target_df,
        source_df[['owner', 'repo', 'github_stars', 'download_count', 'experimental_ai_generated_description']],
        on=common_keys,
        how='left'
    )

    # Drop empty-named columns post-merge (just in case)
    merged_df = merged_df.loc[:, merged_df.columns.str.strip().astype(bool)]

    # Save
    merged_df.to_csv('official_repos_with_metadata.csv', index=False)


if __name__ == "__main__":
    readme_path = "mcp_README_official.md"  # Adjust path as needed
    repositories = extract_github_repos_from_readme(readme_path)
    output_csv = "official_repos_name.csv"
    write_repos_to_csv(repositories, output_csv)
    print(f"Extracted {len(repositories)} repos to {output_csv}")

    source_df = 'get_all_servers.csv'
    target_df = 'official_repos_name.csv'

    generate_official_repos(source_df, target_df)