import requests
import re
from typing import Dict, List, Tuple, Any, Optional
from bs4 import BeautifulSoup

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

# Documentation quality assessment metrics (as per the scorecard)
class DocumentationEvaluator:
    """Evaluates the documentation quality of a repository based on the provided scorecard."""
    
    # Documentation sections to look for
    ALL_SECTIONS = [
        "installation", "usage", "configuration", "api_reference", "error_handling",
        "license", "project_info", "examples", "contributing", "faq", "troubleshooting",
        "architecture", "security", "changelog", "roadmap"
    ]
    
    # Essential sections with higher importance
    ESSENTIAL_SECTIONS = [
        "installation", "usage", "configuration", "api_reference", 
        "error_handling", "license", "project_info"
    ]
    
    def __init__(self, github_token: Optional[str] = None):
        """Initialize the documentation evaluator with optional GitHub token."""
        self.github_token = github_token
        self.headers = {"Authorization": f"token {github_token}"} if github_token else {}
    
    def fetch_readme(self, owner: str, repo: str) -> Optional[str]:
        """Fetch README content from a GitHub repository."""
        url = f"https://api.github.com/repos/{owner}/{repo}/readme"
        
        try:
            headers = {"Accept": "application/vnd.github.v3.raw"}
            if self.github_token:
                headers["Authorization"] = f"token {self.github_token}"
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                return response.text
            else:
                print(f"Error fetching README for {owner}/{repo}: Status code {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching README for {owner}/{repo}: {e}")
            return None
    
    def count_sections(self, readme_content: str) -> Tuple[List[str], List[str]]:
        """Count the number of standard and essential sections in the documentation."""
        # Convert to lowercase for case-insensitive matching
        content_lower = readme_content.lower()
        
        # Find all sections present in the README
        found_sections = []
        for section in self.ALL_SECTIONS:
            # Check for section using different patterns (headings, bold text, etc.)
            patterns = [
                f"#{{{1,6}}}\\s+{section}\\b",  # Markdown headings
                f"\\b{section}\\b",             # Standard word match
                f"\\*\\*{section}\\*\\*"        # Bold text
            ]
            
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    found_sections.append(section)
                    break
        
        # Find essential sections present in the README
        found_essential_sections = [s for s in found_sections if s in self.ESSENTIAL_SECTIONS]
        
        return found_sections, found_essential_sections
    
    def calculate_readability_score(self, readme_content: str) -> float:
        """Calculate readability score based on average sentence length."""
        # Split content into sentences
        sentences = re.split(r'(?<=[.!?])\s+', readme_content)
        
        # Ignore very short sentences (likely headings or not full sentences)
        sentences = [s for s in sentences if len(s.split()) >= 3]
        
        if not sentences:
            return 0.0
        
        # Calculate average sentence length
        total_words = sum(len(s.split()) for s in sentences)
        avg_sentence_length = total_words / len(sentences)
        
        # Score based on average sentence length
        if 15 <= avg_sentence_length <= 20:
            return 5.0  # Optimal
        elif avg_sentence_length < 10:
            return 3.5  # Too short
        elif 20 < avg_sentence_length <= 30:
            return 3.5  # Getting wordy
        elif avg_sentence_length > 30:
            return 2.0  # Too complex
        else:
            return 4.0  # Reasonable
    
    def calculate_heading_structure_bonus(self, readme_content: str) -> float:
        """Evaluate the heading structure for organization quality."""
        # Convert to html for easier parsing of headers
        html_content = readme_content  # Assuming markdown; convert if needed
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract headings (approximate for markdown)
        headings = re.findall(r'#{1,6}\s+.+', readme_content)
        sections = len(headings)
        
        # Calculate hierarchy depth
        h1_count = len(re.findall(r'^#\s+', readme_content, re.MULTILINE))
        h2_count = len(re.findall(r'^##\s+', readme_content, re.MULTILINE))
        h3_count = len(re.findall(r'^###\s+', readme_content, re.MULTILINE))
        
        good_hierarchy = (h1_count >= 1 and h2_count >= 2) or (h1_count + h2_count + h3_count >= 4)
        
        # Section sizing analysis (approximate)
        content_blocks = re.split(r'#{1,6}\s+.+', readme_content)
        avg_section_size = sum(len(block) for block in content_blocks) / max(1, len(content_blocks))
        good_sizing = 300 <= avg_section_size <= 1500
        
        # Calculate score components
        good_sections_score = 2.0 if sections >= 5 else 0.0
        good_hierarchy_score = 2.0 if good_hierarchy else 0.0
        good_sizing_score = 1.0 if good_sizing else 0.0
        
        return good_sections_score + good_hierarchy_score + good_sizing_score
    
    def count_code_examples(self, readme_content: str) -> int:
        """Count the number of code blocks in the README."""
        # Look for markdown code blocks
        code_blocks = re.findall(r'```[\s\S]*?```', readme_content)
        inline_code = re.findall(r'`[^`]+`', readme_content)
        
        return len(code_blocks) + (len(inline_code) // 5)  # Count 5 inline codes as 1 code example
    
    def evaluate(self, owner: str, repo: str) -> Dict[str, Any]:
        """Evaluate documentation quality for a GitHub repository."""
        readme_content = self.fetch_readme(owner, repo)
        
        if not readme_content:
            return {
                "score": 0.0,
                "details": {
                    "base_section_score": 0.0,
                    "essential_sections_bonus": 0.0,
                    "readability_score": 0.0,
                    "size_bonus": 0.0,
                    "heading_structure_bonus": 0.0,
                    "code_examples_bonus": 0.0,
                    "completeness_bonus": 0.0
                },
                "rating": "Very Poor"
            }
        
        # 1. Base Section Score (30%)
        found_sections, found_essential_sections = self.count_sections(readme_content)
        base_section_count = min(15, len(found_sections))
        base_section_score = base_section_count / 15.0 * 30.0
        
        # 2. Essential Sections Bonus (20%)
        essential_sections_count = len(found_essential_sections)
        essential_sections_bonus = essential_sections_count / len(self.ESSENTIAL_SECTIONS) * 20.0
        
        # 3. Readability Score (15%)
        readability_raw = self.calculate_readability_score(readme_content)
        readability_score = readability_raw / 5.0 * 15.0
        
        # 4. Size Bonus (10%)
        size_bonus_raw = min(5.0, len(readme_content) / 1500.0)
        size_bonus = size_bonus_raw / 5.0 * 10.0
        
        # 5. Heading Structure Bonus (10%)
        heading_structure_raw = self.calculate_heading_structure_bonus(readme_content)
        heading_structure_bonus = heading_structure_raw / 5.0 * 10.0
        
        # 6. Code Examples Bonus (10%)
        code_blocks_count = self.count_code_examples(readme_content)
        code_examples_raw = min(3.0, code_blocks_count / 2.0)
        code_examples_bonus = code_examples_raw / 3.0 * 10.0
        
        # 7. Completeness Bonus (5%)
        completeness_raw = len(found_sections) / len(self.ALL_SECTIONS) * 10.0 * 0.5
        completeness_bonus = min(5.0, completeness_raw)
        
        # Total score (0-10 scale)
        total_score = (base_section_score + essential_sections_bonus + readability_score + 
                        size_bonus + heading_structure_bonus + code_examples_bonus + 
                        completeness_bonus) / 10.0
        
        # Determine rating
        if 8.0 <= total_score <= 10.0:
            rating = "Excellent"
        elif 6.0 <= total_score < 8.0:
            rating = "Good"
        elif 4.0 <= total_score < 6.0:
            rating = "Average"
        elif 2.0 <= total_score < 4.0:
            rating = "Poor"
        else:
            rating = "Very Poor"
        
        return {
            "score": round(total_score, 2),
            "details": {
                "base_section_score": round(base_section_score, 2),
                "essential_sections_bonus": round(essential_sections_bonus, 2),
                "readability_score": round(readability_score, 2),
                "size_bonus": round(size_bonus, 2),
                "heading_structure_bonus": round(heading_structure_bonus, 2),
                "code_examples_bonus": round(code_examples_bonus, 2),
                "completeness_bonus": round(completeness_bonus, 2)
            },
            "rating": rating
        }

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

# Function to test the evaluator with a sample repository
def test_documentation_evaluation(owner: str, repo: str):
    """Test the documentation evaluator with a single repository."""
    evaluator = DocumentationEvaluator()
    result = evaluator.evaluate(owner, repo)
    
    print(f"Documentation Evaluation for {owner}/{repo}:")
    print(f"Overall Score: {result['score']}/10 - {result['rating']}")
    print("\nDetailed Scores:")
    for metric, score in result["details"].items():
        print(f"- {metric.replace('_', ' ').title()}: {score}")
    print("-" * 50)

# Run documentation evaluation
evaluate_documentation()

# Example usage
if __name__ == "__main__":
    # Test with a few example repositories
    for repo_info in [
        ("modelcontextprotocol", "servers"),
        ("modelcontextprotocol", "python-sdk")
    ]:
        test_documentation_evaluation(*repo_info)
