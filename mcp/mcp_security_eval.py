import requests
import re
from typing import Dict, List, Any, Optional, Tuple
import os

class SecurityEvaluator:
    """
    Evaluates the security practices of a repository based on the scorecard criteria:
    1. Authentication Methods (40%)
    2. Network Behavior (30%)
    3. Configuration Security (30%)
    """
    
    def __init__(self, github_token: Optional[str] = None):
        """Initialize with optional GitHub token for API access."""
        self.github_token = github_token
        self.headers = {"Authorization": f"token {github_token}"} if github_token else {}
    
    def fetch_repo_content(self, owner: str, repo: str, path: str = "") -> List[Dict[str, Any]]:
        """Fetch repository contents from a specific path."""
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error fetching repo contents for {owner}/{repo}/{path}: Status code {response.status_code}")
                return []
        except Exception as e:
            print(f"Error fetching repo contents for {owner}/{repo}/{path}: {e}")
            return []
    
    def fetch_file_content(self, file_url: str) -> Optional[str]:
        """Fetch the content of a specific file from GitHub."""
        try:
            response = requests.get(file_url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                content_data = response.json()
                if 'content' in content_data and content_data.get('encoding') == 'base64':
                    import base64
                    return base64.b64decode(content_data['content']).decode('utf-8')
                return None
            else:
                print(f"Error fetching file content from {file_url}: Status code {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching file content from {file_url}: {e}")
            return None
    
    def check_oauth_implementation(self, code_files: List[str]) -> Tuple[bool, float]:
        """
        Check if the repository implements OAuth authentication.
        
        Returns:
            Tuple of (has_oauth, score)
        """
        oauth_patterns = [
            r'oauth',
            r'authorize_url',
            r'access_token',
            r'refresh_token',
            r'redirect_uri',
            r'openid',
            r'client_id.*client_secret'
        ]
        
        oauth_found = False
        for code in code_files:
            if not code:
                continue
                
            for pattern in oauth_patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    oauth_found = True
                    break
            
            if oauth_found:
                break
        
        # OAuth implementation positive security adjustment: 20%
        return oauth_found, 0.2 if oauth_found else 0.0
    
    def check_direct_api_tokens(self, code_files: List[str]) -> Tuple[bool, float]:
        """
        Check if the repository uses direct API tokens.
        
        Returns:
            Tuple of (has_direct_tokens, score)
        """
        token_patterns = [
            r'api[_\s]*key',
            r'api[_\s]*token',
            r'auth[_\s]*token',
            r'bearer[_\s]*token',
            r'access[_\s]*token',
            r'secret[_\s]*key',
            r'credentials',
            r'\.headers\[["\']Authorization["\']\]'
        ]
        
        tokens_found = False
        for code in code_files:
            if not code:
                continue
                
            for pattern in token_patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    tokens_found = True
                    break
            
            if tokens_found:
                break
        
        # Direct API token usage negative security adjustment: 20%
        return tokens_found, 0.2 if tokens_found else 0.0
    
    def check_configuration_security(self, code_files: List[str]) -> float:
        """
        Check for potential security risks in configuration handling.
        
        Returns:
            Risk score (0.0-0.3, higher is worse)
        """
        risk_patterns = [
            (r'eval\s*\(', 0.1),                   # Code execution via eval
            (r'exec\s*\(', 0.1),                   # Code execution via exec
            (r'base64\.decode|atob', 0.05),        # Potential obfuscation
            (r'\.innerHtml\s*=|document\.write', 0.05),  # Potential XSS
            (r'require\([\'"]crypto[\'"]', -0.05), # Proper crypto usage (positive)
            (r'bcrypt|scrypt|argon2', -0.05),      # Proper password hashing (positive)
            (r'jwt\.sign|verify', -0.03),          # Proper JWT handling (positive)
            (r'Math\.random\(\)', 0.05),           # Insecure randomness
            (r'new\s+Date\(\)\.getTime\(\)', 0.03) # Time-based randomness (weak)
        ]
        
        risk_score = 0.0
        
        for code in code_files:
            if not code:
                continue
                
            for pattern, weight in risk_patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    risk_score += weight
        
        # Normalize the score between 0 and 0.3 (30%)
        return min(0.3, max(0.0, risk_score))
    
    def analyze_outbound_connections(self, code_files: List[str]) -> float:
        """
        Analyze outbound network connections for security risks.
        
        Returns:
            Risk score (0.0-0.3, higher is worse)
        """
        connection_patterns = [
            (r'https?:\/\/', 0.02),                     # General HTTP requests
            (r'fetch\(|axios|http\.get|requests\.', 0.03),  # API clients
            (r'websocket|socket\.io', 0.04),           # WebSocket connections
            (r'tracking|analytics|telemetry', 0.05),   # Tracking services
            (r'cdn\.', 0.03),                          # CDN usage
            (r'adtech|ads\.', 0.05),                   # Ad networks
            (r'cdn\.jsdelivr\.net|unpkg\.com', 0.01),  # Common JS CDNs (lower risk)
            (r'api\.openai\.com', 0.02),               # OpenAI API usage
            (r'firebase|firestore', 0.03),             # Firebase services
            (r'amazonaws\.com|cloudfront\.net', 0.02), # AWS services
            (r'mixpanel|segment\.io|amplitude', 0.05), # Analytics services
            (r'https:\/\/localhost|127\.0\.0\.1', -0.03)  # Local development (reduce score)
        ]
        
        risk_score = 0.0
        
        for code in code_files:
            if not code:
                continue
                
            for pattern, weight in connection_patterns:
                matches = re.findall(pattern, code, re.IGNORECASE)
                if matches:
                    # Add weight for each unique match, with diminishing returns
                    risk_score += min(0.1, weight * min(len(set(matches)), 3))
        
        # Normalize the score between 0 and 0.3 (30%)
        return min(0.3, max(0.0, risk_score))
    
    def evaluate(self, owner: str, repo: str) -> Dict[str, Any]:
        """
        Evaluate the security practices of a GitHub repository.
        
        Returns:
            Dictionary with security assessment details and overall score
        """
        # Fetch key files from the repository
        repo_contents = self.fetch_repo_content(owner, repo)
        
        # Collect file URLs and paths for analysis
        file_urls = []
        
        for item in repo_contents:
            if item['type'] == 'file' and any(item['name'].endswith(ext) for ext in 
                                             ['.js', '.py', '.ts', '.go', '.java', '.php', '.rb', '.c', '.cpp', '.h']):
                file_urls.append(item['download_url'])
            elif item['type'] == 'dir' and item['name'] in ['src', 'lib', 'app', 'server', 'auth', 'config']:
                # Recursively scan important directories
                dir_contents = self.fetch_repo_content(owner, repo, item['path'])
                for dir_item in dir_contents:
                    if dir_item['type'] == 'file' and any(dir_item['name'].endswith(ext) for ext in 
                                                        ['.js', '.py', '.ts', '.go', '.java', '.php', '.rb', '.c', '.cpp', '.h']):
                        file_urls.append(dir_item['download_url'])
        
        # Limit to avoid excessive API calls
        file_urls = file_urls[:20]
        
        # Fetch file contents for analysis
        code_files = []
        for url in file_urls:
            content = self.fetch_file_content(url)
            if content:
                code_files.append(content)
        
        # Security evaluation based on the three key metrics
        has_oauth, oauth_score = self.check_oauth_implementation(code_files)
        has_direct_tokens, tokens_score = self.check_direct_api_tokens(code_files)
        
        # Authentication Methods (40%)
        # If both OAuth and direct tokens are present, consider it mixed (better than just tokens)
        if has_oauth and has_direct_tokens:
            auth_score = 0.40 - 0.10  # 30%
        elif has_oauth:
            auth_score = 0.40  # 40%
        elif has_direct_tokens:
            auth_score = 0.40 - 0.20  # 20%
        else:
            auth_score = 0.40 - 0.30  # 10% (couldn't determine authentication method)
        
        # Configuration Security (30%)
        config_risk_score = self.check_configuration_security(code_files)
        config_security_score = 0.30 - config_risk_score
        
        # Network Behavior (30%)
        network_risk_score = self.analyze_outbound_connections(code_files)
        network_security_score = 0.30 - network_risk_score
        
        # Calculate final security score (0-1 scale)
        security_score = auth_score + config_security_score + network_security_score
        
        # Determine security risk level
        if security_score <= 0.25:
            risk_level = "HIGH"
        elif security_score <= 0.50:
            risk_level = "MEDIUM"
        elif security_score <= 0.75:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
        
        return {
            "score": round(security_score, 2),
            "risk_level": risk_level,
            "details": {
                "authentication_score": round(auth_score, 2),
                "configuration_security_score": round(config_security_score, 2),
                "network_security_score": round(network_security_score, 2),
                "oauth_implementation": has_oauth,
                "direct_api_tokens": has_direct_tokens
            }
        }

def test_security_evaluation(owner: str, repo: str):
    """Test the security evaluator with a single repository."""
    github_token = os.getenv('GITHUB_TOKEN')
    evaluator = SecurityEvaluator(github_token)
    result = evaluator.evaluate(owner, repo)
    
    print(f"Security Evaluation for {owner}/{repo}:")
    print(f"Overall Security Score: {result['score']} - Risk Level: {result['risk_level']}")
    print("\nSecurity Details:")
    for metric, score in result["details"].items():
        print(f"- {metric.replace('_', ' ').title()}: {score}")
    print("-" * 50)

# Example usage
if __name__ == "__main__":
    # Test with a few example repositories
    for repo_info in [
        ("modelcontextprotocol", "servers"),
        ("modelcontextprotocol", "python-sdk")
    ]:
        test_security_evaluation(*repo_info)
