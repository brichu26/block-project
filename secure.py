import csv
from datetime import datetime, timezone
import os
import re
import requests
import time
import subprocess
import json
from urllib.parse import urlparse
from collections import defaultdict
from tqdm import tqdm

HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

# Popularity thresholds 
STAR_THRESHOLDS = [10000, 5000, 1000, 500, 100]
STAR_SCORES = [10, 8, 6, 4, 2]
FORK_THRESHOLDS = [5000, 1000, 500, 100, 50]
FORK_SCORES = [10, 8, 6, 4, 2]
COMMIT_THRESHOLDS = [30, 90, 180, 365, 730]  # Days since last commit
COMMIT_SCORES = [10, 8, 6, 4, 2]
DOWNLOAD_THRESHOLDS = [100000, 50000, 10000, 5000, 1000]
DOWNLOAD_SCORES = [10, 8, 6, 4, 2]

class MCPAnalyzer:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.servers = []
        self.results = []
        self.suspicious_patterns = [
            r"eval\s*\(",
            r"exec\s*\(",
            r"base64\.b64decode",
            r"send.*data",
            r"collect.*data",
            r"tracking",
            r"fetch\s*\(",
            r"axios\.post",
            r"request\.post",
            r"\.send\(",
            r"store_all_queries",
            r"log_all",
        ]
        self.suspicious_domains = [
            "data-collector",
            "analytics",
            "tracker",
            "metrics",
            "telemetry",
            "logging",
        ]
        # Documentation quality indicators with enhanced keywords
        self.doc_quality_indicators = {
            # Installation and setup
            'installation': [
                r'install(?:ation|ing)?', 
                r'set(?:\s)?up', 
                r'getting started',
                r'quick start', 
                r'prerequisites', 
                r'dependencies',
                r'npm install', 
                r'pip install',
                r'docker (?:run|pull|build)',
                r'brew install',
                r'apt(?:\-get)? install',
                r'yarn add',
                r'composer require',
                r'requirements\.txt',
                r'package\.json'
            ],
            
            # Usage examples
            'usage': [
                r'usage', 
                r'how to use', 
                r'(?:example|basic) usage', 
                r'examples?',
                r'getting started with',
                r'quick tutorial',
                r'sample code',
                r'demo',
                r'walkthrough',
                r'step[- ]by[- ]step',
                r'guide'
            ],
            
            # Configuration options
            'configuration': [
                r'config(?:uration)?', 
                r'settings', 
                r'options', 
                r'parameters',
                r'environment variables?',
                r'\.env',
                r'(?:json|yaml|xml|ini) config',
                r'customiz(?:e|ing|ation)',
                r'preferences',
                r'\.config\.',
                r'flags',
                r'arguments',
                r'optional inputs'
            ],
            
            # API documentation
            'api_reference': [
                r'api reference', 
                r'api documentation',
                r'api',
                r'methods',
                r'functions?', 
                r'endpoints?',
                r'routes?',
                r'sdk',
                r'interface',
                r'(?:public|exposed) methods',
                r'class reference',
                r'parameters?',
                r'return values?',
                r'inputs?',
                r'outputs?',
                r'required inputs?',
                r'optional inputs?'
            ],
            
            # Security information
            'security': [
                r'security', 
                r'authentication', 
                r'permissions', 
                r'auth(?:orization)?',
                r'token',
                r'api key',
                r'credentials',
                r'oauth',
                r'jwt',
                r'secret',
                r'encryption',
                r'privacy',
                r'access control',
                r'scope[s]?',
                r'cors'
            ],
            
            # Error handling
            'error_handling': [
                r'error handling', 
                r'errors?', 
                r'exceptions?', 
                r'troubleshoot(?:ing)?',
                r'debugging',
                r'common problems',
                r'issue[s]? and solutions',
                r'faq',
                r'solutions',
                r'limitations',
                r'known issues',
                r'status codes',
                r'error codes',
                r'retry',
                r'validation'
            ],
            
            # Command line interface
            'cli': [
                r'cli', 
                r'command[- ]line', 
                r'terminal', 
                r'shell',
                r'console',
                r'bash',
                r'zsh',
                r'powershell',
                r'cmd',
                r'commands',
                r'flags',
                r'arguments',
                r'npx',
                r'executable'
            ],
            
            # Visual documentation
            'screenshots': [
                r'!\[.*?\]', 
                r'<img',
                r'screenshot',
                r'diagram',
                r'image',
                r'figure',
                r'\.(?:png|jpe?g|gif|svg)',
                r'visual',
                r'ui',
                r'interface preview'
            ],
            
            # Code examples
            'code_examples': [
                r'```[\s\S]*?```',  # Matches entire code blocks
                r'~~~[\s\S]*?~~~',  # Alternative code block
                r'<code>[\s\S]*?</code>', 
                r'<pre>[\s\S]*?</pre>',
                r'example code',
                r'code snippet',
                r'sample implementation',
                r'implementation example',
                r'example usage',
                r'usage example',
                r'working example'
            ],
            
            # Contribution guidelines
            'contribution': [
                r'contribut(?:e|ing|ion|ors)',
                r'pull request', 
                r'pr',
                r'issue',
                r'development',
                r'developers',
                r'community',
                r'collaborate',
                r'participation',
                r'guidelines',
                r'code of conduct',
                r'coding standards?',
                r'style guide',
                r'commit',
                r'branch',
                r'fork'
            ],
            
            # License information
            'license': [
                r'licen[sc]e', 
                r'copyright', 
                r'mit', 
                r'apache', 
                r'gpl',
                r'bsd',
                r'proprietary',
                r'©',
                r'all rights reserved',
                r'open source',
                r'legal',
                r'terms',
                r'permissions'
            ],
            
            # Project information
            'project_info': [
                r'about',
                r'overview',
                r'introduction',
                r'description',
                r'features',
                r'motivation',
                r'philosophy',
                r'architecture',
                r'version',
                r'release notes',
                r'changelog',
                r'roadmap',
                r'status'
            ],
            
            # Tool/Function reference
            'tools_reference': [
                r'tools?',
                r'functions?',
                r'capabilities',
                r'components',
                r'modules',
                r'services',
                r'plugins',
                r'extensions',
                r'actions',
                r'commands',
                r'operations'
            ],
            
            # Requirements
            'requirements': [
                r'requirements?',
                r'prerequisites?',
                r'dependencies',
                r'compatibility',
                r'supported',
                r'system requirements',
                r'minimum requirements',
                r'recommended',
                r'needed',
                r'necessary'
            ],
            
            # Testing information
            'testing': [
                r'test(?:s|ing)?',
                r'unit tests',
                r'integration tests',
                r'e2e',
                r'end to end',
                r'coverage',
                r'jest',
                r'mocha',
                r'pytest',
                r'rspec',
                r'assertions?',
                r'mocks?',
                r'stubs?'
            ],
            
            # Advanced usage
            'advanced_usage': [
                r'advanced',
                r'custom(?:ization)?',
                r'extending',
                r'plugins?',
                r'hooks',
                r'middleware',
                r'internals',
                r'under the hood',
                r'performance tuning',
                r'optimization',
                r'best practices',
                r'tips'
            ]
        }

    def load_servers(self):
        """Load MCP server data from CSV file"""
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.servers = list(reader)
            print(f"Loaded {len(self.servers)} MCP servers")
            return True
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return False

    def check_github_repo(self, owner, repo):
        """Check GitHub repository for suspicious code and configuration"""
        if not owner or not repo:
            return {"status": "error", "message": "Missing owner or repo"}
            
        result = {
            "owner": owner,
            "repo": repo,
            "suspicious_files": [],
            "oauth_implementation": False,
            "direct_api_tokens": False,
            "outbound_connections": [],
            "risk_score": 0,
            "risk_factors": [],
            "doc_quality_score": 0,
            "doc_quality_details": {}
        }
        
        # Clone repository to temporary directory
        temp_dir = f"temp_{owner}_{repo}"
        try:
            if os.path.exists(temp_dir):
                subprocess.run(["rm", "-rf", temp_dir], check=True)
                
            print(f"Cloning {owner}/{repo}...")
            subprocess.run(
                ["git", "clone", f"https://github.com/{owner}/{repo}.git", temp_dir, "--depth", "1"], 
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            
            # Check documentation quality
            readme_quality = self._analyze_readme_quality(temp_dir)
            result["doc_quality_score"] = readme_quality["score"]
            result["doc_quality_details"] = readme_quality["details"]
            
            # Check for OAuth implementation
            result["oauth_implementation"] = self._check_for_oauth(temp_dir)
            
            # Check for direct API tokens
            result["direct_api_tokens"] = self._check_for_api_tokens(temp_dir)
            
            # Find configuration files
            config_files = self._find_config_files(temp_dir)
            
            # Analyze each configuration file
            for file_path in config_files:
                file_analysis = self._analyze_file(file_path)
                if file_analysis["suspicious"]:
                    result["suspicious_files"].append(file_analysis)
                    result["outbound_connections"].extend(file_analysis.get("outbound_connections", []))
                    result["risk_score"] += file_analysis["risk_score"]
                    result["risk_factors"].extend(file_analysis.get("risk_factors", []))
            
            # Check for network-related code
            network_analysis = self._analyze_network_code(temp_dir)
            result["outbound_connections"].extend(network_analysis.get("outbound_connections", []))
            result["risk_score"] += network_analysis["risk_score"]
            result["risk_factors"].extend(network_analysis.get("risk_factors", []))
            
            # Remove duplicates
            result["outbound_connections"] = list(set(result["outbound_connections"]))
            result["risk_factors"] = list(set(result["risk_factors"]))
            
        except Exception as e:
            result["status"] = "error"
            result["message"] = str(e)
        finally:
            # Clean up
            if os.path.exists(temp_dir):
                subprocess.run(["rm", "-rf", temp_dir], check=True)
        
        return result
        
    def _analyze_readme_quality(self, repo_path):
        """Analyze the quality of README documentation"""
        result = {
            "score": 0,
            "details": {
                "readme_found": False,
                "readme_size_bytes": 0,
                "sections_found": [],
                "missing_sections": [],
                "has_code_examples": False,
                "has_screenshots": False,
                "has_security_info": False,
                "has_error_handling": False,
                "has_cli_docs": False,
                "has_license_info": False,
                "has_project_info": False,
                "has_tools_reference": False,
                "has_requirements": False,
                "has_testing_info": False,
                "has_advanced_usage": False,
                "keyword_matches": {},  # Store count of matches per keyword
                "readability_score": 0,
                "headings_count": 0,
                "code_blocks_count": 0,
                "documentation_completeness": 0  # New metric for completeness
            }
        }
        
        # Look for README files (case insensitive) with expanded search patterns
        readme_patterns = [
            # Main directory patterns
            "README.md", "Readme.md", "readme.md", "README.txt", "readme.txt", "README", "readme",
            # Documentation directory patterns
            "docs/README.md", "docs/index.md", "docs/getting-started.md", "docs/overview.md", "docs/introduction.md",
            # Additional common patterns
            "documentation.md", "DOCUMENTATION.md", "guide.md", "GUIDE.md", "index.md", "manual.md", "MANUAL.md"
        ]
        readme_path = None
        
        for pattern in readme_patterns:
            potential_path = os.path.join(repo_path, pattern)
            if os.path.exists(potential_path):
                readme_path = potential_path
                result["details"]["readme_found"] = True
                break
                
        if not readme_path:
            # If no main README, check if there's a docs directory with documentation
            docs_dir = os.path.join(repo_path, "docs")
            if os.path.exists(docs_dir) and os.path.isdir(docs_dir):
                # Check for any markdown file in docs directory
                for file in os.listdir(docs_dir):
                    if file.endswith(".md") or file.endswith(".txt"):
                        potential_path = os.path.join(docs_dir, file)
                        readme_path = potential_path
                        result["details"]["readme_found"] = True
                        break
        
        if not readme_path:
            return result
            
        try:
            with open(readme_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                
            # Analyze size and content
            result["details"]["readme_size_bytes"] = len(content)
            
            # No documentation is worse than minimal documentation
            if len(content) < 100:
                result["score"] = 0
                return result
                
            # Check for presence of each documentation indicator
            base_score = 0
            total_matches = 0
            total_sections = len(self.doc_quality_indicators)
            
            for section, patterns in self.doc_quality_indicators.items():
                matches_found = 0
                section_strength = 0  # Measure of how strong this section's presence is
                
                for pattern in patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        matches_found += len(matches)
                        section_strength += min(5, len(matches))  # Cap at 5 to prevent over-counting
                        
                        if section not in result["details"]["sections_found"]:
                            result["details"]["sections_found"].append(section)
                            base_score += 1
                            
                        # Special case checks with expanded logic
                        if section == 'code_examples':
                            result["details"]["has_code_examples"] = True
                            # Count actual code blocks (not just mentions)
                            code_blocks = len(re.findall(r'```[\s\S]*?```', content))
                            result["details"]["code_blocks_count"] = code_blocks
                        elif section == 'screenshots':
                            result["details"]["has_screenshots"] = True
                        elif section == 'security':
                            result["details"]["has_security_info"] = True
                        elif section == 'license':
                            result["details"]["has_license_info"] = True
                        elif section == 'error_handling':
                            result["details"]["has_error_handling"] = True
                        elif section == 'cli':
                            result["details"]["has_cli_docs"] = True
                        elif section == 'project_info':
                            result["details"]["has_project_info"] = True
                        elif section == 'tools_reference':
                            result["details"]["has_tools_reference"] = True
                        elif section == 'requirements':
                            result["details"]["has_requirements"] = True
                        elif section == 'testing':
                            result["details"]["has_testing_info"] = True
                        elif section == 'advanced_usage':
                            result["details"]["has_advanced_usage"] = True
                
                # Record keyword match counts with strength
                if matches_found > 0:
                    result["details"]["keyword_matches"][section] = {
                        "count": matches_found,
                        "strength": min(10, section_strength)  # Scale 0-10
                    }
                    total_matches += matches_found
            
            # Calculate documentation completeness (percentage of sections covered)
            covered_sections = len(result["details"]["sections_found"])
            result["details"]["documentation_completeness"] = round((covered_sections / total_sections) * 10, 1)
            
            # Calculate readability score - enhanced version
            sentences = re.split(r'[.!?]', content)
            words = re.findall(r'\w+', content)
            
            # Avoid division by zero
            if len(sentences) > 0 and len(words) > 0:
                avg_sentence_length = len(words) / len(sentences)
                # Optimal sentence length is around 15-20 words
                if avg_sentence_length < 10:
                    readability_score = 3.5  # Too short/fragmented
                elif avg_sentence_length < 20:
                    readability_score = 5.0  # Optimal
                elif avg_sentence_length < 30:
                    readability_score = 3.5  # Getting wordy
                else:
                    readability_score = 2.0  # Too complex
            else:
                readability_score = 1.0
                
            result["details"]["readability_score"] = round(readability_score, 1)
            
            # Enhanced heading analysis (good documentation has clear headers)
            # Extract all headings with their levels
            all_headings = re.findall(r'^(#+)\s+(.+)', content, re.MULTILINE)
            result["details"]["headings_count"] = len(all_headings)
            
            # Check for heading hierarchy - good docs have proper hierarchy (h1 → h2 → h3)
            heading_levels = [len(h[0]) for h in all_headings]
            good_hierarchy = True
            
            if heading_levels:
                # Check if heading levels follow a logical progression
                for i in range(1, len(heading_levels)):
                    if heading_levels[i] > heading_levels[i-1] + 1:
                        good_hierarchy = False
                        break
                        
            heading_quality = 0
            if len(all_headings) > 5:
                heading_quality += 2  # Good number of sections
            if good_hierarchy:
                heading_quality += 2  # Good hierarchy
            if len(all_headings) > 0 and len(content) / len(all_headings) < 1000:
                heading_quality += 1  # Good section size (not too lengthy between headings)
                
            # Calculate final score with improved weighting
            # Base score from sections (up to 15 now with new categories)
            base_section_score = min(15, base_score) 
            
            # Size bonus (relative to complexity)
            size_bonus = min(5, result["details"]["readme_size_bytes"] / 1500)
            
            # Heading structure bonus
            heading_bonus = min(5, heading_quality + (len(all_headings) / 4))
            
            # Code examples bonus
            code_bonus = min(3, result["details"]["code_blocks_count"] / 2)
            
            # Completeness bonus
            completeness_bonus = result["details"]["documentation_completeness"] * 0.5
            
            # Check for essential sections
            key_sections = [
                'installation', 'usage', 'configuration', 'api_reference', 
                'error_handling', 'license', 'project_info'
            ]
            
            essential_sections_found = 0
            for key_section in key_sections:
                if key_section not in result["details"]["sections_found"]:
                    result["details"]["missing_sections"].append(key_section)
                else:
                    essential_sections_found += 1
                    
            # Essential sections bonus
            essential_sections_bonus = (essential_sections_found / len(key_sections)) * 5
            
            # Final score calculation
            final_score = (
                base_section_score * 0.3 +      # 30% weight - has critical sections
                essential_sections_bonus * 0.2 + # 20% weight - has essential sections
                readability_score * 0.15 +      # 15% weight - readable
                size_bonus * 0.1 +              # 10% weight - substantial
                heading_bonus * 0.1 +           # 10% weight - well structured
                code_bonus * 0.1 +              # 10% weight - has examples
                completeness_bonus * 0.05       # 5% weight - overall completeness
            )
            
            result["score"] = min(10, round(final_score, 1))  # Scale to 0-10
                
        except Exception as e:
            print(f"Error analyzing README: {e}")
            result["details"]["error"] = str(e)
            
        return result

    def _check_for_oauth(self, repo_path):
        """Check if the repository implements OAuth"""
        oauth_patterns = [
            r"oauth",
            r"authorize\s*\(",
            r"authentication_url",
            r"authorization_url",
            r"client_id",
            r"client_secret",
            r"redirect_uri",
            r"authorization_code",
            r"access_token",
            r"refresh_token",
        ]
        
        # Check for OAuth-related files and code
        for pattern in oauth_patterns:
            command = f"grep -r -i -l '{pattern}' {repo_path} --include='*.py' --include='*.js' --include='*.ts' --include='*.json' --include='*.yaml' --include='*.yml' 2>/dev/null"
            result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)
            if result.stdout.strip():
                return True
                
        return False

    def _check_for_api_tokens(self, repo_path):
        """Check if the repository uses direct API tokens"""
        token_patterns = [
            r"api_token",
            r"api_key",
            r"bearer_token",
            r"authorization:\s*Bearer",
            r"authorization:\s*Token",
            r"x-api-key",
        ]
        
        for pattern in token_patterns:
            command = f"grep -r -i -l '{pattern}' {repo_path} --include='*.py' --include='*.js' --include='*.ts' --include='*.json' --include='*.yaml' --include='*.yml' 2>/dev/null"
            result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)
            if result.stdout.strip():
                return True
                
        return False

    def _find_config_files(self, repo_path):
        """Find configuration files in the repository"""
        config_patterns = [
            "*.json",
            "*.yaml",
            "*.yml",
            "*.toml",
            "*.ini",
            "*.config",
            "config*",
            "settings*",
        ]
        
        config_files = []
        for pattern in config_patterns:
            command = f"find {repo_path} -name '{pattern}' -type f 2>/dev/null"
            result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)
            if result.stdout.strip():
                config_files.extend(result.stdout.strip().split('\n'))
                
        return config_files

    def _analyze_file(self, file_path):
        """Analyze a file for suspicious patterns"""
        result = {
            "file": os.path.basename(file_path),
            "path": file_path,
            "suspicious": False,
            "found_patterns": [],
            "outbound_connections": [],
            "risk_score": 0,
            "risk_factors": []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for suspicious patterns
            for pattern in self.suspicious_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    result["suspicious"] = True
                    result["found_patterns"].append(pattern)
                    result["risk_score"] += len(matches)
                    result["risk_factors"].append(f"Found pattern: {pattern}")
            
            # Check for URLs and endpoints
            urls = re.findall(r'https?://[^\s"\'\)\}]+', content)
            for url in urls:
                parsed_url = urlparse(url)
                domain = parsed_url.netloc
                
                # Check for suspicious domains
                for suspicious in self.suspicious_domains:
                    if suspicious in domain:
                        result["suspicious"] = True
                        result["outbound_connections"].append(url)
                        result["risk_score"] += 2
                        result["risk_factors"].append(f"Suspicious domain: {domain}")
                        break
                
                # Add all external domains as potential connections
                if domain not in ("localhost", "127.0.0.1"):
                    result["outbound_connections"].append(url)
            
            # Check for potential data exfiltration
            data_exfil_patterns = [
                r"send.*data",
                r"upload.*data",
                r"store.*queries",
                r"log.*queries",
                r"POST\s+https?://",
                r"\.post\s*\(",
            ]
            
            for pattern in data_exfil_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    result["suspicious"] = True
                    result["risk_score"] += 3
                    result["risk_factors"].append(f"Potential data exfiltration: {pattern}")
            
            # Check for obfuscated code
            obfuscation_patterns = [
                r"eval\s*\(.*\)",
                r"base64\.b64decode\(",
                r"exec\s*\(",
                r"atob\s*\(",
                r"decodeURIComponent\s*\(",
            ]
            
            for pattern in obfuscation_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    result["suspicious"] = True
                    result["risk_score"] += 5
                    result["risk_factors"].append(f"Obfuscated code: {pattern}")
                    
        except Exception as e:
            result["error"] = str(e)
            
        return result

    def _analyze_network_code(self, repo_path):
        """Analyze code files for network-related operations"""
        result = {
            "outbound_connections": [],
            "risk_score": 0,
            "risk_factors": []
        }
        
        # Find files with potential network code
        network_patterns = [
            r"requests\.post",
            r"requests\.get",
            r"axios\.",
            r"fetch\(",
            r"http\.",
            r"\.send\(",
            r"WebSocket",
            r"\.emit\(",
        ]
        
        for pattern in network_patterns:
            command = f"grep -r -l '{pattern}' {repo_path} --include='*.py' --include='*.js' --include='*.ts' 2>/dev/null"
            files = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True).stdout.strip()
            
            if files:
                for file_path in files.split('\n'):
                    if file_path:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        # Extract URLs
                        urls = re.findall(r'https?://[^\s"\'\)\}]+', content)
                        for url in urls:
                            parsed_url = urlparse(url)
                            domain = parsed_url.netloc
                            
                            # Check for suspicious domains
                            for suspicious in self.suspicious_domains:
                                if suspicious in domain:
                                    result["outbound_connections"].append(url)
                                    result["risk_score"] += 2
                                    result["risk_factors"].append(f"Suspicious network connection to {domain}")
                                    break
                            
                            # Add all external domains
                            if domain not in ("localhost", "127.0.0.1"):
                                result["outbound_connections"].append(url)
        
        return result
    
    def calculate_score(self, value, thresholds, scores):
        """Assign a score based on predefined thresholds."""
        for i, threshold in enumerate(thresholds):
            if value >= threshold:
                return scores[i]
        return 0
    
    def safe_get_json(self, url):
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Request failed for {url}: {e}")
            return {}

    def calculate_popularity_score(self, owner, repo, stars, downloads):
        repo_url = f"https://api.github.com/repos/{owner}/{repo}"
        commits_url = f"https://api.github.com/repos/{owner}/{repo}/commits"

        repo_data = self.safe_get_json(repo_url)
        commits_data = self.safe_get_json(commits_url)

        # Set forks to 0 if we couldn't fetch the repo data
        forks = repo_data.get("forks_count") if repo_data else None

        # Try to get last commit date
        latest_commit_date = None
        last_commit_days = None
        if isinstance(commits_data, list) and commits_data:
            try:
                latest_commit_date = commits_data[0]["commit"]["committer"]["date"]
                last_commit_days = (datetime.now(timezone.utc) - datetime.strptime(
                    latest_commit_date, "%Y-%m-%dT%H:%M:%SZ")).days
            except (KeyError, ValueError, TypeError):
                last_commit_days = None 

        # Assign scores — use 0 if data is missing
        star_score = self.calculate_score(stars, STAR_THRESHOLDS, STAR_SCORES)
        fork_score = self.calculate_score(forks or 0, FORK_THRESHOLDS, FORK_SCORES)
        downloads_score = self.calculate_score(downloads, DOWNLOAD_THRESHOLDS, DOWNLOAD_SCORES)
        commit_score = self.calculate_score(last_commit_days or 9999, COMMIT_THRESHOLDS, COMMIT_SCORES)

        # Weighted score
        popularity_score = round(
            (0.15 * star_score) +
            (0.15 * fork_score) +
            (0.10 * commit_score) +
            (0.10 * downloads_score), 2
        )

        return {
            "github_stars": stars,
            "download_count": downloads,
            "forks": forks,  # could be 99999
            "last_commit_days": last_commit_days,  # could be 0
            "star_score": star_score,
            "fork_score": fork_score,
            "commit_score": commit_score,
            "popularity_score": popularity_score
        }

    #adjust weights here 
    def analyze_servers(self):
        """Analyze all MCP servers with progress bar, save to CSV, and keep results in memory."""
        output_file = "analysis_results.csv"
        fieldnames = [
            "owner", "repo", "popularity_score", "risk_score",
            "doc_quality_score", "doc_quality_adjustment", "adjusted_risk_score",
            "risk_category", "description"
        ]

        self.results = []
        existing_rows = set()

        # Track processed servers
        if os.path.exists(output_file):
            with open(output_file, mode="r", newline='', encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_rows.add((row["owner"], row["repo"]))

        servers_to_analyze = [
            s for s in self.servers
            if (s.get('owner', '').strip(), s.get('repo', '').strip()) not in existing_rows
        ]

        try:
            with open(output_file, mode="a", newline='', encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)

                if os.path.getsize(output_file) == 0:
                    writer.writeheader()

                for server in tqdm(servers_to_analyze, desc="Analyzing servers", unit="server"):
                    owner = server.get('owner', '').strip()
                    repo = server.get('repo', '').strip()

                    stars = int(float(server.get('github_stars', 0) or 0))
                    download_count = server.get('download_count', '0') or '0'
                    downloads = int(download_count) if download_count.isdigit() else 0
                    description = server.get('experimental_ai_generated_description', '')

                    analysis = self.check_github_repo(owner, repo)

                    popularity_data = self.calculate_popularity_score(owner, repo, stars, downloads)
                    if popularity_data:
                        analysis.update(popularity_data)
                    analysis['github_stars'] = stars
                    analysis['download_count'] = downloads
                    popularity_score = popularity_data['popularity_score']
                    risk_score = analysis['risk_score']

                    if analysis["oauth_implementation"]:
                        risk_score -= 5
                    if analysis["direct_api_tokens"]:
                        risk_score += 5

                    doc_quality = analysis.get('doc_quality_score', 0)
                    doc_quality_factor = doc_quality / 10.0
                    doc_quality_adjustment = -3 * doc_quality_factor
                    risk_score += doc_quality_adjustment

                    adjusted_risk = popularity_score - abs(risk_score / 10)

                    if adjusted_risk > 0.2:
                        risk_category = "MINIMAL"
                    elif adjusted_risk > 0:
                        risk_category = "LOW"
                    elif adjusted_risk > -0.5:
                        risk_category = "MEDIUM"
                    else:
                        risk_category = "HIGH"

                    analysis['adjusted_risk_score'] = adjusted_risk
                    analysis['risk_category'] = risk_category
                    analysis['doc_quality_adjustment'] = doc_quality_adjustment
                    analysis['description'] = description

                    self.results.append(analysis)

                    row = {
                        "owner": owner,
                        "repo": repo,
                        "popularity_score": popularity_score,
                        "risk_score": analysis['risk_score'],
                        "doc_quality_score": doc_quality,
                        "doc_quality_adjustment": doc_quality_adjustment,
                        "adjusted_risk_score": adjusted_risk,
                        "risk_category": risk_category,
                        "description": description
                    }
                    writer.writerow(row)
                    f.flush()
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nInterrupted by user. Progress saved.")
        except Exception as e:
            print(f"\nError: {e}. Progress saved.")
        return self.results


    def generate_report(self, output_file="mcp_report.json"):
        """Generate a JSON report of the analysis results"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
            
        print(f"Report saved to {output_file}")
        
        # Also print a summary to console
        self._print_summary()
        
    def _print_summary(self):
        """Print a summary of the analysis results"""
        print("\n" + "="*80)
        print("MCP SERVER SECURITY ANALYSIS SUMMARY")
        print("="*80)
        
        risk_categories = defaultdict(list)
        for result in self.results:
            risk_categories[result.get('risk_category', 'UNKNOWN')].append(
                (result.get('owner', ''), result.get('repo', ''))
            )
        
        print(f"\nTotal servers analyzed: {len(self.results)}")
        
        # Print documentation quality summary
        total_doc_score = sum(r.get('doc_quality_score', 0) for r in self.results)
        avg_doc_score = total_doc_score / max(1, len(self.results))
        print(f"\nAverage documentation quality score: {avg_doc_score:.2f}/10")
        
        doc_quality_counts = {
            'Excellent (8-10)': sum(1 for r in self.results if r.get('doc_quality_score', 0) >= 8),
            'Good (6-7.9)': sum(1 for r in self.results if 6 <= r.get('doc_quality_score', 0) < 8),
            'Average (4-5.9)': sum(1 for r in self.results if 4 <= r.get('doc_quality_score', 0) < 6),
            'Poor (2-3.9)': sum(1 for r in self.results if 2 <= r.get('doc_quality_score', 0) < 4),
            'Very Poor (0-1.9)': sum(1 for r in self.results if r.get('doc_quality_score', 0) < 2)
        }
        
        print("Documentation quality breakdown:")
        for quality, count in doc_quality_counts.items():
            print(f"  - {quality}: {count} servers")
        
        for category in ['HIGH', 'MEDIUM', 'LOW', 'MINIMAL']:
            servers = risk_categories.get(category, [])
            print(f"\n{category} RISK: {len(servers)} servers")
            for owner, repo in servers:
                print(f"  - {owner}/{repo}")
                
        print("\nDetailed results are available in the JSON report.")
        print("="*80)

    def get_high_risk_servers(self):
        """Return list of high risk servers"""
        return [r for r in self.results if r.get('risk_category') == 'HIGH']
        
    def generate_csv_report(self, output_file="mcp_analysis.csv"):
        """Generate an updated CSV with the analysis results including documentation quality"""
        # Define the fields for the new CSV
        fieldnames = [
            # Base repository info
            'owner', 'repo', 'github_stars', 'download_count', 
            'experimental_ai_generated_description', 
            # Risk assessment
            'risk_category', 'adjusted_risk_score', 'popularity_score', 
            # Documentation quality metrics
            'doc_quality_score', 'doc_quality_adjustment', 'readme_found',
            'has_installation_docs', 'has_usage_docs', 'has_api_reference', 
            'has_security_docs', 'has_error_handling', 'has_cli_docs',
            'has_code_examples', 'has_screenshots', 'has_license_info',
            'readme_size_bytes', 'headings_count', 'code_blocks_count',
            'readability_score', 'missing_sections',
            # Security indicators
            'oauth_implementation', 'direct_api_tokens', 
            'main_risk_factors', 'outbound_connections'
        ]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                # Get doc quality details
                doc_details = result.get('doc_quality_details', {})
                
                # Create base row with repository info
                row = {
                    'owner': result.get('owner', ''),
                    'repo': result.get('repo', ''),
                    'github_stars': result.get('github_stars', 0),
                    'download_count': result.get('download_count', 0),
                    'experimental_ai_generated_description': result.get('description', ''),
                    'risk_category': result.get('risk_category', 'UNKNOWN'),
                    'adjusted_risk_score': round(result.get('adjusted_risk_score', 0), 2),
                    'popularity_score': round(result.get('popularity_score', 0), 2),
                    'doc_quality_score': result.get('doc_quality_score', 0),
                    'doc_quality_adjustment': round(result.get('doc_quality_adjustment', 0), 2),
                    'oauth_implementation': 'Yes' if result.get('oauth_implementation', False) else 'No',
                    'direct_api_tokens': 'Yes' if result.get('direct_api_tokens', False) else 'No',
                    'main_risk_factors': '; '.join(result.get('risk_factors', [])[:5]),  # Top 5 risk factors
                    'outbound_connections': '; '.join(result.get('outbound_connections', [])[:5]),  # Top 5 connections
                }
                
                # Add detailed documentation metrics
                row.update({
                    'readme_found': 'Yes' if doc_details.get('readme_found', False) else 'No',
                    'has_installation_docs': 'Yes' if 'installation' in doc_details.get('sections_found', []) else 'No',
                    'has_usage_docs': 'Yes' if 'usage' in doc_details.get('sections_found', []) else 'No',
                    'has_api_reference': 'Yes' if 'api_reference' in doc_details.get('sections_found', []) else 'No',
                    'has_security_docs': 'Yes' if doc_details.get('has_security_info', False) else 'No',
                    'has_error_handling': 'Yes' if doc_details.get('has_error_handling', False) else 'No',
                    'has_cli_docs': 'Yes' if doc_details.get('has_cli_docs', False) else 'No',
                    'has_code_examples': 'Yes' if doc_details.get('has_code_examples', False) else 'No',
                    'has_screenshots': 'Yes' if doc_details.get('has_screenshots', False) else 'No',
                    'has_license_info': 'Yes' if doc_details.get('has_license_info', False) else 'No',
                    'readme_size_bytes': doc_details.get('readme_size_bytes', 0),
                    'headings_count': doc_details.get('headings_count', 0),
                    'code_blocks_count': doc_details.get('code_blocks_count', 0),
                    'readability_score': doc_details.get('readability_score', 0),
                    'missing_sections': '; '.join(doc_details.get('missing_sections', []))
                })
                
                writer.writerow(row)
                
        print(f"CSV report saved to {output_file}")
        return output_file
    
    def generate_doc_quality_report(self, output_file="mcp_documentation_quality.csv"):
        """Generate a detailed report on documentation quality"""
        fieldnames = [
            'owner', 'repo', 'doc_quality_score', 'readme_found', 'readme_size_bytes',
            'sections_found', 'missing_sections', 'has_code_examples', 'has_screenshots',
            'has_security_info', 'has_license_info', 'readability_score'
        ]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                doc_details = result.get('doc_quality_details', {})
                
                row = {
                    'owner': result.get('owner', ''),
                    'repo': result.get('repo', ''),
                    'doc_quality_score': result.get('doc_quality_score', 0),
                    'readme_found': 'Yes' if doc_details.get('readme_found', False) else 'No',
                    'readme_size_bytes': doc_details.get('readme_size_bytes', 0),
                    'sections_found': '; '.join(doc_details.get('sections_found', [])),
                    'missing_sections': '; '.join(doc_details.get('missing_sections', [])),
                    'has_code_examples': 'Yes' if doc_details.get('has_code_examples', False) else 'No',
                    'has_screenshots': 'Yes' if doc_details.get('has_screenshots', False) else 'No',
                    'has_security_info': 'Yes' if doc_details.get('has_security_info', False) else 'No',
                    'has_license_info': 'Yes' if doc_details.get('has_license_info', False) else 'No',
                    'readability_score': doc_details.get('readability_score', 0)
                }
                writer.writerow(row)
                
        print(f"Documentation quality report saved to {output_file}")
        return output_file

if __name__ == "__main__":
    #csv_file = os.path.join(os.getcwd(), "data", "sorted_data.csv")
    csv_file = os.path.join(os.getcwd(), "sorted_data_by_stars.csv")
    
    analyzer = MCPAnalyzer(csv_file)
    if analyzer.load_servers():
        analyzer.analyze_servers()
        # Generate both JSON and CSV reports
        analyzer.generate_report()
        analyzer._print_summary()
        csv_output = analyzer.generate_csv_report()
        #doc_output = analyzer.generate_doc_quality_report()
        print(f"\nUpdated CSV file with security analysis: {csv_output}")
        #print(f"Documentation quality report: {doc_output}")

