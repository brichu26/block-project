import csv
import os
import re
import requests
import time
import subprocess
import json
from urllib.parse import urlparse
from collections import defaultdict

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
            "risk_factors": []
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

    #adjust weights here 
    def analyze_servers(self):
        """Analyze all MCP servers in the list"""
        for server in self.servers:
            owner = server.get('owner', '').strip()
            repo = server.get('repo', '').strip()
            stars = int(float(server.get('github_stars', 0)))
            download_count = server.get('download_count', '0')
            downloads = int(download_count) if download_count.isdigit() else 0
            description = server.get('experimental_ai_generated_description', '')
            
            print(f"Analyzing {owner}/{repo}...")
            analysis = self.check_github_repo(owner, repo)
            
            # Add metadata
            analysis['github_stars'] = stars
            analysis['download_count'] = downloads
            analysis['description'] = description
            
            # Calculate trust score based on popularity 
            
            # Statistics for github_stars
            mean_stars = 16131.0
            std_dev_stars = 16584.53
            
            # Statistics for download_count
            mean_downloads = 182107.79
            std_dev_downloads = 220439.47
            
            # Standardize stars and downloads & calculate trust score 
            #higher is better
            standardized_stars = (stars - mean_stars) / std_dev_stars
            standardized_downloads = (downloads - mean_downloads) / std_dev_downloads
            #trust_score = (min(5, standardized_stars) * 1.5) + (min(5, standardized_downloads) * 1.5)  # Increase weight
            trust_score = (standardized_stars + standardized_downloads) / 2
            trust_score = min(15, trust_score)  # Cap at 15

            #risk assessment
            risk_score = analysis['risk_score'] #higher is worse counts "risky" words 
            if(analysis["oauth_implementation"]): #good
                risk_score -= 5
            if(analysis["direct_api_tokens"]): #bad
                risk_score += 5
                                    
            # Adjust risk score based on trust
            adjusted_risk = trust_score - abs(risk_score/10)
            
            # Risk categories
            if adjusted_risk > 0.2:
                risk_category = "MINIMAL"
            elif adjusted_risk > 0:
                risk_category = "LOW"
            elif adjusted_risk > -0.5:
                risk_category = "MEDIUM"
            else:
                risk_category = "HIGH"
                
            analysis['trust_score'] = trust_score
            analysis['adjusted_risk_score'] = adjusted_risk
            analysis['risk_category'] = risk_category
            
            self.results.append(analysis)
            
            # manual rate limiting
            time.sleep(0.1)
            
        return self.results


    def generate_report(self, output_file="mcp_security_report.json"):
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
        
    def generate_csv_report(self, output_file="mcp_security_analysis.csv"):
        """Generate an updated CSV with the analysis results"""
        # Define the fields for the new CSV
        fieldnames = [
            'owner', 'repo', 'github_stars', 'download_count', 
            'experimental_ai_generated_description', 'risk_category', 
            'adjusted_risk_score', 'trust_score', 'oauth_implementation', 
            'direct_api_tokens', 'main_risk_factors', 'outbound_connections'
        ]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                row = {
                    'owner': result.get('owner', ''),
                    'repo': result.get('repo', ''),
                    'github_stars': result.get('github_stars', 0),
                    'download_count': result.get('download_count', 0),
                    'experimental_ai_generated_description': result.get('description', ''),
                    'risk_category': result.get('risk_category', 'UNKNOWN'),
                    'adjusted_risk_score': round(result.get('adjusted_risk_score', 0), 2),
                    'trust_score': round(result.get('trust_score', 0), 2),
                    'oauth_implementation': 'Yes' if result.get('oauth_implementation', False) else 'No',
                    'direct_api_tokens': 'Yes' if result.get('direct_api_tokens', False) else 'No',
                    'main_risk_factors': '; '.join(result.get('risk_factors', [])[:5]),  # Top 5 risk factors
                    'outbound_connections': '; '.join(result.get('outbound_connections', [])[:5])  # Top 5 connections
                }
                writer.writerow(row)
                
        print(f"CSV report saved to {output_file}")
        return output_file


if __name__ == "__main__":
    csv_file = os.path.join(os.getcwd(), "pulsemcp_servers_all.csv")
    
    analyzer = MCPAnalyzer(csv_file)
    if analyzer.load_servers():
        analyzer.analyze_servers()
        
        # Generate both JSON and CSV reports
        analyzer.generate_report()
        analyzer._print_summary()
        csv_output = analyzer.generate_csv_report()
        print(f"\nUpdated CSV file with security analysis: {csv_output}")

