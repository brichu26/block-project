import os
import re 
import requests
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin
import random
import base64

#working on extracting api endpoints from documentation files to check if they are using JSON-RPC 2.0
def extract_endpoints_from_docs(self, directory: str):
        """Extract endpoint information from documentation files"""
        doc_files = ["README.md", "README.rst", "docs/index.md", "docs/api.md", "API.md"]
        
        for doc_file in doc_files:
            file_path = os.path.join(directory, doc_file)
            if os.path.exists(file_path):
                self.log(f"Examining documentation: {doc_file}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Look for endpoint patterns in documentation
                    endpoint_patterns = [
                        r'`(\/[a-zA-Z0-9_\/-]+)`',  # Markdown code with path
                        r'"(\/[a-zA-Z0-9_\/-]+)"',  # Quoted paths
                        r'endpoint[:\s]+"([^"]+)"',  # endpoint: "path"
                        r'endpoint[:\s]+\'([^\']+)\'',  # endpoint: 'path'
                        r'@app\.[a-z]+\([\'"]([^\'"]+)[\'"]',  # Flask/FastAPI decorators
                    ]
                    
                    for pattern in endpoint_patterns:
                        for match in re.finditer(pattern, content):
                            endpoint = match.group(1)
                            if endpoint.startswith('/'):
                                self.endpoints.add(endpoint)



def test_oauth_implementation(server_url, client_id=None, client_secret=None):
    """
    Test if an MCP server implements OAuth 2.1 with appropriate security measures.
    
    Args:
        server_url: Base URL of the MCP server
        client_id: Optional client ID for testing authenticated endpoints
        client_secret: Optional client secret for confidential client flow
        
    Returns:
        Dictionary with test results and compliance status
    """
    results = {
        "server_url": server_url,
        "oauth_implemented": False,
        "oauth_version": None,
        "security_measures": {
            "uses_pkce": False,
            "supports_confidential_clients": False,
            "supports_public_clients": False,
            "uses_https": False,
            "proper_token_handling": False,
            "secure_redirect_handling": False
        },
        "endpoints": {
            "authorization": None,
            "token": None,
            "revocation": None,
            "introspection": None,
            "userinfo": None
        },
        "compliance_issues": []
    }
    
    # 1. Check for HTTPS
    results["security_measures"]["uses_https"] = server_url.startswith("https://")
    if not results["security_measures"]["uses_https"]:
        results["compliance_issues"].append("Server does not use HTTPS")
    
    # 2. Discover OAuth endpoints (try standard discovery first)
    discovery_url = f"{server_url}/.well-known/oauth-authorization-server"
    oauth_config = None
    
    try:
        response = requests.get(discovery_url, timeout=10)
        if response.status_code == 200:
            oauth_config = response.json()
            results["oauth_implemented"] = True
            
            # Extract endpoints
            for endpoint_type in ["authorization_endpoint", "token_endpoint", 
                                 "revocation_endpoint", "introspection_endpoint"]:
                if endpoint_type in oauth_config:
                    key = endpoint_type.split("_")[0]
                    results["endpoints"][key] = oauth_config[endpoint_type]
    except:
        # Try alternative discovery methods
        pass
    
    # If discovery failed, check common endpoint locations
    if not oauth_config:
        common_paths = [
            "/oauth/authorize", "/oauth2/authorize", "/oauth/auth", 
            "/oauth/token", "/oauth2/token"
        ]
        
        for path in common_paths:
            try:
                response = requests.get(f"{server_url}{path}", timeout=5)
                # If responds with 401/403/302, might be OAuth endpoint
                if response.status_code in [401, 403, 302]:
                    path_type = "authorization" if "auth" in path else "token"
                    results["endpoints"][path_type] = f"{server_url}{path}"
                    results["oauth_implemented"] = True
            except:
                pass
    
    # 3. Determine OAuth version
    if results["oauth_implemented"]:
        # Check for OAuth 2.1 specific features in response headers or docs
        oauth_version = "2.0"  # Default assumption
        
        # Look for OAuth 2.1 specific requirements
        has_pkce_requirement = False
        
        # If we have a token endpoint, check if it requires PKCE
        token_endpoint = results["endpoints"].get("token")
        if token_endpoint and client_id:
            try:
                # Try token request without PKCE (should fail in 2.1)
                minimal_data = {
                    "grant_type": "authorization_code",
                    "client_id": client_id,
                    "code": "dummy_code",
                    "redirect_uri": "https://example.com/callback"
                }
                response = requests.post(token_endpoint, data=minimal_data)
                
                # In OAuth 2.1, this should fail without PKCE for public clients
                if "code_verifier" in response.text or "PKCE" in response.text:
                    has_pkce_requirement = True
            except:
                pass
        
        if has_pkce_requirement:
            oauth_version = "2.1"
            results["security_measures"]["uses_pkce"] = True
        
        results["oauth_version"] = oauth_version
    
    # 4. Test PKCE support explicitly
    if results["oauth_implemented"] and results["endpoints"]["authorization"]:
        auth_endpoint = results["endpoints"]["authorization"]
        
        try:
            # Generate PKCE parameters
            code_verifier = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(64))
            code_challenge = base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest()).decode().rstrip('=')
            
            # Try auth request with PKCE
            params = {
                "client_id": client_id or "test_client",
                "response_type": "code",
                "redirect_uri": "https://example.com/callback",
                "code_challenge": code_challenge,
                "code_challenge_method": "S256"
            }
            
            response = requests.get(auth_endpoint, params=params, allow_redirects=False)
            
            # If redirects or provides an auth page, PKCE is likely supported
            if response.status_code in [200, 302]:
                results["security_measures"]["uses_pkce"] = True
        except:
            pass
    
    # 5. Test confidential client support (if credentials provided)
    if results["oauth_implemented"] and client_id and client_secret and results["endpoints"]["token"]:
        token_endpoint = results["endpoints"]["token"]
        
        try:
            # Test client credentials grant (only for confidential clients)
            auth = (client_id, client_secret)
            data = {"grant_type": "client_credentials"}
            
            response = requests.post(token_endpoint, auth=auth, data=data)
            
            if response.status_code == 200 and "access_token" in response.json():
                results["security_measures"]["supports_confidential_clients"] = True
        except:
            pass
    
    # 6. Test public client support
    if results["oauth_implemented"] and results["endpoints"]["authorization"]:
        # Public clients use authorization code flow with PKCE
        results["security_measures"]["supports_public_clients"] = results["security_measures"]["uses_pkce"]
    
    # 7. Check redirect handling security
    if results["oauth_implemented"] and results["endpoints"]["authorization"] and client_id:
        auth_endpoint = results["endpoints"]["authorization"]
        
        try:
            # Test with invalid redirect URI
            params = {
                "client_id": client_id,
                "response_type": "code",
                "redirect_uri": "https://malicious-site.com/steal"
            }
            
            response = requests.get(auth_endpoint, params=params, allow_redirects=False)
            
            # Should not redirect to the malicious URI (should error instead)
            if response.status_code != 302 or "malicious-site.com" not in response.headers.get("Location", ""):
                results["security_measures"]["secure_redirect_handling"] = True
        except:
            pass
    
    # 8. Evaluate overall OAuth 2.1 compliance
    if results["oauth_implemented"]:
        # Core OAuth 2.1 requirements
        oauth21_requirements = [
            ("uses_https", "OAuth 2.1 requires HTTPS"),
            ("uses_pkce", "OAuth 2.1 requires PKCE for authorization code flow"), 
            ("secure_redirect_handling", "OAuth 2.1 requires strict redirect URI validation")
        ]
        
        for req, message in oauth21_requirements:
            if not results["security_measures"][req]:
                results["compliance_issues"].append(message)
        
        # Either confidential or public clients must be supported
        if not (results["security_measures"]["supports_confidential_clients"] or 
                results["security_measures"]["supports_public_clients"]):
            results["compliance_issues"].append(
                "OAuth 2.1 server must support either confidential or public clients"
            )
    return results
