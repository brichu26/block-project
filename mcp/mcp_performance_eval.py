import requests
import time
import statistics
from typing import Dict, Any, List, Optional, Tuple

# List of MCP servers to test
mcp_servers = [
    {"name": "MCP Servers", "url": "https://mcp.example.com/api"},  # Replace with actual URL
    {"name": "Python SDK Server", "url": "https://python-sdk.example.com/api"},
    {"name": "TypeScript SDK Server", "url": "https://typescript-sdk.example.com/api"},
    {"name": "Kotlin SDK Server", "url": "https://kotlin-sdk.example.com/api"},
    {"name": "Java SDK Server", "url": "https://java-sdk.example.com/api"}
]

# Performance scoring thresholds
LATENCY_THRESHOLDS = [50, 200, 500, 1000, float('inf')]  # Latency in milliseconds
THROUGHPUT_THRESHOLDS = [10000, 5000, 1000, 500, 0]  # Requests per second

LATENCY_SCORES = [10, 8, 6, 4, 2]
THROUGHPUT_SCORES = [10, 8, 6, 4, 2]

class PerformanceEvaluator:
    """
    Evaluates the performance characteristics of an MCP server.
    Note: This is an optional evaluation component not included in the core scorecard.
    """
    
    # Performance thresholds
    LATENCY_THRESHOLDS = {
        "excellent": 50,    # ms
        "good": 200,        # ms
        "average": 500,     # ms
        "poor": 1000        # ms
    }
    
    THROUGHPUT_THRESHOLDS = {
        "excellent": 100,   # req/sec
        "good": 50,         # req/sec
        "average": 20,      # req/sec
        "poor": 5           # req/sec
    }
    
    def __init__(self, timeout: int = 5, num_requests: int = 10, concurrency: int = 5):
        """Initialize the performance evaluator with configuration parameters."""
        self.timeout = timeout
        self.num_requests = num_requests
        self.concurrency = concurrency
    
    def test_endpoint(self, url: str, method: str = "GET", headers: Dict[str, str] = None, data: Any = None) -> Dict[str, Any]:
        """
        Test the performance of a specific API endpoint.
        
        Args:
            url: The endpoint URL to test
            method: HTTP method (GET, POST, etc.)
            headers: Request headers
            data: Request data for POST/PUT
            
        Returns:
            Dictionary with performance metrics
        """
        if headers is None:
            headers = {}
        
        latencies = []
        errors = 0
        start_time = time.time()
        
        for _ in range(self.num_requests):
            try:
                request_start = time.time()
                
                if method.upper() == "GET":
                    response = requests.get(url, headers=headers, timeout=self.timeout)
                elif method.upper() == "POST":
                    response = requests.post(url, headers=headers, json=data, timeout=self.timeout)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                request_end = time.time()
                
                if response.status_code == 200:
                    latencies.append((request_end - request_start) * 1000)  # Convert to ms
                else:
                    print(f"Warning: {url} returned status code {response.status_code}")
                    errors += 1
            
            except requests.exceptions.RequestException as e:
                print(f"Error: Failed to connect to {url} - {e}")
                errors += 1
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Calculate metrics
        if not latencies:
            return {
                "success": False,
                "error_rate": 1.0,
                "latency": {
                    "avg": float('inf'),
                    "p50": float('inf'),
                    "p95": float('inf'),
                    "min": float('inf'),
                    "max": float('inf')
                },
                "throughput": 0,
                "errors": errors,
                "requests": self.num_requests
            }
        
        # Calculate latency statistics
        avg_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        # Calculate p95 (95th percentile) if we have enough samples
        if len(latencies) >= 5:
            sorted_latencies = sorted(latencies)
            p95_index = int(len(sorted_latencies) * 0.95)
            p95_latency = sorted_latencies[p95_index]
        else:
            p95_latency = max_latency
        
        # Calculate throughput (successful requests per second)
        successful_requests = len(latencies)
        throughput = successful_requests / total_duration if total_duration > 0 else 0
        error_rate = errors / self.num_requests if self.num_requests > 0 else 1.0
        
        return {
            "success": True,
            "error_rate": error_rate,
            "latency": {
                "avg": avg_latency,
                "p50": median_latency,
                "p95": p95_latency,
                "min": min_latency,
                "max": max_latency
            },
            "throughput": throughput,
            "errors": errors,
            "requests": self.num_requests
        }
    
    def evaluate(self, base_url: str, endpoints: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate the performance of an MCP server.
        
        Args:
            base_url: The base URL of the MCP server
            endpoints: List of endpoints to test (default: ["/"])
            
        Returns:
            Dictionary with performance evaluation results
        """
        if endpoints is None:
            endpoints = ["/"]
        
        results = {}
        
        for endpoint in endpoints:
            url = f"{base_url.rstrip('/')}{endpoint}"
            print(f"Testing endpoint: {url}")
            
            result = self.test_endpoint(url)
            results[endpoint] = result
        
        # Calculate average scores
        avg_latency = 0
        avg_throughput = 0
        total_endpoints = len(results)
        successful_endpoints = 0
        
        for endpoint, result in results.items():
            if result["success"]:
                avg_latency += result["latency"]["avg"]
                avg_throughput += result["throughput"]
                successful_endpoints += 1
        
        if successful_endpoints > 0:
            avg_latency /= successful_endpoints
            avg_throughput /= successful_endpoints
        else:
            avg_latency = float('inf')
            avg_throughput = 0
        
        # Score latency (lower is better)
        if avg_latency <= self.LATENCY_THRESHOLDS["excellent"]:
            latency_score = 10
        elif avg_latency <= self.LATENCY_THRESHOLDS["good"]:
            latency_score = 8
        elif avg_latency <= self.LATENCY_THRESHOLDS["average"]:
            latency_score = 6
        elif avg_latency <= self.LATENCY_THRESHOLDS["poor"]:
            latency_score = 4
        else:
            latency_score = 2
        
        # Score throughput (higher is better)
        if avg_throughput >= self.THROUGHPUT_THRESHOLDS["excellent"]:
            throughput_score = 10
        elif avg_throughput >= self.THROUGHPUT_THRESHOLDS["good"]:
            throughput_score = 8
        elif avg_throughput >= self.THROUGHPUT_THRESHOLDS["average"]:
            throughput_score = 6
        elif avg_throughput >= self.THROUGHPUT_THRESHOLDS["poor"]:
            throughput_score = 4
        else:
            throughput_score = 2
        
        # Calculate overall performance score
        overall_score = (latency_score * 0.5) + (throughput_score * 0.5)
        
        # Determine rating
        if overall_score >= 9:
            rating = "Excellent"
        elif overall_score >= 7:
            rating = "Good"
        elif overall_score >= 5:
            rating = "Average"
        elif overall_score >= 3:
            rating = "Poor"
        else:
            rating = "Very Poor"
        
        return {
            "score": round(overall_score, 2),
            "rating": rating,
            "details": {
                "latency_score": latency_score,
                "throughput_score": throughput_score,
                "avg_latency": round(avg_latency, 2),
                "avg_throughput": round(avg_throughput, 2),
                "endpoints": results,
                "successful_endpoints": successful_endpoints,
                "total_endpoints": total_endpoints
            }
        }

def test_performance_evaluation(server_url: str, endpoints: List[str] = None):
    """Test the performance evaluator with a single server."""
    if endpoints is None:
        endpoints = ["/"]
    
    evaluator = PerformanceEvaluator(num_requests=5)  # Reduced for testing
    result = evaluator.evaluate(server_url, endpoints)
    
    print(f"Performance Evaluation for {server_url}:")
    print(f"Overall Score: {result['score']}/10 - {result['rating']}")
    print(f"Average Latency: {result['details']['avg_latency']} ms (Score: {result['details']['latency_score']})")
    print(f"Average Throughput: {result['details']['avg_throughput']} req/sec (Score: {result['details']['throughput_score']})")
    print(f"Successful Endpoints: {result['details']['successful_endpoints']}/{result['details']['total_endpoints']}")
    
    print("\nEndpoint Details:")
    for endpoint, data in result['details']['endpoints'].items():
        if data["success"]:
            print(f"- {endpoint}: Avg Latency: {data['latency']['avg']:.2f} ms, Throughput: {data['throughput']:.2f} req/sec")
        else:
            print(f"- {endpoint}: Failed to connect")
    
    print("-" * 50)

# Example usage
if __name__ == "__main__":
    # Test with sample MCP server URLs
    test_performance_evaluation("https://example.com/api", ["/", "/health", "/models"])
