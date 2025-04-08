import requests
import time
import statistics

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

def measure_latency_and_throughput(url, num_requests=10, concurrency=5):
    """Measure latency (response time) and throughput (requests per second) for an MCP server."""
    latencies = []
    start_time = time.time()

    for _ in range(num_requests):
        try:
            request_start = time.time()
            response = requests.get(url, timeout=5)  # Send request with 5-second timeout
            request_end = time.time()

            if response.status_code == 200:
                latencies.append((request_end - request_start) * 1000)  # Convert to ms
            else:
                print(f"Warning: {url} returned status code {response.status_code}")

        except requests.exceptions.RequestException as e:
            print(f"Error: Failed to connect to {url} - {e}")

    end_time = time.time()
    total_duration = end_time - start_time

    # Calculate average latency and throughput
    avg_latency = statistics.mean(latencies) if latencies else float('inf')
    throughput = num_requests / total_duration if total_duration > 0 else 0

    return avg_latency, throughput

def calculate_score(value, thresholds, scores):
    """Assign a score based on predefined thresholds."""
    for i, threshold in enumerate(thresholds):
        if value >= threshold:
            return scores[i]
    return 0

def evaluate_mcp_server_performance():
    """Evaluate performance (latency & throughput) for multiple MCP servers."""
    for server in mcp_servers:
        print(f"Testing {server['name']} ({server['url']})...")
        latency, throughput = measure_latency_and_throughput(server['url'])

        # Assign scores
        latency_score = calculate_score(latency, LATENCY_THRESHOLDS, LATENCY_SCORES)
        throughput_score = calculate_score(throughput, THROUGHPUT_THRESHOLDS, THROUGHPUT_SCORES)

        # Compute final performance score (weighted sum)
        performance_score = (0.10 * latency_score) + (0.10 * throughput_score)

        print(f"Results for {server['name']}:")
        print(f"- Avg Latency: {latency:.2f} ms")
        print(f"- Throughput: {throughput:.2f} requests/sec")
        print(f"- Latency Score: {latency_score}/10")
        print(f"- Throughput Score: {throughput_score}/10")
        print(f"- Total Performance Score: {performance_score}/20")
        print("-" * 50)

# Run the performance evaluation
evaluate_mcp_server_performance()
