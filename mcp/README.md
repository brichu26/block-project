# MCP (Model Context Protocol) Server Evaluation

A toolkit for evaluating and scoring MCP servers based on multiple criteria:

- Documentation quality
- Security practices
- Popularity & community support
- Performance metrics

## Components

- **mcp_scoring.py**: Central scoring system that combines all evaluation criteria
- **mcp_documentation_eval.py**: Evaluates documentation quality and completeness
- **mcp_security_eval.py**: Assesses security practices and risk levels
- **mcp_popularity_eval.py**: Measures community adoption and support
- **mcp_performance_eval.py**: Evaluates server performance characteristics

## Requirements

```
requests>=2.28.0
pandas>=1.4.0
numpy>=1.22.0
beautifulsoup4>=4.11.0
matplotlib>=3.5.0
python-dotenv>=0.20.0
```

## Usage

To evaluate MCP servers:

```python
from mcp_scoring import MCPServerScorer

# Initialize the scorer
scorer = MCPServerScorer()

# Evaluate a specific server
results = scorer.score_server("owner", "repo")

# Generate a human-readable scorecard
scorecard = scorer.generate_scorecard(results)
print(scorecard)

# Export results to CSV or JSON
from mcp_scoring import export_results_to_csv, export_results_to_json
export_results_to_csv([results])
export_results_to_json([results])
```

## Environment Variables

- `GITHUB_TOKEN`: GitHub API token (recommended to avoid rate limiting)

## Output

The evaluation generates detailed scorecards with metrics across all evaluation categories and provides an overall score for each MCP server. 