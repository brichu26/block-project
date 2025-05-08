# MCP Server Evaluation System

A comprehensive evaluation framework for Model Context Protocol (MCP) servers. This system analyzes GitHub repositories based on documentation quality, security practices, and popularity metrics.

[![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Table of Contents

- [Overview](#overview)
- [Evaluation Metrics](#evaluation-metrics)
  - [Documentation Quality](#documentation-quality-assessment)
  - [Security Practices](#security-practices-assessment)
  - [Popularity & Community Support](#popularity--community-support)
  - [Performance (Optional)](#performance-evaluation-optional)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## Overview

This toolset provides quantitative evaluation of Model Context Protocol (MCP) server implementations based on their GitHub repositories. The evaluation system applies a standardized scorecard to assess key aspects of each implementation's quality, security, and community adoption.

Each server is evaluated across three primary domains:
1. **Documentation Quality** - Measures comprehensiveness, organization, and readability of documentation
2. **Security Practices** - Evaluates authentication methods, configuration security, and network behavior
3. **Popularity & Community Support** - Assesses GitHub stars, forks, download counts, and contribution activity

The results are consolidated into detailed scorecards and exportable to CSV and JSON formats.

## Evaluation Metrics

### Documentation Quality Assessment

Documentation quality is evaluated based on 7 key components (weighted):

| Component | Weight | Description |
|-----------|--------|-------------|
| Base Section Score | 30% | Points awarded for each documentation section found (max 15 points) |
| Essential Sections Bonus | 20% | Extra points for including key sections (installation, usage, configuration, API reference, error handling, license, project info) |
| Readability Score | 15% | Based on average sentence length and clarity |
| Size Bonus | 10% | Based on total README size |
| Heading Structure Bonus | 10% | Evaluates number of headings, hierarchy depth, and section sizing |
| Code Examples Bonus | 10% | Based on the number of code blocks |
| Completeness Bonus | 5% | Reflects the percentage of total possible sections covered |

Final documentation scores range from 0-10 with ratings:
- **Excellent (8-10)**: Comprehensive, well-structured documentation with abundant examples
- **Good (6-7.9)**: Solid documentation covering most essential topics
- **Average (4-5.9)**: Basic documentation with some gaps
- **Poor (2-3.9)**: Minimal documentation with significant omissions
- **Very Poor (0-1.9)**: Severely lacking or absent documentation

### Security Practices Assessment

Security assessment focuses on identifying potential risks in the codebase:

| Metric | Weight | Description |
|--------|--------|-------------|
| Authentication Methods | 40% | Implementation of OAuth versus direct API tokens |
| Network Behavior | 30% | Analysis of outbound connections and data transmission |
| Configuration Security | 30% | Detection of suspicious patterns in configuration files |

Security risk levels are classified as:
- **MINIMAL Risk**: Score > 0.75
- **LOW Risk**: Score > 0.50
- **MEDIUM Risk**: Score > 0.25
- **HIGH Risk**: Score ≤ 0.25

### Popularity & Community Support

Popularity metrics provide insight into community validation using z-score statistics from 4000+ repositories:

| Metric | Weight | Description |
|--------|--------|-------------|
| GitHub Stars | 15% | Total repository stars with z-score based scoring |
| Fork Count | 15% | Number of repository forks with z-score based scoring |
| Recent Commit Activity | 10% | Frequency of contributions (days since last commit) |
| Download Count | 10% | Installation/download metrics when available |

### Performance Evaluation (Optional)

While not part of the core scorecard, the system also includes performance evaluation tools:

| Metric | Description |
|--------|-------------|
| Latency | Response time in milliseconds |
| Throughput | Request handling capacity (requests per second) |
| Error Rate | Percentage of failed requests |

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mcp-server-evaluation.git
cd mcp-server-evaluation

# Install dependencies
pip install -r requirements.txt

# Set up GitHub API token (optional but recommended to avoid rate limits)
export GITHUB_TOKEN=your_github_token
```

## Usage

### Basic Usage

```bash
# Evaluate repositories listed in the MCP Community README
python mcp/mcp_scoring.py

# Evaluate specific repositories
python mcp/mcp_scoring.py --repos username/repo1 username/repo2

# Generate detailed scorecards
python mcp/mcp_scoring.py --verbose

# Custom output filenames
python mcp/mcp_scoring.py --output-csv results.csv --output-json results.json
```

### Evaluating a Single Component

```bash
# Evaluate just documentation
python mcp/mcp_documentation_eval.py

# Evaluate just security
python mcp/mcp_security_eval.py

# Evaluate just popularity
python mcp/mcp_popularity_eval.py

# Evaluate performance (requires running servers)
python mcp/mcp_performance_eval.py
```

## Examples

### Example Scorecard

```
================================================================================
MCP SERVER SCORECARD: modelcontextprotocol/servers
================================================================================

OVERALL SCORE: 8.45/10

1. DOCUMENTATION QUALITY: 9.2/10 - Excellent
----------------------------------------
- Base Section Score: 28.0 (30% weight)
- Essential Sections Bonus: 17.14 (20% weight)
- Readability Score: 12.0 (15% weight)
- Size Bonus: 8.5 (10% weight)
- Heading Structure Bonus: 8.0 (10% weight)
- Code Examples Bonus: 8.33 (10% weight)
- Completeness Bonus: 4.67 (5% weight)

2. SECURITY PRACTICES: 0.78 - Risk Level: MINIMAL
----------------------------------------
- Authentication Score: 0.3 (40% weight)
- Configuration Security Score: 0.24 (30% weight)
- Network Security Score: 0.24 (30% weight)
- OAuth Implementation: Yes
- Direct API Tokens: Yes

3. POPULARITY & COMMUNITY SUPPORT: 7.8/10
----------------------------------------
- GitHub Stars: 286 (z-score: -0.01)
- Forks: 42 (z-score: -0.05)
- Days Since Last Commit: 3 (z-score: -0.88)
- Download Count: 2154 (z-score: 5.83)

Stars Score: 0.75 (15% weight)
Forks Score: 0.75 (15% weight)
Recent Activity Score: 0.8 (10% weight)
Download Score: 0.6 (10% weight)

================================================================================
```

## Project Structure

```
mcp/
├── mcp_documentation_eval.py   # Documentation quality evaluation
├── mcp_security_eval.py        # Security practices assessment
├── mcp_popularity_eval.py      # Popularity and community support evaluation
├── mcp_performance_eval.py     # Performance evaluation (optional)
└── mcp_scoring.py              # Main scoring system integrating all components
```

## Requirements

- Python 3.8+
- Dependencies:
  - requests
  - beautifulsoup4
  - pandas
  - numpy

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Model Context Protocol (MCP) community
- All MCP server maintainers who have contributed to the ecosystem

