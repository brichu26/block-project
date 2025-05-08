import pandas as pd
from typing import List, Optional
import csv
import random

"""
This file is used to summarize the analysis results of the CSV file.
It is used to get the mean popularity score, mean documentation quality score, and the distribution of the popularity and documentation quality scores.
It also gets the distribution of the risk categories.

"""

def analyze_csv_metrics(csv_file_path):
    """
    Analyze a CSV file for popularity scores, documentation quality, and risk categories.
    
    Args:
        csv_file_path (str): Path to the CSV file
        
    Returns:
        dict: Dictionary containing all the analysis results
    """
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            if not rows:
                return {"error": "CSV file is empty"}
            
            # Calculate means
            popularity_scores = [float(row.get('popularity_score', 0)) for row in rows]
            doc_quality_scores = [float(row.get('doc_quality_score', 0)) for row in rows]
            
            mean_popularity = sum(popularity_scores) / len(popularity_scores)
            mean_doc_quality = sum(doc_quality_scores) / len(doc_quality_scores)
            
            # Popularity score distribution
            popularity_ranges = {
                'Excellent (4-5)': sum(1 for score in popularity_scores if 4 <= score <= 5),
                'Good (3-3.9)': sum(1 for score in popularity_scores if 3 <= score < 4),
                'Average (2-2.9)': sum(1 for score in popularity_scores if 2 <= score < 3),
                'Below Average (1-1.9)': sum(1 for score in popularity_scores if 1 <= score < 2),
                'Poor (0-0.9)': sum(1 for score in popularity_scores if score < 1)
            }
            
            # Documentation quality breakdown
            doc_quality_ranges = {
                'Excellent (8-10)': sum(1 for score in doc_quality_scores if 8 <= score <= 10),
                'Good (6-7.9)': sum(1 for score in doc_quality_scores if 6 <= score < 8),
                'Average (4-5.9)': sum(1 for score in doc_quality_scores if 4 <= score < 6),
                'Poor (2-3.9)': sum(1 for score in doc_quality_scores if 2 <= score < 4),
                'Very Poor (0-1.9)': sum(1 for score in doc_quality_scores if score < 2)
            }
            
            # Risk categories
            risk_categories = {
                'HIGH': sum(1 for row in rows if row.get('risk_category') == 'HIGH'),
                'MEDIUM': sum(1 for row in rows if row.get('risk_category') == 'MEDIUM'),
                'LOW': sum(1 for row in rows if row.get('risk_category') == 'LOW'),
                'MINIMAL': sum(1 for row in rows if row.get('risk_category') == 'MINIMAL')
            }
            
            # Calculate percentages
            total_rows = len(rows)
            
            popularity_percentages = {
                category: (count / total_rows * 100) 
                for category, count in popularity_ranges.items()
            }
            
            doc_quality_percentages = {
                category: (count / total_rows * 100) 
                for category, count in doc_quality_ranges.items()
            }
            
            risk_percentages = {
                category: (count / total_rows * 100) 
                for category, count in risk_categories.items()
            }
            
            return {
                "mean_popularity_score": round(mean_popularity, 2),
                "mean_doc_quality_score": round(mean_doc_quality, 2),
                "popularity_distribution": {
                    "counts": popularity_ranges,
                    "percentages": popularity_percentages
                },
                "doc_quality_distribution": {
                    "counts": doc_quality_ranges,
                    "percentages": doc_quality_percentages
                },
                "risk_distribution": {
                    "counts": risk_categories,
                    "percentages": risk_percentages
                }
            }
            
    except Exception as e:
        return {"error": f"Error analyzing CSV file: {str(e)}"}

def print_analysis_results(results):
    """
    Print the analysis results in a formatted way.
    
    Args:
        results (dict): Results from analyze_csv_metrics
    """
    if "error" in results:
        print(f"Error: {results['error']}")
        return
        
    print("\n" + "="*80)
    print("CSV ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nMean Popularity Score: {results['mean_popularity_score']}/5")
    print(f"Mean Documentation Quality Score: {results['mean_doc_quality_score']}/10")
    
    print("\nPopularity Score Distribution:")
    for category, count in results['popularity_distribution']['counts'].items():
        percentage = results['popularity_distribution']['percentages'][category]
        print(f"  - {category}: {count} servers ({percentage:.1f}%)")
    
    print("\nDocumentation Quality Breakdown:")
    for category, count in results['doc_quality_distribution']['counts'].items():
        percentage = results['doc_quality_distribution']['percentages'][category]
        print(f"  - {category}: {count} servers ({percentage:.1f}%)")
    
    print("\nRisk Categories:")
    for category, count in results['risk_distribution']['counts'].items():
        percentage = results['risk_distribution']['percentages'][category]
        print(f"  - {category} RISK: {count} servers ({percentage:.1f}%)")
    
    print("\n" + "="*80)