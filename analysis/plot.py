#!/usr/bin/env python3
"""
VMEvalKit Analysis Tool

Analyzes evaluation results to show model performance by domain and overall rankings.
Only scores 4 and 5 are considered "correct" (successful).

Usage:
    python analysis/plot.py --eval-folder data/evaluations/human-eval/
    python analysis/plot.py --eval-folder data/evaluations/gpt4o-eval/
"""

import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import numpy as np
from datetime import datetime
import matplotlib.colors as mcolors

# Set up plotting style with sophisticated color scheme
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

# Define color schemes
MAIN_PALETTE = sns.color_palette("deep", 10)  # For distinct models
SUCCESS_CMAP = plt.cm.RdYlGn  # Red-Yellow-Green for success rates
DOMAIN_COLORS = {
    'chess': '#2E86AB',     # Deep blue
    'maze': '#A23B72',      # Purple
    'raven': '#F18F01',     # Orange
    'rotation': '#C73E1D',  # Red
    'sudoku': '#6A994E'     # Green
}

# Model signature colors (will be assigned dynamically)
MODEL_COLORS = {}

def load_evaluation_data(eval_folder: Path) -> list:
    """Load all evaluation JSON files from the specified folder."""
    evaluations = []
    
    for json_file in eval_folder.rglob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract relevant information
            if "metadata" in data and "result" in data:
                eval_data = {
                    "model_name": data["metadata"].get("model_name", "unknown"),
                    "task_type": data["metadata"].get("task_type", "unknown"),
                    "task_id": data["metadata"].get("task_id", "unknown"),
                    "score": data["result"].get("solution_correctness_score", 0),
                    "evaluator": data["metadata"].get("evaluator", "unknown"),
                    "annotator": data["metadata"].get("annotator", "unknown")
                }
                evaluations.append(eval_data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse {json_file}: {e}")
    
    return evaluations

def calculate_domain_performance(evaluations: list) -> pd.DataFrame:
    """Calculate performance by model and domain (task_type)."""
    results = []
    
    # Group by model and domain
    grouped = defaultdict(lambda: defaultdict(list))
    for eval_data in evaluations:
        model = eval_data["model_name"]
        domain = eval_data["task_type"].replace("_task", "")  # Remove "_task" suffix
        score = eval_data["score"]
        grouped[model][domain].append(score)
    
    # Calculate performance metrics
    for model, domains in grouped.items():
        for domain, scores in domains.items():
            total_tasks = len(scores)
            if total_tasks > 0:
                # Count scores 4 and 5 as correct
                correct_tasks = sum(1 for score in scores if score >= 4)
                success_rate = (correct_tasks / total_tasks) * 100
                avg_score = np.mean(scores)
                
                results.append({
                    "model": model,
                    "domain": domain,
                    "total_tasks": total_tasks,
                    "correct_tasks": correct_tasks,
                    "success_rate": success_rate,
                    "average_score": avg_score,
                    "scores": scores
                })
    
    return pd.DataFrame(results)

def calculate_overall_performance(evaluations: list) -> pd.DataFrame:
    """Calculate overall performance ranking for all models."""
    results = []
    
    # Group by model
    grouped = defaultdict(list)
    for eval_data in evaluations:
        model = eval_data["model_name"]
        score = eval_data["score"]
        grouped[model].append(score)
    
    # Calculate overall metrics
    for model, scores in grouped.items():
        total_tasks = len(scores)
        if total_tasks > 0:
            correct_tasks = sum(1 for score in scores if score >= 4)
            success_rate = (correct_tasks / total_tasks) * 100
            avg_score = np.mean(scores)
            
            results.append({
                "model": model,
                "total_tasks": total_tasks,
                "correct_tasks": correct_tasks,
                "success_rate": success_rate,
                "average_score": avg_score
            })
    
    # Sort by success rate
    df = pd.DataFrame(results)
    df = df.sort_values("success_rate", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    
    return df

def create_overall_model_figure(overall_df: pd.DataFrame):
    """Create a clean bar chart showing overall model performance."""
    
    # Professional typography settings
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 11
    
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    ax.set_facecolor('white')
    
    # Data is already sorted by success rate
    models = overall_df["model"].values
    success_rates = overall_df["success_rate"].values
    
    # Create gradient colors based on success rate
    colors = [SUCCESS_CMAP(rate/100) for rate in success_rates]
    
    # Create horizontal bar chart for better readability of model names
    y_pos = np.arange(len(models))
    bars = ax.barh(y_pos, success_rates, color=colors,
                   edgecolor='#333333', linewidth=1.2, height=0.7,
                   alpha=0.85)
    
    # Set labels and title
    ax.set_xlabel('Success Rate', fontsize=13, color='#333333', fontweight='bold')
    ax.set_ylabel('Model', fontsize=13, color='#333333', fontweight='bold')
    ax.set_title('Overall Model Performance Ranking', 
                fontsize=16, fontweight='bold', pad=20, color='#333333')
    
    # Set y-axis
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.invert_yaxis()  # Highest at the top
    
    # Set x-axis (0-100%)
    ax.set_xlim(0, 105)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_xticklabels([f'{int(x)}%' for x in np.arange(0, 101, 10)])
    
    # Add subtle grid
    ax.grid(axis='x', alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    # Add value labels
    for i, (bar, row) in enumerate(zip(bars, overall_df.itertuples())):
        rate = row.success_rate
        # Success rate at end of bar
        ax.text(rate + 1, i, f'{rate:.1f}%', 
               va='center', ha='left', fontsize=10, 
               fontweight='bold', color='#333333')
        
        # Task count in the middle of bar
        label_color = 'white' if rate > 50 else '#333333'
        ax.text(rate/2, i, f'{row.correct_tasks}/{row.total_tasks}',
               va='center', ha='center', fontsize=9,
               color=label_color, fontweight='bold')
        
        # Rank on the left
        ax.text(-2, i, f'#{row.rank}', 
               va='center', ha='right', fontsize=10,
               fontweight='bold', color='#666666')
    
    # Add summary statistics
    total_evals = sum(overall_df["total_tasks"])
    total_correct = sum(overall_df["correct_tasks"])
    overall_rate = (total_correct / total_evals * 100) if total_evals > 0 else 0
    
    summary_text = (f"Total Evaluations: {total_evals:,} | "
                   f"Overall Success Rate: {overall_rate:.1f}% | "
                   f"Models Evaluated: {len(overall_df)}")
    ax.text(0.5, -0.08, summary_text, transform=ax.transAxes,
           ha='center', va='top', fontsize=10, color='#666666')
    
    plt.tight_layout()
    
    # Save figure
    figures_dir = Path(__file__).parent / "figures"
    output_path = figures_dir / "overall_model_ranking.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"📊 Overall model ranking figure saved to: {output_path}")

def create_overall_domain_figure(domain_df: pd.DataFrame):
    """Create a clean bar chart showing overall domain performance."""
    
    # Professional typography settings
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    ax.set_facecolor('white')
    
    # Calculate domain statistics
    domain_stats = domain_df.groupby("domain").agg({
        "success_rate": "mean",
        "total_tasks": "sum",
        "correct_tasks": "sum"
    }).sort_values("success_rate", ascending=False)
    
    domains = domain_stats.index
    success_rates = domain_stats["success_rate"].values
    
    # Use domain-specific colors
    colors = [DOMAIN_COLORS.get(d, MAIN_PALETTE[i % len(MAIN_PALETTE)]) 
             for i, d in enumerate(domains)]
    
    # Create vertical bar chart
    x_pos = np.arange(len(domains))
    bars = ax.bar(x_pos, success_rates, color=colors,
                 edgecolor='#333333', linewidth=1.2, width=0.7,
                 alpha=0.85)
    
    # Set labels and title
    ax.set_xlabel('Domain', fontsize=13, color='#333333', fontweight='bold')
    ax.set_ylabel('Average Success Rate', fontsize=13, color='#333333', fontweight='bold')
    ax.set_title('Domain Difficulty Analysis', 
                fontsize=16, fontweight='bold', pad=20, color='#333333')
    
    # Set x-axis
    ax.set_xticks(x_pos)
    ax.set_xticklabels([d.capitalize() for d in domains], rotation=0, ha='center')
    
    # Set y-axis (0-100%)
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticklabels([f'{int(y)}%' for y in np.arange(0, 101, 20)])
    
    # Add subtle grid
    ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    # Add value labels and difficulty indicators
    for i, (bar, rate, domain) in enumerate(zip(bars, success_rates, domains)):
        # Success rate on top
        ax.text(i, rate + 1, f'{rate:.1f}%', 
               ha='center', va='bottom', fontsize=11,
               fontweight='bold', color='#333333')
        
        # Task count in middle of bar
        total = domain_stats.loc[domain, "total_tasks"]
        correct = domain_stats.loc[domain, "correct_tasks"]
        label_color = 'white' if rate > 40 else '#333333'
        ax.text(i, rate/2, f'{int(correct)}/{int(total)}',
               ha='center', va='center', fontsize=9,
               color=label_color, fontweight='bold')
        
        # Difficulty level below x-axis
        difficulty = "Easy" if rate > 70 else "Medium" if rate > 40 else "Hard"
        color_diff = "#2E7D32" if rate > 70 else "#FFA726" if rate > 40 else "#D32F2F"
        ax.text(i, -8, difficulty, ha='center', va='top',
               color=color_diff, fontweight='bold', fontsize=10)
    
    # Add horizontal reference lines
    ax.axhline(y=70, color='#2E7D32', linestyle=':', alpha=0.3, linewidth=1)
    ax.axhline(y=40, color='#FFA726', linestyle=':', alpha=0.3, linewidth=1)
    
    # Add legend for difficulty levels
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E7D32', alpha=0.7, label='Easy (>70%)'),
        Patch(facecolor='#FFA726', alpha=0.7, label='Medium (40-70%)'),
        Patch(facecolor='#D32F2F', alpha=0.7, label='Hard (<40%)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', 
             frameon=True, fancybox=False, shadow=False,
             edgecolor='#CCCCCC', facecolor='white', fontsize=9)
    
    # Add summary statistics
    total_tasks = domain_stats["total_tasks"].sum()
    total_correct = domain_stats["correct_tasks"].sum()
    overall_rate = (total_correct / total_tasks * 100) if total_tasks > 0 else 0
    
    summary_text = (f"Total Tasks: {int(total_tasks):,} | "
                   f"Overall Success Rate: {overall_rate:.1f}% | "
                   f"Number of Domains: {len(domains)}")
    ax.text(0.5, -0.12, summary_text, transform=ax.transAxes,
           ha='center', va='top', fontsize=10, color='#666666')
    
    plt.tight_layout()
    
    # Save figure
    figures_dir = Path(__file__).parent / "figures"
    output_path = figures_dir / "overall_domain_difficulty.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"📊 Overall domain difficulty figure saved to: {output_path}")

def create_individual_model_plots(domain_df: pd.DataFrame, overall_df: pd.DataFrame):
    """Create clean individual bar plots for each model's performance across domains."""
    
    # Create figures/models directory
    models_dir = Path(__file__).parent / "figures" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Professional typography settings
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    
    # Assign colors to models if not already assigned
    unique_models = overall_df["model"].unique()
    for i, model in enumerate(unique_models):
        if model not in MODEL_COLORS:
            MODEL_COLORS[model] = MAIN_PALETTE[i % len(MAIN_PALETTE)]
    
    # Create individual plot for each model - SIMPLE BAR CHART
    for model in unique_models:
        model_data = domain_df[domain_df["model"] == model]
        
        if model_data.empty:
            continue
        
        # Create figure with white background - SINGLE PLOT ONLY
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
        ax.set_facecolor('white')
        
        # Sort domains by success rate for better visual
        model_data_sorted = model_data.sort_values("success_rate", ascending=False)
        
        # Prepare data
        domains = model_data_sorted["domain"].values
        success_rates = model_data_sorted["success_rate"].values
        
        # Create gradient colors based on success rate
        colors = [SUCCESS_CMAP(rate/100) for rate in success_rates]
        
        # Create bar chart (vertical bars for domains)
        x_pos = np.arange(len(domains))
        bars = ax.bar(x_pos, success_rates, color=colors, 
                      edgecolor='#333333', linewidth=1.2, width=0.7,
                      alpha=0.85)
        
        # Set labels and title
        ax.set_xlabel('Domain', fontsize=13, color='#333333')
        ax.set_ylabel('Success Rate', fontsize=13, color='#333333')
        ax.set_title(f'{model} Performance by Domain', 
                    fontsize=15, fontweight='bold', pad=20, color='#333333')
        
        # Set x-axis
        ax.set_xticks(x_pos)
        ax.set_xticklabels([d.capitalize() for d in domains], rotation=0, ha='center')
        
        # Set y-axis (0-100%)
        ax.set_ylim(0, 105)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_yticklabels([f'{int(y)}%' for y in np.arange(0, 101, 20)])
        
        # Add subtle grid
        ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        
        # Add value labels on top of bars
        for i, (bar, rate) in enumerate(zip(bars, success_rates)):
            # Display success rate percentage
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{rate:.1f}%', ha='center', va='bottom',
                   fontsize=10, fontweight='bold', color='#333333')
            
            # Add task count below percentage
            tasks = model_data_sorted.iloc[i]["total_tasks"]
            correct = model_data_sorted.iloc[i]["correct_tasks"]
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                   f'{correct}/{tasks}', ha='center', va='center',
                   fontsize=9, color='white' if rate > 50 else '#333333',
                   fontweight='bold')
        
        # Add overall stats as subtitle
        overall_stats = overall_df[overall_df["model"] == model].iloc[0]
        subtitle_text = (f"Overall Success: {overall_stats['success_rate']:.1f}% | "
                        f"Rank: #{overall_stats['rank']}/{len(overall_df)} | "
                        f"Average Score: {overall_stats['average_score']:.2f}")
        ax.text(0.5, -0.12, subtitle_text, transform=ax.transAxes,
               ha='center', va='top', fontsize=10, color='#666666')
        
        plt.tight_layout()
        
        # Save with sanitized filename
        safe_model_name = model.replace("/", "_").replace(" ", "_")
        output_path = models_dir / f"{safe_model_name}_performance.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"  📊 Created plot for {model}: {output_path}")

def create_visualizations(domain_df: pd.DataFrame, overall_df: pd.DataFrame, output_path: str):
    """Create comprehensive visualizations with improved bar plots."""
    
    # Create figures directory if it doesn't exist
    figures_dir = Path(__file__).parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Assign colors to models
    unique_models = overall_df["model"].unique()
    for i, model in enumerate(unique_models):
        if model not in MODEL_COLORS:
            MODEL_COLORS[model] = MAIN_PALETTE[i % len(MAIN_PALETTE)]
    
    # Create main comparison figure
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Grouped bar chart for model performance by domain
    ax1 = plt.subplot(2, 2, 1)
    pivot_success = domain_df.pivot(index="domain", columns="model", values="success_rate")
    pivot_success.plot(kind='bar', ax=ax1, width=0.8, edgecolor='white', linewidth=2,
                       color=[MODEL_COLORS[m] for m in pivot_success.columns])
    
    ax1.set_xlabel("Domain", fontweight='bold', fontsize=12)
    ax1.set_ylabel("Success Rate (%)", fontweight='bold', fontsize=12)
    ax1.set_title("Model Performance Comparison by Domain", fontsize=14, fontweight='bold')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.legend(title="Models", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Add value labels on bars
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.0f', fontsize=8, padding=2)
    
    # 2. Overall Model Ranking (horizontal bar)
    ax2 = plt.subplot(2, 2, 2)
    colors_rank = [SUCCESS_CMAP(rate/100) for rate in overall_df["success_rate"]]
    bars = ax2.barh(range(len(overall_df)), overall_df["success_rate"], 
                    color=colors_rank, edgecolor='white', linewidth=2, height=0.6)
    
    ax2.set_yticks(range(len(overall_df)))
    ax2.set_yticklabels(overall_df["model"])
    ax2.set_xlabel("Success Rate (%)", fontweight='bold', fontsize=12)
    ax2.set_title("Overall Model Ranking (All Domains)", fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 100)
    ax2.grid(axis='x', alpha=0.3)
    ax2.invert_yaxis()
    
    # Add value and rank labels
    for i, (bar, row) in enumerate(zip(bars, overall_df.iterrows())):
        rate = row[1]["success_rate"]
        correct = row[1]["correct_tasks"]
        total = row[1]["total_tasks"]
        # Add percentage in the bar
        label_color = 'white' if rate > 50 else 'black'
        ax2.text(rate/2, i, f'{rate:.1f}%', va='center', ha='center', 
                fontweight='bold', color=label_color, fontsize=11)
        # Add task count outside
        ax2.text(rate + 1, i, f'({correct}/{total})', va='center', ha='left', 
                fontsize=9, alpha=0.7)
        # Add rank on the left
        ax2.text(-2, i, f'#{row[1]["rank"]}', va='center', ha='right', 
                fontweight='bold', fontsize=10)
    
    # 3. Domain Difficulty Ranking (vertical bar)
    ax3 = plt.subplot(2, 2, 3)
    domain_stats = domain_df.groupby("domain").agg({
        "success_rate": "mean",
        "total_tasks": "sum",
        "correct_tasks": "sum"
    }).sort_values("success_rate")
    
    domains = domain_stats.index
    success_rates = domain_stats["success_rate"].values
    colors_domain = [DOMAIN_COLORS.get(d, MAIN_PALETTE[i % len(MAIN_PALETTE)]) 
                    for i, d in enumerate(domains)]
    
    bars = ax3.bar(range(len(domains)), success_rates, 
                  color=colors_domain, edgecolor='white', linewidth=2, width=0.6)
    
    ax3.set_xticks(range(len(domains)))
    ax3.set_xticklabels([d.capitalize() for d in domains], rotation=45, ha='right')
    ax3.set_ylabel("Average Success Rate (%)", fontweight='bold', fontsize=12)
    ax3.set_title("Domain Difficulty Analysis", fontsize=14, fontweight='bold')
    ax3.set_ylim(0, max(success_rates) * 1.2)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels and difficulty indicators
    for i, (bar, rate, domain) in enumerate(zip(bars, success_rates, domains)):
        # Add percentage on top
        ax3.text(i, rate + 1, f'{rate:.1f}%', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
        # Add total tasks below
        total = domain_stats.loc[domain, "total_tasks"]
        correct = domain_stats.loc[domain, "correct_tasks"]
        ax3.text(i, rate/2, f'{int(correct)}/{int(total)}', ha='center', va='center',
                color='white' if rate > 30 else 'black', fontweight='bold', fontsize=9)
        # Add difficulty level
        difficulty = "Easy" if rate > 70 else "Medium" if rate > 40 else "Hard"
        color_diff = "green" if rate > 70 else "orange" if rate > 40 else "red"
        ax3.text(i, -5, difficulty, ha='center', va='top', 
                color=color_diff, fontweight='bold', fontsize=9)
    
    # 4. Score Distribution Across All Models
    ax4 = plt.subplot(2, 2, 4)
    all_scores = []
    for eval_data in domain_df.itertuples():
        all_scores.extend(eval_data.scores)
    
    score_counts = np.bincount(all_scores, minlength=6)[1:]  # Scores 1-5
    score_labels = ['Score 1', 'Score 2', 'Score 3', 'Score 4\n(Success)', 'Score 5\n(Success)']
    score_colors = [SUCCESS_CMAP(i/4) for i in range(5)]
    
    bars = ax4.bar(range(5), score_counts, color=score_colors, 
                  edgecolor='white', linewidth=2, width=0.6)
    
    ax4.set_xticks(range(5))
    ax4.set_xticklabels(score_labels)
    ax4.set_ylabel("Count", fontweight='bold', fontsize=12)
    ax4.set_title("Overall Score Distribution", fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # Add count labels and percentage
    total_scores = sum(score_counts)
    for i, (bar, count) in enumerate(zip(bars, score_counts)):
        if count > 0:
            ax4.text(i, count + 5, str(int(count)), ha='center', va='bottom', 
                    fontweight='bold', fontsize=10)
            percentage = (count / total_scores) * 100
            ax4.text(i, count/2, f'{percentage:.1f}%', ha='center', va='center',
                    color='white' if count > 50 else 'black', fontweight='bold', fontsize=9)
    
    # Add success threshold line
    ax4.axvline(x=2.5, color='red', linestyle='--', alpha=0.5, linewidth=2, 
               label='Success Threshold')
    ax4.legend(loc='upper left')
    
    # Add super title
    success_count = score_counts[3] + score_counts[4]
    overall_success = (success_count / total_scores) * 100 if total_scores > 0 else 0
    fig.suptitle(f"VMEvalKit Comprehensive Analysis\n" + 
                f"Total Evaluations: {total_scores} | Success Rate: {overall_success:.1f}% | " +
                f"Models: {len(unique_models)} | Domains: {len(domain_df['domain'].unique())}",
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Main visualization saved to: {output_path}")
    plt.close()

def print_detailed_results(domain_df: pd.DataFrame, overall_df: pd.DataFrame):
    """Print comprehensive statistics in table-friendly format for paper."""
    
    print("\n" + "="*100)
    print("🎯 VMEVALKIT COMPREHENSIVE EVALUATION STATISTICS")
    print("="*100)
    
    # Table 1: Overall Model Performance Ranking
    print("\n📊 TABLE 1: OVERALL MODEL PERFORMANCE RANKING")
    print("-" * 100)
    print(f"{'Rank':<6} {'Model':<30} {'Success Rate':<15} {'Correct/Total':<15} {'Avg Score':<12} {'Std Dev':<10}")
    print("-" * 100)
    
    for _, row in overall_df.iterrows():
        # Calculate std dev for this model
        model_scores = []
        for _, domain_row in domain_df[domain_df["model"] == row["model"]].iterrows():
            model_scores.extend(domain_row["scores"])
        std_dev = np.std(model_scores) if model_scores else 0
        
        print(f"{row['rank']:<6} {row['model']:<30} {row['success_rate']:>6.2f}%{'':<8} "
              f"{row['correct_tasks']:>3d}/{row['total_tasks']:<3d}{'':<8} "
              f"{row['average_score']:>6.3f}{'':<6} {std_dev:>6.3f}")
    
    # Table 2: Domain-wise Performance Matrix
    print("\n\n📊 TABLE 2: MODEL PERFORMANCE BY DOMAIN (Success Rate %)")
    print("-" * 100)
    
    pivot_table = domain_df.pivot(index="model", columns="domain", values="success_rate")
    
    # Print header
    header = f"{'Model':<30}"
    for domain in sorted(pivot_table.columns):
        header += f" {domain.capitalize():<12}"
    header += f" {'Average':<12}"
    print(header)
    print("-" * 100)
    
    # Print data
    for model in overall_df["model"]:
        row_str = f"{model:<30}"
        domain_scores = []
        for domain in sorted(pivot_table.columns):
            if model in pivot_table.index and domain in pivot_table.columns:
                value = pivot_table.loc[model, domain]
                if pd.notna(value):
                    row_str += f" {value:>6.1f}%{'':<5}"
                    domain_scores.append(value)
                else:
                    row_str += f" {'N/A':<12}"
            else:
                row_str += f" {'N/A':<12}"
        
        # Add average
        if domain_scores:
            avg_score = np.mean(domain_scores)
            row_str += f" {avg_score:>6.1f}%"
        else:
            row_str += f" {'N/A':<12}"
        print(row_str)
    
    # Table 3: Domain Statistics
    print("\n\n📊 TABLE 3: DOMAIN-LEVEL STATISTICS")
    print("-" * 100)
    print(f"{'Domain':<15} {'Avg Success':<15} {'Std Dev':<12} {'Min Score':<12} {'Max Score':<12} {'Total Tasks':<12} {'Difficulty':<12}")
    print("-" * 100)
    
    domain_stats = domain_df.groupby("domain").agg({
        "success_rate": ["mean", "std", "min", "max"],
        "total_tasks": "sum"
    }).round(2)
    
    for domain in domain_stats.index:
        avg_rate = domain_stats.loc[domain, ("success_rate", "mean")]
        std_rate = domain_stats.loc[domain, ("success_rate", "std")]
        min_rate = domain_stats.loc[domain, ("success_rate", "min")]
        max_rate = domain_stats.loc[domain, ("success_rate", "max")]
        total_tasks = int(domain_stats.loc[domain, ("total_tasks", "sum")])
        
        difficulty = "Easy" if avg_rate > 70 else "Medium" if avg_rate > 40 else "Hard"
        
        print(f"{domain.capitalize():<15} {avg_rate:>6.2f}%{'':<8} {std_rate:>8.2f}{'':<4} "
              f"{min_rate:>6.2f}%{'':<5} {max_rate:>6.2f}%{'':<5} {total_tasks:>8}{'':<4} {difficulty:<12}")
    
    # Table 4: Score Distribution Analysis
    print("\n\n📊 TABLE 4: SCORE DISTRIBUTION ANALYSIS")
    print("-" * 100)
    
    all_scores = []
    score_by_model = defaultdict(list)
    for _, row in domain_df.iterrows():
        all_scores.extend(row["scores"])
        score_by_model[row["model"]].extend(row["scores"])
    
    score_counts = np.bincount(all_scores, minlength=6)[1:]  # Scores 1-5
    total_scores = len(all_scores)
    
    print(f"{'Score':<10} {'Count':<10} {'Percentage':<15} {'Cumulative %':<15} {'Classification':<20}")
    print("-" * 100)
    
    cumulative = 0
    for i, count in enumerate(score_counts, 1):
        percentage = (count / total_scores) * 100 if total_scores > 0 else 0
        cumulative += percentage
        classification = "Success" if i >= 4 else "Failure"
        print(f"Score {i:<4} {count:<10} {percentage:>6.2f}%{'':<8} {cumulative:>6.2f}%{'':<8} {classification:<20}")
    
    # Table 5: Model-Domain Detailed Breakdown
    print("\n\n📊 TABLE 5: DETAILED MODEL-DOMAIN BREAKDOWN")
    print("-" * 100)
    print(f"{'Model':<30} {'Domain':<15} {'Tasks':<8} {'Correct':<10} {'Success%':<12} {'Avg Score':<12}")
    print("-" * 100)
    
    for model in sorted(domain_df["model"].unique()):
        model_data = domain_df[domain_df["model"] == model].sort_values("domain")
        first_row = True
        for _, row in model_data.iterrows():
            model_name = model if first_row else ""
            first_row = False
            print(f"{model_name:<30} {row['domain'].capitalize():<15} {row['total_tasks']:<8} "
                  f"{row['correct_tasks']:<10} {row['success_rate']:>6.2f}%{'':<5} {row['average_score']:>6.3f}")
        
        # Add model summary
        model_overall = overall_df[overall_df["model"] == model].iloc[0]
        print(f"{'  TOTAL':<30} {'All Domains':<15} {model_overall['total_tasks']:<8} "
              f"{model_overall['correct_tasks']:<10} {model_overall['success_rate']:>6.2f}%{'':<5} "
              f"{model_overall['average_score']:>6.3f}")
        print("-" * 100)
    
    # Table 6: Statistical Summary
    print("\n\n📊 TABLE 6: STATISTICAL SUMMARY")
    print("-" * 100)
    
    print(f"Total Evaluations: {total_scores:,}")
    print(f"Total Correct (Scores 4-5): {sum(score_counts[3:]):,}")
    print(f"Total Failed (Scores 1-3): {sum(score_counts[:3]):,}")
    print(f"Overall Success Rate: {(sum(score_counts[3:]) / total_scores * 100):.2f}%")
    print(f"Number of Models Evaluated: {len(overall_df)}")
    print(f"Number of Domains: {len(domain_df['domain'].unique())}")
    print(f"Average Tasks per Model: {total_scores / len(overall_df):.1f}")
    print(f"Average Tasks per Domain: {total_scores / len(domain_df['domain'].unique()):.1f}")
    
    # Best performers
    print("\n🏆 TOP PERFORMERS:")
    print("-" * 50)
    
    # Best overall model
    best_model = overall_df.iloc[0]
    print(f"Best Overall Model: {best_model['model']} ({best_model['success_rate']:.2f}%)")
    
    # Best model per domain
    for domain in sorted(domain_df["domain"].unique()):
        domain_best = domain_df[domain_df["domain"] == domain].nlargest(1, "success_rate").iloc[0]
        print(f"Best in {domain.capitalize()}: {domain_best['model']} ({domain_best['success_rate']:.2f}%)")
    
    # Difficulty analysis
    print("\n📊 DOMAIN DIFFICULTY RANKING:")
    print("-" * 50)
    
    domain_difficulty = domain_df.groupby("domain")["success_rate"].mean().sort_values(ascending=False)
    
    for rank, (domain, avg_rate) in enumerate(domain_difficulty.items(), 1):
        difficulty_level = "Easy" if avg_rate > 70 else "Medium" if avg_rate > 40 else "Hard"
        print(f"{rank}. {domain.capitalize():<15} - Average Success: {avg_rate:>6.2f}% ({difficulty_level})")
    
    # Table 7: Performance Variance Analysis
    print("\n\n📊 TABLE 7: PERFORMANCE VARIANCE ANALYSIS")
    print("-" * 100)
    print(f"{'Model':<30} {'Min Domain %':<15} {'Max Domain %':<15} {'Variance':<12} {'Consistency':<15}")
    print("-" * 100)
    
    for model in overall_df["model"]:
        model_domain_scores = domain_df[domain_df["model"] == model]["success_rate"].values
        if len(model_domain_scores) > 0:
            min_score = model_domain_scores.min()
            max_score = model_domain_scores.max()
            variance = max_score - min_score
            consistency = "High" if variance < 20 else "Medium" if variance < 40 else "Low"
            
            print(f"{model:<30} {min_score:>6.2f}%{'':<8} {max_score:>6.2f}%{'':<8} "
                  f"{variance:>6.2f}{'':<6} {consistency:<15}")
    
    print("\n" + "="*100)
    print("END OF COMPREHENSIVE STATISTICS")
    print("="*100)

def main():
    parser = argparse.ArgumentParser(
        description="Analyze VMEvalKit evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analysis/plot.py --eval-folder data/evaluations/human-eval/
  python analysis/plot.py --eval-folder data/evaluations/gpt4o-eval/
        """
    )
    
    parser.add_argument("--eval-folder", required=True, type=str,
                      help="Path to evaluation folder (e.g., data/evaluations/human-eval/)")
    
    args = parser.parse_args()
    
    eval_folder = Path(args.eval_folder)
    if not eval_folder.exists():
        print(f"❌ Error: Evaluation folder not found: {eval_folder}")
        return
    
    # Load and analyze data
    print(f"📂 Loading evaluations from: {eval_folder}")
    evaluations = load_evaluation_data(eval_folder)
    
    if not evaluations:
        print(f"❌ No evaluation files found in {eval_folder}")
        return
    
    print(f"✅ Loaded {len(evaluations)} evaluations")
    
    # Calculate performance metrics
    domain_df = calculate_domain_performance(evaluations)
    overall_df = calculate_overall_performance(evaluations)
    
    # Print detailed results
    print_detailed_results(domain_df, overall_df)
    
    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    
    # Extract evaluation type from folder path
    eval_type = "unknown"
    if "human-eval" in str(eval_folder):
        eval_type = "human-eval"
    elif "gpt4o-eval" in str(eval_folder):
        eval_type = "gpt4o-eval"
    elif "custom-eval" in str(eval_folder):
        eval_type = "custom-eval"
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"analysis/figures/vmevalkit_{eval_type}_{timestamp}.png"
    
    # Create main comprehensive visualization (4-panel grid)
    create_visualizations(domain_df, overall_df, output_path)
    
    # Create clean individual figures
    print(f"\n📊 Creating clean individual visualizations...")
    
    # 1. Overall model ranking figure
    create_overall_model_figure(overall_df)
    
    # 2. Overall domain difficulty figure
    create_overall_domain_figure(domain_df)
    
    # 3. Individual model performance plots
    print(f"\n📊 Creating individual model performance plots...")
    create_individual_model_plots(domain_df, overall_df)

if __name__ == "__main__":
    main()