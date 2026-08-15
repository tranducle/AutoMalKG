"""Render plots from archived manuscript-summary artifacts."""

import json
import matplotlib.pyplot as plt
import os

# Apply basic academic style overrides
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "axes.edgecolor": "#94A3B8",
    "axes.linewidth": 0.8,
    "grid.color": "#E2E8F0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 7.5,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

def plot_exp1():
    with open("../data/processed/exp1_results.json", "r") as f:
        data = json.load(f)
        
    metrics = data["metrics"]
    labels = list(metrics.keys())
    f1_scores = [metrics[k]["f1"] for k in labels]
    
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.grid(True, zorder=1)
    
    colors = ['#E2E8F0', '#CBD5E1', '#94A3B8', '#475569', '#0F766E']
    bars = ax.bar(labels, f1_scores, color=colors, zorder=3)
    
    ax.set_title('Archived Extraction Benchmark Summary')
    ax.set_ylabel('F1 Score')
    ax.set_ylim(0, 1.0)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.3f}", ha='center',
                fontsize=7.5, fontweight="semibold",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5), zorder=5)
        
    fig.savefig("../data/processed/exp1_plot.pdf", bbox_inches="tight")
    fig.savefig("../data/processed/exp1_plot.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

def plot_exp2():
    with open("../data/processed/exp2_results.json", "r") as f:
        data = json.load(f)
        
    # Plot 2A: Prompting Ablation
    labels_2a = list(data["Ablation_2A_Prompting"].keys())
    f1_scores = [data["Ablation_2A_Prompting"][k]["f1"] for k in labels_2a]
    
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.grid(True, zorder=1)
    bars = ax.bar(labels_2a, f1_scores, color=['#94A3B8', '#0F766E'], zorder=3)
    ax.set_title('Archived Prompting Ablation Summary')
    ax.set_ylabel('F1 Score')
    ax.set_ylim(0, 1.0)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.3f}", ha='center',
                fontsize=7.5, fontweight="semibold",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5), zorder=5)
    fig.savefig("../data/processed/exp2a_plot.pdf", bbox_inches="tight")
    fig.savefig("../data/processed/exp2a_plot.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    # Plot 2B: Reasoning Ablation
    labels_2b = list(data["Ablation_2B_Reasoning"].keys())
    tpr_scores = [data["Ablation_2B_Reasoning"][k]["TPR"] for k in labels_2b]
    
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.grid(True, zorder=1)
    bars = ax.bar(labels_2b, tpr_scores, color=['#94A3B8', '#0F766E'], zorder=3)
    ax.set_title('Archived Reasoning Ablation Summary')
    ax.set_ylabel('True Positive Rate (Hallucinations Caught)')
    ax.set_ylim(0, 1.1)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center',
                fontsize=7.5, fontweight="semibold",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5), zorder=5)
    fig.savefig("../data/processed/exp2b_plot.pdf", bbox_inches="tight")
    fig.savefig("../data/processed/exp2b_plot.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

def plot_exp3():
    with open("../data/processed/exp3_results.json", "r") as f:
        data = json.load(f)
        
    vendors = [k for k in data.keys() if "Vendor" in k]
    mean_f1 = [data[k]["mean_f1"] for k in vendors]
    std_dev = [data[k]["std_dev"] for k in vendors]
    
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.grid(True, zorder=1)
    bars = ax.bar(vendors, mean_f1, yerr=std_dev, capsize=5, color='#0F766E', alpha=0.85, zorder=3, error_kw=dict(ecolor='#334155', lw=1.5, capsize=5, capthick=1.5))
    ax.set_title('Archived Exploratory Vendor Summary')
    ax.set_ylabel('F1 Score')
    ax.set_ylim(0.8, 1.0)
    ax.text(1, 0.82, f"Archived ANOVA p-value: {data['ANOVA_p_value']:.3f}", ha='center',
            fontsize=7.5, fontweight="semibold",
            bbox=dict(facecolor='white', alpha=0.85, edgecolor="#E2E8F0", pad=2.0), zorder=5)
    
    fig.savefig("../data/processed/exp3_plot.pdf", bbox_inches="tight")
    fig.savefig("../data/processed/exp3_plot.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

def plot_exp4():
    with open("../data/processed/exp4_results.json", "r") as f:
        data = json.load(f)

    curve = data.get("token_volume_curve", data.get("scalability_curve", []))
    reports = [p["reports_processed"] for p in curve]
    token_millions = [p.get("estimated_tokens_millions", p.get("tokens_millions")) for p in curve]
    
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.grid(True, zorder=1)
    ax.plot(reports, token_millions, marker='o', linestyle='-', color='#0F766E', linewidth=1.5, markersize=5, zorder=3)
    ax.set_title('Analytical Token-Volume Scaling')
    ax.set_xlabel('Number of Reports Processed')
    ax.set_ylabel('Token Volume (Millions)')
    
    fig.savefig("../data/processed/exp4_plot.pdf", bbox_inches="tight")
    fig.savefig("../data/processed/exp4_plot.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    os.makedirs("../data/processed", exist_ok=True)
    plot_exp1()
    plot_exp2()
    plot_exp3()
    plot_exp4()
    print("Plots generated successfully in ../data/processed/")
