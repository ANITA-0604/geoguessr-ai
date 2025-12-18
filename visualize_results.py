import matplotlib.pyplot as plt
import numpy as np

def plot_quantitative_results(mean_error, median_error, acc_25, acc_100, acc_500):
    """
    Creates a professional figure combining a Bar Chart and a Table
    for your slide presentation.
    """
    # 1. Setup Data
    metrics_labels = ['Within 25km', 'Within 100km', 'Within 500km']
    metrics_values = [acc_25 * 100, acc_100 * 100, acc_500 * 100] # Convert to %
    
    colors = ['#ff9999', '#66b3ff', '#99ff99'] # Light Red, Blue, Green

    # 2. Create Figure
    fig, (ax_chart, ax_table) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    # --- TOP: BAR CHART ---
    bars = ax_chart.bar(metrics_labels, metrics_values, color=colors, edgecolor='black', alpha=0.7)
    
    # Add number labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax_chart.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax_chart.set_ylim(0, 100)
    ax_chart.set_ylabel('Accuracy (%)', fontsize=12)
    ax_chart.set_title('Geolocation Accuracy by Distance Threshold', fontsize=14, fontweight='bold')
    
    # Add a horizontal line for Random Chance (approx 1/20 states = 5% or similar)
    # Optional: Just keeps the chart clean.
    
    # --- BOTTOM: TABLE ---
    ax_table.axis('off') # Hide the graph part for the table
    
    table_data = [
        ["Mean Error", f"{mean_error:.2f} km"],
        ["Median Error", f"{median_error:.2f} km"],
        ["Accuracy < 25 km", f"{acc_25:.2%}"],
        ["Accuracy < 100 km", f"{acc_100:.2%}"],
        ["Accuracy < 500 km", f"{acc_500:.2%}"]
    ]
    
    # Create the table
    table = ax_table.table(cellText=table_data, 
                           colLabels=["Metric", "Value"], 
                           loc='center', 
                           cellLoc='center')
    
    # Styling the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2) # Stretch row heights
    
    # Make header bold (optional manual styling)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e0e0e0') # Grey header background

    plt.tight_layout()
    plt.savefig('quantitative_results.png', dpi=300)
    print("Saved 'quantitative_results.png'")

if __name__ == "__main__":
    # REPLACE THESE NUMBERS WITH YOUR EXACT RESULTS FROM evaluate_distance.py
    MEAN_ERROR = 863.46
    MEDIAN_ERROR = 527.83
    ACC_25 = 0.1409
    ACC_100 = 0.2510
    ACC_500 = 0.4817
    
    plot_quantitative_results(MEAN_ERROR, MEDIAN_ERROR, ACC_25, ACC_100, ACC_500)