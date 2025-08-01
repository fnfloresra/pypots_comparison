import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Data from the provided JSON
methods = ["KNN", "BRITS", "SAITS"]
metrics_data = {
    "Train Time": [10, 180, 300],
    "Complexity": [2, 7, 9],
    "Accuracy": [6, 8, 9],
    "Memory Usage": [3, 6, 8],
    "Interpret": [9, 5, 3]
}

# Brand colors for the three methods
colors = ['#1FB8CD', '#2E8B57', '#DB4545']  # Strong cyan, Sea green, Bright red

# Create the grouped bar chart
fig = go.Figure()

# Add bars for each method
for i, method in enumerate(methods):
    values = [metrics_data[metric][i] for metric in metrics_data.keys()]
    fig.add_trace(go.Bar(
        name=method,
        x=list(metrics_data.keys()),
        y=values,
        marker_color=colors[i]
    ))

# Update layout
fig.update_layout(
    title="PyPOTS Methods Comparison",
    xaxis_title="Metrics",
    yaxis_title="Score/Time",
    barmode='group',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5),
    xaxis=dict(showgrid=True),
    yaxis=dict(showgrid=True)
)

# Save the chart
fig.write_image("pypots_methods_comparison.png")