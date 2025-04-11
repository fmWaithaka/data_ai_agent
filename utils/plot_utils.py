import plotly.io as pio

# Configure default corporate theme
pio.templates.default = "plotly_white"

CORPORATE_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

def apply_corporate_style(fig):
    """Apply consistent styling to Plotly figures"""
    fig.update_layout(
        font=dict(family="Arial", size=12, color="#333333"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        colorway=CORPORATE_COLORS,
        xaxis=dict(
            showline=True,
            linecolor="lightgray",
            showgrid=False
        ),
        yaxis=dict(
            showline=True,
            linecolor="lightgray",
            showgrid=True,
            gridcolor="#f0f0f0"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig