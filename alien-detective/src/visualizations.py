"""
visualizations.py
All Plotly chart builders for the alien signal mystery game.
Each function returns a go.Figure ready for st.plotly_chart().
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ── Shared theme ───────────────────────────────────────────────────────────────
DARK_BG    = "#0a0e1a"
PANEL_BG   = "#111827"
GRID_COLOR = "#1e2d40"
TEXT_COLOR = "#c8d8e8"
ACCENT     = "#00e5ff"

CLASS_COLORS = {
    "Beacon":          "#4fc3f7",
    "Warning":         "#ff7043",
    "DataTransmission":"#66bb6a",
    "Anomaly":         "#f48fb1",
    "Interference":    "#9e9e9e",
}

CLUSTER_PALETTE = px.colors.qualitative.Bold


def _base_layout(title: str = "", height: int = 450) -> dict:
    return dict(
        title=dict(text=title, font=dict(color=ACCENT, size=15), x=0.02),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT_COLOR, size=12),
        height=height,
        margin=dict(l=50, r=20, t=50, b=40),
        xaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL LAB CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def signal_scatter(df: pd.DataFrame, color_col: str = "signal_class") -> go.Figure:
    """
    2D scatter: sector_x vs sector_y, coloured by class or cluster.
    """
    fig = go.Figure()
    classes = df[color_col].unique()
    palette = CLASS_COLORS if color_col == "signal_class" else {}

    for i, cls in enumerate(classes):
        mask  = df[color_col] == cls
        color = palette.get(cls, CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)])
        sub   = df[mask]

        hover = (
            "<b>Signal ID:</b> %{customdata[0]}<br>"
            "<b>Freq:</b> %{customdata[1]} MHz<br>"
            "<b>Amplitude:</b> %{customdata[2]} dB<br>"
            "<b>Class:</b> " + str(cls) + "<br>"
            "<extra></extra>"
        )
        fig.add_trace(go.Scatter(
            x=sub["sector_x"], y=sub["sector_y"],
            mode="markers",
            name=str(cls),
            marker=dict(color=color, size=7, opacity=0.75,
                        line=dict(width=0.5, color="white")),
            customdata=sub[["signal_id", "frequency_mhz", "amplitude_db"]].values,
            hovertemplate=hover,
        ))

    fig.update_layout(**_base_layout("Signal Origin Map — Sector Grid", height=460))
    fig.update_xaxes(title="Sector X")
    fig.update_yaxes(title="Sector Y")
    return fig


def frequency_histogram(df: pd.DataFrame) -> go.Figure:
    """Histogram of signal frequencies coloured by class."""
    fig = go.Figure()
    for cls, color in CLASS_COLORS.items():
        sub = df[df["signal_class"] == cls]
        if sub.empty:
            continue
        fig.add_trace(go.Histogram(
            x=sub["frequency_mhz"],
            name=cls,
            marker_color=color,
            opacity=0.75,
            nbinsx=30,
        ))
    fig.update_layout(
        **_base_layout("Signal Frequency Distribution (MHz)", height=380),
        barmode="overlay",
        xaxis_title="Frequency (MHz)",
        yaxis_title="Count",
        legend=dict(bgcolor=PANEL_BG, bordercolor=GRID_COLOR),
    )
    return fig


def timeline_chart(df: pd.DataFrame) -> go.Figure:
    """Time series: signals per day, stacked by class."""
    df2 = df.copy()
    df2["date"] = df2["timestamp"].dt.date
    grouped = df2.groupby(["date", "signal_class"]).size().reset_index(name="count")

    fig = go.Figure()
    for cls, color in CLASS_COLORS.items():
        sub = grouped[grouped["signal_class"] == cls]
        fig.add_trace(go.Scatter(
            x=sub["date"], y=sub["count"],
            name=cls,
            mode="lines",
            stackgroup="one",
            line=dict(color=color, width=1),
            fillcolor=color,
        ))
    fig.update_layout(
        **_base_layout("Signal Activity Over Time", height=350),
        xaxis_title="Date",
        yaxis_title="Signals / Day",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  CLASSIFICATION CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def confusion_matrix_heatmap(cm: list, labels: list) -> go.Figure:
    """Annotated confusion matrix heatmap."""
    cm_arr = np.array(cm)
    fig = go.Figure(go.Heatmap(
        z=cm_arr,
        x=labels,
        y=labels,
        colorscale=[[0, PANEL_BG], [1, ACCENT]],
        text=cm_arr,
        texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(
        **_base_layout("Confusion Matrix", height=420),
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )
    return fig


def feature_importance_bar(feature_importance: dict) -> go.Figure:
    """Horizontal bar chart of RandomForest feature importances."""
    items  = sorted(feature_importance.items(), key=lambda x: x[1])
    feats  = [i[0] for i in items]
    imps   = [i[1] for i in items]
    colors = [ACCENT if imp == max(imps) else "#4a90d9" for imp in imps]

    fig = go.Figure(go.Bar(
        x=imps, y=feats,
        orientation="h",
        marker_color=colors,
    ))
    fig.update_layout(
        **_base_layout("Feature Importance (RandomForest)", height=380),
        xaxis_title="Importance",
    )
    return fig


def class_distribution_pie(df: pd.DataFrame, col: str = "signal_class") -> go.Figure:
    """Donut chart of signal class distribution."""
    counts = df[col].value_counts()
    colors = [CLASS_COLORS.get(c, "#888") for c in counts.index]

    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.45,
        marker_colors=colors,
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout("Signal Class Distribution", height=360),
        legend=dict(bgcolor=PANEL_BG),
    )
    return fig


def confidence_histogram(df: pd.DataFrame) -> go.Figure:
    """Distribution of classifier confidence scores."""
    if "confidence" not in df.columns:
        return go.Figure()

    fig = go.Figure(go.Histogram(
        x=df["confidence"],
        nbinsx=20,
        marker_color=ACCENT,
        opacity=0.8,
    ))
    fig.update_layout(
        **_base_layout("Classifier Confidence Distribution", height=320),
        xaxis_title="Confidence Score",
        yaxis_title="Count",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  CLUSTERING CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def cluster_scatter(df: pd.DataFrame,
                    cluster_col: str = "kmeans_cluster",
                    highlight_anomalies: bool = True) -> go.Figure:
    """
    Sector scatter coloured by cluster label.
    Optionally overlays Anomaly signals with a red glow.
    """
    fig = go.Figure()
    clusters = sorted(df[cluster_col].unique())

    for i, cl in enumerate(clusters):
        label  = f"Cluster {cl}" if cl >= 0 else "Noise / Outlier"
        color  = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)] if cl >= 0 else "#555"
        sub    = df[df[cluster_col] == cl]
        fig.add_trace(go.Scatter(
            x=sub["sector_x"], y=sub["sector_y"],
            mode="markers",
            name=label,
            marker=dict(color=color, size=7, opacity=0.65,
                        line=dict(width=0.4, color="white")),
            hovertemplate=(
                f"<b>{label}</b><br>"
                "X: %{x:.1f}  Y: %{y:.1f}<br>"
                "<extra></extra>"
            ),
        ))

    if highlight_anomalies and "signal_class" in df.columns:
        anomalies = df[df["signal_class"] == "Anomaly"]
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies["sector_x"], y=anomalies["sector_y"],
                mode="markers",
                name="⚠ Anomaly",
                marker=dict(
                    color="#ff4444", size=11, opacity=1.0,
                    symbol="diamond",
                    line=dict(width=1.5, color="#ff8888"),
                ),
                hovertemplate="<b>ANOMALY</b><br>X: %{x:.1f}  Y: %{y:.1f}<extra></extra>",
            ))

    fig.update_layout(**_base_layout("Cluster Map — Origin Sectors", height=460))
    fig.update_xaxes(title="Sector X")
    fig.update_yaxes(title="Sector Y")
    return fig


def pca_scatter(df: pd.DataFrame) -> go.Figure:
    """2D PCA projection scatter."""
    if "pca_x" not in df.columns:
        return go.Figure()

    color_col = "signal_class" if "signal_class" in df.columns else "kmeans_cluster"
    classes   = df[color_col].unique()

    fig = go.Figure()
    for i, cls in enumerate(classes):
        sub   = df[df[color_col] == cls]
        color = CLASS_COLORS.get(str(cls),
                                  CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)])
        fig.add_trace(go.Scatter(
            x=sub["pca_x"], y=sub["pca_y"],
            mode="markers", name=str(cls),
            marker=dict(color=color, size=6, opacity=0.75),
        ))

    ev = df["explained_variance"].iloc[0] if "explained_variance" in df.columns else 0
    fig.update_layout(
        **_base_layout(f"PCA Projection (variance explained: {ev:.1%})", height=400),
        xaxis_title="PC 1", yaxis_title="PC 2",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  CASE BOARD CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def anomaly_radar(df: pd.DataFrame) -> go.Figure:
    """
    Radar chart comparing average feature values for each signal class.
    Helps players spot how Anomaly signals differ.
    """
    features = ["frequency_mhz", "amplitude_db", "duration_sec",
                 "pulse_rate_hz", "noise_ratio"]

    # Normalise each feature to 0-1 for display
    df2 = df.copy()
    for f in features:
        mn, mx = df2[f].min(), df2[f].max()
        df2[f + "_norm"] = (df2[f] - mn) / (mx - mn + 1e-9)

    norm_feats = [f + "_norm" for f in features]
    means = df2.groupby("signal_class")[norm_feats].mean()

    fig = go.Figure()
    for cls in means.index:
        vals  = means.loc[cls].tolist()
        vals += vals[:1]   # close the polygon
        cats  = features + [features[0]]
        color = CLASS_COLORS.get(cls, "#aaa")

        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats,
            fill="toself", name=cls,
            line_color=color,
            fillcolor=color.replace(")", ",0.15)").replace("rgb(", "rgba("),
            opacity=0.8,
        ))

    fig.update_layout(
        **_base_layout("Signal Profile Radar — Class Comparison", height=440),
        polar=dict(
            bgcolor=PANEL_BG,
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=GRID_COLOR),
            angularaxis=dict(gridcolor=GRID_COLOR),
        ),
    )
    return fig


def mystery_score_gauge(score: float) -> go.Figure:
    """Gauge chart showing how 'anomalous' the anomaly cluster is."""
    color = "#ff4444" if score > 0.7 else ("#ffaa00" if score > 0.4 else "#66bb6a")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score * 100,
        title=dict(text="Anomaly Concentration Score", font=dict(color=TEXT_COLOR)),
        delta=dict(reference=50, increasing=dict(color="#ff4444")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=TEXT_COLOR),
            bar=dict(color=color),
            bgcolor=PANEL_BG,
            borderwidth=1,
            bordercolor=GRID_COLOR,
            steps=[
                dict(range=[0,  40], color="#1a2a1a"),
                dict(range=[40, 70], color="#2a2a1a"),
                dict(range=[70,100], color="#2a1a1a"),
            ],
            threshold=dict(line=dict(color=ACCENT, width=3), value=70),
        ),
        number=dict(suffix="%", font=dict(color=TEXT_COLOR)),
    ))
    fig.update_layout(
        paper_bgcolor=DARK_BG,
        font=dict(color=TEXT_COLOR),
        height=280,
        margin=dict(l=30, r=30, t=60, b=10),
    )
    return fig
