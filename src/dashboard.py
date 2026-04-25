import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_FILE = "reports/live_packets.csv"
MODEL_COMPARISON_FILE = "reports/model_comparison.csv"
TUNING_RESULTS_FILE = "reports/tuning_results.csv"


def inject_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(13, 148, 136, 0.14), transparent 30%),
                radial-gradient(circle at top right, rgba(59, 130, 246, 0.14), transparent 28%),
                linear-gradient(180deg, #06131a 0%, #0b1720 55%, #0f1e28 100%);
            color: #e6f1f5;
        }
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2rem;
        }
        .hero-card, .panel-card {
            background: rgba(10, 22, 29, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.16);
            box-shadow: 0 24px 70px rgba(0, 0, 0, 0.25);
            backdrop-filter: blur(12px);
            border-radius: 24px;
            padding: 1.15rem 1.25rem;
        }
        .hero-title {
            font-size: 2.15rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin-bottom: 0.2rem;
        }
        .hero-subtitle {
            color: #9fb5c0;
            font-size: 0.98rem;
            margin-bottom: 0;
        }
        .metric-label {
            color: #8aa3b0;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.72rem;
        }
        .metric-value {
            font-size: 1.75rem;
            font-weight: 800;
            margin-top: 0.18rem;
        }
        .metric-delta {
            color: #7dd3fc;
            font-size: 0.88rem;
            margin-top: 0.15rem;
        }
        .insight-good {
            border-left: 4px solid #14b8a6;
            padding-left: 0.85rem;
            margin-bottom: 0.85rem;
        }
        .insight-warn {
            border-left: 4px solid #f59e0b;
            padding-left: 0.85rem;
            margin-bottom: 0.85rem;
        }
        .insight-info {
            border-left: 4px solid #60a5fa;
            padding-left: 0.85rem;
            margin-bottom: 0.85rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 1.35rem;
            overflow-x: auto;
            overflow-y: hidden;
            flex-wrap: nowrap;
            padding: 0 0.35rem 0.25rem;
            scrollbar-width: none;
        }
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
            display: none;
        }
        .stTabs [data-baseweb="tab"] {
            flex: 0 0 auto;
            width: auto;
            min-width: max-content;
            height: auto;
            padding: 0.8rem 0 0.95rem;
            margin: 0;
            background: transparent;
        }
        .stTabs [data-baseweb="tab"] p {
            font-size: 1rem;
            line-height: 1.2;
            white-space: nowrap;
        }
        @media (max-width: 900px) {
            .stTabs [data-baseweb="tab-list"] {
                gap: 0.9rem;
            }
            .stTabs [data-baseweb="tab"] p {
                font-size: 0.92rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_dataset():
    if not os.path.exists(DATA_FILE):
        st.warning("No live packet data found yet. Run realtime_capture.py first.")
        st.stop()

    df = pd.read_csv(DATA_FILE)
    if df.empty:
        st.warning("Packet file exists, but it is empty.")
        st.stop()

    numeric_columns = ["src_port", "dst_port", "packet_size", "ttl"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    if df.empty:
        st.warning("Packet log has no valid timestamps to visualize.")
        st.stop()

    df["protocol"] = df["protocol"].fillna("OTHER").astype(str).str.upper()
    df["flags"] = df["flags"].fillna("N/A").astype(str)
    df["src_ip"] = df["src_ip"].fillna("Unknown").astype(str)
    df["dst_ip"] = df["dst_ip"].fillna("Unknown").astype(str)
    df["hour"] = df["timestamp"].dt.strftime("%H:00")
    df["minute_bucket"] = df["timestamp"].dt.floor("min")
    return df.sort_values("timestamp")


def build_model_snapshot():
    comparison = None
    tuning = None

    if os.path.exists(MODEL_COMPARISON_FILE):
        comparison = pd.read_csv(MODEL_COMPARISON_FILE)
    if os.path.exists(TUNING_RESULTS_FILE):
        tuning = pd.read_csv(TUNING_RESULTS_FILE)

    return comparison, tuning


def metric_card(label, value, delta_text=None):
    delta_html = f'<div class="metric-delta">{delta_text}</div>' if delta_text else ""
    st.markdown(
        f"""
        <div class="panel-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_controls(df):
    st.sidebar.header("Dashboard Controls")
    if st.sidebar.button("Refresh now", use_container_width=True):
        st.rerun()

    max_rows = st.sidebar.slider("Rows to inspect", 50, 2000, 300, step=50)
    protocol_options = sorted(df["protocol"].dropna().unique().tolist())
    selected_protocols = st.sidebar.multiselect(
        "Protocol filter",
        options=protocol_options,
        default=protocol_options,
    )
    source_query = st.sidebar.text_input("Source IP contains", value="").strip()

    latest_timestamp = df["timestamp"].max()
    st.sidebar.caption(
        f"Last packet timestamp: {latest_timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    filtered = df[df["protocol"].isin(selected_protocols)].copy()
    if source_query:
        filtered = filtered[
            filtered["src_ip"].str.contains(source_query, case=False, na=False)
        ]

    if filtered.empty:
        st.warning("No rows match the current filters.")
        st.stop()

    return filtered.tail(max_rows)


def packet_insights(df):
    protocol_share = df["protocol"].value_counts(normalize=True).mul(100)
    dominant_protocol = protocol_share.index[0]
    dominant_share = protocol_share.iloc[0]
    avg_size = df["packet_size"].mean()
    peak_source = df["src_ip"].value_counts().idxmax()
    top_port = df["dst_port"].astype(int).value_counts().idxmax()

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("Operational Analysis")
    st.markdown(
        f'<div class="insight-info"><strong>Traffic profile</strong><br>{dominant_protocol} leads the current stream at {dominant_share:.1f}% of captured packets.</div>',
        unsafe_allow_html=True,
    )
    severity_class = "insight-warn" if avg_size > 700 else "insight-good"
    st.markdown(
        f'<div class="{severity_class}"><strong>Packet sizing</strong><br>Average packet size is {avg_size:.0f} bytes in the active viewing window.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="insight-info"><strong>Top talker</strong><br>{peak_source} is the most active source, and destination port {top_port} is the busiest service target.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def overview_tab(df):
    latest_timestamp = df["timestamp"].max()
    avg_packet_size = df["packet_size"].mean()
    unique_sources = df["src_ip"].nunique()
    unique_destinations = df["dst_ip"].nunique()
    top_source = df["src_ip"].value_counts().idxmax()
    top_source_count = df["src_ip"].value_counts().iloc[0]

    hero_left, hero_right = st.columns([1.6, 1])
    with hero_left:
        st.markdown(
            """
            <div class="hero-card">
                <div class="hero-title">Network Telemetry Command Center</div>
                <p class="hero-subtitle">
                    Review packet health, endpoint concentration, and protocol behavior from a cleaner operational cockpit.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with hero_right:
        st.markdown(
            f"""
            <div class="hero-card">
                <div class="metric-label">Latest packet seen</div>
                <div class="metric-value" style="font-size:1.3rem">
                    {latest_timestamp.strftime("%Y-%m-%d %H:%M:%S")}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("Total Packets", f"{len(df):,}")
    with m2:
        metric_card("Average Size", f"{avg_packet_size:.0f} bytes")
    with m3:
        metric_card("Unique Sources", f"{unique_sources}")
    with m4:
        metric_card("Unique Destinations", f"{unique_destinations}", f"Top source: {top_source} ({top_source_count})")

    left, right = st.columns([1.15, 0.85])
    with left:
        protocol_counts = df["protocol"].value_counts().reset_index()
        protocol_counts.columns = ["Protocol", "Count"]
        fig = px.pie(
            protocol_counts,
            names="Protocol",
            values="Count",
            hole=0.55,
            color="Protocol",
            color_discrete_sequence=["#14b8a6", "#38bdf8", "#f59e0b", "#ef4444"],
            title="Protocol Mix",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        packet_insights(df)

    c1, c2 = st.columns(2)
    with c1:
        trend = (
            df.groupby("minute_bucket", as_index=False)
            .agg(
                packet_count=("packet_size", "size"),
                avg_packet_size=("packet_size", "mean"),
            )
            .sort_values("minute_bucket")
        )
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=trend["minute_bucket"],
                y=trend["packet_count"],
                name="Packets",
                marker_color="#0ea5e9",
                opacity=0.75,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=trend["minute_bucket"],
                y=trend["avg_packet_size"],
                name="Avg packet size",
                line=dict(color="#14b8a6", width=3),
                yaxis="y2",
            )
        )
        fig.update_layout(
            title="Traffic Volume and Packet Size",
            yaxis=dict(title="Packets"),
            yaxis2=dict(title="Avg size", overlaying="y", side="right"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Minute",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        top_pairs = (
            df.assign(flow=df["src_ip"] + " -> " + df["dst_ip"])
            .groupby("flow", as_index=False)
            .agg(
                packet_count=("packet_size", "size"),
                avg_packet_size=("packet_size", "mean"),
            )
            .sort_values("packet_count", ascending=False)
            .head(10)
            .sort_values("packet_count", ascending=True)
        )
        fig = px.bar(
            top_pairs,
            x="packet_count",
            y="flow",
            orientation="h",
            color="avg_packet_size",
            color_continuous_scale="Tealgrn",
            title="Top Source to Destination Flows",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)


def protocol_tab(df):
    left, right = st.columns(2)
    with left:
        flags = df["flags"].value_counts().head(12).reset_index()
        flags.columns = ["Flag", "Count"]
        fig = px.bar(
            flags,
            x="Flag",
            y="Count",
            color="Count",
            color_continuous_scale="Blues",
            title="Transport Flag Distribution",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.histogram(
            df,
            x="ttl",
            nbins=20,
            color="protocol",
            barmode="overlay",
            color_discrete_sequence=["#14b8a6", "#38bdf8", "#f59e0b", "#ef4444"],
            title="TTL Distribution by Protocol",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    pivot = (
        df.pivot_table(
            index="hour",
            columns="protocol",
            values="packet_size",
            aggfunc="count",
            fill_value=0,
        )
        .sort_index()
    )
    heatmap = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="Teal",
        )
    )
    heatmap.update_layout(
        title="Protocol Activity Heatmap by Hour",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(heatmap, use_container_width=True)


def endpoints_tab(df):
    left, right = st.columns(2)
    with left:
        top_sources = df["src_ip"].value_counts().head(12).reset_index()
        top_sources.columns = ["Source IP", "Packets"]
        fig = px.bar(
            top_sources.sort_values("Packets"),
            x="Packets",
            y="Source IP",
            orientation="h",
            color="Packets",
            color_continuous_scale="Mint",
            title="Top Source IP Addresses",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        top_ports = df["dst_port"].astype(int).astype(str).value_counts().head(12).reset_index()
        top_ports.columns = ["Destination Port", "Packets"]
        fig = px.treemap(
            top_ports,
            path=["Destination Port"],
            values="Packets",
            color="Packets",
            color_continuous_scale="Bluyl",
            title="Destination Port Utilization",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    table = (
        df.groupby(["src_ip", "dst_ip", "protocol"], as_index=False)
        .agg(packet_count=("packet_size", "size"), avg_size=("packet_size", "mean"))
        .sort_values("packet_count", ascending=False)
        .head(25)
    )
    st.subheader("High-Volume Endpoint Pairs")
    st.dataframe(table, use_container_width=True, hide_index=True)


def model_analysis_tab(comparison, tuning):
    if comparison is None:
        st.info("Model analysis files are not available yet. Run training first.")
        return

    left, right = st.columns([1.1, 0.9])
    with left:
        melted = comparison.melt(
            id_vars=["Model"],
            value_vars=["Accuracy", "Precision", "Recall", "F1 Score"],
            var_name="Metric",
            value_name="Score",
        )
        fig = px.bar(
            melted,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            color_discrete_sequence=["#14b8a6", "#38bdf8", "#f59e0b"],
            title="Current Model Benchmark",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        leader = comparison.sort_values("F1 Score", ascending=False).iloc[0]
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("Deployment Recommendation")
        st.markdown(
            f"""
            <div class="insight-good"><strong>Champion model</strong><br>{leader['Model']} currently leads with F1 {leader['F1 Score']:.3f} and recall {leader['Recall']:.3f}.</div>
            <div class="insight-info"><strong>Threshold</strong><br>Operational threshold is {leader['Threshold']:.2f}, tuned for IDS-style recall sensitivity.</div>
            <div class="insight-warn"><strong>Operator note</strong><br>Monitor recall and F1 first. Missed attacks are more costly than isolated false positives.</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if tuning is not None and not tuning.empty:
        top_tuning = tuning.head(15).copy()
        fig = px.scatter(
            top_tuning,
            x="Recall",
            y="F1 Score",
            color="Model",
            size="Accuracy",
            hover_data=["k", "threshold"],
            color_discrete_sequence=["#14b8a6", "#38bdf8", "#f59e0b"],
            title="Top Tuning Candidates",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top Cross-Validation Candidates")
        st.dataframe(top_tuning, use_container_width=True, hide_index=True)


def main():
    st.set_page_config(
        page_title="AI Intrusion Detection Dashboard",
        layout="wide",
    )
    inject_styles()

    raw_df = load_dataset()
    comparison, tuning = build_model_snapshot()
    filtered = sidebar_controls(raw_df)

    tabs = st.tabs(
        [
            "Executive Overview",
            "Protocol & Payloads",
            "Endpoints & Ports",
            "Model Analysis",
        ]
    )

    with tabs[0]:
        overview_tab(filtered)
    with tabs[1]:
        protocol_tab(filtered)
    with tabs[2]:
        endpoints_tab(filtered)
    with tabs[3]:
        model_analysis_tab(comparison, tuning)

    st.subheader("Raw Packet Log")
    export_df = filtered.sort_values("timestamp", ascending=False).copy()
    export_df["timestamp"] = export_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    st.download_button(
        "Download filtered packet log",
        export_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_live_packets.csv",
        mime="text/csv",
    )
    st.dataframe(export_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
