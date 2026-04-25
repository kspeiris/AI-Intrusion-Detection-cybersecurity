import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_FILE = "reports/live_detection.csv"
MODEL_METADATA_FILE = "models/model_metadata.json"
MODEL_COMPARISON_FILE = "reports/model_comparison.csv"


def inject_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(239, 68, 68, 0.14), transparent 28%),
                radial-gradient(circle at top right, rgba(245, 158, 11, 0.12), transparent 24%),
                linear-gradient(180deg, #12090b 0%, #1c0f13 45%, #21161c 100%);
            color: #fff1f2;
        }
        .block-container {
            padding-top: 3.4rem;
            padding-bottom: 2rem;
        }
        .hero-card, .panel-card {
            background: rgba(27, 15, 20, 0.76);
            border: 1px solid rgba(251, 191, 36, 0.18);
            box-shadow: 0 24px 70px rgba(0, 0, 0, 0.28);
            backdrop-filter: blur(14px);
            border-radius: 24px;
            padding: 1.15rem 1.25rem;
        }
        .hero-title {
            font-size: 2.05rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin-bottom: 0.25rem;
        }
        .hero-subtitle {
            color: #fbcfe8;
            opacity: 0.78;
        }
        .metric-label {
            color: #fdba74;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.72rem;
        }
        .metric-value {
            font-size: 1.75rem;
            font-weight: 800;
            margin-top: 0.2rem;
        }
        .metric-delta {
            color: #fecdd3;
            font-size: 0.88rem;
            margin-top: 0.15rem;
        }
        .insight-critical {
            border-left: 4px solid #ef4444;
            padding-left: 0.85rem;
            margin-bottom: 0.9rem;
        }
        .insight-watch {
            border-left: 4px solid #f59e0b;
            padding-left: 0.85rem;
            margin-bottom: 0.9rem;
        }
        .insight-ok {
            border-left: 4px solid #22c55e;
            padding-left: 0.85rem;
            margin-bottom: 0.9rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 1.35rem;
            overflow-x: auto;
            overflow-y: visible;
            flex-wrap: nowrap;
            padding: 0.45rem 0.35rem 0.25rem;
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


def load_detections():
    if not os.path.exists(DATA_FILE):
        st.warning("No detection data found. Run realtime_detector.py first.")
        st.stop()

    df = pd.read_csv(DATA_FILE)
    if df.empty:
        st.warning("Detection file is empty.")
        st.stop()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    if df.empty:
        st.warning("Detection log has no valid timestamps to visualize.")
        st.stop()

    df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0)
    df["src_port"] = pd.to_numeric(df["src_port"], errors="coerce").fillna(0).astype(int)
    df["dst_port"] = pd.to_numeric(df["dst_port"], errors="coerce").fillna(0).astype(int)
    df["prediction"] = df["prediction"].fillna("UNKNOWN").astype(str).str.upper()
    df["protocol"] = df["protocol"].fillna("UNKNOWN").astype(str).str.upper()
    df["src_ip"] = df["src_ip"].fillna("Unknown").astype(str)
    df["dst_ip"] = df["dst_ip"].fillna("Unknown").astype(str)
    df["hour"] = df["timestamp"].dt.strftime("%H:00")
    df["minute_bucket"] = df["timestamp"].dt.floor("min")
    df["risk_band"] = pd.cut(
        df["risk_score"],
        bins=[0, 0.35, 0.6, 0.8, 1.0],
        labels=["Low", "Guarded", "Elevated", "Critical"],
        include_lowest=True,
    )
    df["risk_mismatch"] = (
        ((df["prediction"] == "NORMAL") & (df["risk_score"] >= 0.6))
        | ((df["prediction"] == "ATTACK") & (df["risk_score"] < 0.35))
    )
    return df.sort_values("timestamp")


def load_model_context():
    metadata = None
    comparison = None
    if os.path.exists(MODEL_METADATA_FILE):
        with open(MODEL_METADATA_FILE, "r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
    if os.path.exists(MODEL_COMPARISON_FILE):
        comparison = pd.read_csv(MODEL_COMPARISON_FILE)
    return metadata, comparison


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
    st.sidebar.header("Alert Controls")
    if st.sidebar.button("Refresh now", use_container_width=True):
        st.rerun()

    window = st.sidebar.slider("Rows to inspect", 100, 5000, 500, step=100)
    prediction_filter = st.sidebar.multiselect(
        "Prediction filter",
        options=sorted(df["prediction"].unique().tolist()),
        default=sorted(df["prediction"].unique().tolist()),
    )
    protocol_filter = st.sidebar.multiselect(
        "Protocol filter",
        options=sorted(df["protocol"].unique().tolist()),
        default=sorted(df["protocol"].unique().tolist()),
    )
    min_risk = st.sidebar.slider("Minimum risk score", 0.0, 1.0, 0.0, 0.05)

    latest_timestamp = df["timestamp"].max()
    st.sidebar.caption(
        f"Last detection timestamp: {latest_timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    filtered = df[
        df["prediction"].isin(prediction_filter)
        & df["protocol"].isin(protocol_filter)
        & (df["risk_score"] >= min_risk)
    ].copy()
    if filtered.empty:
        st.warning("No rows match the current filters.")
        st.stop()

    return filtered.tail(window)


def hero(metadata, attacks, total):
    attack_rate = (attacks / total * 100) if total else 0
    model_name = metadata["best_model_name"] if metadata else "Unavailable"
    threshold = metadata["best_threshold"] if metadata else "N/A"

    left, right = st.columns([1.55, 1])
    with left:
        st.markdown(
            """
            <div class="hero-card">
                <div class="hero-title">Alert Monitoring</div>
                <p class="hero-subtitle">
                    Review live model results, suspicious sources, and risk trends.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"""
            <div class="hero-card">
                <div class="metric-label">Detection engine</div>
                <div class="metric-value" style="font-size:1.15rem">{model_name}</div>
                <div style="color:#fecdd3">Threshold: {threshold}</div>
                <div style="color:#fdba74;margin-top:0.35rem">Attack rate: {attack_rate:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def overview_tab(df, metadata):
    total = len(df)
    attacks = len(df[df["prediction"] == "ATTACK"])
    avg_risk = df["risk_score"].mean()
    critical = len(df[df["risk_score"] >= 0.8])
    mismatches = int(df["risk_mismatch"].sum())

    hero(metadata, attacks, total)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Decisions", f"{total:,}")
    with c2:
        metric_card("Attack Alerts", f"{attacks:,}")
    with c3:
        metric_card("Average Risk", f"{avg_risk:.2f}")
    with c4:
        metric_card("Critical Events", f"{critical:,}", f"Risk mismatches: {mismatches}")

    recent_attacks = df[df["prediction"] == "ATTACK"].sort_values(
        ["risk_score", "timestamp"], ascending=[False, False]
    ).head(10)

    left, right = st.columns([1.1, 0.9])
    with left:
        if attacks > 0:
            st.error("Active attack decisions detected in the current viewing window.")
            st.subheader("Highest-Risk Attack Alerts")
            st.dataframe(recent_attacks, use_container_width=True, hide_index=True)
        else:
            st.success("No attacks detected in the current viewing window.")

    with right:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("Analyst Summary")
        dominant_source = (
            df[df["prediction"] == "ATTACK"]["src_ip"].value_counts().idxmax()
            if attacks > 0
            else "None"
        )
        threat_markup = (
            f'<div class="insight-critical"><strong>Threat ratio</strong><br>{(attacks / total * 100):.1f}% of scored packets are currently classified as attacks.</div>'
            if attacks > 0
            else '<div class="insight-ok"><strong>Threat ratio</strong><br>No attack-classified packets are present in the current log.</div>'
        )
        st.markdown(threat_markup, unsafe_allow_html=True)
        st.markdown(
            f'<div class="insight-watch"><strong>Average model confidence</strong><br>Mean risk score is {avg_risk:.2f}, with {critical} events in the critical band.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="insight-watch"><strong>Dominant suspicious source</strong><br>{dominant_source} is the most frequent attack-classified source in this window.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        pie_source = df["prediction"].value_counts().reset_index()
        pie_source.columns = ["prediction", "count"]
        fig = px.pie(
            pie_source,
            names="prediction",
            values="count",
            hole=0.58,
            color="prediction",
            color_discrete_map={"ATTACK": "#ef4444", "NORMAL": "#22c55e"},
            title="Normal vs Attack Distribution",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with g2:
        trend = (
            df.groupby(["minute_bucket", "prediction"], as_index=False)
            .agg(avg_risk=("risk_score", "mean"), event_count=("risk_score", "size"))
            .sort_values("minute_bucket")
        )
        fig = px.line(
            trend,
            x="minute_bucket",
            y="avg_risk",
            color="prediction",
            markers=True,
            color_discrete_map={"ATTACK": "#ef4444", "NORMAL": "#22c55e"},
            title="Average Risk Over Time",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)


def threat_analysis_tab(df):
    left, right = st.columns(2)
    with left:
        fig = px.histogram(
            df,
            x="risk_score",
            color="prediction",
            nbins=30,
            barmode="overlay",
            color_discrete_map={"ATTACK": "#ef4444", "NORMAL": "#22c55e"},
            title="Risk Score Distribution",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        band_counts = df["risk_band"].value_counts(dropna=False).reset_index()
        band_counts.columns = ["Risk Band", "Count"]
        fig = px.bar(
            band_counts,
            x="Risk Band",
            y="Count",
            color="Risk Band",
            color_discrete_map={
                "Low": "#22c55e",
                "Guarded": "#f59e0b",
                "Elevated": "#fb7185",
                "Critical": "#ef4444",
            },
            title="Risk Band Classification",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    attack_protocol = (
        df.groupby(["protocol", "prediction"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    fig = px.bar(
        attack_protocol,
        x="protocol",
        y="count",
        color="prediction",
        barmode="stack",
        color_discrete_map={"ATTACK": "#ef4444", "NORMAL": "#22c55e"},
        title="Protocol Composition by Classification",
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


def entity_tab(df):
    left, right = st.columns(2)
    attack_df = df[df["prediction"] == "ATTACK"]

    with left:
        if not attack_df.empty:
            top_attackers = attack_df["src_ip"].value_counts().head(12).reset_index()
            top_attackers.columns = ["Source IP", "Attack Count"]
            fig = px.bar(
                top_attackers.sort_values("Attack Count"),
                x="Attack Count",
                y="Source IP",
                orientation="h",
                color="Attack Count",
                color_continuous_scale="Reds",
                title="Top Suspicious Source IPs",
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No suspicious source IPs yet.")

    with right:
        if not attack_df.empty:
            top_ports = attack_df["dst_port"].astype(int).astype(str).value_counts().head(12).reset_index()
            top_ports.columns = ["Destination Port", "Alerts"]
            fig = px.treemap(
                top_ports,
                path=["Destination Port"],
                values="Alerts",
                color="Alerts",
                color_continuous_scale="Sunsetdark",
                title="Attack-Associated Destination Ports",
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No attack-linked destination ports yet.")

    flow_table = (
        df.groupby(["src_ip", "dst_ip", "protocol", "prediction"], as_index=False)
        .agg(events=("risk_score", "size"), avg_risk=("risk_score", "mean"))
        .sort_values(["avg_risk", "events"], ascending=[False, False])
        .head(25)
    )
    st.subheader("Most Important Detection Flows")
    st.dataframe(flow_table, use_container_width=True, hide_index=True)


def model_tab(metadata, comparison):
    if not metadata:
        st.info("Model metadata is unavailable.")
        return

    model_name = metadata["best_model_name"]
    model_details = metadata["models"][model_name]

    left, right = st.columns([1.1, 0.9])
    with left:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("Deployed Model Profile")
        st.markdown(
            f"""
            <div class="insight-ok"><strong>Active model</strong><br>{model_name} is the deployed scoring engine.</div>
            <div class="insight-watch"><strong>Selection basis</strong><br>{metadata.get('selection_basis', 'unknown')} with threshold {metadata['best_threshold']:.2f}.</div>
            <div class="insight-watch"><strong>Test F1</strong><br>{model_details['test_metrics']['F1 Score']:.3f} with recall {model_details['test_metrics']['Recall']:.3f}.</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        metrics_df = pd.DataFrame(
            {
                "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
                "Cross-Validated": [
                    model_details["cv_metrics"]["Accuracy"],
                    model_details["cv_metrics"]["Precision"],
                    model_details["cv_metrics"]["Recall"],
                    model_details["cv_metrics"]["F1 Score"],
                ],
                "Test": [
                    model_details["test_metrics"]["Accuracy"],
                    model_details["test_metrics"]["Precision"],
                    model_details["test_metrics"]["Recall"],
                    model_details["test_metrics"]["F1 Score"],
                ],
            }
        )
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Cross-Validated",
                x=metrics_df["Metric"],
                y=metrics_df["Cross-Validated"],
                marker_color="#fb7185",
            )
        )
        fig.add_trace(
            go.Bar(
                name="Test",
                x=metrics_df["Metric"],
                y=metrics_df["Test"],
                marker_color="#f59e0b",
            )
        )
        fig.update_layout(
            barmode="group",
            title="Deployed Model Performance Snapshot",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    if comparison is not None:
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
            color_discrete_sequence=["#ef4444", "#f59e0b", "#22c55e"],
            title="Model Comparison",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Selected Features")
    st.dataframe(
        pd.DataFrame({"selected_feature": model_details["selected_features"]}),
        use_container_width=True,
        hide_index=True,
    )


def main():
    st.set_page_config(
        page_title="Live IDS Alert Dashboard",
        layout="wide",
    )
    inject_styles()

    raw_df = load_detections()
    metadata, comparison = load_model_context()
    filtered = sidebar_controls(raw_df)

    tabs = st.tabs(
        [
            "Executive Overview",
            "Threat Analysis",
            "Sources & Targets",
            "Model Operations",
        ]
    )

    with tabs[0]:
        overview_tab(filtered, metadata)
    with tabs[1]:
        threat_analysis_tab(filtered)
    with tabs[2]:
        entity_tab(filtered)
    with tabs[3]:
        model_tab(metadata, comparison)

    st.subheader("Live Detection Log")
    export_df = filtered.sort_values("timestamp", ascending=False).copy()
    export_df["timestamp"] = export_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    st.download_button(
        "Download filtered detection log",
        export_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_live_detection.csv",
        mime="text/csv",
    )
    st.dataframe(export_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
