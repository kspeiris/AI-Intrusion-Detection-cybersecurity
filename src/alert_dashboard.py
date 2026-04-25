import os

import pandas as pd
import plotly.express as px
import streamlit as st

DATA_FILE = "reports/live_detection.csv"


def main():
    st.set_page_config(
        page_title="Live IDS Alert Dashboard",
        layout="wide",
    )

    st.title("Live AI Intrusion Detection Dashboard")

    if not os.path.exists(DATA_FILE):
        st.warning("No detection data found. Run realtime_detector.py first.")
        st.stop()

    df = pd.read_csv(DATA_FILE)

    if df.empty:
        st.warning("Detection file is empty.")
        st.stop()

    total = len(df)
    attacks = len(df[df["prediction"] == "ATTACK"])
    normal = len(df[df["prediction"] == "NORMAL"])

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Packets", total)
    col2.metric("Normal Traffic", normal)
    col3.metric("Detected Attacks", attacks)

    st.divider()

    recent_attacks = df[df["prediction"] == "ATTACK"].tail(10)

    if attacks > 0:
        st.error("ATTACK DETECTED")
        st.subheader("Recent Attack Alerts")
        st.dataframe(recent_attacks, use_container_width=True)
    else:
        st.success("No attacks detected yet.")

    left, right = st.columns(2)

    with left:
        st.subheader("Traffic Classification")
        fig = px.pie(
            df,
            names="prediction",
            title="Normal vs Attack",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Risk Score Over Time")
        fig = px.line(
            df.tail(200),
            x="timestamp",
            y="risk_score",
            color="prediction",
            title="Live Risk Score",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Suspicious Source IPs")
    attack_df = df[df["prediction"] == "ATTACK"]

    if not attack_df.empty:
        top_attackers = attack_df["src_ip"].value_counts().head(10).reset_index()
        top_attackers.columns = ["Source IP", "Attack Count"]

        fig = px.bar(
            top_attackers,
            x="Source IP",
            y="Attack Count",
            title="Top Attack Sources",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No suspicious source IPs yet.")

    st.subheader("Live Detection Logs")
    st.dataframe(df.tail(200), use_container_width=True)


if __name__ == "__main__":
    main()
