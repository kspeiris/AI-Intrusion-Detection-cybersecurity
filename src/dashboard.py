import os

import pandas as pd
import plotly.express as px
import streamlit as st

DATA_FILE = "reports/live_packets.csv"

def main():
    st.set_page_config(
        page_title="AI Intrusion Detection Dashboard",
        layout="wide",
    )

    st.title("AI-Based Intrusion Detection Dashboard")
    st.caption("Real-time network traffic monitoring using Scapy + Streamlit")

    if not os.path.exists(DATA_FILE):
        st.warning("No live packet data found yet. Run realtime_capture.py first.")
        st.stop()

    df = pd.read_csv(DATA_FILE)

    if df.empty:
        st.warning("Packet file exists, but it is empty.")
        st.stop()

    st.sidebar.header("Dashboard Settings")
    refresh = st.sidebar.checkbox("Auto refresh", value=True)
    max_rows = st.sidebar.slider("Rows to display", 10, 500, 100)

    if refresh:
        st.sidebar.info("Refresh the browser page to update latest packets.")

    total_packets = len(df)
    tcp_packets = len(df[df["protocol"] == "TCP"])
    udp_packets = len(df[df["protocol"] == "UDP"])
    icmp_packets = len(df[df["protocol"] == "ICMP"])

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Packets", total_packets)
    col2.metric("TCP Packets", tcp_packets)
    col3.metric("UDP Packets", udp_packets)
    col4.metric("ICMP Packets", icmp_packets)

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Protocol Distribution")
        protocol_counts = df["protocol"].value_counts().reset_index()
        protocol_counts.columns = ["Protocol", "Count"]

        fig = px.pie(
            protocol_counts,
            names="Protocol",
            values="Count",
            title="Protocol Usage",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Packet Size Over Time")
        fig = px.line(
            df.tail(max_rows),
            x="timestamp",
            y="packet_size",
            title="Packet Size Trend",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Source IP Addresses")
    top_sources = df["src_ip"].value_counts().head(10).reset_index()
    top_sources.columns = ["Source IP", "Packets"]

    fig = px.bar(
        top_sources,
        x="Source IP",
        y="Packets",
        title="Top 10 Source IPs",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Destination Ports")
    top_ports = df["dst_port"].value_counts().head(10).reset_index()
    top_ports.columns = ["Destination Port", "Packets"]

    fig = px.bar(
        top_ports,
        x="Destination Port",
        y="Packets",
        title="Top 10 Destination Ports",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Live Packet Table")
    st.dataframe(df.tail(max_rows), use_container_width=True)


if __name__ == "__main__":
    main()
