import csv
import os
import time

from scapy.all import ICMP, IP, TCP, UDP, sniff
from config import resolve_project_path

OUTPUT_FILE = "reports/live_packets.csv"

os.makedirs(resolve_project_path("reports"), exist_ok=True)

CSV_COLUMNS = [
    "timestamp",
    "src_ip",
    "dst_ip",
    "protocol",
    "src_port",
    "dst_port",
    "packet_size",
    "ttl",
    "flags",
]


def packet_to_features(packet):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    src_ip = "N/A"
    dst_ip = "N/A"
    protocol = "OTHER"
    src_port = 0
    dst_port = 0
    ttl = 0
    flags = "N/A"

    if IP in packet:
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        ttl = packet[IP].ttl

        if TCP in packet:
            protocol = "TCP"
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport
            flags = str(packet[TCP].flags)
        elif UDP in packet:
            protocol = "UDP"
            src_port = packet[UDP].sport
            dst_port = packet[UDP].dport
        elif ICMP in packet:
            protocol = "ICMP"

    packet_size = len(packet)

    return [
        timestamp,
        src_ip,
        dst_ip,
        protocol,
        src_port,
        dst_port,
        packet_size,
        ttl,
        flags,
    ]


def save_packet(row):
    output_path = resolve_project_path(OUTPUT_FILE)
    file_exists = os.path.exists(output_path)

    with open(output_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(CSV_COLUMNS)

        writer.writerow(row)


def process_packet(packet):
    row = packet_to_features(packet)
    save_packet(row)

    print(
        f"[{row[0]}] {row[3]} {row[1]}:{row[4]} -> {row[2]}:{row[5]} "
        f"size={row[6]} ttl={row[7]} flags={row[8]}"
    )


def start_capture():
    print("Starting real-time packet capture...")
    print("Press CTRL + C to stop.")
    try:
        sniff(
            prn=process_packet,
            store=False,
        )
    except RuntimeError as exc:
        message = str(exc).lower()
        if "winpcap is not installed" in message or "not available at layer 2" in message:
            print("Npcap/WinPcap-compatible packet capture is not installed or not available.")
            print("Install Npcap with WinPcap compatibility, then rerun as Administrator.")
            return
        raise
    except PermissionError:
        print("Packet capture permission denied. Run the terminal as Administrator.")
    except OSError as exc:
        print(f"Packet capture failed: {exc}")


if __name__ == "__main__":
    start_capture()
