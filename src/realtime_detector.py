import csv
import json
import random
import os
import time
from collections import defaultdict, deque

import joblib
import pandas as pd
from scapy.all import ICMP, IP, TCP, UDP, sniff

from config import COLUMNS, ENCODERS_PATH, MODEL_METADATA_PATH, SCALER_PATH
from logging_utils import get_logger

OUTPUT_FILE = "reports/live_detection.csv"
NSL_FEATURES = COLUMNS[:-2]
RECENT_WINDOW_SECONDS = 2
HOST_HISTORY_WINDOW_SECONDS = 60
DEMO_SLEEP_SECONDS = 1.0

logger = get_logger("ids.detector")

os.makedirs("reports", exist_ok=True)

detector_state = {
    "loaded": False,
    "error": None,
    "model": None,
    "scaler": None,
    "selector": None,
    "encoders": None,
    "threshold": None,
}

recent_packets = deque(maxlen=500)
connection_history = defaultdict(lambda: deque(maxlen=255))


def map_protocol(packet):
    if TCP in packet:
        return "tcp"
    if UDP in packet:
        return "udp"
    if ICMP in packet:
        return "icmp"
    return "tcp"


def map_service(packet):
    if TCP in packet or UDP in packet:
        port = packet[TCP].dport if TCP in packet else packet[UDP].dport

        common_ports = {
            20: "ftp_data",
            21: "ftp",
            22: "ssh",
            23: "telnet",
            25: "smtp",
            53: "domain_u",
            67: "eco_i",
            68: "eco_i",
            69: "tftp_u",
            80: "http",
            110: "pop_3",
            123: "ntp_u",
            143: "imap4",
            443: "http",
            3306: "mysql",
        }

        return common_ports.get(port, "private")

    if ICMP in packet:
        return "eco_i"

    return "private"


def map_flag(packet):
    if TCP in packet:
        flags = str(packet[TCP].flags)

        if "S" in flags and "A" not in flags:
            return "S0"
        if "S" in flags and "A" in flags:
            return "SF"
        if "R" in flags:
            return "REJ"
        if "F" in flags:
            return "SF"

    return "SF"


def safe_encode(value, encoder):
    value = str(value)

    if value not in encoder.classes_:
        value = "unknown" if "unknown" in encoder.classes_ else encoder.classes_[0]

    return encoder.transform([value])[0]


def load_detector_artifacts():
    with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    rf_metadata = metadata["models"]["Random Forest"]
    detector_state["model"] = joblib.load(rf_metadata["model_path"])
    detector_state["scaler"] = joblib.load(SCALER_PATH)
    detector_state["selector"] = joblib.load(rf_metadata["selector_path"])
    detector_state["encoders"] = joblib.load(ENCODERS_PATH)
    detector_state["threshold"] = rf_metadata["threshold"]
    detector_state["loaded"] = True
    detector_state["error"] = None
    logger.info(
        "detector artifacts loaded",
        extra={"threshold": detector_state["threshold"], "model_path": rf_metadata["model_path"]},
    )


def ensure_detector_ready():
    if not detector_state["loaded"]:
        raise RuntimeError(detector_state["error"] or "Detector artifacts are not loaded.")


def prune_recent_packets(now):
    while recent_packets and now - recent_packets[0]["time"] > RECENT_WINDOW_SECONDS:
        recent_packets.popleft()


def prune_connection_history(ip_address, now):
    history = connection_history[ip_address]
    while history and now - history[0] > HOST_HISTORY_WINDOW_SECONDS:
        history.popleft()


def extract_nsl_features(packet):
    now = time.time()

    src_ip = "0.0.0.0"
    dst_ip = "0.0.0.0"
    src_port = 0
    dst_port = 0

    if IP in packet:
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst

    if TCP in packet:
        src_port = packet[TCP].sport
        dst_port = packet[TCP].dport
    elif UDP in packet:
        src_port = packet[UDP].sport
        dst_port = packet[UDP].dport

    protocol_type = map_protocol(packet)
    service = map_service(packet)
    flag = map_flag(packet)

    packet_size = len(packet)

    recent_packets.append(
        {
            "time": now,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "service": service,
            "dst_port": dst_port,
            "flag": flag,
            "src_port": src_port,
        }
    )

    connection_history[dst_ip].append(now)
    connection_history[src_ip].append(now)
    prune_recent_packets(now)
    prune_connection_history(dst_ip, now)
    prune_connection_history(src_ip, now)

    recent_window = list(recent_packets)

    same_host_packets = [p for p in recent_window if p["dst_ip"] == dst_ip]
    same_service_packets = [p for p in recent_window if p["service"] == service]
    same_src_port_packets = [
        p for p in same_host_packets if p["src_port"] == src_port and src_port != 0
    ]
    same_service_host_packets = [
        p for p in same_host_packets if p["service"] == service
    ]

    count = len(same_host_packets)
    srv_count = len(same_service_packets)

    serror_count = len([p for p in same_host_packets if p["flag"] == "S0"])
    rerror_count = len([p for p in same_host_packets if p["flag"] == "REJ"])

    serror_rate = serror_count / count if count else 0
    rerror_rate = rerror_count / count if count else 0

    same_srv_rate = srv_count / count if count else 0
    diff_srv_rate = 1 - same_srv_rate if count else 0
    srv_diff_host_rate = (
        len([p for p in same_service_packets if p["dst_ip"] != dst_ip]) / srv_count
        if srv_count
        else 0
    )

    dst_host_count = min(len(connection_history[dst_ip]), 255)
    dst_host_srv_count = min(len(same_service_host_packets), 255)

    features = {
        "duration": 0,
        "protocol_type": protocol_type,
        "service": service,
        "flag": flag,
        "src_bytes": packet_size,
        "dst_bytes": 0,
        "land": 1 if src_ip == dst_ip and src_port == dst_port else 0,
        "wrong_fragment": 0,
        "urgent": 0,
        "hot": 0,
        "num_failed_logins": 0,
        "logged_in": 0,
        "num_compromised": 0,
        "root_shell": 0,
        "su_attempted": 0,
        "num_root": 0,
        "num_file_creations": 0,
        "num_shells": 0,
        "num_access_files": 0,
        "num_outbound_cmds": 0,
        "is_host_login": 0,
        "is_guest_login": 0,
        "count": count,
        "srv_count": srv_count,
        "serror_rate": serror_rate,
        "srv_serror_rate": serror_rate,
        "rerror_rate": rerror_rate,
        "srv_rerror_rate": rerror_rate,
        "same_srv_rate": same_srv_rate,
        "diff_srv_rate": diff_srv_rate,
        "srv_diff_host_rate": srv_diff_host_rate,
        "dst_host_count": dst_host_count,
        "dst_host_srv_count": dst_host_srv_count,
        "dst_host_same_srv_rate": same_srv_rate,
        "dst_host_diff_srv_rate": diff_srv_rate,
        "dst_host_same_src_port_rate": (
            len(same_src_port_packets) / count if count else 0
        ),
        "dst_host_srv_diff_host_rate": (
            len([p for p in same_service_host_packets if p["src_ip"] != src_ip]) / dst_host_srv_count
            if dst_host_srv_count
            else 0
        ),
        "dst_host_serror_rate": serror_rate,
        "dst_host_srv_serror_rate": serror_rate,
        "dst_host_rerror_rate": rerror_rate,
        "dst_host_srv_rerror_rate": rerror_rate,
    }

    return features, src_ip, dst_ip, src_port, dst_port, protocol_type


def save_detection(row):
    file_exists = os.path.exists(OUTPUT_FILE)

    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(
                [
                    "timestamp",
                    "src_ip",
                    "dst_ip",
                    "src_port",
                    "dst_port",
                    "protocol",
                    "prediction",
                    "risk_score",
                ]
            )

        writer.writerow(row)


def save_demo_detection():
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    protocol = random.choice(["tcp", "udp", "icmp"])
    src_ip = random.choice(["192.168.1.10", "192.168.1.15", "10.0.0.22", "172.16.0.8"])
    dst_ip = random.choice(["8.8.8.8", "1.1.1.1", "192.168.1.1", "10.0.0.1"])
    src_port = random.choice([0, 44321, 51820, 60333])
    dst_port = random.choice([53, 80, 443, 22, 3389])
    result = random.choices(["NORMAL", "ATTACK"], weights=[0.7, 0.3], k=1)[0]
    risk_score = round(random.uniform(0.05, 0.95), 4)

    if result == "ATTACK":
        logger.warning(
            f"DEMO ATTACK DETECTED | {timestamp} | "
            f"{protocol.upper()} {src_ip}:{src_port} -> {dst_ip}:{dst_port} | Risk={risk_score:.2f}"
        )
    else:
        logger.info(
            f"DEMO NORMAL | {timestamp} | "
            f"{protocol.upper()} {src_ip}:{src_port} -> {dst_ip}:{dst_port} | Risk={risk_score:.2f}"
        )

    save_detection(
        [
            timestamp,
            src_ip,
            dst_ip,
            src_port,
            dst_port,
            protocol,
            result,
            risk_score,
        ]
    )


def run_demo_mode():
    logger.warning("Npcap/WinPcap-compatible capture is unavailable. Starting detector demo mode.")
    logger.warning("Demo mode writes synthetic alerts to reports/live_detection.csv for dashboard presentations.")
    logger.warning("Press CTRL + C to stop demo mode.")

    try:
        while True:
            save_demo_detection()
            time.sleep(DEMO_SLEEP_SECONDS)
    except KeyboardInterrupt:
        logger.info("Demo mode stopped by user.")


def _predict_packet(packet):
    features, src_ip, dst_ip, src_port, dst_port, protocol = extract_nsl_features(packet)

    df = pd.DataFrame([features], columns=NSL_FEATURES)

    for col in ["protocol_type", "service", "flag"]:
        df[col] = safe_encode(df[col].iloc[0], detector_state["encoders"][col])

    scaled = detector_state["scaler"].transform(df)
    selected = detector_state["selector"].transform(scaled)

    probability = 0.0
    if hasattr(detector_state["model"], "predict_proba"):
        probability = float(detector_state["model"].predict_proba(selected)[0][1])
        prediction = int(probability >= detector_state["threshold"])
    else:
        prediction = int(detector_state["model"].predict(selected)[0])

    result = "ATTACK" if prediction == 1 else "NORMAL"

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    if result == "ATTACK":
        logger.warning(
            f"ATTACK DETECTED | {timestamp} | "
            f"{protocol.upper()} {src_ip}:{src_port} -> {dst_ip}:{dst_port} | Risk={probability:.2f}"
        )
    else:
        logger.info(
            f"NORMAL | {timestamp} | "
            f"{protocol.upper()} {src_ip}:{src_port} -> {dst_ip}:{dst_port} | Risk={probability:.2f}"
        )

    save_detection(
        [
            timestamp,
            src_ip,
            dst_ip,
            src_port,
            dst_port,
            protocol,
            result,
            round(probability, 4),
        ]
    )


def predict_packet(packet):
    try:
        ensure_detector_ready()
        _predict_packet(packet)
    except Exception as exc:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.exception(f"DETECTION ERROR | {timestamp} | {exc}")


def start_detection():
    logger.info("Starting real-time AI intrusion detection...")
    try:
        load_detector_artifacts()
    except Exception as exc:
        detector_state["loaded"] = False
        detector_state["error"] = str(exc)
        logger.exception(f"Failed to load detector artifacts: {exc}")
        return

    logger.info(f"Using Random Forest with threshold {detector_state['threshold']}")
    logger.info("Press CTRL + C to stop.")
    try:
        sniff(
            prn=predict_packet,
            store=False,
        )
    except RuntimeError as exc:
        message = str(exc)
        if "winpcap is not installed" in message.lower() or "not available at layer 2" in message.lower():
            logger.error("Npcap/WinPcap-compatible packet capture is not installed or not available.")
            logger.error("Install Npcap with WinPcap compatibility, then rerun as Administrator for live sniffing.")
            run_demo_mode()
            return
        logger.exception(f"Packet capture runtime failure: {exc}")
    except PermissionError:
        logger.error("Packet capture permission denied. Run the terminal as Administrator and ensure Npcap/libpcap is installed.")
    except OSError as exc:
        logger.exception(f"Packet capture failed: {exc}")


if __name__ == "__main__":
    start_detection()
