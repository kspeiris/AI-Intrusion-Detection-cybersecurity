# AI-Intrusion-Detection

This project is an AI-based Intrusion Detection System using the NSL-KDD
dataset. It includes preprocessing, model training, evaluation, API-based
prediction, real-time packet capture, and a live dashboard.

## Project Structure

```text
AI-Intrusion-Detection/
├── data/
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
├── models/
├── reports/
├── src/
│   ├── app.py
│   ├── config.py
│   ├── dashboard.py
│   ├── evaluate.py
│   ├── preprocess.py
│   ├── realtime_capture.py
│   ├── train.py
│   └── visualize.py
├── requirements.txt
└── README.md
```

## Install

```bash
pip install -r requirements.txt
```

## Run Tests

```bash
python -m pytest tests/test_api.py
```

## Run Training

```bash
python src/train.py
```

Training writes model artifacts into `models/` and report files into `reports/`.

## Run Evaluation

```bash
python src/evaluate.py
```

## Run API

```bash
uvicorn src.app:app --reload
```

Open `http://127.0.0.1:8000/docs`.

## Run Real-Time Packet Capture

```bash
python src/realtime_capture.py
```

This writes live packet rows to `reports/live_packets.csv`.

On Windows, run VS Code as Administrator if packet capture permissions fail.

## Run Dashboard

```bash
streamlit run src/dashboard.py
```

Open `http://localhost:8501`.

## Pinned Windows Run Scripts

These PowerShell launchers use the exact Python interpreter configured in this
workspace:

```powershell
.\scripts\run_detector.ps1
.\scripts\run_alert_dashboard.ps1
```

If Npcap is missing, `run_detector.ps1` will now fall back to a demo mode that
writes synthetic alerts into `reports/live_detection.csv` so the alert dashboard
can still be presented.

## Important Note

The trained NSL-KDD models do not directly classify raw Scapy packets because
the dataset uses engineered features rather than raw packet fields.

Recommended project wording:

> The trained ML model detects intrusions using NSL-KDD features. The real-time
> module captures live packets and visualizes traffic. Future work will map live
> packet statistics into NSL-KDD-style features for real-time classification.
