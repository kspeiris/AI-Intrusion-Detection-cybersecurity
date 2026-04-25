# AI-Intrusion-Detection

This project is an AI-based Intrusion Detection System using the NSL-KDD
dataset. It includes data preprocessing, model training, evaluation, and a
FastAPI app for prediction.

## Project Structure

```text
AI-Intrusion-Detection/
├── data/
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
├── models/
├── reports/
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── app.py
├── requirements.txt
└── README.md
```

## Install

```bash
pip install -r requirements.txt
```

## Run Training

```bash
python src/train.py
```

## Run Evaluation

```bash
python src/evaluate.py
```

## Run API

```bash
uvicorn src.app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Important Note

The API expects scaled numeric features. For a better production API, the
scaler and encoders should also be loaded so raw NSL-KDD style input can be
accepted directly.
