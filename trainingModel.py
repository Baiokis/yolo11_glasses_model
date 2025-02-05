from ultralytics import YOLO
from pathlib import Path

def main():
    model = YOLO("yolo11n.yaml")

    HYPERPARAMETERS = {
        "epochs": 1000,
        "batch_size": 8,
        "imgsz": 768,
        "optimizer": "SGD",
        "device": "0",
        "workers": 12,
        "label_smoothing": 0.5,
        "patience": 100
    }

    results = model.train(
        data="data.yaml",
        epochs=HYPERPARAMETERS["epochs"],
        batch=HYPERPARAMETERS["batch_size"],
        imgsz=HYPERPARAMETERS["imgsz"],
        optimizer=HYPERPARAMETERS["optimizer"],
        device=HYPERPARAMETERS["device"],
        workers=HYPERPARAMETERS["workers"],
        label_smoothing=HYPERPARAMETERS["label_smoothing"],
        pretrained=False,
        patience=HYPERPARAMETERS["patience"]
    )
    metrics = model.val(data="data.yaml", device=0, batch=64)
    print(metrics)

if __name__ == "__main__":
    main()
