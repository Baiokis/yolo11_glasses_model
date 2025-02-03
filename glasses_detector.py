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

#validar o modelo
def validate_images():#
    model = YOLO("modelos/last.pt")
    
    image_folder = Path("fotos")
    
    if not image_folder.exists() or not image_folder.is_dir():
        print(f"A pasta {image_folder} não existe ou não é um diretório válido.")
        return

    image_files = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png"))
    
    if not image_files:
        print(f"Nenhuma imagem encontrada na pasta {image_folder}.")
        return

    for image_path in image_files:
        print(f"Validando imagem: {image_path}")
        results = model(image_path)
        
        for result in results:
            result.show()
            print(result)

if __name__ == "__main__":
    main()
