from ultralytics import YOLO
import multiprocessing as mp

if __name__ == "__main__":
    mp.freeze_support()

    MODEL_PATH = "runs/detect/weights/best.pt"
    DATA_YAML = "dataset/merged_data.yaml"

    model = YOLO(MODEL_PATH)

    model.val(
        data=DATA_YAML,
        split="test",
        imgsz=640,
        device=0,
        workers=8,
    )
