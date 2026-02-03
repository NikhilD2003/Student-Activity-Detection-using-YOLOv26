from ultralytics import YOLO
import multiprocessing as mp
if __name__ == "__main__":
    mp.freeze_support()
    MODEL_WEIGHTS = "yolo26m.pt"
    DATA_YAML = "dataset/merged_data.yaml"

    EPOCHS = 15
    IMG_SIZE = 640
    BATCH_SIZE = 12
    DEVICE = 0

    RUN_NAME = "student_activity_8class"
    model = YOLO(MODEL_WEIGHTS)
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        name=RUN_NAME,
        pretrained=True,
        workers=8,
        cache=False,
        project="runs/detect",
        patience=15,
        save=True,
        val=True,
    )

    print("\nTraining finished successfully.")
