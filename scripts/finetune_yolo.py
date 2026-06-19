"""STEP 7: Fine-tune YOLO detector on combined dataset."""
import multiprocessing

def main():
    from ultralytics import YOLO, settings
    import mlflow

    # Disable built-in MLflow logging to prevent SQLite parenthesized metric crashes
    settings.update({"mlflow": False})
    
    print("Loading pre-trained OralGuard YOLO model...")
    model = YOLO(r"C:\Users\enosh\oralguard\src\detector\weights\oralguard_det\weights\best.pt")

    print("Starting fine-tuning on combined dataset (10 epochs for faster execution)...")
    results = model.train(
        data=r"C:\Users\enosh\oralguard\data\combined\dental_combined.yaml",
        epochs=10,
        imgsz=1024,
        batch=8,
        device=0,
        lr0=0.0001,
        lrf=0.01,
        warmup_epochs=3,
        patience=10,
        save=True,
        project=r"C:\Users\enosh\oralguard\src\detector\weights",
        name="oralguard_finetuned",
        pretrained=True,
        exist_ok=True,
        verbose=True,
        workers=0,  # Safest for Windows multiprocessing
    )

    print("\n=== Fine-tuning complete ===")
    metrics = results.results_dict
    print("Overall mAP50:", metrics.get("metrics/mAP50(B)", "N/A"))
    print("Overall mAP50-95:", metrics.get("metrics/mAP50-95(B)", "N/A"))

    for key, val in sorted(metrics.items()):
        print(f"  {key}: {val}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
