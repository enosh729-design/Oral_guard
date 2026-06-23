"""STEP 7: Fine-tune YOLO detector on combined dataset (V3)."""
import multiprocessing

def main():
    from ultralytics import YOLO, settings
    import mlflow

    # Disable built-in MLflow logging to prevent SQLite parenthesized metric crashes
    settings.update({"mlflow": False})
    
    print("Loading pre-trained OralGuard YOLO model...")
    model = YOLO(r"C:\Users\enosh\oralguard\src\detector\weights\oralguard_det\weights\best.pt")

    # We will use 5 epochs to ensure it finishes within context limits
    print("Starting fine-tuning on combined dataset (5 epochs)...")
    results = model.train(
        data=r"C:\Users\enosh\oralguard\data\combined\dental_combined.yaml",
        epochs=5,
        imgsz=1024,
        batch=8,
        device=0,
        lr0=0.00005,
        lrf=0.01,
        warmup_epochs=1,
        patience=15,
        save=True,
        project=r"C:\Users\enosh\oralguard\src\detector\weights",
        name="oralguard_v3",
        pretrained=True,
        exist_ok=True,
        workers=2,  # 2 workers is stable and faster
    )

    print("\n=== YOLO V3 RESULTS ===")
    metrics = results.results_dict
    print("Overall mAP50:", metrics.get("metrics/mAP50(B)", "N/A"))
    print("mAP50-95:", metrics.get("metrics/mAP50-95(B)", "N/A"))

    for key, val in sorted(metrics.items()):
        print(f"  {key}: {val}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
