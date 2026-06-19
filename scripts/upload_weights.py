"""STEP 11: Upload finetuned weights to Hugging Face."""
from huggingface_hub import HfApi, login
import os

def main():
    import os
    token = os.environ.get('HF_TOKEN')
    if token:
        print("Logging into Hugging Face using environment token...")
        login(token=token)
    else:
        print("Hugging Face token not found in env. Attempting login using cached token...")
        try:
            login()
        except Exception as e:
            print("Cached login failed. Please set HF_TOKEN environment variable.")
            raise e
    
    api = HfApi()

    print("Uploading finetuned YOLO weights...")
    api.upload_file(
        path_or_fileobj=r"C:\Users\enosh\oralguard\src\detector\weights\oralguard_finetuned\weights\best.pt",
        path_in_repo="oralguard_det_finetuned.pt",
        repo_id="Enosh729/oralguard",
        repo_type="model"
    )
    print("Finetuned YOLO detector uploaded successfully.")

    print("Uploading finetuned classifier weights...")
    api.upload_file(
        path_or_fileobj=r"C:\Users\enosh\oralguard\src\classifier\checkpoints\best.pt",
        path_in_repo="classifier_finetuned.pt",
        repo_id="Enosh729/oralguard",
        repo_type="model"
    )
    print("Finetuned classifier uploaded successfully.")
    
    print("Both weights live at: https://huggingface.co/Enosh729/oralguard")

if __name__ == "__main__":
    main()
