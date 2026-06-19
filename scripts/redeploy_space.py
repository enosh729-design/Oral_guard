"""STEP 12: Redeploy the Streamlit app to Hugging Face Space."""
from huggingface_hub import HfApi, login

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
    print("Uploading updated app.py to Hugging Face Space...")
    api.upload_file(
        path_or_fileobj=r"C:\Users\enosh\oralguard\app.py",
        path_in_repo="app.py",
        repo_id="Enosh729/oralguard-demo",
        repo_type="space"
    )
    print("Updated app deployed successfully.")
    print("Live: https://huggingface.co/spaces/Enosh729/oralguard-demo")

if __name__ == "__main__":
    main()
