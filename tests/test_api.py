"""
Integration test for the OralGuard FastAPI server.
Starts the server, sends a test X-ray image to /predict, and verifies the response.
"""

import os
import sys
import time
import subprocess
import requests
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST_IMAGE = ROOT / "data" / "dentex" / "images" / "test" / "train_130.png"
PORT = 8011  # Avoid port conflicts

def run_test():
    print("=== OralGuard API Integration Test ===")
    
    # 1. Start uvicorn server in background
    cmd = [
        str(ROOT / ".venv" / "Scripts" / "python"),
        "-m", "uvicorn",
        "api.main:app",
        "--host", "127.0.0.1",
        "--port", str(PORT),
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    
    print(f"Starting server on port {PORT}...")
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # 2. Wait for server to start
    health_url = f"http://127.0.0.1:{PORT}/health"
    predict_url = f"http://127.0.0.1:{PORT}/predict"
    
    success = False
    for i in range(20):  # Wait up to 20 seconds
        time.sleep(1.0)
        try:
            r = requests.get(health_url, timeout=2)
            if r.status_code == 200:
                print("Server is up and healthy!")
                print("Health status:", r.json())
                success = True
                break
        except requests.exceptions.RequestException:
            pass
            
    if not success:
        print("Error: Server failed to start or respond to /health in 20 seconds.")
        # Print server logs if failed
        stdout, stderr = proc.communicate(timeout=2)
        print("STDOUT:", stdout.decode())
        print("STDERR:", stderr.decode())
        proc.kill()
        sys.exit(1)
        
    # 3. Send inference request
    if not TEST_IMAGE.exists():
        print(f"Error: Test image not found at {TEST_IMAGE}")
        proc.kill()
        sys.exit(1)
        
    print(f"Sending test image {TEST_IMAGE.name} to /predict...")
    try:
        with open(TEST_IMAGE, "rb") as f:
            files = {"file": (TEST_IMAGE.name, f, "image/png")}
            r = requests.post(predict_url, files=files, timeout=60)
            
        if r.status_code != 200:
            print(f"Error: Predict request failed with status code {r.status_code}")
            print("Response:", r.text)
            proc.kill()
            sys.exit(1)
            
        res = r.json()
        print("\n--- Predict Response ---")
        print(f"Request ID : {res.get('request_id')}")
        print(f"Image File : {res.get('image_filename')}")
        print(f"Teeth Det  : {res.get('num_teeth_detected')}")
        print(f"Time Taken : {res.get('processing_time_ms')} ms")
        print("\nFirst 3 findings (if any):")
        for finding in res.get("findings", [])[:3]:
            print(f"  Tooth {finding.get('tooth_id')} ({finding.get('tooth_label')}):")
            print(f"    Findings   : {finding.get('findings')}")
            print(f"    Uncertain  : {finding.get('uncertain')}")
            print(f"    Confidence : {finding.get('confidence')}")
            print(f"    GradCAM    : {finding.get('gradcam_path')}")
            
        assert "request_id" in res, "Missing request_id"
        assert "findings" in res, "Missing findings"
        print("\nAssertion passed: Predict response structure is valid.")
        
    except Exception as e:
        print(f"Error during API request/assertion: {e}")
        proc.kill()
        sys.exit(1)
        
    finally:
        # 4. Shutdown server
        print("Shutting down server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            
    print("=== Integration Test Completed Successfully ===")

if __name__ == "__main__":
    run_test()
