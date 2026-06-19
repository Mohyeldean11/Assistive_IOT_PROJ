"""Example Raspberry Pi sender. Replace sample_payload() with your sensor/CV output."""
import os, time, requests
from datetime import datetime

SERVER_URL = os.getenv("MONITOR_URL", "http://127.0.0.1:5000/api/readings")
API_KEY = os.getenv("DEVICE_API_KEY", "replace-with-a-device-api-key")

def sample_payload():
    now = datetime.now().strftime("%a %b %d %H:%M:%S %Y")
    return {"deviceId":"rpi-01","dht22":{"gpio_pin":4,"timestamp":now,"temperature_celsius":23.4,"humidity_percent":55},"PIR501":{"value":1,"gpio_pin":3,"timestamp":now},"pose":"STANDING","health_status":"Normal","emergency":False,"stroke_risk":{"risk_level":"LOW","facial_droop":False,"arm_weakness":False,"sudden_collapse":False,"pose_freeze":False,"gradual_deterioration":False,"signs_count":0}}

while True:
    try:
        response = requests.post(SERVER_URL, json=sample_payload(), headers={"X-API-Key": API_KEY}, timeout=15)
        print(response.status_code, response.text)
    except requests.RequestException as exc:
        print("Upload failed:", exc)
    time.sleep(120)
