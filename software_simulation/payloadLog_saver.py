import json
import time
import os

nowtime = time.strftime("%Y_%m_%d_%H_%M", time.gmtime())
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)
target_file = os.path.join(log_dir, f"{nowtime}_logs.json")


def save_logs(payloads) -> bool:
    existing = []
    
    if os.path.exists(target_file):
        with open(target_file, "r", encoding="utf-8") as fs:
            try:
                existing = json.load(fs)
            except json.JSONDecodeError:
                existing = []
    
    if isinstance(payloads, list):
        existing.extend(payloads)
    else:
        existing.append(payloads)

    with open(target_file, "w", encoding="utf-8") as fs:
        json.dump(existing, fs, indent=2)
    
    return True






