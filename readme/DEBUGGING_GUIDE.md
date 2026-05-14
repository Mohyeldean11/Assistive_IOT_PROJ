# Module Debugging and Troubleshooting Guide

## Quick Reference

| Issue | Module | Solution |
|-------|--------|----------|
| Camera not detected | Mediapipe_Class | Change camera index in `captureframe()` |
| Ollama not responding | AI_Module_layer | Start Ollama service, check model |
| Azure connection fails | Azure_handle | Verify connection strings |
| No stroke detection | stroke_detector | Adjust threshold values |
| Sensor data invalid | sensorfusion | Check GPIO pins and sensor wiring |

---

## Module-Specific Debugging

### 1. **stroke_detector.py** - Not Detecting Strokes

**Problem**: High-risk situations not being detected

**Troubleshooting Steps**:
```python
# Check individual detection methods
detector = StrokeDetector()
landmarks = {...}  # Your pose data

# Debug each check separately
print("Facial droop:", detector.check_facial_droop(landmarks))
print("Arm weakness:", detector.check_arm_weakness(landmarks))
print("Sudden collapse:", detector.check_sudden_collapse())
print("Pose freeze:", detector.check_pose_freeze())

# Check risk level
result = detector.get_stroke_risk(landmarks)
print("Risk level:", result['risk_level'])
print("Signs detected:", result['signs_count'])
```

**Threshold Adjustments**:
- Lower thresholds = More sensitive (more false positives)
- Higher thresholds = Less sensitive (may miss strokes)

```python
# In stroke_detector.py
ASYMMETRY_THRESHOLD = 0.03  # Decrease to 0.02 for more sensitivity
ARM_DROP_THRESHOLD = 0.05   # Decrease to 0.03 for more sensitivity
COLLAPSE_THRESHOLD = 0.10   # Decrease to 0.07 for more sensitivity
```

---

### 2. **AI_Module_layer.py** - AI Not Responding

**Problem**: Ollama timeouts or invalid responses

**Solution 1**: Verify Ollama is running
```bash
# Check if Ollama service is active
ollama list  # Lists available models

# If not running, start it
ollama serve  # In terminal/background
```

**Solution 2**: Verify model is loaded
```bash
ollama pull llama3.2  # Download model if missing
ollama run llama3.2   # Test model directly
```

**Solution 3**: Debug prompt response
```python
ai_layer = AI_LAYER()
prompt = ai_layer.Prompt_builder()
print("Prompt:", prompt)

# Test Ollama directly
import ollama
response = ollama.generate(model="llama3.2", prompt="respond with one word: SITTING or STANDING")
print("Response:", response.response.strip())
```

**Solution 4**: Handle non-word responses
```python
# Add validation to Get_Elder_Pose()
response = ollama.generate(model=self.model, prompt=prompt)
pose = response.response.strip().upper()

# Only accept valid poses
if pose not in poses:
    pose = self.old_pose  # Fall back to previous pose
```

---

### 3. **Mediapipe_Class.py** - Camera Issues

**Problem**: Camera not accessible or returning empty landmarks

**Solution 1**: Check camera index
```python
# Try different camera indices
for i in range(5):
    try:
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera {i} is available")
            cap.release()
    except:
        pass
```

**Solution 2**: Verify camera permissions
```bash
# Linux/Mac: Check permissions
ls -la /dev/video*

# Windows: Check device manager
# Device Manager > Imaging devices > Check camera status
```

**Solution 3**: Debug landmark extraction
```python
processor = PoseProcessor()
frame_rgb = processor.captureframe()
pose_data = processor.processframe()

if not pose_data:
    print("No person detected!")
else:
    print(f"Detected {len(pose_data)} landmarks")
    for name, coords in pose_data.items():
        if coords['presence'] < 0.5:
            print(f"Low confidence: {name}")
```

---

### 4. **sensorfusion.py** - Sensor Data Problems

**Problem**: Invalid or missing sensor readings

**Debug Sensor Data**:
```python
# Test individual sensors
pir_data = read_PIR501()
print("PIR Data:", pir_data)

dht_data = read_dht22()
print("DHT22 Data:", dht_data)

# Verify data ranges
temp = dht_data['temperature_celsius']
humidity = dht_data['humidity_percent']
motion = pir_data['value']

assert 18 <= temp <= 28, f"Temp out of range: {temp}"
assert 20 <= humidity <= 80, f"Humidity out of range: {humidity}"
assert motion in [0, 1], f"Invalid motion value: {motion}"
```

**GPIO Configuration**:
```python
# In sensorfusion.py, verify GPIO pins
pir_output = {
    'value': 1,
    'gpio_pin': 3,  # Verify this matches your hardware
}

dht22_raw = {
    'gpio_pin': 4,  # Verify this matches your hardware
}
```

---

### 5. **Emergency.py** - False Alarms

**Problem**: Too many false emergency alerts

**Calibration**:
```python
# Adjust temperature range (default: 18-24°C)
if not (16 <= temp <= 26):  # Wider range = fewer false alerts

# Adjust humidity range (default: 40-60%)
if not (30 <= humidity <= 70):  # Wider range = fewer false alerts

# Add motion threshold history
# Only alert if no motion for N consecutive readings
```

**Debug Emergency Detection**:
```python
from Emergency import EmergencyALgorithm

payloads = [{
    'dht22': {'temperature_celsius': 22, 'humidity_percent': 50},
    'PIR501': {'value': 1},
    'pose': 'SITTING'
}]

result = EmergencyALgorithm(payloads)
print("Emergency triggered:", result)
```

---

### 6. **Azure_handle.py** - Cloud Connection Issues

**Problem**: Cannot connect to Azure

**Solution 1**: Verify connection strings
```python
# In Azure_handle.py
CONNECTION_STRING = "your_iot_hub_connection_string"
STORAGE_CONNECTION_STRING = "your_storage_connection_string"

# Test connection
try:
    admin = AzureAdmin_IOT()
    if admin.client is None:
        print("Failed to connect to IoT Hub")
except Exception as e:
    print(f"Connection error: {e}")
```

**Solution 2**: Check network connectivity
```bash
# Ping Azure endpoints
ping *.azure-devices.net
ping *.table.core.windows.net
```

**Solution 3**: Debug payload structure
```python
payload = {
    'deviceId': 'rpi-01',
    'dht22': {'temperature_celsius': 25, 'humidity_percent': 50},
    'PIR501': {'value': 1},
    'pose': 'SITTING',
    'health_status': 'Normal',
    'emergency': False,
    'stroke_risk': {'risk_level': 'LOW'},
    'timestamp': '2023-01-01__12:00'
}

# Verify JSON serializable
import json
json_str = json.dumps(payload)
print("Valid JSON:", json_str)
```

---

### 7. **payloadLog_saver.py** - Logging Issues

**Problem**: Logs not being saved

**Debug**:
```python
import os
from payloadLog_saver import log_dir, target_file, save_logs

print("Log directory:", log_dir)
print("Target file:", target_file)
print("Directory exists:", os.path.exists(log_dir))

# Test saving
result = save_logs({'test': 'data'})
print("Save result:", result)
print("File exists:", os.path.exists(target_file))
```

---

## Common Error Messages

### Error: "cannot access attribute 'X' for class 'Y'"
**Cause**: Method not found in class  
**Fix**: Check method name spelling, verify inheritance

### Error: "TypeError: path should be string, bytes, os.PathLike or integer"
**Cause**: None passed as file path  
**Fix**: Verify mock setup in tests, check variable initialization

### Error: "connection string is invalid"
**Cause**: Malformed Azure connection string  
**Fix**: Re-copy connection string from Azure portal, ensure no extra quotes

### Error: "No module named 'ollama'"
**Cause**: Ollama not installed  
**Fix**: `pip install ollama`

### Error: "Cannot find pose_landmarker_lite.task"
**Cause**: Model file not in correct path  
**Fix**: Verify `Helper_Models/` directory exists with model file

---

## Performance Optimization

### Slow Pose Detection
```python
# Reduce frame resolution
frame = cv2.resize(frame, (640, 480))  # Instead of full resolution

# Process every Nth frame
frame_count = 0
if frame_count % 3 == 0:  # Process every 3rd frame
    result = process_frame()
frame_count += 1
```

### Reduce Memory Usage
```python
# Clear old pose history more aggressively
if len(self.pose_history) > 3:
    self.pose_history = self.pose_history[-3:]

# Don't store full frame data
# Store only landmark coordinates, not images
```

---

## Testing Checklist

Before deployment, verify:

- [ ] All 46 unit tests pass
- [ ] Ollama service is running with llama3.2
- [ ] Camera is accessible and working
- [ ] Azure connection strings are valid
- [ ] GPIO pins are correctly configured (if using real sensors)
- [ ] Log directory has write permissions
- [ ] No hardcoded test paths in production code

---

## Additional Resources

### Run Tests
```bash
cd software_simulation
python -m unittest discover -p "test_*.py" -v
```

### Check Code Quality
```bash
# Install linting tools
pip install pylint flake8

# Check code style
pylint *.py
flake8 *.py
```

### Monitor Logs
```bash
# Watch log directory for new files
cd logs
ls -lah *.json | tail -n 5
```

---

**Last Updated**: May 13, 2026  
**For Issues**: Check module docstrings and test files for examples