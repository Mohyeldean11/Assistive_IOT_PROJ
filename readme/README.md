# Stroke Detection and Elderly Monitoring System

This system provides real-time monitoring and stroke detection for elderly individuals using computer vision, sensor data, and AI analysis.

## System Architecture

The system consists of several modules that work together to collect sensor data, analyze pose for stroke indicators, and send alerts to Azure cloud services.

## Modules Overview

### 1. `sensorfusion.py` - Sensor Data Collection
**Purpose**: Collects environmental and motion sensor data from DHT22 and PIR sensors.

**Key Functions**:
- `init_sensors()`: Initialize all sensors
- `read_PIR501()`: Read motion detection data
- `read_dht22()`: Read temperature and humidity data
- `retreive_sensor_Data()`: Collect multiple sensor readings
- `fuse_data_with_pose()`: Combine sensor data with pose information

**Usage Steps**:
1. Call `init_sensors()` to initialize hardware
2. Use `retreive_sensor_Data()` to collect sensor readings
3. Data includes temperature, humidity, and motion detection

### 2. `Mediapipe_Class.py` - Pose Detection
**Purpose**: Captures video frames and extracts body pose landmarks using MediaPipe.

**Key Classes**:
- `PoseProcessor`: Handles camera capture and pose detection
- `frame_classifier`: Processes and categorizes body parts

**Usage Steps**:
1. Initialize `PoseProcessor` with model path
2. Call `processframe()` to capture and analyze pose
3. Use `frame_classifier` to get face, body, or whole person landmarks
4. Landmarks include x, y, z coordinates and presence confidence

### 3. `AI_Module_layer.py` - AI Analysis Layer
**Purpose**: Uses Ollama AI to analyze pose and health status.

**Key Features**:
- Pose classification (SITTING, STANDING, LAYING, etc.)
- Health status assessment from sensor data
- Stroke risk evaluation using rule-based detection

**Usage Steps**:
1. Initialize `AI_LAYER` with model name
2. Call `Get_Elder_Pose()` for pose analysis
3. Call `Get_Elder_status()` with sensor data for health assessment
4. Call `Get_Stroke_Risk()` for stroke detection analysis

### 4. `stroke_detector.py` - Stroke Detection Algorithm
**Purpose**: Implements rule-based stroke detection using pose landmarks.

**Detection Criteria**:
- Facial droop (mouth/eye asymmetry)
- Arm weakness (uneven arm positions)
- Sudden collapse (rapid position changes)
- Pose freeze (lack of movement)
- Gradual deterioration (slow worsening)

**Usage Steps**:
1. Initialize `StrokeDetector`
2. Call `update()` with pose landmarks for each frame
3. Call `get_stroke_risk()` to get risk assessment
4. Returns risk level (LOW/MEDIUM/HIGH) and individual signs

### 5. `Emergency.py` - Emergency Detection
**Purpose**: Analyzes sensor and pose data to detect emergency situations.

**Emergency Triggers**:
- Abnormal temperature (outside 18-24°C)
- Abnormal humidity (outside 40-60%)
- Fall detection (laying on floor without motion)
- No movement detected

**Usage Steps**:
1. Call `EmergencyALgorithm()` with payload data
2. Returns boolean indicating emergency status

### 6. `Azure_handle.py` - Cloud Integration
**Purpose**: Handles communication with Azure IoT Hub and Table Storage.

**Key Classes**:
- `AzureAdmin_IOT`: Sends telemetry to IoT Hub
- `AzureAdmin_DATABASE`: Stores data in Table Storage

**Usage Steps**:
1. Initialize `AzureAdmin_IOT()` and `AzureAdmin_DATABASE()`
2. Use `Initiate_Azure_connection_send()` to send messages
3. Use `save_to_table()` to store data in Azure Tables
4. Call `parse_payloads()` to prepare data for Azure

### 7. `payloadLog_saver.py` - Local Logging
**Purpose**: Saves payload data to local JSON files for backup and analysis.

**Key Functions**:
- `save_logs()`: Append payload data to timestamped log files

**Usage Steps**:
1. Call `save_logs()` with payload data
2. Logs are saved in `logs/` directory with timestamps

### 8. `main.py` - Main Application
**Purpose**: Orchestrates all modules to run the complete monitoring system.

**Execution Flow**:
1. Retrieve sensor data
2. Initialize AI layer
3. Analyze pose and health status
4. Check for emergencies
5. Send data to Azure
6. Save local logs

## Installation and Setup

### Prerequisites
- Python 3.8+
- MediaPipe
- Azure IoT SDK
- Ollama with llama3.2 model
- Camera access
- DHT22 and PIR sensors (simulated in code)

### Installation Steps
1. Install Python dependencies:
   ```bash
   pip install mediapipe azure-iot-device azure-data-tables ollama
   ```

2. Set up Azure connection strings in `Azure_handle.py`

3. Ensure camera is accessible (camera index 1)

4. Start Ollama service with llama3.2 model

### Running the System
1. Run the main application:
   ```bash
   python main.py
   ```

2. The system will:
   - Collect sensor data
   - Analyze pose from camera
   - Assess health and stroke risk
   - Send data to Azure
   - Save local logs

## Testing

Run unit tests for individual modules:
```bash
python -m unittest test_*.py
```

### Test Coverage
- `test_AI_Module_layer.py`: AI analysis functions
- `test_stroke_detector.py`: Stroke detection algorithms
- `test_Mediapipe_Class.py`: Pose detection
- `test_sensorfusion.py`: Sensor data collection
- `test_payloadLog_saver.py`: Logging functionality
- `test_Emergency.py`: Emergency detection
- `test_Azure_handle.py`: Cloud integration

## Configuration

### Thresholds (in `stroke_detector.py`)
- `ASYMMETRY_THRESHOLD`: Facial asymmetry sensitivity (default: 0.03)
- `ARM_DROP_THRESHOLD`: Arm weakness sensitivity (default: 0.05)
- `COLLAPSE_THRESHOLD`: Fall detection sensitivity (default: 0.10)
- `FREEZE_THRESHOLD`: Movement detection sensitivity (default: 0.002)

### AI Prompts (in `AI_Module_layer.py`)
- Customize prompts for different analysis requirements
- Adjust model parameters for performance vs accuracy

## Troubleshooting

### Common Issues
1. **Camera not accessible**: Check camera index and permissions
2. **Azure connection failed**: Verify connection strings
3. **Ollama not responding**: Ensure Ollama service is running
4. **Sensor data invalid**: Check sensor initialization

### Debug Mode
- Enable print statements in modules for debugging
- Check local log files in `logs/` directory
- Monitor Azure telemetry for data flow

## Future Enhancements

- Machine learning model for improved stroke detection
- Real-time video streaming
- Mobile app integration
- Multi-camera support
- Advanced temporal analysis
- Integration with emergency services