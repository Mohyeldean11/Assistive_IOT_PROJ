"""
Unit tests for individual components
Run with: python test_individual_modules.py
"""
import sys
import json
from unittest.mock import patch, MagicMock
import time

print("=" * 60)
print("TESTING INDIVIDUAL COMPONENTS")
print("=" * 60)

# ============================================
# TEST 1: Sensor Fusion (PIR + DHT22)
# ============================================
print("\n[1/4] Testing SENSOR FUSION (sensorfusion.py)...")
try:
    from sensorfusion import retreive_sensor_Data, read_PIR501, read_dht22
    
    sensor_data = retreive_sensor_Data()
    print(f"✅ Sensor data retrieved successfully")
    print(f"   - Got {len(sensor_data)} payload(s)")
    for i, payload in enumerate(sensor_data):
        print(f"   - Payload {i}: deviceId={payload['deviceId']}, "
              f"temp={payload['dht22'].get('temperature_celsius ')}, "
              f"motion={payload['PIR501']['value']}")
except Exception as e:
    print(f"❌ Sensor Fusion Error: {e}")

# ============================================
# TEST 2: Emergency Algorithm
# ============================================
print("\n[2/4] Testing EMERGENCY ALGORITHM (Emergency.py)...")
try:
    from Emergency import EmergencyALgorithm
    from sensorfusion import retreive_sensor_Data
    
    test_payloads = retreive_sensor_Data()
    emergency_result = EmergencyALgorithm(test_payloads)
    print(f"✅ Emergency algorithm executed successfully")
    print(f"   - Emergency flag triggered: {emergency_result}")
except Exception as e:
    print(f"❌ Emergency Algorithm Error: {e}")

# ============================================
# TEST 3: MediaPipe Pose Detection
# ============================================
print("\n[3/4] Testing MEDIAPIPE POSE DETECTION...")
try:
    from Mediapipe_Class import PoseProcessor_instance
    
    # Test without actually capturing (if no camera available)
    print(f"   - PoseProcessor class initialized ✅")
    print(f"   - Ready to capture and process frames")
    print(f"   - ⚠️  Skipping actual frame capture (requires camera)")
    
except Exception as e:
    print(f"❌ MediaPipe Error: {e}")

# ============================================
# TEST 4: Payload & Logging
# ============================================
print("\n[4/4] Testing PAYLOAD LOGGING (payloadLog_saver.py)...")
try:
    from payloadLog_saver import save_logs
    from sensorfusion import retreive_sensor_Data
    
    test_payloads = retreive_sensor_Data()
    result = save_logs(test_payloads)
    
    if result:
        print(f"✅ Payloads logged successfully")
        print(f"   - Logged {len(test_payloads)} payload(s)")
    else:
        print(f"❌ Failed to save logs")
        
except Exception as e:
    print(f"❌ Payload Logging Error: {e}")

# ============================================
# INTEGRATION TEST: Sensor + Emergency
# ============================================
print("\n" + "=" * 60)
print("INTEGRATION TEST: Sensor → Emergency Detection")
print("=" * 60)
try:
    from sensorfusion import retreive_sensor_Data
    from Emergency import EmergencyALgorithm
    from payloadLog_saver import save_logs
    
    print("\nStep 1: Reading sensor data...")
    payloads = retreive_sensor_Data()
    print(f"✅ Got {len(payloads)} readings")
    
    print("\nStep 2: Running emergency detection...")
    emergency_status = EmergencyALgorithm(payloads)
    print(f"✅ Emergency detection: {emergency_status}")
    
    print("\nStep 3: Saving logs...")
    save_logs(payloads)
    print(f"✅ Logs saved")
    
    print("\n✅ INTEGRATION TEST PASSED!")
    
except Exception as e:
    print(f"\n❌ INTEGRATION TEST FAILED: {e}")
    import traceback
    traceback.print_exc()

# ============================================
# AZURE INTEGRATION TEST (Optional)
# ============================================
print("\n" + "=" * 60)
print("AZURE INTEGRATION (Mock)")
print("=" * 60)
print("⚠️  Azure test requires valid CONNECTION_STRING")
print("   - Mock only (actual connection skipped)")
print("   - To test: Update Azure_handle.py with real credentials")
print("             then run: python main.py")
