import unittest
import sys
import os

# Add the software_simulation directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Emergency import EmergencyALgorithm

class TestEmergency(unittest.TestCase):
    def test_emergency_normal_conditions(self):
        payloads = [
            {
                'dht22': {'temperature_celsius': 22, 'humidity_percent': 50},
                'PIR501': {'value': 1},
                'pose': 'SITTING'
            }
        ]
        result = EmergencyALgorithm(payloads)
        self.assertFalse(result)

    def test_emergency_high_temperature(self):
        payloads = [
            {
                'dht22': {'temperature_celsius': 30, 'humidity_percent': 50},  # High temp
                'PIR501': {'value': 1},
                'pose': 'SITTING'
            }
        ]
        result = EmergencyALgorithm(payloads)
        self.assertTrue(result)

    def test_emergency_low_temperature(self):
        payloads = [
            {
                'dht22': {'temperature_celsius': 10, 'humidity_percent': 50},  # Low temp
                'PIR501': {'value': 1},
                'pose': 'SITTING'
            }
        ]
        result = EmergencyALgorithm(payloads)
        self.assertTrue(result)

    def test_emergency_high_humidity(self):
        payloads = [
            {
                'dht22': {'temperature_celsius': 22, 'humidity_percent': 70},  # High humidity
                'PIR501': {'value': 1},
                'pose': 'SITTING'
            }
        ]
        result = EmergencyALgorithm(payloads)
        self.assertTrue(result)

    def test_emergency_low_humidity(self):
        payloads = [
            {
                'dht22': {'temperature_celsius': 22, 'humidity_percent': 30},  # Low humidity
                'PIR501': {'value': 1},
                'pose': 'SITTING'
            }
        ]
        result = EmergencyALgorithm(payloads)
        self.assertTrue(result)

    def test_emergency_fall_detected(self):
        payloads = [
            {
                'dht22': {'temperature_celsius': 22, 'humidity_percent': 50},
                'PIR501': {'value': 0},  # No motion
                'pose': 'LAYING ON THE FLOOR'  # Fall pose
            }
        ]
        result = EmergencyALgorithm(payloads)
        self.assertTrue(result)

    def test_emergency_no_movement(self):
        payloads = [
            {
                'dht22': {'temperature_celsius': 22, 'humidity_percent': 50},
                'PIR501': {'value': 0},  # No motion
                'pose': 'SITTING'
            }
        ]
        result = EmergencyALgorithm(payloads)
        self.assertTrue(result)

    def test_emergency_multiple_payloads(self):
        payloads = [
            {
                'dht22': {'temperature_celsius': 22, 'humidity_percent': 50},
                'PIR501': {'value': 1},
                'pose': 'SITTING'
            },
            {
                'dht22': {'temperature_celsius': 30, 'humidity_percent': 50},  # High temp in second payload
                'PIR501': {'value': 1},
                'pose': 'STANDING'
            }
        ]
        result = EmergencyALgorithm(payloads)
        self.assertTrue(result)  # Should trigger due to high temp

    def test_emergency_no_pose_data(self):
        payloads = [
            {
                'dht22': {'temperature_celsius': 22, 'humidity_percent': 50},
                'PIR501': {'value': 1}
                # No pose data
            }
        ]
        result = EmergencyALgorithm(payloads)
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()