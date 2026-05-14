import unittest
from unittest.mock import patch
import sys
import os

# Add the software_simulation directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sensorfusion import (
    init_sensors, read_PIR501, read_dht22, retreive_sensor_Data,
    fuse_data_with_pose, PayloadGroup, CurrentPAYLOAD
)

class TestSensorFusion(unittest.TestCase):
    def setUp(self):
        # Clear global state between tests
        PayloadGroup.clear()
        CurrentPAYLOAD.clear()

    @patch('sensorfusion.Init_env_sensor')
    @patch('sensorfusion.Init_PIR_sensor')
    @patch('sensorfusion.Init_BUTTON_MECHANISM')
    def test_init_sensors_success(self, mock_button, mock_pir, mock_env):
        result = init_sensors()
        self.assertTrue(result)
        mock_env.assert_called_once()
        mock_pir.assert_called_once()
        mock_button.assert_called_once()

    @patch('sensorfusion.Init_env_sensor')
    def test_init_sensors_failure(self, mock_env):
        mock_env.side_effect = Exception("Init failed")
        result = init_sensors()
        self.assertFalse(result)

    @patch('sensorfusion.time.asctime')
    @patch('sensorfusion.random.randrange')
    def test_read_PIR501(self, mock_randrange, mock_asctime):
        mock_asctime.return_value = 'Mon Jan 1 00:00:00 2023'
        mock_randrange.return_value = 1

        result = read_PIR501()
        expected = {
            'value': 1,
            'gpio_pin': 3,
            'timestamp': 'Mon Jan 1 00:00:00 2023'
        }
        self.assertEqual(result, expected)

    @patch('sensorfusion.time.asctime')
    @patch('sensorfusion.random.randrange')
    def test_read_dht22(self, mock_randrange, mock_asctime):
        mock_asctime.return_value = 'Mon Jan 1 00:00:00 2023'
        mock_randrange.side_effect = [25, 60]  # temperature, humidity

        result = read_dht22()
        expected = {
            'gpio_pin': 4,
            'timestamp': 'Mon Jan 1 00:00:00 2023',
            'temperature_celsius': 25,
            'humidity_percent': 60
        }
        self.assertEqual(result, expected)

    @patch('sensorfusion.time.sleep')
    @patch('sensorfusion.read_dht22')
    @patch('sensorfusion.read_PIR501')
    def test_retreive_sensor_Data(self, mock_pir, mock_dht, mock_sleep):
        mock_dht.return_value = {'temperature_celsius': 25, 'humidity_percent': 60}
        mock_pir.return_value = {'value': 1, 'gpio_pin': 3}

        result = retreive_sensor_Data()
        self.assertEqual(len(result), 3)
        for payload in result:
            self.assertEqual(payload['deviceId'], 'rpi-01')
            self.assertIn('dht22', payload)
            self.assertIn('PIR501', payload)

    def test_fuse_data_with_pose(self):
        payloads = [
            {'deviceId': 'rpi-01', 'dht22': {'temp': 25}},
            {'deviceId': 'rpi-02', 'dht22': {'temp': 26}}
        ]
        pose = 'SITTING'
        result = fuse_data_with_pose(payloads, pose)

        self.assertEqual(len(result), 2)
        for payload in result:
            self.assertEqual(payload['pose'], 'SITTING')

if __name__ == '__main__':
    unittest.main()