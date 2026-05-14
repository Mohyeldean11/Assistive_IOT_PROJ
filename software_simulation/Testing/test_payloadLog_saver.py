import unittest
import json
import os
import tempfile
from unittest.mock import patch, MagicMock
import sys

# Add the software_simulation directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from payloadLog_saver import save_logs

class TestPayloadLogSaver(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_log_file = os.path.join(self.temp_dir, 'test_logs.json')

    def tearDown(self):
        # Clean up temporary files
        if os.path.exists(self.test_log_file):
            os.remove(self.test_log_file)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_save_logs_new_file(self):
        with patch('payloadLog_saver.target_file', self.test_log_file):
            payload = {'deviceId': 'rpi-01', 'temperature': 25}
            result = save_logs(payload)
            self.assertTrue(result)

            # Check file was created and contains data
            self.assertTrue(os.path.exists(self.test_log_file))
            with open(self.test_log_file, 'r') as f:
                data = json.load(f)
                self.assertEqual(data, [payload])

    def test_save_logs_append_to_existing(self):
        with patch('payloadLog_saver.target_file', self.test_log_file):
            # Create existing file with data
            existing_data = [{'deviceId': 'rpi-01', 'temperature': 24}]
            with open(self.test_log_file, 'w') as f:
                json.dump(existing_data, f)

            # Add new payload
            new_payload = {'deviceId': 'rpi-01', 'temperature': 25}
            result = save_logs(new_payload)
            self.assertTrue(result)

            # Check data was appended
            with open(self.test_log_file, 'r') as f:
                data = json.load(f)
                self.assertEqual(len(data), 2)
                self.assertEqual(data[0], existing_data[0])
                self.assertEqual(data[1], new_payload)

    def test_save_logs_list_payload(self):
        with patch('payloadLog_saver.target_file', self.test_log_file):
            payloads = [
                {'deviceId': 'rpi-01', 'temperature': 25},
                {'deviceId': 'rpi-02', 'temperature': 26}
            ]
            result = save_logs(payloads)
            self.assertTrue(result)

            # Check all payloads were saved
            with open(self.test_log_file, 'r') as f:
                data = json.load(f)
                self.assertEqual(data, payloads)

    def test_save_logs_creates_directory(self):
        with patch('payloadLog_saver.target_file', self.test_log_file):
            with patch('payloadLog_saver.os.makedirs') as mock_makedirs:
                payload = {'deviceId': 'rpi-01', 'temperature': 25}
                result = save_logs(payload)
                self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()