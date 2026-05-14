import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the software_simulation directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Azure_handle import AzureAdmin_IOT, AzureAdmin_DATABASE, parse_payloads

class TestAzureAdminIOT(unittest.TestCase):
    @patch('Azure_handle.IoTHubDeviceClient.create_from_connection_string')
    def test_init_success(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client

        admin = AzureAdmin_IOT()
        self.assertEqual(admin.client, mock_client)
        mock_client.connect.assert_called_once()

    @patch('Azure_handle.IoTHubDeviceClient.create_from_connection_string')
    def test_init_failure(self, mock_create):
        mock_create.side_effect = ValueError("Invalid connection string")

        admin = AzureAdmin_IOT()
        self.assertIsNone(admin.client)

    @patch('Azure_handle.IoTHubDeviceClient.create_from_connection_string')
    def test_send_messages(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client

        admin = AzureAdmin_IOT()
        messages = [
            {'deviceId': 'rpi-01', 'temperature': 25},
            {'deviceId': 'rpi-02', 'temperature': 26}
        ]

        admin.Initiate_Azure_connection_send(messages)
        self.assertEqual(mock_client.send_message.call_count, 2)

    @patch('Azure_handle.IoTHubDeviceClient.create_from_connection_string')
    def test_send_messages_no_client(self, mock_create):
        mock_create.side_effect = ValueError("Invalid connection string")

        admin = AzureAdmin_IOT()
        messages = [{'deviceId': 'rpi-01', 'temperature': 25}]

        # Should not raise exception
        admin.Initiate_Azure_connection_send(messages)

    @patch('Azure_handle.IoTHubDeviceClient.create_from_connection_string')
    def test_send_messages_no_messages(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client

        admin = AzureAdmin_IOT()

        # Should not raise exception
        admin.Initiate_Azure_connection_send(None)
        admin.Initiate_Azure_connection_send([])

class TestAzureAdminDatabase(unittest.TestCase):
    @patch('Azure_handle.TableServiceClient.from_connection_string')
    def test_init_success(self, mock_create):
        mock_service = MagicMock()
        mock_table_client = MagicMock()
        mock_service.get_table_client.return_value = mock_table_client
        mock_create.return_value = mock_service

        admin = AzureAdmin_DATABASE()
        self.assertEqual(admin.table_service, mock_service)
        self.assertEqual(admin.table_client, mock_table_client)

    @patch('Azure_handle.TableServiceClient.from_connection_string')
    def test_init_failure(self, mock_create):
        mock_create.side_effect = Exception("Invalid connection string")

        admin = AzureAdmin_DATABASE()
        self.assertIsNone(admin.table_service)
        self.assertIsNone(admin.table_client)

    @patch('Azure_handle.TableServiceClient.from_connection_string')
    @patch('Azure_handle.uuid.uuid4')
    @patch('Azure_handle.time.strftime')
    def test_save_to_table(self, mock_strftime, mock_uuid, mock_create):
        mock_service = MagicMock()
        mock_table_client = MagicMock()
        mock_service.get_table_client.return_value = mock_table_client
        mock_create.return_value = mock_service

        mock_uuid.return_value = 'test-uuid'
        mock_strftime.return_value = '2023-01-01__12:00'

        admin = AzureAdmin_DATABASE()
        payloads = [
            {
                'deviceId': 'rpi-01',
                'PIR501': {'value': 1},
                'dht22': {'temperature_celsius': 25, 'humidity_percent': 50},
                'pose': 'SITTING',
                'health_status': 'Normal',
                'emergency': False,
                'stroke_risk': {'risk_level': 'LOW'}
            }
        ]

        admin.save_to_table(payloads)
        mock_table_client.create_entity.assert_called_once()

    @patch('Azure_handle.TableServiceClient.from_connection_string')
    def test_save_to_table_no_client(self, mock_create):
        mock_create.side_effect = Exception("Invalid connection string")

        admin = AzureAdmin_DATABASE()
        payloads = [{'deviceId': 'rpi-01'}]

        # Should not raise exception
        admin.save_to_table(payloads)

class TestParsePayloads(unittest.TestCase):
    def test_parse_payloads(self):
        payloads = [
            {'deviceId': 'rpi-01', 'temperature': 25},
            {'deviceId': 'rpi-02', 'temperature': 26}
        ]
        health_status = 'Normal'
        emergency_flag = False
        stroke_risk = {'risk_level': 'LOW', 'facial_droop': False}

        result = parse_payloads(payloads, health_status, emergency_flag, stroke_risk)

        self.assertEqual(len(result), 2)
        for payload in result:
            self.assertEqual(payload['health_status'], 'Normal')
            self.assertEqual(payload['emergency'], False)
            self.assertEqual(payload['stroke_risk'], stroke_risk)

if __name__ == '__main__':
    unittest.main()