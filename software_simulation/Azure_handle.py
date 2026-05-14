
from azure.iot.device import IoTHubDeviceClient, Message
from azure.data.tables import TableServiceClient
import time, uuid,json

CONNECTION_STRING = "YOUR IOT ID"
STORAGE_CONNECTION_STRING = "YOUR DB ID"
TABLE_NAME = "telemetry"

class AzureAdmin:
    def __init__(self):
        pass


class AzureAdmin_IOT(AzureAdmin):
    client  = None
    def __init__(self):
        try:
            self.client = IoTHubDeviceClient.create_from_connection_string(connection_string=CONNECTION_STRING)
            self.client.connect()
            print('connected')
        except ValueError as e:
            print(f'⚠️  Azure connection skipped: {e}')
            self.client = None

    def Initiate_Azure_connection_send(self, messages=None) -> None:
        if not messages or not self.client:
            return

        for message in messages:
            payload = json.dumps(message)
            self.client.send_message(Message(payload))
    

    



class AzureAdmin_DATABASE(AzureAdmin):

    table_service = None
    table_client =None

    def __init__(self):
        try:
            self.table_service = TableServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
            self.table_client = self.table_service.get_table_client(TABLE_NAME)
        except Exception as e:
            print(f'⚠️  Azure table storage skipped: {e}')
            self.table_service = None
            self.table_client = None

    def save_to_table(self,payload: dict):
        if not self.table_client:
            print("⚠️  Skipping database save (no connection)")
            return
            
        entity ={
        "PartitionKey": payload.get("deviceId", "rpi-01"),
        "RowKey": str(uuid.uuid4()),
        "motion": str(payload.get("PIR501", {}).get("value", "")),
        "temperature": str(payload.get("dht22", {}).get("temperature_celsius", "")),
        "humidity": str(payload.get("dht22", {}).get("humidity_percent", "")),
        "pose": payload.get("pose", ""),
        "health_status": payload.get("health_status", ""),
        "emergency": str(payload.get("emergency", False)),
        "timestamp": time.strftime("%Y-%m-%d__%H:%M", time.gmtime())
        }
        self.table_client.create_entity(entity)
        print(f"Saved: {entity}")

    def view_data(self):
        if not self.table_client:
            print("No table client available")
            return
            
        entities = self.table_client.list_entities()
        for entity in entities:
            print(entity)


def parse_payload(payload: dict, health_status: str, emergency_flag: bool, stroke_risk: dict) -> dict:
    payload['health_status'] = health_status
    payload['emergency'] = emergency_flag
    payload['stroke_risk'] = stroke_risk
    return payload


