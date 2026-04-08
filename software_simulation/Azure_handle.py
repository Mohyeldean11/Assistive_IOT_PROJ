
from azure.iot.device import IoTHubDeviceClient, Message
from azure.data.tables import TableServiceClient
import time, uuid,json

CONNECTION_STRING = "keys are removed for now"
STORAGE_CONNECTION_STRING = "keys are removed for now"
TABLE_NAME = "telemetry"

class AzureAdmin:
    def __init__(self):
        pass


class AzureAdmin_IOT(AzureAdmin):
    client  = None
    def __init__(self):
        self.client = IoTHubDeviceClient.create_from_connection_string(connection_string=CONNECTION_STRING)
        self.client.connect()
        print('connected')

    def Initiate_Azure_connection_send(self, messages=None) -> None:
        if not messages:
            return

        for message in messages:
            payload = json.dumps(message)
            self.client.send_message(Message(payload))
    

    



class AzureAdmin_DATABASE(AzureAdmin):

    table_service = None
    table_client =None

    def __init__(self):
       self.table_service = TableServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
       self.table_client = self.table_service.get_table_client(TABLE_NAME)

    def save_to_table(self,payloadgp: list):
        for payload in payloadgp:          
            entity ={
            "PartitionKey": payload.get("deviceId", "rpi-01"),
            "RowKey": str(uuid.uuid4()),
            "motion": str(payload.get("PIR501", {}).get("value", "")),
            "temperature": str(payload.get("dht22", {}).get("temperature_celsius", "")),
            "humidity": str(payload.get("dht22", {}).get("humidity_percent", "")),
            "timestamp": time.strftime("%Y-%m-%d__%H:%M", time.gmtime())
        }
            self.table_client.create_entity(entity)
            print(f"Saved: {entity}")

    def view_data(self):
        entities = self.table_client.list_entities()
        for entity in entities:
            print(entity)


