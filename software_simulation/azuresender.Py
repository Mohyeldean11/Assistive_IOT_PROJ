
from azure.iot.device import IoTHubDeviceClient, Message
from azure.data.tables import TableServiceClient
import time, uuid

CONNECTION_STRING = "HostName=elderlyMonitor-trial.azure-devices.net;DeviceId=rpi-01;SharedAccessKey=OOJ+dOV4DERHdfZFdedCeFLtJxB3R7yc4s1j3syBJQY="
STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=assistivestorage01;AccountKey=mubMA9lZOYyaHPJIWsAUZsW+6LErrm1GByblaRxEOvVhcjifwdMYiHQ8bErOVsJJw4r57HWX/WTp+ASt/ZwtWA==;EndpointSuffix=core.windows.net"
TABLE_NAME = "telemetry"

table_service = TableServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
table_client = table_service.get_table_client(TABLE_NAME)
client = IoTHubDeviceClient.create_from_connection_string(connection_string=CONNECTION_STRING)


def Initiate_Azure_connection_send(message:str ="None")->None:
    client.connect()
    print("done azure")
    client.send_message(message)



def save_to_table(payload: dict):
    entity = {
        "PartitionKey": payload.get("deviceId", "rpi-01"),
        "RowKey": str(uuid.uuid4()),
        "motion": str(payload.get("PIR501", {}).get("value", "")),
        "temperature": str(payload.get("dht22", {}).get("temperature_celsius", "")),
        "humidity": str(payload.get("dht22", {}).get("humidity_percent", "")),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    table_client.create_entity(entity)
    print(f"Saved: {entity}")


def view_data():
    entities = table_client.list_entities()
    for entity in entities:
        print(entity)