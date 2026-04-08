from sensorfusion import *
from Emergency import *
from payloadLog_saver import *
from Azure_handle import *


if __name__ == '__main__':

    Payloads = retreive_sensor_Data()
    Currentclient = AzureAdmin_IOT()
    Currentclient.Initiate_Azure_connection_send(Payloads)
    Currentclient = AzureAdmin_DATABASE()
    Currentclient.save_to_table(Payloads)    
    EmergencyALgorithm(Payloads)
    save_logs(Payloads)
  
